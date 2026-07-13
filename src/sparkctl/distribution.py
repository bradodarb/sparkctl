"""Model distribution: verified HF downloads on the head + rsync mirroring over the fabric.

Every shard is sha256-verified (blob name == content hash) before it can ever reach a serve;
corrupt shards are deleted and re-fetched. See docs/backends.md for the rationale."""
import json
import sys
import time

from sparkctl import config, remote
from sparkctl.engines.ollama import ollama_ensure
from sparkctl.recipes import svc_provider


def dl_cname(model):
    return f"{config.PFX}-dl-{model.replace('/', '_')}"


def _dl_env():
    """Research-backed robustness env for HF downloads: the modern hf-xet backend has known stalls,
    and the plain downloader honors HF_HUB_DOWNLOAD_TIMEOUT (a hung read -> retryable TimeoutError).
    Disable xet by default + set finite timeouts. Pure/testable."""
    dl = config.DL
    e = [f"-e HF_HUB_DOWNLOAD_TIMEOUT={dl.get('request_timeout', 60)}",
         f"-e HF_HUB_ETAG_TIMEOUT={dl.get('etag_timeout', 30)}"]
    if not dl.get("use_xet", False):
        e.append("-e HF_HUB_DISABLE_XET=1")
    return " ".join(e)


def hf_download_start(model, cache=None):
    """Start a detached, resumable, authenticated HF download container on HEAD. Returns its name.
    Runs as the invoking user (cache stays user-owned for the fabric mirror). Sources the HF token
    if present (non-interactive shells don't read ~/.bashrc) and passes it via `-e HF_TOKEN` (name
    only — value never lands on the command line/ps/logs). Robustness env from _dl_env().
    `cache` overrides the node cache — e.g. a NAS mount, so downloads land there directly."""
    name = dl_cname(model)
    inner = ('[ -f "$HOME/.hugging_face.sh" ] && . "$HOME/.hugging_face.sh"; '
             f'docker rm -f {name} >/dev/null 2>&1 || true; '
             f'docker run -d --name {name} --user "$(id -u):$(id -g)" '
             f'-e HOME=/cache -e HF_HOME=/cache -e XDG_CACHE_HOME=/cache '
             f'${{HF_TOKEN:+-e HF_TOKEN}} {_dl_env()} -v {cache or config.CACHE}:/cache {config.IMAGE} '
             f'hf download {model} >/dev/null')
    remote.on(config.HEAD, f"bash -lc {json.dumps(inner)}")
    return name


def _wait_with_watchdog(name, model, cache=None):
    """Supervise a running download: return 'exited' when the container stops, or kill it and return
    'stalled' if the cache stops growing for stall_timeout (turns any hang into a resumable retry)."""
    stall = config.DL.get("stall_timeout", 180)
    cache_dir = f"{cache or config.CACHE}/hub/models--{model.replace('/', '--')}"
    last_size, last_change = -1, time.time()
    while True:
        st = remote.on(config.HEAD, f"docker inspect -f '{{{{.State.Status}}}}' {name} 2>/dev/null",
                       capture=True, check=False).stdout.strip()
        if st != "running":
            return "exited"
        out = remote.on(config.HEAD, f"du -sb {cache_dir} 2>/dev/null | cut -f1",
                        capture=True, check=False).stdout.strip()
        size = int(out) if out.isdigit() else last_size
        if size != last_size:
            last_size, last_change = size, time.time()
        elif time.time() - last_change > stall:
            print(f"[pull:hf] no progress for {stall}s — killing {name} to resume")
            remote.on(config.HEAD, f"docker rm -f {name}", check=False)
            return "stalled"
        time.sleep(15)


def verify_model(node, model, delete_bad=False, cache=None):
    """Integrity gate: every LFS shard's sha256 must equal its blob filename. True if clean.
    delete_bad=True removes corrupt blobs so a re-download re-fetches them."""
    flag = " --delete-bad" if delete_bad else ""
    r = remote.on(node, f"python3 {config.REMOTE}/bin/verify-model.py {model} "
                        f"--cache {cache or config.CACHE}{flag}", check=False)
    return r.returncode == 0


def _pull_hf(svc, cache=None):
    """Download once on HEAD, then VERIFY every shard (sha256 == blob name). Interrupted downloads
    can leave full-size-but-corrupt shards that hf's size/presence checks miss; corrupt shards are
    deleted and re-fetched (up to 3 attempts) BEFORE we ever mirror. Returns True -> needs mirror."""
    m = svc["model"]
    name = dl_cname(m)
    attempts = config.DL.get("max_attempts", 4)
    for attempt in range(1, attempts + 1):
        print(f"[pull:hf] {m} -> {config.HEAD} (download attempt {attempt}/{attempts})")
        hf_download_start(m, cache=cache)
        if _wait_with_watchdog(name, m, cache=cache) == "stalled":
            print("[pull:hf] stalled — resuming from partial")
            continue                                 # watchdog already removed the container
        rc = remote.on(config.HEAD, f"docker inspect -f '{{{{.State.ExitCode}}}}' {name}",
                       capture=True, check=False).stdout.strip()
        if rc != "0":
            remote.on(config.HEAD, f"docker logs --tail 10 {name}", check=False)
        remote.on(config.HEAD, f"docker rm {name}", check=False)
        if rc != "0":
            print(f"[pull:hf] download exit={rc}; retrying")
            continue
        print("[pull:hf] verifying integrity (sha256 == blob name) ...")
        if verify_model(config.HEAD, m, cache=cache):
            print(f"[pull:hf] {m} verified clean ✅")
            return True
        print("[pull:hf] corrupt shard(s) detected — deleting + re-downloading")
        verify_model(config.HEAD, m, delete_bad=True, cache=cache)
    sys.exit(f"[pull:hf] FAILED to obtain a verified-clean copy of {m} after {attempts} attempts")


def _pull_ollama(svc):
    node = svc.get("node", config.HEAD)
    print(f"[pull:ollama] {svc['model']} -> {node} (node-local)")
    ollama_ensure(node)
    remote.on(node, f"docker exec {config.PFX}-ollama ollama pull {svc['model']}", check=False)
    return False


PULLERS = {"hf": _pull_hf, "ollama": _pull_ollama}


def _rsync_hub(ip, checksum=False):
    # --no-owner/--no-group: don't chgrp (user can't; everything is user-owned on write anyway).
    # exclude .locks/ (transient) + *.incomplete (partials). -c: content checksum (repair mode).
    c = "c" if checksum else ""
    remote.on(config.HEAD, f"rsync -a{c} --no-owner --no-group --delete "
                           f"--exclude '.locks/' --exclude '*.incomplete' "
                           f"{config.CACHE}/hub/ {config.USER}@{ip}:{config.CACHE}/hub/")


# ---------------------------------------------------------------- ensure (nodes -> NAS -> download)
RSYNC_FLAGS = "-a --no-owner --no-group --delete --exclude '.locks/' --exclude '*.incomplete'"


def _hub_dir(model):
    return f"hub/models--{model.replace('/', '--')}"


def _nas():
    return config.CFG.get("nas") or None


def _nas_mode(nas):
    return nas.get("mode", "path")


def _present_cmd(cache, model):
    """Fast structural presence check: a populated snapshot and no partial downloads. Full sha256
    verification stays where it belongs — after every transfer, not on every apply."""
    hub = f"{cache}/{_hub_dir(model)}"
    return (f'test -d {hub}/snapshots && [ -n "$(ls -A {hub}/snapshots 2>/dev/null)" ] '
            f'&& [ -z "$(find {hub} -name \'*.incomplete\' -print -quit 2>/dev/null)" ]')


def model_present(node, model):
    return remote.on(node, _present_cmd(config.CACHE, model), check=False, capture=True).returncode == 0


def model_on_nas(model):
    nas = _nas()
    if not nas:
        return False
    if _nas_mode(nas) == "ssh":   # NAS is an ssh endpoint — check from the head
        user = nas.get("user", config.USER)
        cmd = _present_cmd(nas["remote_path"], model)
        return remote.on(config.HEAD, f"ssh -o BatchMode=yes {user}@{nas['host']} "
                                      f"{json.dumps(cmd)}", check=False, capture=True).returncode == 0
    # path mode: the NAS mount is visible on the head
    return remote.on(config.HEAD, _present_cmd(nas["path"], model),
                     check=False, capture=True).returncode == 0


def replicate_model(model, node, source, checksum=False):
    """Per-model rsync from `source` ('head' | 'nas') to a node. NAS path mode pushes from the
    head (which sees the mount); NAS ssh mode pulls on the destination node (node -> NAS ssh)."""
    c = RSYNC_FLAGS if not checksum else RSYNC_FLAGS.replace("-a ", "-ac ")
    hub = _hub_dir(model)
    nas = _nas() or {}
    remote.on(node, f"mkdir -p {config.CACHE}/{hub}", check=False)
    if source == "nas" and _nas_mode(nas) == "ssh":
        user = nas.get("user", config.USER)
        remote.on(node, f"rsync {c} {user}@{nas['host']}:{nas['remote_path']}/{hub}/ "
                        f"{config.CACHE}/{hub}/")
        return
    src = f"{nas['path']}/{hub}/" if source == "nas" else f"{config.CACHE}/{hub}/"
    if node == config.HEAD:   # head receiving from a mounted NAS: local copy
        remote.on(config.HEAD, f"rsync {c} {src} {config.CACHE}/{hub}/")
    else:
        ip = config.NODES[node]["fabric_ip"]
        remote.on(config.HEAD, f"rsync {c} {src} {config.USER}@{ip}:{config.CACHE}/{hub}/")


def _replicate_verified(model, node, source):
    """Replicate + sha256-verify, with one checksum-mode repair pass — a corrupt copy can never
    silently reach a serve (same guarantee as mirror_to_others)."""
    print(f"[ensure] {model}: {source} -> {node}")
    replicate_model(model, node, source)
    if verify_model(node, model):
        return
    print(f"[ensure] {node}: {model} failed verify — re-syncing with checksum")
    replicate_model(model, node, source, checksum=True)
    if not verify_model(node, model):
        sys.exit(f"[ensure] {node}: {model} STILL corrupt after checksum re-sync")


def push_to_nas(model):
    """Archive a freshly-downloaded model from the head onto the NAS (so future clusters/rebuilds
    replicate instead of re-downloading). Best-effort."""
    nas = _nas()
    if not nas or model_on_nas(model):
        return
    hub = _hub_dir(model)
    print(f"[ensure] {model}: archiving head -> NAS")
    if _nas_mode(nas) == "ssh":
        user = nas.get("user", config.USER)
        remote.on(config.HEAD, f"rsync {RSYNC_FLAGS} {config.CACHE}/{hub}/ "
                               f"{user}@{nas['host']}:{nas['remote_path']}/{hub}/", check=False)
    else:
        remote.on(config.HEAD, f"mkdir -p {nas['path']}/{hub} && "
                               f"rsync {RSYNC_FLAGS} {config.CACHE}/{hub}/ {nas['path']}/{hub}/",
                  check=False)


def ensure_models(recipe):
    """`apply`'s pull-if-missing: for each model, check the nodes, then the NAS, then download —
    replicating from whichever source has the weights. Every transfer is sha256-verified.
    With no `nas:` configured this reduces to today's head-download + fabric-mirror flow."""
    nas = _nas()
    for svc in recipe["services"]:
        provider = svc_provider(svc)
        if provider == "ollama":
            _pull_ollama(svc)                    # incremental — a no-op when already present
            continue
        if provider != "hf":
            print(f"[ensure] provider '{provider}' unsupported — skipping {svc.get('name')}")
            continue
        m = svc["model"]
        missing = [n for n in config.NODES if not model_present(n, m)]
        if not missing:
            print(f"[ensure] {m}: present on all nodes ✅")
            continue
        if nas and model_on_nas(m):
            print(f"[ensure] {m}: found on NAS — replicating to {', '.join(missing)}")
            for node in missing:
                _replicate_verified(m, node, source="nas")
            continue
        # nowhere yet -> download. With a mounted NAS and download_to: nas, land it there directly;
        # otherwise download to the head (then archive a copy to the NAS if one is configured).
        if nas and _nas_mode(nas) == "path" and nas.get("download_to", "nas") == "nas":
            print(f"[ensure] {m}: downloading to NAS ({nas['path']})")
            _pull_hf(svc, cache=nas["path"])
            for node in missing:
                _replicate_verified(m, node, source="nas")
            continue
        if config.HEAD in missing:
            _pull_hf(svc)                        # verified download on the head
            missing.remove(config.HEAD)
        for node in missing:
            _replicate_verified(m, node, source="head")
        if nas:
            push_to_nas(m)


def presence_matrix(recipe):
    """model -> {'services': [...], 'nodes': {node: bool}, 'nas': bool|None} — powers `get models`."""
    out = {}
    for svc in recipe["services"]:
        if svc_provider(svc) != "hf":
            continue
        m = svc["model"]
        if m not in out:
            out[m] = {"services": [], "nodes": {n: model_present(n, m) for n in config.NODES},
                      "nas": model_on_nas(m) if _nas() else None}
        out[m]["services"].append(svc["name"])
    return out


def mirror_to_others(models=None):
    """Distribute HEAD's cache to the other nodes over the fabric, then VERIFY each mirrored model
    (sha256 == blob name). On failure, re-sync with checksum and re-verify — a corrupt mirror can
    never silently reach a serve."""
    for node in [n for n in config.NODES if n != config.HEAD]:
        ip = config.NODES[node]["fabric_ip"]
        print(f"[mirror] {config.HEAD} -> {node} over fabric ({ip})")
        _rsync_hub(ip)
        for m in (models or []):
            if verify_model(node, m):
                continue
            print(f"[mirror] {node}: {m} failed verify — re-syncing with checksum")
            _rsync_hub(ip, checksum=True)
            if not verify_model(node, m):
                sys.exit(f"[mirror] {node}: {m} STILL corrupt after checksum re-sync")
        if models:
            print(f"[mirror] {node}: all models verified ✅")
