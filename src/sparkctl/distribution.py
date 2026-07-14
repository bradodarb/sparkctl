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
    Runs as the invoking user (cache stays user-owned for the fabric mirror). Sources secrets
    (`sparkctl secret set HF_TOKEN`; legacy ~/.hugging_face.sh still honored) and passes the token
    via `-e HF_TOKEN` (name only — value never lands on the command line/ps/logs). Robustness env
    from _dl_env(). `cache` overrides the node cache — e.g. a NAS mount, so downloads land there."""
    name = dl_cname(model)
    inner = ('[ -f "$HOME/.hugging_face.sh" ] && . "$HOME/.hugging_face.sh"; '
             'set -a; [ -f "$HOME/.sparkctl/secrets.env" ] && . "$HOME/.sparkctl/secrets.env"; set +a; '
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


def _ollama_manifest(model):
    """Relative manifest path inside the ollama store for a model ref: 'nomic-embed-text' lives
    under registry.ollama.ai/library/, while refs with a registry path ('hf.co/org/repo:Q4')
    keep it. Tag defaults to 'latest'."""
    name, _, tag = model.partition(":")
    if "/" not in name:
        name = f"registry.ollama.ai/library/{name}"
    return f"models/manifests/{name}/{tag or 'latest'}"


def _ollama_ref(manifest_path):
    """Inverse of _ollama_manifest: a path under models/manifests/ back to a model ref."""
    parts = manifest_path.strip("/").split("/")
    if len(parts) < 2:
        return None
    name, tag = "/".join(parts[:-1]), parts[-1]
    if name.startswith("registry.ollama.ai/library/"):
        name = name[len("registry.ollama.ai/library/"):]
    return name if tag == "latest" else f"{name}:{tag}"


# ---------------------------------------------------------------- inventory (what's installed where)
def _list_hf_cmd(cache):
    """Emit the hub-dir name of every COMPLETE model in an HF cache (populated snapshot, no partial
    downloads) — one command per store, the inventory analogue of _present_cmd."""
    return (f'cd {cache}/hub 2>/dev/null && for d in models--*; do '
            f'[ -n "$(ls -A "$d/snapshots" 2>/dev/null)" ] && '
            f'[ -z "$(find "$d" -name \'*.incomplete\' -print -quit 2>/dev/null)" ] && '
            f'echo "$d"; done; true')


def _hub_dir_to_model(d):
    return d[len("models--"):].replace("--", "/")


def _parse_hf_listing(out):
    return {_hub_dir_to_model(line) for line in (out or "").splitlines()
            if line.startswith("models--")}


def list_hf_models(node):
    return _parse_hf_listing(remote.on(node, _list_hf_cmd(config.CACHE),
                                       check=False, capture=True).stdout)


def list_nas_models():
    nas = _nas()
    if not nas:
        return set()
    if _nas_mode(nas) == "ssh":
        user = nas.get("user", config.USER)
        cmd = _list_hf_cmd(nas["remote_path"])
        out = remote.on(config.HEAD, f"ssh -o BatchMode=yes {user}@{nas['host']} "
                                     f"{json.dumps(cmd)}", check=False, capture=True).stdout
    else:
        out = remote.on(config.HEAD, _list_hf_cmd(nas["path"]), check=False, capture=True).stdout
    return _parse_hf_listing(out)


def list_ollama_models(node):
    out = remote.on(node, f'cd {config.CACHE}/ollama/models/manifests 2>/dev/null && '
                          f'find . -type f | sed "s|^\\./||"; true',
                    check=False, capture=True).stdout or ""
    return {r for r in (_ollama_ref(line.strip()) for line in out.splitlines() if line.strip())
            if r}


def _hf_precision(cfg):
    """Human tag for a model's precision from its config.json: the quantization format when
    quantized (NVFP4/FP8/AWQ/...), else the checkpoint dtype (BF16/FP16/...)."""
    q = cfg.get("quantization_config") or {}
    if q:
        blob = json.dumps(q).lower()
        for tag in ("nvfp4", "fp8", "fp4", "int8", "int4", "awq", "gptq"):
            if tag in blob:
                return tag.upper()
        return str(q.get("quant_method", "quantized")).upper()
    dt = str(cfg.get("torch_dtype") or cfg.get("dtype") or "").replace("torch.", "")
    return dt.upper().replace("BFLOAT16", "BF16").replace("FLOAT16", "FP16") \
             .replace("FLOAT32", "FP32") or "-"


def hf_model_meta(node, model, cache=None):
    """(size_bytes|None, precision) for an HF model read from a cache on `node`: du of the hub dir
    + the snapshot's config.json. One ssh round-trip; (None, '-') when absent. `cache` overrides
    the node cache — e.g. a NAS mount read from the head."""
    hub = f"{cache or config.CACHE}/{_hub_dir(model)}"
    out = remote.on(node, f'du -sb {hub} 2>/dev/null | cut -f1; '
                          f'cat "$(ls -1d {hub}/snapshots/* 2>/dev/null | head -1)/config.json" '
                          f'2>/dev/null',
                    check=False, capture=True).stdout or ""
    first, _, rest = out.partition("\n")
    size = int(first.strip()) if first.strip().isdigit() else None
    try:
        cfg = json.loads(rest)
    except ValueError:
        cfg = {}
    return size, (_hf_precision(cfg) if cfg else "-")


def ollama_model_meta(node, model):
    """(size_bytes|None, precision) for an ollama model from the node's store: manifest layer
    sizes + the config blob's file_type (e.g. F16, Q4_K_M)."""
    store = f"{config.CACHE}/ollama"
    out = remote.on(node, f"cat {store}/{_ollama_manifest(model)} 2>/dev/null",
                    check=False, capture=True).stdout or ""
    try:
        man = json.loads(out)
    except ValueError:
        return None, "-"
    size = sum(layer.get("size", 0) for layer in man.get("layers", [])) or None
    digest = (man.get("config") or {}).get("digest", "").replace(":", "-")
    precision = "-"
    if digest:
        blob = remote.on(node, f"cat {store}/models/blobs/{digest} 2>/dev/null",
                         check=False, capture=True).stdout or ""
        try:
            precision = json.loads(blob).get("file_type") or "-"
        except ValueError:
            pass
    return size, precision


def inventory(recipe):
    """Everything installed anywhere — each node's HF cache, each node's ollama store, the NAS —
    unioned with the current recipe's requirements. model -> {'services', 'source',
    'nodes': {node: bool|None}, 'nas': bool|None}; powers `get models`."""
    nas_cfg = _nas()
    hf_on = {n: list_hf_models(n) for n in config.NODES}
    ol_on = {n: list_ollama_models(n) for n in config.NODES}
    on_nas = list_nas_models() if nas_cfg else set()
    svc, served = {}, {}
    for s in recipe["services"]:
        key = (s["model"], svc_provider(s))
        svc.setdefault(key, []).append(s["name"])
        # the gateway alias this model answers to (ollama serves under its own model name)
        served.setdefault(key, set()).add(s.get("served_name") or s["model"])
    out = {}
    for m in set().union(*hf_on.values(), on_nas, {m for (m, p) in svc if p == "hf"}):
        out[m] = {"served": sorted(served.get((m, "hf"), ())),
                  "services": svc.get((m, "hf"), []), "source": "hf",
                  "nodes": {n: m in hf_on[n] for n in config.NODES},
                  "nas": (m in on_nas) if nas_cfg else None}
    for m in set().union(*ol_on.values(), {m for (m, p) in svc if p == "ollama"}):
        out[m] = {"served": sorted(served.get((m, "ollama"), ())),
                  "services": svc.get((m, "ollama"), []), "source": "ollama",
                  "nodes": {n: m in ol_on[n] for n in config.NODES},
                  "nas": None}
    for (m, p), names in svc.items():               # unknown providers: listed, presence unknown
        if p not in ("hf", "ollama"):
            out[m] = {"served": sorted(served.get((m, p), ())), "services": names, "source": p,
                      "nodes": {n: None for n in config.NODES}, "nas": None}
    # current recipe's models first, then the rest of the library alphabetically
    return dict(sorted(out.items(), key=lambda kv: (not kv[1]["services"], kv[0].lower())))


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
        # nowhere yet -> download. A mounted NAS is always the download target (that's the point
        # of configuring one); ssh mode can't be mounted into the download container, so it
        # downloads to the head and archives a copy back to the NAS below.
        if nas and _nas_mode(nas) == "path":
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
