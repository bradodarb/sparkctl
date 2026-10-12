"""Resource-oriented verbs: get / describe / apply / delete / top.

kubectl-style grammar over the cluster's nouns — nodes, services, recipes. `apply` is the one
mutating verb for deployments: validate -> tear down -> repoint `current` -> bring up -> manifest.
"""
import base64
import json
import re
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from sparkctl import config, progress, remote
from sparkctl.backends import get_backend
from sparkctl.distribution import ensure_models, hf_model_meta, inventory, ollama_model_meta
from sparkctl.engines.vllm import multinode_serve_alive, svc_cname
from sparkctl.server.lifecycle import refresh_or_start_server
from sparkctl.manifest import verify_deployment, write_active_manifest
from sparkctl.recipes import (SERVED_ENGINES, current_recipe, load_recipe, recipe_hash,
                              services_by_node, svc_provider, svc_world)


# ---------------------------------------------------------------- shared helpers
def _table(headers, rows):
    widths = [max(len(str(c)) for c in col) for col in zip(headers, *rows)] if rows else \
             [len(h) for h in headers]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*headers))
    for r in rows:
        print(fmt.format(*(str(c) for c in r)))


def _svc_placement(svc):
    """(node, port, container_name) a service actually serves from — mirrors services_by_node."""
    if svc["engine"] in SERVED_ENGINES:
        node = svc.get("node", config.HEAD) if svc_world(svc) <= 1 else config.HEAD
        port = svc.get("port", config.CFG["cluster"]["api_port"])
        cname = svc_cname(svc["name"], "head" if svc_world(svc) > 1 else None)
    else:
        node = svc.get("node", config.HEAD)
        port = svc.get("port", 11434)
        cname = f"{config.PFX}-ollama"
    return node, port, cname


# ---------------------------------------------------------------- get
def _get_nodes(output):
    rows = []
    for node, v in config.NODES.items():
        role = "head" if node == config.HEAD else "worker"
        reachable = remote.on(node, "true", check=False, capture=True).returncode == 0
        row = {"name": node, "host": v.get("host", ""), "role": role,
               "reachable": "yes" if reachable else "NO"}
        if output == "wide":
            n = remote.on(node, f"docker ps -q --filter name={config.PFX}- | wc -l",
                          capture=True, check=False).stdout.strip() if reachable else "-"
            row.update({"lan_ip": v.get("lan_ip", ""), "fabric_ip": v.get("fabric_ip", ""),
                        "containers": n})
        rows.append(row)
    if output == "json":
        print(json.dumps(rows, indent=2))
        return
    headers = list(rows[0].keys()) if rows else []
    _table([h.upper() for h in headers], [[r[h] for h in headers] for r in rows])


def _fmt_size(nbytes):
    if not nbytes:
        return "-"
    gb = nbytes / 1e9
    return f"{gb:.0f}GB" if gb >= 10 else f"{gb:.1f}GB"


def _get_services(output):
    recipe_name = current_recipe()
    recipe = load_recipe(recipe_name)
    status_by_node = get_backend().status()
    rows, meta = [], {}
    for svc in recipe["services"]:
        node, port, cname = _svc_placement(svc)
        state = status_by_node.get(node, {}).get(cname, "not running")
        # a multinode container is `sleep infinity` — Up says nothing about the exec'd serve
        if svc["engine"] == "vllm" and svc_world(svc) > 1 and "Up" in state \
                and not multinode_serve_alive(cname):
            state += "  ⚠️ vllm serve DEAD"
        key = (node, svc["model"])
        if key not in meta:              # size/precision read from the serving node's store
            probe = ollama_model_meta if svc_provider(svc) == "ollama" else hf_model_meta
            meta[key] = probe(node, svc["model"])
        size, precision = meta[key]
        row = {"name": svc["name"], "served": svc.get("served_name") or svc["model"],
               "engine": svc["engine"], "model": svc["model"],
               "size": _fmt_size(size), "precision": precision,
               "node": node, "port": port, "status": state}
        if output == "wide":
            row.update({"container": cname, "recipe": recipe_name})
        rows.append(row)
    if output == "json":
        print(json.dumps(rows, indent=2))
        return
    headers = list(rows[0].keys()) if rows else []
    _table([h.upper() for h in headers], [[r[h] for h in headers] for r in rows])


def _get_recipes(output):
    cur = current_recipe()
    rows = [{"current": "*" if p.stem == cur else "", "name": p.stem}
            for p in sorted((config.ROOT / "recipes").glob("*.yaml"))]
    if output == "json":
        print(json.dumps(rows, indent=2))
        return
    _table(["CURRENT", "NAME"], [[r["current"], r["name"]] for r in rows])


def _get_models(output):
    matrix = inventory(load_recipe(current_recipe()))
    nas = config.CFG.get("nas") or {}
    rows = []
    def mark(v):
        return "-" if v is None else ("✓" if v else "✗")
    def meta(model, info):
        """size/precision from the first node holding the model, else a mounted NAS via the head."""
        node = next((n for n, present in info["nodes"].items() if present), None)
        if node and info["source"] in ("hf", "ollama"):
            probe = ollama_model_meta if info["source"] == "ollama" else hf_model_meta
            return probe(node, model)
        if info["nas"] and nas.get("mode", "path") == "path":
            return hf_model_meta(config.HEAD, model, cache=nas["path"])
        return None, "-"
    for model, info in matrix.items():
        size, precision = meta(model, info)
        row = {"model": model, "name": ",".join(info["served"]) or "-",
               "source": info["source"], "size": _fmt_size(size),
               "precision": precision, "services": ",".join(info["services"]) or "-"}
        for node, present in info["nodes"].items():
            row[node] = mark(present)
        row["nas"] = mark(info["nas"])
        rows.append(row)
    if output == "json":
        print(json.dumps(rows, indent=2))
        return
    headers = list(rows[0].keys()) if rows else ["model"]
    _table([h.upper() for h in headers], [[r[h] for h in headers] for r in rows])


def cmd_get(args):
    {"nodes": _get_nodes, "services": _get_services, "recipes": _get_recipes,
     "models": _get_models}[args.resource](args.output)


# ---------------------------------------------------------------- describe
def _describe_node(name):
    if name not in config.NODES:
        sys.exit(f"no such node: {name}")
    v = dict(config.NODES[name])
    v["role"] = "head" if name == config.HEAD else "worker"
    print(yaml.safe_dump({name: v}, sort_keys=False).rstrip())
    rec, ok = verify_deployment(name)
    badge = "matches current" if ok else ("DRIFT" if rec else "nothing active")
    print(f"\nactive recipe: {rec or '(none)'}  [{badge}]")
    print("\ncontainers:")
    remote.on(name, f"docker ps --filter name={config.PFX}- "
                    f"--format 'table {{{{.Names}}}}\\t{{{{.Status}}}}'", check=False)
    print("\nmodel cache:")
    remote.on(name, f"df -h {config.CACHE} | tail -1", check=False)


def _describe_service(name):
    recipe = load_recipe(current_recipe())
    svc = next((s for s in recipe["services"] if s["name"] == name), None)
    if not svc:
        sys.exit(f"no service '{name}' in current recipe")
    node, port, cname = _svc_placement(svc)
    print(yaml.safe_dump({"service": svc}, sort_keys=False).rstrip())
    print(f"\nresolved: node={node} port={port} container={cname} provider={svc_provider(svc)}")
    print("\ncontainer:")
    remote.on(node, f"docker ps --filter name={cname} "
                    f"--format 'table {{{{.Names}}}}\\t{{{{.Status}}}}'", check=False)
    print("\nrecent logs:")
    remote.on(node, f"docker logs --tail 10 {cname} 2>&1 || true", check=False)


def _describe_recipe(name):
    recipe = load_recipe(name)
    print(f"# {name}  (sha {recipe_hash(name)[:12]})")
    print(yaml.safe_dump(recipe, sort_keys=False).rstrip())
    print("\n# resolved placement:")
    for node, svcs in services_by_node(recipe).items():
        for s in svcs:
            print(f"#   {s['name']:<16} -> {node}:{s['port']} ({s['engine']}, serves '{s['served_name']}')")


def cmd_describe(args):
    {"node": _describe_node, "service": _describe_service, "recipe": _describe_recipe}[args.kind](args.name)


# ---------------------------------------------------------------- boot arm/disarm state
# Two head-local markers under {STATE} gate the systemd boot apply (`sparkctl apply --boot`):
#   paused        — present => boot serves NOTHING (disarmed). Set by `stop` or a self-heal.
#   boot_attempt  — a recipe name written *before* load and cleared only once the deployment is
#                   confirmed healthy. If a boot finds it still set, the previous boot started that
#                   recipe and never became healthy (it OOM'd/crashed the node) — so we auto-disarm
#                   instead of retrying the same killer recipe forever.
PAUSED, BOOT_ATTEMPT = "paused", "boot_attempt"


def _state_write(name, content=""):
    b64 = base64.b64encode(content.encode()).decode()
    remote.on(config.HEAD, f"mkdir -p {config.STATE} && echo {b64} | base64 -d > {config.STATE}/{name}",
              check=False)


def _state_read(name):
    r = remote.on(config.HEAD, f"cat {config.STATE}/{name} 2>/dev/null", capture=True, check=False)
    return (r.stdout or "").strip() if r.returncode == 0 else None


def _state_clear(name):
    remote.on(config.HEAD, f"rm -f {config.STATE}/{name}", check=False)


# ---------------------------------------------------------------- apply / delete
def _services_down():
    get_backend().down()
    for node in config.NODES:   # clear the active manifest — nothing is deployed anymore
        remote.on(node, f"rm -f {config.STATE}/active.json", check=False)
    print("all services stopped")


def _free_page_cache():
    """Best-effort: flush clean page cache on every node right before a load, so vLLM sees real RAM
    headroom and keeps weight-prefetch enabled. vLLM disables prefetch when the checkpoint exceeds
    ~90% of *available* RAM — and the download/verify/mirror we just ran leaves the weights hot in
    page cache, starving the loader (58GB gemma load went from prefetched to ~285s/shard). Drops
    page cache only (reclaimable); needs passwordless sudo, skipped silently otherwise."""
    for node in config.NODES:
        remote.on(node, "sync 2>/dev/null; sudo sysctl -q vm.drop_caches=1 2>/dev/null "
                        "|| sudo sh -c 'echo 1 > /proc/sys/vm/drop_caches' 2>/dev/null || true",
                  check=False)


def _recipe_up(name):
    print(f"== bringing up recipe: {name} ==")
    get_backend().up(load_recipe(name))
    write_active_manifest(name)
    print(f"[manifest] active.json written on all nodes (recipe {name} @ {recipe_hash(name)[:12]})")
    refresh_or_start_server()   # start/repoint the unified server (on its host) at the new deploy


def resolve_apply_target(args):
    """Recipe name to apply: an explicit name, a -f manifest (copied into recipes/ — the deployable
    source of truth), or the current pointer."""
    if getattr(args, "filename", None):
        src = Path(args.filename)
        if not src.exists():
            sys.exit(f"no such file: {src}")
        dst = config.ROOT / "recipes" / f"{src.stem}.yaml"
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
            print(f"[apply] {src} -> {dst}")
        return src.stem
    return args.recipe or current_recipe()


_TEARDOWN_NOISE = ("weakref", "_del_library", "_clear_torch_ops", "_get_packet",
                   "jit_get_operation", "atexit")


_EXC_RE = re.compile(r"[A-Za-z_]+(Error|Exception):")


def _is_teardown_exc(lines, idx):
    """Is the exception at lines[idx] interpreter-teardown noise? The giveaway usually isn't on the
    exception line itself (e.g. `UnicodeDecodeError: invalid start byte` is markerless) but in the
    traceback FRAMES just above it (torch's _del_library / _jit_get_operation finalizer). Scan back
    to the enclosing 'Traceback' header (bounded) for any teardown marker."""
    if any(n in lines[idx] for n in _TEARDOWN_NOISE):
        return True
    for j in range(idx - 1, max(-1, idx - 16), -1):
        if "Traceback (most recent call last)" in lines[j]:
            break
        if any(n in lines[j] for n in _TEARDOWN_NOISE):
            return True
    return False


def _extract_cause(lines):
    """The real, actionable exception from a crashed engine's log tail — skipping the teardown
    traceback that trails it. Prefer vLLM's engine-core ERROR lines, then any exception anywhere;
    in both passes ignore teardown-noise exceptions. Pure (list[str] -> str) so it's unit-testable."""
    def scan(idxs):
        for i in idxs:
            if _EXC_RE.search(lines[i]) and not _is_teardown_exc(lines, i):
                return lines[i].split("]", 1)[-1].strip()
        return ""

    core = [i for i, l in enumerate(lines) if "core.py:1165]" in l or "EngineCore failed" in l]
    return scan(reversed(core)) or scan(range(len(lines) - 1, -1, -1))


def _crash_report(node, cname):
    """Exit code, OOMKilled flag, the CONCRETE cause line, and a log tail. The cause is the real
    exception vLLM logged for the failure — not the interpreter-teardown traceback that trails it,
    which otherwise buries the actual reason."""
    insp = remote.on(node, f"docker inspect {cname} "
                           f"--format '{{{{.State.ExitCode}}}}|{{{{.State.OOMKilled}}}}' 2>/dev/null",
                     capture=True, check=False).stdout.strip()
    code, _, oom = insp.partition("|")
    lines = (remote.on(node, f"docker logs --tail 200 {cname} 2>&1",
                       capture=True, check=False).stdout or "").splitlines()
    return (code or "?"), (oom.strip().lower() == "true"), _extract_cause(lines), \
        "\n".join(lines[-15:]).rstrip()


def _fail_service(svc, node, cname, why, context=()):
    """Print an actionable crash report for one service, then exit non-zero. Called from the wait
    loop the moment a container is found dead — the deployment is not serving, say so loudly.
    `context` lines (memguard kills, last free-memory readings) print right under the headline."""
    code, oom, cause, tail = _crash_report(node, cname)
    print(f"[wait] ❌ {svc['name']} on {node} {why} (exit={code}, OOMKilled={oom})", flush=True)
    for line in context:
        print(f"[wait]    {line}", flush=True)
    if cause:
        print(f"[wait]    cause: {cause}", flush=True)
    if oom:
        print("[wait]    → OOM during load. On unified memory the GPU reservation "
              "(gpu_memory_utilization × 128GB) and the host-side weight load share ONE pool — "
              "lower gpu_memory_utilization, drop --safetensors-load-strategy=eager, or use fewer "
              "replicas per node.", flush=True)
    if tail:
        print(f"[wait]    last log lines from {cname}:", flush=True)
        print("\n".join("        " + ln for ln in tail.splitlines()), flush=True)
    sys.exit(f"[wait] deployment FAILED — '{svc['name']}' is not serving (full log: sparkctl logs {svc['name']})")


MEM_WARN_GIB = 15          # warn when free memory is under this AND falling fast...
MEM_WARN_RATE = -5         # ...faster than this many GiB/min,
MEM_CRIT_GIB = 8           # or unconditionally under this (memguard kills below its floor, 6 GiB)


class _LoadWatch:
    """Per-apply telemetry state for the wait loop: memory history per node, memguard kills seen,
    one-time engine facts already printed. See sparkctl.progress for the parsing."""

    def __init__(self, recipe, t0):
        self.t0 = t0
        self.served = [s for s in recipe["services"] if s["engine"] in SERVED_ENGINES]
        nodes = set()
        for s in self.served:
            nodes |= {config.HEAD, config.WORKER} if svc_world(s) > 1 else {_svc_placement(s)[0]}
        self.nodes = sorted(n for n in nodes if n)
        self.mem = {n: [] for n in self.nodes}            # node -> [(t, gib), ...]
        self.kills, self.facts, self.last_warn = set(), {}, {}

    def sample_memory(self):
        """Every poll: record free memory, warn on a fast fall, surface memguard kills at once."""
        now = time.time()
        for node in self.nodes:
            gib, kills = progress.probe_node(node, self.t0)
            for k in kills:
                if (node, k) not in self.kills:
                    self.kills.add((node, k))
                    print(f"[wait] 🛑 {node} memguard: {k}", flush=True)
            if gib is None:
                continue
            hist = self.mem[node]
            hist.append((now, gib))
            del hist[:-60]
            rate = progress.mem_trend(hist, gib)
            falling = rate is not None and rate < MEM_WARN_RATE
            if (gib < MEM_CRIT_GIB or (gib < MEM_WARN_GIB and falling)) \
                    and now - self.last_warn.get(node, 0) >= 30:
                self.last_warn[node] = now
                trend = f", falling {-rate:.0f} GiB/min" if falling else ""
                print(f"[wait] ⚠️  {node}: {gib:.1f} GiB free{trend} — unified memory is nearly "
                      f"exhausted (memguard kills serve containers below 6 GiB)", flush=True)

    def heartbeat(self, elapsed, pending_names):
        """Every 30s: each still-loading service's phase + new one-time facts, and free memory."""
        mem = ", ".join(f"{n} {h[-1][1]:.0f} GiB free" for n, h in self.mem.items() if h)
        phases = []
        for svc in self.served:
            if svc.get("served_name") not in pending_names:
                continue
            node, _, cname = _svc_placement(svc)
            phase, facts = progress.parse_engine_log(progress.engine_log(node, cname))
            for key, msg in facts.items():
                if (svc["name"], key) not in self.facts:
                    self.facts[(svc["name"], key)] = msg
                    print(f"[wait] {svc['name']}: {msg}", flush=True)
            phases.append(f"{svc['name']}: {phase or 'starting'}")
        detail = " · ".join(p for p in (", ".join(phases), mem) if p)
        print(f"[wait] {elapsed:.0f}s — {detail or ', '.join(sorted(pending_names))}", flush=True)

    def context(self, node):
        """Crash-report lines: memguard kills on that node and its recent free-memory readings."""
        lines = [f"memguard on {n} killed it: {k}" for n, k in sorted(self.kills) if n == node]
        hist = self.mem.get(node) or []
        if hist:
            lines.append(f"{node} free memory, last readings: "
                         + " → ".join(f"{g:.0f}" for _, g in hist[-6:]) + " GiB")
        return lines


def _wait_ready(recipe, timeout):
    """Block until every endpoint of the deployment answers /v1/models — vLLM finishes loading
    weights well after its container reports 'Up'. Polls every 5s with a 30s heartbeat that shows
    each service's load phase, key facts (weights/rank, KV capacity) and per-node free memory;
    warns when unified memory is running out, fails FAST (with a crash report) if a service's
    container dies, exits non-zero on timeout."""
    backend = get_backend()
    served_from = config.SELF or config.SERVER.get("host", "local")
    pending = {(served, base)
               for served, bases in backend.endpoints(recipe, served_from).items()
               for base in bases}
    print(f"[wait] waiting on {len(pending)} endpoint(s), timeout {timeout}s "
          f"(big models take minutes to load)", flush=True)
    t0, last_note = time.time(), 0.0
    watch = _LoadWatch(recipe, t0)
    while pending:
        for tgt in sorted(pending):
            served, base = tgt
            if remote.sh(f"curl -sf --max-time 3 {base}/models -o /dev/null",
                         check=False, capture=True).returncode == 0:
                print(f"[wait] {served} @ {base} ready ({time.time() - t0:.0f}s)", flush=True)
                pending.discard(tgt)
        if not pending:
            break
        elapsed = time.time() - t0
        if elapsed > timeout:
            sys.exit("[wait] TIMEOUT — still not answering: "
                     + ", ".join(base for _, base in sorted(pending)))
        watch.sample_memory()
        status = backend.status()   # docker ps lists running only — a vanished container crashed
        for svc in recipe["services"]:
            node, _, cname = _svc_placement(svc)
            if not status.get(node, {}).get(cname):
                _fail_service(svc, node, cname, "exited during load", watch.context(node))
            if svc["engine"] == "vllm" and svc_world(svc) > 1 \
                    and not multinode_serve_alive(cname):
                where = ("its Ray container" if svc.get("distributed_backend", "ray") == "ray"
                         else "its container")
                _fail_service(svc, node, cname, f"serve process died inside {where}",
                              watch.context(node))
        if elapsed - last_note >= 30:
            last_note = elapsed
            watch.heartbeat(elapsed, {s for s, _ in pending})
        time.sleep(5)
    print(f"[wait] deployment ready in {time.time() - t0:.0f}s ✅")


def _ensure_images(recipe):
    """Before tearing down the running deployment, make sure every pinned (non-default) `image:` is
    present on the nodes that will run it — building or pulling it if not, per its cluster.images
    declaration. We only ever provision DECLARED images (a known dockerfile:/pull: spec), so this is
    materializing exactly what the operator wrote in cluster.yaml, not guessing; an UNDECLARED pinned
    image is still a hard stop. The cluster default image is a build/deploy invariant → skip it."""
    from sparkctl.cli.main import _image_targets
    specs = {tag: (df, pull) for tag, df, pull in _image_targets()}
    for svc in recipe["services"]:
        if svc["engine"] not in SERVED_ENGINES:   # ollama uses its own fixed image, nothing to provision
            continue
        img = svc.get("image", config.IMAGE)
        if img == config.IMAGE:
            continue
        nodes = [config.HEAD, config.WORKER] if svc_world(svc) > 1 else [svc.get("node", config.HEAD)]
        for node in [n for n in nodes if n]:
            if remote.on(node, f"docker image inspect {img} >/dev/null 2>&1",
                         check=False, capture=True).returncode == 0:
                continue
            if img not in specs:
                sys.exit(f"[apply] pinned image '{img}' is missing on {node} and not declared under "
                         f"cluster.images — declare it (dockerfile:/pull:) or build it, then re-apply.")
            df, pull = specs[img]
            if pull:
                print(f"[apply] image '{img}' missing on {node} — pulling {pull}", flush=True)
                remote.on(node, f"docker pull {pull} && docker tag {pull} {img}")
            else:
                print(f"[apply] image '{img}' missing on {node} — building ({df})", flush=True)
                remote.on(node, f"docker build -t {img} -f {config.REMOTE}/docker/{df} {config.REMOTE}/docker")


def cmd_apply(args):
    # --boot is the systemd auto-restore path; a human `apply` never carries it, so human intent
    # always wins (re-arms boot, ignores the disarm markers).
    boot = getattr(args, "boot", False)
    if boot:
        if _state_read(PAUSED) is not None:
            print("[boot] disarmed (paused) — serving nothing. `sparkctl apply <recipe>` re-arms.")
            return
        stale = _state_read(BOOT_ATTEMPT)
        if stale:   # last boot started this recipe and never confirmed healthy -> it crashed us
            _state_write(PAUSED, f"auto-disarmed: {stale} never became healthy last boot (likely OOM)")
            _state_clear(BOOT_ATTEMPT)
            print(f"[boot] ⚠️  previous boot of '{stale}' never became healthy — auto-disarming to "
                  f"break the crash loop.\n"
                  f"       Fix it (lower gpu_memory_utilization / max_model_len), then: "
                  f"sparkctl apply {stale}")
            return

    name = resolve_apply_target(args)
    recipe = load_recipe(name)         # validate before tearing anything down
    _ensure_images(recipe)             # a pinned variant image must exist before we stop the old one
    ensure_models(recipe)              # pull-if-missing: nodes -> NAS -> download, verified
    _services_down()
    (config.ROOT / "current").write_text(name + "\n")
    print(f"current -> {name}")
    _state_clear(PAUSED)               # an explicit/boot apply re-arms; success clears boot_attempt
    if boot:
        _state_write(BOOT_ATTEMPT, name)   # dirty flag: survives a hard OOM that kills this process
    _free_page_cache()                     # RAM headroom for the loader -> vLLM re-enables prefetch
    _recipe_up(name)
    # Verify the deployment actually SERVES before reporting success — a launched container is not a
    # loaded model (vLLM can OOM minutes into loading). Waiting is the default; --detach opts out.
    if not getattr(args, "detach", False):
        try:
            _wait_ready(recipe, getattr(args, "timeout", 1800))
        except SystemExit:
            if boot:   # load never came up healthy — disarm so the next boot doesn't retry the OOM
                _state_write(PAUSED, f"auto-disarmed: {name} failed to become ready last boot")
                _state_clear(BOOT_ATTEMPT)
            raise
    else:
        print("[apply] launched (detached) — NOT verified serving yet. Containers can still die "
              "during load; confirm with `sparkctl status` or re-run without --detach to block "
              "until every endpoint answers.")
    _state_clear(BOOT_ATTEMPT)         # confirmed healthy (or detached) -> clear the dirty flag


def cmd_stop(args):
    """Tear down every service AND disarm boot: the next boot serves nothing until the next
    `sparkctl apply <recipe>`. This is the inverse of `apply` and the way to break an OOM/crash
    loop — `delete services --all` keeps `current`, so the systemd boot unit re-applies it."""
    _services_down()
    _state_clear(BOOT_ATTEMPT)
    _state_write(PAUSED, "stopped via `sparkctl stop`")
    print("boot disarmed — next boot serves nothing. Re-arm with `sparkctl apply <recipe>`.")


def cmd_delete(args):
    if args.kind == "services":
        if not args.all:
            sys.exit("refusing to delete all services without --all")
        _services_down()
        return
    if not args.name:
        sys.exit("usage: sparkctl delete service <name>")
    recipe = load_recipe(current_recipe())
    svc = next((s for s in recipe["services"] if s["name"] == args.name), None)
    if not svc:
        sys.exit(f"no service '{args.name}' in current recipe")
    node, _, cname = _svc_placement(svc)
    if svc["engine"] == "vllm" and svc_world(svc) > 1:
        remote.on(config.HEAD, f"docker rm -f {svc_cname(svc['name'], 'head')}", check=False)
        remote.on(config.WORKER, f"docker rm -f {svc_cname(svc['name'], 'worker')}", check=False)
    else:
        remote.on(node, f"docker rm -f {cname}", check=False)
    print(f"service '{args.name}' deleted (recipe unchanged — `sparkctl apply` restores it)")


# ---------------------------------------------------------------- top
TOP_GAUGES = {"vllm:num_requests_running": "RUNNING", "vllm:num_requests_waiting": "WAITING",
              "vllm:kv_cache_usage_perc": "KV-CACHE"}   # renamed from gpu_cache_usage_perc in vLLM 0.22.1
TOP_COUNTERS = {"vllm:prompt_tokens_total": "PROMPT-TOK/S",
                "vllm:generation_tokens_total": "GEN-TOK/S"}


def parse_metrics(text, names):
    """Sum each named metric across its label sets in a Prometheus exposition body. Pure."""
    out = {}
    for line in (text or "").splitlines():
        if line.startswith("#"):
            continue
        for n in names:
            if line.startswith(n + "{") or line.startswith(n + " "):
                try:
                    out[n] = out.get(n, 0.0) + float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
    return out


def _scrape(url, timeout=3):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except Exception:
        return None


def _top_targets():
    """(label, node, url) per vLLM service — the per-node /metrics endpoints to sample."""
    recipe = load_recipe(current_recipe())
    served_from = "local" if config.SELF is None else config.SELF
    targets = []
    for svc in recipe["services"]:
        if svc["engine"] != "vllm":
            continue
        node, port, _ = _svc_placement(svc)
        host = remote.backend_host(node, served_from)
        targets.append((svc["name"], node, f"http://{host}:{port}/metrics"))
    return targets


def _node_baseline(nodes):
    """Baseline MEM/GPU per node (unified memory + nvidia-smi util), sampled over SSH."""
    from sparkctl.server.metrics import node_stats_cmd, parse_node_stats
    cmd = node_stats_cmd(config.CACHE)
    out = {}
    for node in nodes:
        r = remote.on(node, cmd, capture=True, check=False)
        s = parse_node_stats(r.stdout) if r.returncode == 0 else {}
        mem = (f"{s['mem_used_kb'] / 2**20:.0f}/{s['mem_total_kb'] / 2**20:.0f}Gi"
               if "mem_total_kb" in s else "-")
        gpu = f"{s['gpu_util_pct']:.0f}%" if "gpu_util_pct" in s else "-"
        out[node] = (mem, gpu)
    return out


def cmd_top(args):
    targets = _top_targets()
    if not targets:
        sys.exit("current recipe has no vLLM services to sample")
    names = list(TOP_GAUGES) + list(TOP_COUNTERS)
    prev, prev_t = {}, {}
    try:
        while True:
            rows, now = {}, time.time()
            baseline = _node_baseline(config.NODES) if args.resource == "nodes" else {}
            for label, node, url in targets:
                m = parse_metrics(_scrape(url), names)
                key = node if args.resource == "nodes" else label
                row = rows.setdefault(key, {h: 0.0 for h in
                                            list(TOP_GAUGES.values()) + list(TOP_COUNTERS.values())})
                if not m:
                    row["down"] = True
                    continue
                for metric, header in TOP_GAUGES.items():
                    row[header] += m.get(metric, 0.0)
                for metric, header in TOP_COUNTERS.items():
                    total = m.get(metric, 0.0)
                    pk = (key, metric)
                    if pk in prev:
                        dt = max(now - prev_t[pk], 1e-6)
                        row[header] += max(total - prev[pk], 0.0) / dt
                    prev[pk], prev_t[pk] = total, now
            if args.resource == "nodes":
                for n in config.NODES:      # nodes without vLLM services still get a baseline row
                    rows.setdefault(n, None)
            if not args.once:
                print("\x1b[2J\x1b[H", end="")   # clear + home
            first = args.resource.upper().rstrip("S")
            headers = [first] + list(TOP_GAUGES.values()) + list(TOP_COUNTERS.values())
            if baseline:
                headers += ["MEM", "GPU"]
            table_rows = []
            for key, row in rows.items():
                if row is None or row.get("down"):
                    cells = [key] + ["-"] * (len(TOP_GAUGES) + len(TOP_COUNTERS))
                else:
                    cells = [key, f"{row['RUNNING']:.0f}", f"{row['WAITING']:.0f}",
                             f"{row['KV-CACHE'] * 100:.1f}%",
                             f"{row['PROMPT-TOK/S']:.0f}", f"{row['GEN-TOK/S']:.0f}"]
                if baseline:
                    cells += list(baseline.get(key, ("-", "-")))
                table_rows.append(cells)
            _table(headers, table_rows)
            if args.once:
                return
            print(f"\n(refreshing every {args.interval:g}s — Ctrl-C to exit)")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
