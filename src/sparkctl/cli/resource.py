"""Resource-oriented verbs: get / describe / apply / delete / top.

kubectl-style grammar over the cluster's nouns — nodes, services, recipes. `apply` is the one
mutating verb for deployments: validate -> tear down -> repoint `current` -> bring up -> manifest.
"""
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import yaml

from sparkctl import config, remote
from sparkctl.backends import get_backend
from sparkctl.distribution import ensure_models, hf_model_meta, inventory, ollama_model_meta
from sparkctl.engines.vllm import multinode_serve_alive, svc_cname
from sparkctl.server.lifecycle import refresh_server_if_running
from sparkctl.manifest import verify_deployment, write_active_manifest
from sparkctl.recipes import (current_recipe, load_recipe, recipe_hash, services_by_node,
                              svc_provider, svc_world)


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
    if svc["engine"] == "vllm":
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


# ---------------------------------------------------------------- apply / delete
def _services_down():
    get_backend().down()
    for node in config.NODES:   # clear the active manifest — nothing is deployed anymore
        remote.on(node, f"rm -f {config.STATE}/active.json", check=False)
    print("all services stopped")


def _recipe_up(name):
    print(f"== bringing up recipe: {name} ==")
    get_backend().up(load_recipe(name))
    write_active_manifest(name)
    print(f"[manifest] active.json written on all nodes (recipe {name} @ {recipe_hash(name)[:12]})")
    refresh_server_if_running()   # if a node-hosted server is up here, repoint it at the new deploy


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


def _wait_ready(recipe, timeout):
    """Block until every endpoint of the deployment answers /v1/models — vLLM finishes loading
    weights well after its container reports 'Up'. Polls every 5s with a 30s heartbeat, fails
    FAST if a service's container dies (a crash never becomes ready), exits non-zero on timeout."""
    backend = get_backend()
    served_from = config.SELF or config.SERVER.get("host", "local")
    pending = {(served, base)
               for served, bases in backend.endpoints(recipe, served_from).items()
               for base in bases}
    print(f"[wait] waiting on {len(pending)} endpoint(s), timeout {timeout}s "
          f"(big models take minutes to load)", flush=True)
    t0, last_note = time.time(), 0.0
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
        status = backend.status()   # docker ps lists running only — a vanished container crashed
        for svc in recipe["services"]:
            node, _, cname = _svc_placement(svc)
            if not status.get(node, {}).get(cname):
                sys.exit(f"[wait] service '{svc['name']}' on {node} is no longer running — "
                         f"check: sparkctl logs {svc['name']}")
            if svc["engine"] == "vllm" and svc_world(svc) > 1 \
                    and not multinode_serve_alive(cname):
                sys.exit(f"[wait] service '{svc['name']}' crashed inside its Ray container — "
                         f"check: sparkctl logs {svc['name']}")
        if elapsed - last_note >= 30:
            last_note = elapsed
            print(f"[wait] still loading ({elapsed:.0f}s): "
                  + ", ".join(sorted({s for s, _ in pending})), flush=True)
        time.sleep(5)
    print(f"[wait] deployment ready in {time.time() - t0:.0f}s ✅")


def cmd_apply(args):
    name = resolve_apply_target(args)
    recipe = load_recipe(name)         # validate before tearing anything down
    ensure_models(recipe)              # pull-if-missing: nodes -> NAS -> download, verified
    _services_down()
    (config.ROOT / "current").write_text(name + "\n")
    print(f"current -> {name}")
    _recipe_up(name)
    if getattr(args, "wait", False):
        _wait_ready(recipe, getattr(args, "timeout", 1800))


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
              "vllm:gpu_cache_usage_perc": "KV-CACHE"}
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
