"""DockerBackend — the default: services run as docker containers over SSH, weights distributed
by verified download + fabric rsync, boot persistence via the systemd unit on the head."""
import time

from sparkctl import config, remote
from sparkctl.backends.base import Backend
from sparkctl.distribution import PULLERS, mirror_to_others
from sparkctl.engines import ENGINES
from sparkctl.engines.vllm import svc_cname
from sparkctl.recipes import (SERVED_ENGINES, services_by_node, speculative_model, svc_provider,
                              svc_world)


def _served_nodes(svc):
    """Nodes whose unified memory a served service claims (a multinode service spans head+worker)."""
    if svc_world(svc) > 1:
        return [n for n in (config.HEAD, config.WORKER) if n]
    return [svc.get("node", config.HEAD)]


def launch_waves(services):
    """Split services into launch waves so no two SERVED services load on the same node at once.
    On GB10 unified memory, vLLM/SGLang size their KV pool from memory measured during their own
    startup — a neighbour loading concurrently gets counted as theirs (one comes up with negative KV
    and dies, the other over-claims once it frees). Wave k holds each node's k-th served service;
    ollama (lazy, no profiling) always goes in wave 0. Pure/testable."""
    waves, depth = [], {}
    for svc in services:
        k = 0
        if svc["engine"] in SERVED_ENGINES:
            nodes = _served_nodes(svc)
            k = max(depth.get(n, 0) for n in nodes)
            for n in nodes:
                depth[n] = k + 1
        while len(waves) <= k:
            waves.append([])
        waves[k].append(svc)
    return waves


def _wait_loaded(svcs, timeout):
    """Block until each service answers /v1/models on its node (weights + KV + graphs allocated, so
    its memory footprint is final). A dead container stops blocking — its memory is freed, and
    apply's readiness wait reports the crash; a timeout likewise just stops blocking."""
    pending = {s["name"]: s for s in svcs}
    t0, last_note = time.time(), 0.0
    while pending:
        for name, svc in list(pending.items()):
            node = _served_nodes(svc)[0]
            cname = svc_cname(name, "head" if svc_world(svc) > 1 else None)
            port = svc.get("port", config.CFG["cluster"]["api_port"])
            if remote.on(node, f"curl -sf --max-time 3 http://localhost:{port}/v1/models -o /dev/null",
                         check=False, capture=True).returncode == 0:
                print(f"[up] {name} loaded ({time.time() - t0:.0f}s)", flush=True)
                del pending[name]
            elif remote.on(node, f"docker inspect -f '{{{{.State.Running}}}}' {cname}",
                           check=False, capture=True).stdout.strip() != "true":
                print(f"[up] ⚠️  {name} exited while loading — not waiting on it", flush=True)
                del pending[name]
        if not pending:
            return
        elapsed = time.time() - t0
        if elapsed > timeout:
            print(f"[up] ⚠️  still loading after {timeout}s: {', '.join(pending)} — continuing", flush=True)
            return
        if elapsed - last_note >= 30:
            last_note = elapsed
            print(f"[up] waiting for {', '.join(pending)} to load before starting its node-mates "
                  f"({elapsed:.0f}s)", flush=True)
        time.sleep(5)


class DockerBackend(Backend):
    def up(self, recipe, timeout=1800):
        waves = launch_waves(recipe["services"])
        for i, wave in enumerate(waves):
            if i:   # only block on EARLIER served services that share a node with this wave
                nodes = {n for s in wave if s["engine"] in SERVED_ENGINES for n in _served_nodes(s)}
                _wait_loaded([s for w in waves[:i] for s in w if s["engine"] in SERVED_ENGINES
                              and nodes & set(_served_nodes(s))], timeout)
            for svc in wave:
                ENGINES[svc["engine"]][0](svc)

    def down(self):
        for _, down in ENGINES.values():
            down()

    def status(self):
        out = {}
        for node in config.NODES:
            r = remote.on(node, f"docker ps --filter name={config.PFX}- "
                                f"--format '{{{{.Names}}}}\\t{{{{.Status}}}}'",
                          capture=True, check=False)
            out[node] = {}
            for line in (r.stdout or "").splitlines():
                name, _, status = line.partition("\t")
                out[node][name.strip()] = status.strip()
        return out

    def logs(self, svc, follow=False, tail=80):
        f = "-f " if follow else ""
        if svc["engine"] in SERVED_ENGINES:
            if svc_world(svc) <= 1:   # single-node (all sglang, single-node vllm): plain container logs
                remote.on(svc.get("node", config.HEAD),
                          f"docker logs --tail {tail} {f}{svc_cname(svc['name'])}", check=False)
            elif svc.get("distributed_backend", "ray") == "mp":
                # mp: `vllm serve` is the head container's PID 1 -> docker logs captures it directly
                remote.on(config.HEAD, f"docker logs --tail {tail} {f}{svc_cname(svc['name'], 'head')}",
                          check=False)
            else:  # Ray: serve runs as a detached exec -> follow its tee'd log file
                t = f"tail -n {tail}" + (" -f" if follow else "")
                remote.on(config.HEAD, f"docker exec {svc_cname(svc['name'], 'head')} "
                                       f"{t} /tmp/vllm-serve.log", check=False)
        else:
            remote.on(svc.get("node", config.HEAD),
                      f"docker logs --tail {tail} {f}{config.PFX}-ollama", check=False)

    def pull(self, recipe):
        hf_models = []
        for svc in recipe["services"]:
            fn = PULLERS.get(svc_provider(svc))
            if not fn:
                print(f"[pull] provider '{svc_provider(svc)}' unsupported — skipping {svc.get('name')}")
                continue
            if fn(svc) and svc_provider(svc) == "hf":
                hf_models.append(svc["model"])
                draft = speculative_model(svc)   # DFlash/EAGLE drafter: download + mirror it too
                if draft and PULLERS["hf"]({**svc, "model": draft}):
                    hf_models.append(draft)
        if hf_models:
            mirror_to_others(hf_models)

    def endpoints(self, recipe, served_from):
        eps = {}
        for node, svcs in services_by_node(recipe).items():
            host = remote.backend_host(node, served_from)
            for s in svcs:
                eps.setdefault(s["served_name"], []).append(f"http://{host}:{s['port']}/v1")
        return eps

    def metrics_targets(self, recipe, served_from):
        targets = []
        for node, svcs in services_by_node(recipe).items():
            host = remote.backend_host(node, served_from)
            for s in svcs:
                if s["engine"] in SERVED_ENGINES:   # vllm + sglang both expose Prometheus at /metrics
                    targets.append({"node": node, "service": s["name"],
                                    "url": f"http://{host}:{s['port']}/metrics"})
        return targets

    def run_workload(self, name, image, command="", node=None, env=None, volumes=None):
        e = " ".join(f"-e {k}={v}" for k, v in (env or {}).items())
        v = " ".join(f"-v {src}:{dst}" for src, dst in (volumes or {}).items())
        remote.on(node or config.HEAD,
                  f"docker rm -f {name} >/dev/null 2>&1 || true; "
                  f"docker run -d --name {name} {e} {v} {image} {command}".strip(), check=False)
