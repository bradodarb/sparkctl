"""DockerBackend — the default: services run as docker containers over SSH, weights distributed
by verified download + fabric rsync, boot persistence via the systemd unit on the head."""
from sparkctl import config, remote
from sparkctl.backends.base import Backend
from sparkctl.distribution import PULLERS, mirror_to_others
from sparkctl.engines import ENGINES
from sparkctl.engines.vllm import svc_cname
from sparkctl.recipes import services_by_node, svc_provider, svc_world


class DockerBackend(Backend):
    def up(self, recipe):
        for svc in recipe["services"]:
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
        if svc["engine"] == "vllm":
            if svc_world(svc) <= 1:
                remote.on(svc.get("node", config.HEAD),
                          f"docker logs --tail {tail} {f}{svc_cname(svc['name'])}", check=False)
            else:  # multinode serve runs as a detached exec -> follow its tee'd log file
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
                if s["engine"] == "vllm":
                    targets.append({"node": node, "service": s["name"],
                                    "url": f"http://{host}:{s['port']}/metrics"})
        return targets

    def run_workload(self, name, image, command="", node=None, env=None, volumes=None):
        e = " ".join(f"-e {k}={v}" for k, v in (env or {}).items())
        v = " ".join(f"-v {src}:{dst}" for src, dst in (volumes or {}).items())
        remote.on(node or config.HEAD,
                  f"docker rm -f {name} >/dev/null 2>&1 || true; "
                  f"docker run -d --name {name} {e} {v} {image} {command}".strip(), check=False)
