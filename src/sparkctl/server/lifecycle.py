"""Unified-server lifecycle across run modes (cluster.yaml server.mode):

  local   plain process, no Docker (default) — pidfile-managed background pair
          (uvicorn app + LiteLLM child), deps auto-installed into ~/.sparkctl/venv
  docker  csv-server container (built from docker/server.Dockerfile) + csv-litellm sidecar
  k8s     roadmap stub

Started/stopped via `sparkctl serve` / `sparkctl serve stop` — never its own gateway/metrics
commands; whether metrics/grafana run is declared in cluster.yaml."""
import importlib.util
import json
import os
import shlex
import shutil
import sys
import time
from pathlib import Path

import yaml

from sparkctl import config, remote
from sparkctl.recipes import current_recipe, load_recipe
from sparkctl.server import litellm_bridge
from sparkctl.server.metrics import grafana_prometheus_config, scrape_targets

SRV_DIR = Path.home() / ".sparkctl"
SRV_PID = SRV_DIR / "server.pid"
SRV_LOG = SRV_DIR / "server.log"
VENV = SRV_DIR / "venv"
SRV_NAME = f"{config.PFX}-server"
SIDECAR_NAME = f"{config.PFX}-litellm"
SRV_IMAGE = f"{config.PFX}-server:latest"
PROM_NAME, GRAF_NAME = f"{config.PFX}-prometheus", f"{config.PFX}-grafana"
PROM_IMAGE, GRAF_IMAGE = "prom/prometheus:latest", "grafana/grafana:latest"
NET = f"{config.PFX}-server-net"
UNIT = "sparkctl-server.service"


def _mode():
    m = config.SERVER.get("mode", "local")
    if m == "k8s":
        sys.exit("server.mode: k8s is not yet implemented — see docs/backends.md for the roadmap")
    if m not in ("local", "docker"):
        sys.exit(f"unknown server.mode: {m} (local | docker | k8s)")
    return m


def _have_server_deps():
    return all(importlib.util.find_spec(p) for p in ("fastapi", "uvicorn", "httpx"))


def _ensure_venv():
    """Self-contained deps for local mode: create ~/.sparkctl/venv and install sparkctl[server]
    (fastapi/uvicorn/httpx/litellm) from this checkout. One-time; reused thereafter."""
    py = VENV / "bin" / "python"
    # deps present AND sparkctl resolves into this checkout (editable) — else (re)install
    probe = ("import fastapi, uvicorn, httpx, litellm, sparkctl; print(sparkctl.__file__)")
    r = remote.sh(f"{py} -c '{probe}'", check=False, capture=True) if py.exists() else None
    if r and r.returncode == 0 and str(config.ROOT) in (r.stdout or ""):
        return py
    print("[server] first run: creating venv + installing sparkctl[server] (one-time, ~1-2 min)…",
          flush=True)
    remote.sh(f"python3 -m venv {VENV}")
    r = remote.sh(f"{VENV}/bin/pip install --quiet --upgrade pip && "
                  f"{VENV}/bin/pip install --quiet -e '{config.ROOT}[server]'", check=False)
    if r.returncode != 0:
        sys.exit("[server] failed to install server dependencies — see pip output above")
    return py


def _wait_healthy(timeout=120):
    """Block until /healthz reports the litellm child up AND /v1 answers."""
    port = config.SERVER.get("port", 8080)
    t0 = time.time()
    while time.time() - t0 < timeout:
        h = remote.sh(f"curl -s --max-time 3 http://localhost:{port}/healthz",
                      check=False, capture=True)
        try:
            litellm_up = json.loads(h.stdout or "{}").get("litellm") == "up"
        except ValueError:
            litellm_up = False
        if litellm_up and remote.sh(
                f"curl -sf --max-time 3 http://localhost:{port}/v1/models -o /dev/null",
                check=False, capture=True).returncode == 0:
            print(f"[server] ready in {time.time() - t0:.0f}s ✅")
            return
        time.sleep(2)
    sys.exit(f"[server] not ready after {timeout}s — check {SRV_LOG}")


def serve_start(foreground=False, wait=False):
    mode = _mode()
    load_recipe(current_recipe())          # fail fast on a broken recipe
    if mode == "docker":
        _docker_up()
        if wait:
            _wait_healthy()
        return
    port = config.SERVER.get("port", 8080)
    if not foreground and _unit_active():
        # the boot unit owns the server here: stopping its process makes systemd (Restart=always)
        # respawn it with the new recipe — a nohup'd second copy would just fight it for the port
        serve_stop(quiet=True)
        print(f"[server] restarted via {UNIT} on :{port} (mode: local)")
        if wait:
            _wait_healthy()
        return
    serve_stop(quiet=True)                 # idempotent restart
    if foreground:                         # pidfile = this process (execve keeps the pid), so
        SRV_DIR.mkdir(parents=True, exist_ok=True)    # `serve` restarts / `serve stop` can find it
        SRV_PID.write_text(f"{os.getpid()}\n")
    # the app supervises a litellm child — make sure the CLI exists somewhere we can find it
    if not (VENV / "bin" / "litellm").exists() and not shutil.which("litellm"):
        _ensure_venv()
    if foreground and _have_server_deps():
        _grafana_up_if_enabled()
        from sparkctl.server.__main__ import main as run_server
        return run_server()
    py = Path(sys.executable) if _have_server_deps() else _ensure_venv()
    # `python -m sparkctl.server` needs `sparkctl` importable. In a source checkout it's NOT installed
    # into `py` (only bin/sparkctl injects src onto sys.path), so put src on PYTHONPATH for the spawned
    # process — else it dies with ModuleNotFoundError and leaves a stale pidfile. Harmless when
    # pip-installed (a non-existent src/ is ignored and sparkctl resolves from site-packages).
    src = config.ROOT / "src"
    pythonpath = os.pathsep.join(p for p in (str(src), os.environ.get("PYTHONPATH", "")) if p)
    if foreground:                          # deps live in the venv — re-exec into it
        _grafana_up_if_enabled()
        env = {**os.environ, "SPARKCTL_ROOT": str(config.ROOT), "PYTHONPATH": pythonpath}
        os.execve(str(py), [str(py), "-m", "sparkctl.server"], env)
    SRV_DIR.mkdir(parents=True, exist_ok=True)
    remote.sh(f"SPARKCTL_ROOT={config.ROOT} PYTHONPATH={shlex.quote(pythonpath)} "
              f"nohup {py} -m sparkctl.server > {SRV_LOG} 2>&1 & echo $! > {SRV_PID}", check=False)
    _grafana_up_if_enabled()
    where = "local" if config.SELF is None else config.SELF
    print(f"[server] up on {where}:{port} (mode: local) — /v1 /metrics /dash /healthz")
    print(f"[server] log: {SRV_LOG}   stop: sparkctl serve stop")
    if wait:
        _wait_healthy()


def _unit_active():
    """True when the boot unit (sparkctl-server.service) is running the server on this machine."""
    return shutil.which("systemctl") is not None and remote.sh(
        f"systemctl is-active --quiet {UNIT}", check=False).returncode == 0


def serve_stop(quiet=False):
    if SRV_PID.exists():
        remote.sh(f"kill $(cat {SRV_PID}) 2>/dev/null || true", check=False)
        SRV_PID.unlink(missing_ok=True)
    # csv-gateway is the pre-unified-server container name — clearing it makes takeover seamless
    remote.sh(f"docker rm -f {SRV_NAME} {SIDECAR_NAME} {PROM_NAME} {GRAF_NAME} "
              f"{config.PFX}-gateway >/dev/null 2>&1 || true", check=False)
    if not quiet:
        print("[server] down")
        if _unit_active():
            print(f"[server] note: {UNIT} will respawn it — `sudo systemctl stop {UNIT}` to keep it down")


def serve_status(args=None):
    port = config.SERVER.get("port", 8080)
    if SRV_PID.exists():
        remote.sh(f"ps -p $(cat {SRV_PID}) -o pid=,etime=,comm= 2>/dev/null "
                  f"|| echo 'local server not running (stale pidfile)'", check=False)
    remote.sh(f"docker ps --filter name={config.PFX}-server --filter name={SIDECAR_NAME} "
              f"--filter name={PROM_NAME} --filter name={GRAF_NAME} "
              f"--format 'table {{{{.Names}}}}\\t{{{{.Status}}}}' 2>/dev/null || true", check=False)
    print(f"--- /healthz (:{port}) ---", flush=True)
    remote.sh(f"curl -s --max-time 5 http://localhost:{port}/healthz || echo 'server not ready'",
              check=False)
    print(f"\n--- /v1/models (:{port}) ---", flush=True)
    remote.sh(f"curl -s --max-time 5 http://localhost:{port}/v1/models || echo 'gateway not ready'",
              check=False)


def serve_config(args=None):
    settings = config.SERVER
    recipe = load_recipe(current_recipe())
    served_from = settings.get("host", "local") if config.SELF is None else config.SELF
    print("# LiteLLM route table")
    print(yaml.safe_dump(litellm_bridge.litellm_config(recipe, settings.get("host", "local"), settings),
                         sort_keys=False))
    print("# metrics scrape targets (aggregated on the server's /metrics)")
    for t in scrape_targets(recipe, served_from):
        print(f"#   {t['node']}/{t['service']} -> {t['url']}")
    if settings.get("grafana", {}).get("enabled"):
        print("\n# prometheus config for the optional grafana stack")
        print(yaml.safe_dump(grafana_prometheus_config(_server_scrape_addr(), _scrape_interval()),
                             sort_keys=False))


def serve_tunnel(args=None):
    """Forward localhost:<port> on the dev machine to a node-hosted server."""
    host = config.SERVER.get("host", "local")
    port = config.SERVER.get("port", 8080)
    if host == "local":
        print(f"server host is 'local' — already at http://localhost:{port}")
        return
    print(f"tunneling http://localhost:{port} -> {host}:{port}  (Ctrl-C to stop)", flush=True)
    target = (remote.server_host_target(host) if host in config.SERVER_HOSTS
              else f"{config.USER}@{remote.node_addr(host)}")
    remote.sh(f"ssh -N -L {port}:localhost:{port} {target}", check=False)


def server_running():
    if SRV_PID.exists() and remote.sh(f"kill -0 $(cat {SRV_PID}) 2>/dev/null",
                                      check=False).returncode == 0:
        return True
    return bool((remote.sh(f"docker ps -q --filter name={SRV_NAME} 2>/dev/null", check=False,
                           capture=True).stdout or "").strip())


def _is_server_host():
    """True if the unified server is meant to run on THIS machine — either server.host is 'local'
    and we're the dev machine (config.SELF is None), or server.host names the node we're on."""
    host = config.SERVER.get("host", "local")
    return ((host == "local" and config.SELF is None) or host == config.SELF
            or host == config.SERVER_SELF)


def refresh_or_start_server():
    """Post-apply: make the unified server reflect the new deployment. If it's already running here,
    restart it (so LiteLLM routes + scrape targets match the new recipe); if it's down but this
    machine is the designated server host, START it — an apply with no gateway serving nothing is a
    footgun. No-op on machines that don't host the server (e.g. the head when host is 'local')."""
    if server_running():
        print("[server] running — restarting for the new deployment")
        serve_start()
    elif _is_server_host():
        print("[server] not running — starting it for the new deployment")
        serve_start()


# ---------------------------------------------------------------- docker mode
def _docker_up():
    port = config.SERVER.get("port", 8080)
    settings = config.SERVER
    recipe = load_recipe(current_recipe())
    cfg_file, _ = litellm_bridge.write_config(recipe, settings)
    serve_stop(quiet=True)
    root = config.ROOT if config.SELF is None else config.REMOTE
    if not (remote.sh(f"docker image inspect {SRV_IMAGE} >/dev/null 2>&1", check=False).returncode == 0):
        print(f"[server] building {SRV_IMAGE}…", flush=True)
        remote.sh(f"docker build -t {SRV_IMAGE} -f {root}/docker/server.Dockerfile {root}")
    if config.SELF is None:   # dev machine: user bridge; sidecar reachable by container name
        remote.sh(f"docker network inspect {NET} >/dev/null 2>&1 || docker network create {NET}",
                  check=False)
        remote.sh(f"docker run -d --restart unless-stopped --name {SIDECAR_NAME} --network {NET} "
                  f"-v {cfg_file}:/app/config.yaml {litellm_bridge.LITELLM_IMAGE} "
                  f"--config /app/config.yaml")
        remote.sh(f"docker run -d --restart unless-stopped --name {SRV_NAME} --network {NET} -p {port}:{port} "
                  f"-v {config.ROOT}:/workspace:ro -e SPARKCTL_ROOT=/workspace "
                  f"-e SPARKCTL_LITELLM_URL=http://{SIDECAR_NAME}:4000 {SRV_IMAGE}")
    else:                     # node: host networking; sidecar on loopback internal port
        internal = litellm_bridge.internal_port(settings)
        remote.sh(f"docker run -d --restart unless-stopped --name {SIDECAR_NAME} --network host "
                  f"-v {cfg_file}:/app/config.yaml {litellm_bridge.LITELLM_IMAGE} "
                  f"--config /app/config.yaml --host 127.0.0.1 --port {internal}")
        remote.sh(f"docker run -d --restart unless-stopped --name {SRV_NAME} --network host "
                  f"-v {config.REMOTE}:/workspace:ro -e SPARKCTL_ROOT=/workspace "
                  f"-e SPARKCTL_LITELLM_URL=http://127.0.0.1:{internal} {SRV_IMAGE}")
    _grafana_up_if_enabled()
    where = "local" if config.SELF is None else config.SELF
    print(f"[server] up on {where}:{port} (mode: docker) — /v1 /metrics /dash /healthz")


# ---------------------------------------------------------------- optional grafana extra
def _scrape_interval():
    return config.SERVER.get("metrics", {}).get("scrape_interval_s", 10)


def _server_scrape_addr():
    """Address the Prometheus container uses to reach the unified server's /metrics."""
    port = config.SERVER.get("port", 8080)
    if config.SELF is None:
        if config.SERVER.get("mode", "local") == "docker":
            return f"{SRV_NAME}:{port}"                # same bridge network
        return f"host.docker.internal:{port}"          # container -> host process (Docker Desktop)
    return f"localhost:{port}"                          # node: host networking everywhere


def _grafana_up_if_enabled():
    gr = config.SERVER.get("grafana", {})
    if not gr.get("enabled", False):
        return
    pport, gport = gr.get("prometheus_port", 9090), gr.get("grafana_port", 3000)
    d = SRV_DIR / "metrics" if config.SELF is None else Path(config.STATE) / "metrics"
    (d / "provisioning" / "datasources").mkdir(parents=True, exist_ok=True)
    (d / "provisioning" / "dashboards").mkdir(parents=True, exist_ok=True)
    (d / "prometheus.yml").write_text(
        yaml.safe_dump(grafana_prometheus_config(_server_scrape_addr(), _scrape_interval()),
                       sort_keys=False))
    if config.SELF is None:
        remote.sh(f"docker network inspect {NET} >/dev/null 2>&1 || docker network create {NET}",
                  check=False)
        # host-gateway: Docker Desktop resolves host.docker.internal itself; Linux needs the mapping
        net = f"--network {NET} --add-host=host.docker.internal:host-gateway"
        pmap, gmap = f"-p {pport}:9090", f"-p {gport}:3000"
        prom_listen, graf_port, ds_url = "9090", "3000", f"http://{PROM_NAME}:9090"
    else:
        net, pmap, gmap = "--network host", "", ""
        prom_listen, graf_port, ds_url = str(pport), str(gport), f"http://localhost:{pport}"
    (d / "provisioning" / "datasources" / "ds.yml").write_text(yaml.safe_dump({
        "apiVersion": 1, "datasources": [{"name": "Prometheus", "type": "prometheus",
        "uid": "prometheus", "access": "proxy", "url": ds_url, "isDefault": True}]}))
    (d / "provisioning" / "dashboards" / "provider.yml").write_text(yaml.safe_dump({
        "apiVersion": 1, "providers": [{"name": "sparkctl", "type": "file", "allowUiUpdates": True,
        "options": {"path": "/var/lib/grafana/dashboards"}}]}))
    dash_src = f"{config.REMOTE if config.SELF is not None else config.ROOT}/docker/grafana/dashboards"
    remote.sh(f"docker rm -f {PROM_NAME} {GRAF_NAME} >/dev/null 2>&1 || true", check=False)
    remote.sh(f"docker run -d --restart unless-stopped --name {PROM_NAME} {net} {pmap} "
              f"-v {d}/prometheus.yml:/etc/prometheus/prometheus.yml -v {config.PFX}-prom-data:/prometheus "
              f"{PROM_IMAGE} --config.file=/etc/prometheus/prometheus.yml "
              f"--storage.tsdb.retention.time={gr.get('retention', '15d')} "
              f"--web.listen-address=:{prom_listen}", check=False)
    remote.sh(f"docker run -d --restart unless-stopped --name {GRAF_NAME} {net} {gmap} "
              f"-e GF_AUTH_ANONYMOUS_ENABLED=true -e GF_AUTH_ANONYMOUS_ORG_ROLE=Admin "
              f"-e GF_SERVER_HTTP_PORT={graf_port} "
              f"-v {d}/provisioning:/etc/grafana/provisioning "
              f"-v {dash_src}:/var/lib/grafana/dashboards {GRAF_IMAGE}", check=False)
    print(f"[server] grafana extra: Prometheus :{pport} + Grafana :{gport} (anon admin)")


def cmd_serve(args):
    action = getattr(args, "action", None) or "start"
    {"start": lambda a: serve_start(foreground=getattr(a, "foreground", False),
                                    wait=getattr(a, "wait", False)),
     "stop": lambda a: serve_stop(),
     "status": serve_status,
     "config": serve_config,
     "tunnel": serve_tunnel}[action](args)
