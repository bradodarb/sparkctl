"""Deployment: push the repo to every node (day-to-day) and one-time provisioning (--init).

Replaces the old deploy.sh — node list, user, and paths all come from cluster.yaml through the
same PyYAML load as everything else (no yq, no second config parser)."""
import base64
import shlex
import sys

from sparkctl import config, remote, secrets

EXCLUDES = "--exclude 'state/' --exclude '*.log' --exclude '.git' --exclude '__pycache__'"


def _repo_files():
    root = config.ROOT
    return (f"{root}/bin {root}/src {root}/recipes {root}/docker {root}/systemd "
            f"{root}/pyproject.toml {root}/cluster.yaml {root}/current")


def deploy():
    """rsync the source-of-truth repo to every node. Safe to re-run after any edit."""
    for node in config.NODES:
        a = remote.node_addr(node)
        print(f"[deploy] {node} -> {config.USER}@{a}:{config.REMOTE}")
        r = remote.sh(f"rsync -az --delete {EXCLUDES} {_repo_files()} "
                      f"{config.USER}@{a}:{config.REMOTE}/", check=False)
        if r.returncode != 0:
            sys.exit(f"[deploy] rsync to {node} failed (exit {r.returncode}). If {config.REMOTE} "
                     f"doesn't exist on the node yet, provision it once with: "
                     f"sparkctl deploy --init  (creates it with sudo and installs the boot daemon)")
    secrets.sync_to_nodes()   # no-op when no secrets are set; keeps late-added nodes covered


def deploy_server_host(name):
    """rsync the repo (+ secrets, for the LiteLLM child) to a non-node server host."""
    t, path = remote.server_host_target(name), remote.server_host_path(name)
    print(f"[deploy] {name} (server host) -> {t}:{path}")
    r = remote.sh(f"rsync -az --delete {EXCLUDES} {_repo_files()} {t}:{path}/", check=False)
    if r.returncode != 0:
        sys.exit(f"[deploy] rsync to {name} failed (exit {r.returncode}). If {path} doesn't exist "
                 f"there yet, provision it once with: sparkctl deploy --init")
    if secrets.PATH.exists():
        remote.sh(f"ssh -o BatchMode=yes {t} 'mkdir -p ~/.sparkctl' && "
                  f"rsync -a {secrets.PATH} {t}:{secrets.NODE_PATH} && "
                  f"ssh -o BatchMode=yes {t} 'chmod 600 ~/{secrets.NODE_PATH}'", check=False)


def _render_unit(name, user=None, remote_path=None):
    """Fill the shipped unit template with this cluster's user + remote path."""
    tpl = (config.ROOT / "systemd" / name).read_text()
    return (tpl.replace("{{REMOTE}}", remote_path or config.REMOTE)
               .replace("{{USER}}", user or config.USER))


def _unit_cmd(name, user=None, remote_path=None):
    """Shell fragment installing + enabling a rendered unit (values never hit argv unrendered)."""
    b64 = base64.b64encode(_render_unit(name, user, remote_path).encode()).decode()
    return (f"echo {b64} | base64 -d | sudo tee /etc/systemd/system/{name} >/dev/null && "
            f"sudo systemctl enable {name}")


def deploy_init(server_host_only=False):
    """One-time provisioning: dirs, docker group, PyYAML, boot/server units, legacy retirement —
    ONE interactive ssh session per node, so sudo prompts at most once per node.
    server_host_only: provision just the non-node server host (nodes untouched)."""
    srv_host = config.SERVER.get("host", "local")
    if server_host_only:
        if srv_host not in config.SERVER_HOSTS:
            sys.exit(f"[deploy --init] server.host '{srv_host}' is not a server.hosts entry")
        return _init_server_host(srv_host)
    for node in config.NODES:
        a = remote.node_addr(node)
        print(f"[deploy --init] provisioning {node} ({a}) — sudo may prompt once")
        steps = [
            f"sudo mkdir -p {config.REMOTE}",
            f"sudo chown -R {config.USER}:{config.USER} {config.REMOTE}",
            f"id -nG | tr ' ' '\\n' | grep -qx docker || sudo usermod -aG docker {config.USER}",
            "python3 -c 'import yaml' 2>/dev/null || pip3 install --user --quiet pyyaml "
            "|| sudo apt-get install -y python3-yaml",
            # headless serving node: boot to multi-user (no desktop) to free unified memory
            # and keep gdm from starting — GB10 shares system+GPU RAM, every GB counts.
            "sudo systemctl set-default multi-user.target",
            "sudo systemctl mask gdm.service 2>/dev/null || true",
            # retire a pre-rename install if this cluster still has one
            "sudo systemctl disable cluster-serve.service 2>/dev/null",
            "sudo rm -f /etc/systemd/system/cluster-serve.service",
            "sudo rm -rf /opt/cluster-serve",
        ]
        if node == config.HEAD:
            steps.append(_unit_cmd("sparkctl.service"))
        if node == srv_host:
            steps.append(_unit_cmd("sparkctl-server.service"))
        steps.append("sudo systemctl daemon-reload")
        remote.sh(f"ssh -t {config.USER}@{a} {shlex.quote('; '.join(steps) + '; true')}",
                  check=False)
    if srv_host in config.SERVER_HOSTS:
        _init_server_host(srv_host)
    deploy()
    print("[deploy --init] done — check: systemctl status sparkctl (on the head)")


def _init_server_host(name):
    """Provision a non-node server host: repo dir, docker group (grafana extra), venv support,
    and the boot unit. Not a serving node — no headless target, no gdm masking."""
    t, path = remote.server_host_target(name), remote.server_host_path(name)
    user = t.split("@", 1)[0]
    print(f"[deploy --init] provisioning server host {name} ({t}) — sudo may prompt once")
    steps = [
        f"sudo mkdir -p {path}",
        f"sudo chown -R {user}:{user} {path}",
        f"id -nG | tr ' ' '\\n' | grep -qx docker || sudo usermod -aG docker {user}",
        "python3 -c 'import yaml' 2>/dev/null || sudo apt-get install -y python3-yaml",
        "python3 -c 'import ensurepip' 2>/dev/null || sudo apt-get install -y python3-venv",
        # system engine, not a (possibly stopped) Docker Desktop context — the grafana extra and
        # `serve stop` run docker non-interactively as this user
        "docker context use default >/dev/null 2>&1 || true",
        _unit_cmd("sparkctl-server.service", user, path),
        "sudo systemctl daemon-reload",
    ]
    # an active ufw silently drops the laptops — open the server's ports to its own subnet only
    cidr = config.SERVER_HOSTS[name].get("allow_from")
    if cidr:
        ports = [config.SERVER.get("port", 8080)]
        if config.SERVER.get("grafana", {}).get("enabled"):
            ports.append(config.SERVER["grafana"].get("grafana_port", 3000))
        steps += [f"(! sudo ufw status | grep -q 'Status: active' || "
                  f"sudo ufw allow from {cidr} to any port {p} proto tcp)" for p in ports]
    remote.sh(f"ssh -t {t} {shlex.quote('; '.join(steps) + '; true')}", check=False)
    deploy_server_host(name)
    remote.sh(f"ssh -t {t} 'sudo systemctl restart sparkctl-server'", check=False)
