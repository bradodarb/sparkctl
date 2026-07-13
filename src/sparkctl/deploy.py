"""Deployment: push the repo to every node (day-to-day) and one-time provisioning (--init).

Replaces the old deploy.sh — node list, user, and paths all come from cluster.yaml through the
same PyYAML load as everything else (no yq, no second config parser)."""
import base64
import shlex

from sparkctl import config, remote

EXCLUDES = "--exclude 'state/' --exclude '*.log' --exclude '.git' --exclude '__pycache__'"


def deploy():
    """rsync the source-of-truth repo to every node. Safe to re-run after any edit."""
    root = config.ROOT
    for node in config.NODES:
        a = remote.node_addr(node)
        print(f"[deploy] {node} -> {config.USER}@{a}:{config.REMOTE}")
        remote.sh(f"rsync -az --delete {EXCLUDES} "
                  f"{root}/bin {root}/src {root}/recipes {root}/docker {root}/systemd "
                  f"{root}/pyproject.toml {root}/cluster.yaml {root}/current "
                  f"{config.USER}@{a}:{config.REMOTE}/")


def _render_unit(name):
    """Fill the shipped unit template with this cluster's user + remote path."""
    tpl = (config.ROOT / "systemd" / name).read_text()
    return tpl.replace("{{REMOTE}}", config.REMOTE).replace("{{USER}}", config.USER)


def _install_unit(node, name):
    b64 = base64.b64encode(_render_unit(name).encode()).decode()
    cmd = (f"echo {b64} | base64 -d | sudo tee /etc/systemd/system/{name} >/dev/null && "
           f"sudo systemctl daemon-reload && sudo systemctl enable {name}")
    remote.sh(f"ssh -t {config.USER}@{remote.node_addr(node)} {shlex.quote(cmd)}")


def deploy_init():
    """One-time provisioning: dirs, docker group, PyYAML, boot unit. Interactive — sudo on the
    nodes may prompt for a password (hence ssh -t)."""
    for node in config.NODES:
        a = remote.node_addr(node)
        print(f"[deploy --init] provisioning {node} ({a}) — sudo may prompt for a password")
        cmd = (f"sudo mkdir -p {config.REMOTE} && "
               f"sudo chown -R {config.USER}:{config.USER} {config.REMOTE}; "
               f"id -nG | tr ' ' '\\n' | grep -qx docker || sudo usermod -aG docker {config.USER}; "
               f"python3 -c 'import yaml' 2>/dev/null || pip3 install --user --quiet pyyaml "
               f"|| sudo apt-get install -y python3-yaml")
        remote.sh(f"ssh -t {config.USER}@{a} {shlex.quote(cmd)}")
    deploy()
    print(f"[deploy --init] installing boot unit on {config.HEAD}")
    _install_unit(config.HEAD, "sparkctl.service")
    # retire the pre-rename unit if this cluster still has it
    old = ("sudo systemctl disable cluster-serve.service 2>/dev/null; "
           "sudo rm -f /etc/systemd/system/cluster-serve.service; sudo systemctl daemon-reload")
    remote.sh(f"ssh -t {config.USER}@{remote.node_addr(config.HEAD)} {shlex.quote(old)}",
              check=False)
    srv_host = config.SERVER.get("host", "local")
    if srv_host != "local":
        print(f"[deploy --init] installing server unit on {srv_host}")
        _install_unit(srv_host, "sparkctl-server.service")
    print("[deploy --init] done — check: systemctl status sparkctl (on the head)")
