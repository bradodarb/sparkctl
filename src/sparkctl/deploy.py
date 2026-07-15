"""Deployment: push the repo to every node (day-to-day) and one-time provisioning (--init).

Replaces the old deploy.sh — node list, user, and paths all come from cluster.yaml through the
same PyYAML load as everything else (no yq, no second config parser)."""
import base64
import shlex
import sys

from sparkctl import config, remote, secrets

EXCLUDES = "--exclude 'state/' --exclude '*.log' --exclude '.git' --exclude '__pycache__'"


def deploy():
    """rsync the source-of-truth repo to every node. Safe to re-run after any edit."""
    root = config.ROOT
    for node in config.NODES:
        a = remote.node_addr(node)
        print(f"[deploy] {node} -> {config.USER}@{a}:{config.REMOTE}")
        r = remote.sh(f"rsync -az --delete {EXCLUDES} "
                      f"{root}/bin {root}/src {root}/recipes {root}/docker {root}/systemd "
                      f"{root}/pyproject.toml {root}/cluster.yaml {root}/current "
                      f"{config.USER}@{a}:{config.REMOTE}/", check=False)
        if r.returncode != 0:
            sys.exit(f"[deploy] rsync to {node} failed (exit {r.returncode}). If {config.REMOTE} "
                     f"doesn't exist on the node yet, provision it once with: "
                     f"sparkctl deploy --init  (creates it with sudo and installs the boot daemon)")
    secrets.sync_to_nodes()   # no-op when no secrets are set; keeps late-added nodes covered


def _render_unit(name):
    """Fill the shipped unit template with this cluster's user + remote path."""
    tpl = (config.ROOT / "systemd" / name).read_text()
    return tpl.replace("{{REMOTE}}", config.REMOTE).replace("{{USER}}", config.USER)


def _unit_cmd(name):
    """Shell fragment installing + enabling a rendered unit (values never hit argv unrendered)."""
    b64 = base64.b64encode(_render_unit(name).encode()).decode()
    return (f"echo {b64} | base64 -d | sudo tee /etc/systemd/system/{name} >/dev/null && "
            f"sudo systemctl enable {name}")


def deploy_init():
    """One-time provisioning: dirs, docker group, PyYAML, boot/server units, legacy retirement —
    ONE interactive ssh session per node, so sudo prompts at most once per node."""
    srv_host = config.SERVER.get("host", "local")
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
    deploy()
    print("[deploy --init] done — check: systemctl status sparkctl (on the head)")
