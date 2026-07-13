"""Execution + addressing layer: run commands locally or on nodes over SSH."""
import shlex
import subprocess

from sparkctl import config


def sh(cmd, check=True, capture=False):
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=capture)


def node_addr(node):
    """Address to reach a node. From the control machine use lan_ip (fast, no .local mDNS stall);
    from another node use ssh_host (fabric — .local doesn't resolve node-to-node)."""
    v = config.NODES[node]
    return (v.get("lan_ip") or v["host"]) if config.SELF is None else (v.get("ssh_host") or v["host"])


def on(node, cmd, **kw):
    """Run a shell command on a node — locally if it's the node we're on, else over SSH.
    shlex.quote (single quotes), NOT double quotes: the command must reach the remote shell
    verbatim — $vars and $() must never expand on the calling machine."""
    if node == config.SELF:
        return sh(cmd, **kw)
    return sh(f"ssh -o BatchMode=yes {config.USER}@{node_addr(node)} {shlex.quote(cmd)}", **kw)


def backend_host(node, served_from):
    """Address a gateway/scraper uses to reach a node's API. From the dev machine ('local') use
    lan_ip; from a node use the fabric IP (both peers reachable over the fabric)."""
    if served_from == "local":
        return config.NODES[node].get("lan_ip") or config.NODES[node]["host"]
    return config.NODES[node]["fabric_ip"]
