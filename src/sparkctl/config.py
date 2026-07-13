"""Cluster configuration — cluster.yaml at the repo root is the single source of truth.

Loaded once at import (every command needs it). Root discovery order:
  1. $SPARKCTL_ROOT (set by the bin/sparkctl shim and the systemd unit)
  2. nearest ancestor of the CWD containing cluster.yaml
  3. the repo containing this package (editable install from a checkout)
"""
import getpass
import os
import socket
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("PyYAML missing — run: pip3 install --user pyyaml")


def _find_root():
    env = os.environ.get("SPARKCTL_ROOT")
    if env:
        return Path(env).resolve()
    for d in (Path.cwd(), *Path.cwd().parents):
        if (d / "cluster.yaml").exists():
            return d
    dev = Path(__file__).resolve().parents[2]          # <repo>/src/sparkctl/config.py
    if (dev / "cluster.yaml").exists():
        return dev
    sys.exit("cluster.yaml not found — run from a cluster repo or set SPARKCTL_ROOT")


ROOT = _find_root()
CFG = yaml.safe_load((ROOT / "cluster.yaml").read_text())
if "gateway" in CFG or "metrics" in CFG:
    sys.exit("cluster.yaml uses the removed gateway:/metrics: blocks — the unified server replaced "
             "them; configure server: instead (see cluster.yaml.example)")
SERVER = CFG.get("server", {})
HEAD = CFG["cluster"]["head"]
IMAGE = CFG["cluster"]["container_image"]
CACHE = CFG["cluster"]["model_cache"]
NODES = CFG["nodes"]
FABRIC = CFG.get("fabric", {})       # optional: single-node clusters have no inter-node fabric
PFX = CFG["cluster"].get("container_prefix", "spark")      # container name prefix
DEPLOY = CFG.get("deploy", {})
USER = DEPLOY.get("user", getpass.getuser())
REMOTE = DEPLOY.get("remote_path", "/opt/sparkctl")
WORKER = next((n for n in NODES if n != HEAD), None)       # the (first) non-head node
STATE = f"{REMOTE}/state"                                  # on-node state dir (active.json, etc.)
DL = CFG.get("download", {})                               # download robustness knobs

# Which node are we on? None => control machine (e.g. the Mac).
_HOST = socket.gethostname().split(".")[0]
SELF = next((n for n, v in NODES.items()
             if _HOST == str(v.get("host", n)).split(".")[0] or _HOST == n), None)
