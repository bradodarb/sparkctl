"""Secrets (HF_TOKEN etc.): one env file per machine — ~/.sparkctl/secrets.env, mode 0600, never
in the repo. `sparkctl secret set` writes it locally and syncs it to every node over rsync (values
never touch argv/ps), so the head has what downloads need. Consumers: the HF download container
(sourced), the ollama container (--env-file), and the litellm child (merged into its env)."""
from pathlib import Path

from sparkctl import config, remote

PATH = Path.home() / ".sparkctl" / "secrets.env"
NODE_PATH = ".sparkctl/secrets.env"     # relative to $HOME on nodes


def load(path=None):
    """Parse KEY=value lines (comments/blanks ignored) into a dict. Missing file -> {}."""
    p = Path(path or PATH)
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        out[k.strip()] = v.strip()
    return out


def save(secrets, path=None):
    p = Path(path or PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch(mode=0o600, exist_ok=True)
    p.chmod(0o600)                       # tighten pre-existing files too
    p.write_text("".join(f"{k}={v}\n" for k, v in sorted(secrets.items())))


def sync_to_nodes():
    """rsync the local secrets file to every node (0600). File transfer, not a shell command —
    values stay out of process listings on both ends."""
    if not PATH.exists():
        return
    for node in config.NODES:
        a = remote.node_addr(node)
        remote.on(node, "mkdir -p ~/.sparkctl", check=False)
        # `-a` already preserves the source's 0600; set it explicitly too since older rsync (macOS
        # ships 2.6.9) rejects --chmod. Perms enforced on the node, not via a version-specific flag.
        remote.sh(f"rsync -a {PATH} {config.USER}@{a}:{NODE_PATH}")
        remote.on(node, f"chmod 600 {NODE_PATH}", check=False)
        print(f"[secret] synced -> {node}")
