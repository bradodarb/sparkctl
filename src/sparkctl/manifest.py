"""Per-node active.json manifests — the deployment-drift detection mechanism used by `status`."""
import base64
import datetime
import json

from sparkctl import config, remote
from sparkctl.recipes import current_recipe, load_recipe, recipe_hash, services_by_node


def write_active_manifest(recipe_name):
    """Persist per-node {STATE}/active.json (recipe + sha256 + services + timestamp) so the gateway
    and `status` can verify a node is running the *intended* recipe version, not just 'something up'."""
    recipe = load_recipe(recipe_name)
    h = recipe_hash(recipe_name)
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for node, svcs in services_by_node(recipe).items():
        manifest = {"recipe": recipe_name, "recipe_sha256": h, "node": node,
                    "services": svcs, "started_at": ts}
        b64 = base64.b64encode(json.dumps(manifest).encode()).decode()
        remote.on(node, f"mkdir -p {config.STATE} && echo {b64} | base64 -d > {config.STATE}/active.json",
                  check=False)


def verify_deployment(node):
    """Read a node's active manifest; return (recipe_name_or_None, hash_ok) vs the current recipe."""
    r = remote.on(node, f"cat {config.STATE}/active.json 2>/dev/null", capture=True, check=False)
    if r.returncode != 0 or not (r.stdout or "").strip():
        return (None, False)
    try:
        man = json.loads(r.stdout)
    except Exception:
        return (None, False)
    cur = current_recipe()
    want = recipe_hash(cur) if (config.ROOT / "recipes" / f"{cur}.yaml").exists() else None
    return (man.get("recipe"), man.get("recipe") == cur and man.get("recipe_sha256") == want)
