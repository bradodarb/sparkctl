"""Recipes: load, hash, and map services onto nodes. The `current` file is the active pointer."""
import copy
import hashlib
import sys

import yaml

from sparkctl import config

# Engines that run a per-service, OpenAI-compatible container on its own port with a /metrics
# endpoint (as opposed to ollama's single shared container). The stack treats these uniformly for
# placement, logs, metrics scraping, image provisioning, and gateway registration.
SERVED_ENGINES = ("vllm", "sglang")


def current_recipe():
    return (config.ROOT / "current").read_text().strip()


def load_recipe(name):
    p = config.ROOT / "recipes" / f"{name}.yaml"
    if not p.exists():
        sys.exit(f"no such recipe: {name} ({p})")
    return expand_services(yaml.safe_load(p.read_text()))


def svc_world(svc):
    p = svc.get("parallel", {})
    return p.get("tensor", 1) * p.get("pipeline", 1)   # total ranks (1 GPU/node on Spark)


def expand_services(recipe):
    """Fan out any service that declares `nodes: [a, b, ...]` into one single-node replica per node.

    Each replica is `node:`-pinned, gets a unique name `{name}-{node}` (container names derive from
    the service name, so they must differ), but KEEPS the same `served_name` — that shared name is
    exactly what the gateway pools into a load balancer (see server/litellm_bridge). So one compact
    service block becomes N identical single-node replicas the rest of the stack already understands.

    Services with a scalar `node:` (or neither) pass through untouched → existing recipes are
    unaffected. Called once at load_recipe time so every downstream consumer sees only single-node
    services. Deep-copies each replica so mutating one never aliases another's env/extra_args.
    Exits (not just warns) on the contradictory combinations so a skipped `validate` can't push a
    broken config through `apply`; validate reports the same cases with friendlier phrasing."""
    services = recipe.get("services")
    if not isinstance(services, list):
        return recipe
    out = []
    for svc in services:
        nodes = svc.get("nodes")
        if not nodes:
            out.append(svc)
            continue
        name = svc.get("name")
        if svc.get("node"):
            sys.exit(f"recipe service '{name}': set either 'node:' or 'nodes:', not both")
        if not isinstance(nodes, list):
            sys.exit(f"recipe service '{name}': 'nodes' must be a list of node names, e.g. [coach, ref]")
        if svc_world(svc) > 1:
            sys.exit(f"recipe service '{name}': 'nodes:' replicates a single-node service per node, but "
                     f"parallel.tensor/pipeline>1 already spans nodes — drop 'parallel' or 'nodes:'")
        for n in dict.fromkeys(nodes):      # de-dupe, preserve order (validate warns on dupes)
            replica = copy.deepcopy(svc)
            replica.pop("nodes", None)
            replica["node"] = n
            replica["name"] = f"{name}-{n}"
            out.append(replica)
    return {**recipe, "services": out}


def svc_provider(svc):
    return svc.get("provider") or ("ollama" if svc["engine"] == "ollama" else "hf")


def speculative_model(svc):
    """The speculative drafter's HF repo id, if this service speculates with a SEPARATE model
    (draft_model / EAGLE / DFlash) — it must be downloaded + mirrored to every node like the primary.
    None for self-speculation (mtp/ngram, which reuse the target's own weights) or no speculation."""
    spec = svc.get("speculative")
    return spec.get("model") if isinstance(spec, dict) else None


def recipe_hash(name):
    return hashlib.sha256((config.ROOT / "recipes" / f"{name}.yaml").read_bytes()).hexdigest()


def services_by_node(recipe):
    """Map node -> the services that node actually hosts. A multinode vLLM service is served on the
    Ray HEAD, so it's recorded there."""
    m = {n: [] for n in config.NODES}
    for svc in recipe["services"]:
        if svc["engine"] in SERVED_ENGINES:
            node = svc.get("node", config.HEAD) if svc_world(svc) <= 1 else config.HEAD
            served = svc.get("served_name", "model")
            port = svc.get("port", config.CFG["cluster"]["api_port"])
        else:  # ollama
            node, served, port = svc.get("node", config.HEAD), svc["model"], svc.get("port", 11434)
        m.setdefault(node, []).append(
            {"name": svc["name"], "served_name": served, "engine": svc["engine"], "port": port})
    return m
