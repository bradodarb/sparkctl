"""Recipes: load, hash, and map services onto nodes. The `current` file is the active pointer."""
import hashlib
import sys

import yaml

from sparkctl import config


def current_recipe():
    return (config.ROOT / "current").read_text().strip()


def load_recipe(name):
    p = config.ROOT / "recipes" / f"{name}.yaml"
    if not p.exists():
        sys.exit(f"no such recipe: {name} ({p})")
    return yaml.safe_load(p.read_text())


def svc_world(svc):
    p = svc.get("parallel", {})
    return p.get("tensor", 1) * p.get("pipeline", 1)   # total ranks (1 GPU/node on Spark)


def svc_provider(svc):
    return svc.get("provider") or ("ollama" if svc["engine"] == "ollama" else "hf")


def recipe_hash(name):
    return hashlib.sha256((config.ROOT / "recipes" / f"{name}.yaml").read_bytes()).hexdigest()


def services_by_node(recipe):
    """Map node -> the services that node actually hosts. A multinode vLLM service is served on the
    Ray HEAD, so it's recorded there."""
    m = {n: [] for n in config.NODES}
    for svc in recipe["services"]:
        if svc["engine"] == "vllm":
            node = svc.get("node", config.HEAD) if svc_world(svc) <= 1 else config.HEAD
            served = svc.get("served_name", "model")
            port = svc.get("port", config.CFG["cluster"]["api_port"])
        else:  # ollama
            node, served, port = svc.get("node", config.HEAD), svc["model"], svc.get("port", 11434)
        m.setdefault(node, []).append(
            {"name": svc["name"], "served_name": served, "engine": svc["engine"], "port": port})
    return m
