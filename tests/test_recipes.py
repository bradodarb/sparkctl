"""Recipe/topology logic — provider defaults, per-node service placement, manifest hashing.
Pure logic, no live cluster. Run: pytest -q"""
import hashlib

from sparkctl import config
from sparkctl.recipes import recipe_hash, services_by_node, svc_provider

REPLICATED = {"services": [
    {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "coder",
     "node": config.HEAD, "parallel": {"tensor": 1}, "port": 8000},
    {"name": "agent-ref", "engine": "vllm", "model": "org/M", "served_name": "coder",
     "node": config.WORKER, "parallel": {"tensor": 1}, "port": 8000},
    {"name": "embeddings", "engine": "ollama", "model": "nomic-embed-text",
     "node": config.WORKER, "port": 11434},
]}


def test_svc_provider_defaults():
    assert svc_provider({"engine": "vllm"}) == "hf"
    assert svc_provider({"engine": "ollama"}) == "ollama"
    assert svc_provider({"engine": "vllm", "provider": "hf"}) == "hf"


def test_services_by_node_places_correctly():
    byn = services_by_node(REPLICATED)
    assert any(s["name"] == "agent" for s in byn[config.HEAD])
    assert any(s["engine"] == "ollama" for s in byn[config.WORKER])


def test_manifest_hash_matches_file():
    # recipe_hash of a real recipe equals sha256 of its file bytes
    name = "nemotron-3-super"
    f = config.ROOT / "recipes" / f"{name}.yaml"
    assert recipe_hash(name) == hashlib.sha256(f.read_bytes()).hexdigest()
