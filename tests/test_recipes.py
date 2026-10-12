"""Recipe/topology logic — provider defaults, per-node service placement, manifest hashing.
Pure logic, no live cluster. Run: pytest -q"""
import hashlib

import pytest

from sparkctl import config
from sparkctl.recipes import expand_services, recipe_hash, services_by_node, svc_provider

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


# ---------------------------------------------------------------- expand_services (nodes: fan-out)
def _nodes_recipe(**over):
    svc = {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "coder",
           "nodes": [config.HEAD, config.WORKER], "port": 8000, "env": {"K": "v"}}
    svc.update(over)
    return {"name": "r", "services": [svc]}


def test_expand_passthrough_when_no_nodes():
    r = {"name": "r", "services": [{"name": "agent", "engine": "vllm", "node": config.HEAD}]}
    assert expand_services(r)["services"] == r["services"]


def test_expand_fans_out_one_replica_per_node():
    svcs = expand_services(_nodes_recipe())["services"]
    assert [s["name"] for s in svcs] == [f"agent-{config.HEAD}", f"agent-{config.WORKER}"]
    assert [s["node"] for s in svcs] == [config.HEAD, config.WORKER]
    assert all("nodes" not in s for s in svcs)
    # SAME served_name across replicas is what forms the gateway load-balance pool
    assert {s["served_name"] for s in svcs} == {"coder"}


def test_expand_replicas_are_deepcopied():
    svcs = expand_services(_nodes_recipe())["services"]
    svcs[0]["env"]["K"] = "changed"
    assert svcs[1]["env"]["K"] == "v"          # mutating one replica never aliases another


def test_expand_dedupes_nodes_preserving_order():
    svcs = expand_services(_nodes_recipe(nodes=[config.WORKER, config.HEAD, config.WORKER]))["services"]
    assert [s["node"] for s in svcs] == [config.WORKER, config.HEAD]


def test_expand_rejects_node_and_nodes_together():
    with pytest.raises(SystemExit):
        expand_services(_nodes_recipe(node=config.HEAD))


def test_expand_rejects_nodes_with_multinode_parallel():
    with pytest.raises(SystemExit):
        expand_services(_nodes_recipe(parallel={"tensor": 2}))


def test_expand_rejects_non_list_nodes():
    with pytest.raises(SystemExit):
        expand_services(_nodes_recipe(nodes=config.HEAD))


def test_speculative_model_extraction():
    from sparkctl.recipes import speculative_model
    assert speculative_model({"speculative": {"method": "dflash", "model": "org/d"}}) == "org/d"
    assert speculative_model({"speculative": {"method": "ngram"}}) is None   # self-speculation
    assert speculative_model({}) is None
