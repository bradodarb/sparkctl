"""LiteLLM config generation — routing, load-balance pools, address modes. Pure logic."""
from sparkctl import config
from sparkctl.server.litellm_bridge import litellm_config

# A recipe with a replicated single-node agent (coach+ref, same served_name) + ollama embeddings.
REPLICATED = {"services": [
    {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "coder",
     "node": config.HEAD, "parallel": {"tensor": 1}, "port": 8000},
    {"name": "agent-ref", "engine": "vllm", "model": "org/M", "served_name": "coder",
     "node": config.WORKER, "parallel": {"tensor": 1}, "port": 8000},
    {"name": "embeddings", "engine": "ollama", "model": "nomic-embed-text",
     "node": config.WORKER, "port": 11434},
]}
MULTINODE = {"services": [
    {"name": "agent", "engine": "vllm", "model": "org/Big", "served_name": "big",
     "parallel": {"tensor": 2}, "port": 8000},
]}


def _by_model(cfg):
    out = {}
    for e in cfg["model_list"]:
        out.setdefault(e["model_name"], []).append(e["litellm_params"]["api_base"])
    return out


def test_replicated_creates_lb_pool():
    cfg = litellm_config(REPLICATED, "local", {})
    bym = _by_model(cfg)
    assert len(bym["coder"]) == 2                       # coach + ref -> load-balance pool
    assert bym["coder"][0] != bym["coder"][1]
    assert "nomic-embed-text" in bym                    # embeddings routed too
    assert bym["nomic-embed-text"][0].endswith(":11434/v1")


def test_openai_prefix_and_key():
    e = litellm_config(REPLICATED, "local", {})["model_list"][0]
    assert e["litellm_params"]["model"] == "openai/coder"
    assert e["litellm_params"]["api_key"] == "none"


def test_multinode_single_head_backend():
    cfg = litellm_config(MULTINODE, "local", {})
    bym = _by_model(cfg)
    head = config.NODES[config.HEAD]
    assert bym["big"] == [f"http://{head.get('lan_ip', head['host'])}:8000/v1"]


def test_host_mode_changes_addresses():
    local = litellm_config(MULTINODE, "local", {})["model_list"][0]["litellm_params"]["api_base"]
    onnode = litellm_config(MULTINODE, config.HEAD, {})["model_list"][0]["litellm_params"]["api_base"]
    assert config.NODES[config.HEAD]["fabric_ip"] in onnode        # node-hosted -> fabric IP
    assert config.NODES[config.HEAD]["fabric_ip"] not in local     # dev-machine -> lan_ip/host


def test_routing_strategy_and_master_key():
    cfg = litellm_config(MULTINODE, "local", {"routing_strategy": "least-busy", "master_key": "sk-x"})
    assert cfg["router_settings"]["routing_strategy"] == "least-busy"
    assert cfg["general_settings"]["master_key"] == "sk-x"


def test_router_retries_default_on_and_overridable():
    # connection errors retry on another deployment instead of surfacing a 500
    rs = litellm_config(MULTINODE, "local", {})["router_settings"]
    assert rs["num_retries"] == 2 and rs["cooldown_time"] == 30
    rs = litellm_config(MULTINODE, "local", {"num_retries": 0, "cooldown_time": 5})["router_settings"]
    assert rs["num_retries"] == 0 and rs["cooldown_time"] == 5


def test_sglang_service_registered_in_router():
    # an sglang service is a served OpenAI endpoint -> it must appear in the LiteLLM model_list
    recipe = {"services": [
        {"name": "agent", "engine": "sglang", "model": "org/M", "served_name": "muse",
         "node": config.HEAD, "port": 8000},
    ]}
    bym = _by_model(litellm_config(recipe, "local", {}))
    assert "muse" in bym and bym["muse"][0].endswith(":8000/v1")


def test_multimodal_declares_supports_vision():
    recipe = {"services": [
        {"name": "vlm", "engine": "vllm", "model": "org/VLM", "served_name": "vlm",
         "node": config.HEAD, "parallel": {"tensor": 1}, "port": 8000, "multimodal": True},
        {"name": "txt", "engine": "vllm", "model": "org/T", "served_name": "txt",
         "node": config.HEAD, "parallel": {"tensor": 1}, "port": 8001},
    ]}
    ml = {e["model_name"]: e for e in litellm_config(recipe, "local", {})["model_list"]}
    assert ml["vlm"].get("model_info") == {"supports_vision": True}
    assert "model_info" not in ml["txt"]        # text model: no vision claim


def test_allow_unauthenticated_opts_into_keyless_litellm():
    cfg = litellm_config(MULTINODE, "local", {"allow_unauthenticated": True})
    assert cfg["general_settings"] == {"dangerously_permit_weak_or_unset_master_key": True}
    keyed = litellm_config(MULTINODE, "local", {"master_key": "k", "allow_unauthenticated": True})
    assert keyed["general_settings"] == {"master_key": "k"}       # a key always wins
    assert "general_settings" not in litellm_config(MULTINODE, "local", {})
