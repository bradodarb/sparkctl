"""Recipe-wizard builders — pure, no cluster or config needed."""
import yaml

from sparkctl.cli.create import build_recipe, detect_preset, to_yaml


def test_gpt_oss_preset_has_no_parsers():
    p = detect_preset("openai/gpt-oss-120b")
    assert p["family"] == "gpt-oss"
    assert p["tool_call_parser"] is None and p["reasoning_parser"] is None
    assert p["trust_remote_code"] is False


def test_qwen_coder_and_qwen3_presets():
    assert detect_preset("Qwen/Qwen3-Coder-Next-FP8")["tool_call_parser"] == "qwen3_coder"
    q = detect_preset("Qwen/Qwen3-235B-A22B")
    assert q["tool_call_parser"] == "hermes" and q["reasoning_parser"] == "qwen3"


def test_singlenode_uses_eager_no_mmap_and_no_ray_env():
    r = build_recipe(name="x", model="openai/gpt-oss-20b", tensor=1, node="coach")
    svc = r["services"][0]
    assert svc["node"] == "coach" and "parallel" not in svc
    assert "--load-format=safetensors" in svc["extra_args"]
    assert "--safetensors-load-strategy=eager" in svc["extra_args"]
    assert "RAY_memory_monitor_refresh_ms" not in svc.get("env", {})


def test_multinode_uses_fastsafetensors_and_ray_oom_fix():
    r = build_recipe(name="y", model="RedHatAI/Qwen3-235B-A22B-NVFP4", tensor=2)
    svc = r["services"][0]
    assert svc["parallel"]["tensor"] == 2
    assert "--load-format=fastsafetensors" in svc["extra_args"]
    assert svc["env"]["RAY_memory_monitor_refresh_ms"] == "0"
    assert svc["env"]["VLLM_USE_FLASHINFER_MOE_FP4"] == "1"   # NVFP4 in the model name


def test_embeddings_service_appended():
    r = build_recipe(name="z", model="openai/gpt-oss-20b", tensor=1, node="coach",
                     embeddings={"model": "nomic-embed-text", "node": "ref", "port": 11434})
    assert [s["name"] for s in r["services"]] == ["agent", "embeddings"]
    emb = r["services"][1]
    assert emb["engine"] == "ollama" and emb["model"] == "nomic-embed-text" and emb["node"] == "ref"


def test_to_yaml_roundtrips_to_valid_recipe():
    r = build_recipe(name="rt", model="openai/gpt-oss-20b", tensor=1, node="coach",
                     description="round trip")
    parsed = yaml.safe_load(to_yaml(r))
    assert parsed["name"] == "rt"
    assert parsed["services"][0]["served_name"] == "rt"
