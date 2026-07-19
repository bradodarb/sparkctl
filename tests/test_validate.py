"""Recipe linter — pure rule checks, no cluster/config needed."""
from sparkctl.cli.validate import lint_recipe


def _msgs(findings, level=None):
    return [f.message for f in findings if level is None or f.level == level]


def _levels(findings):
    return {f.level for f in findings}


def _apply_all_fixes(recipe, **kw):
    """Apply every auto-fix once, then re-lint (mirrors `--fix`)."""
    for f in lint_recipe(recipe, **kw):
        if f.fix:
            f.fix()
    return lint_recipe(recipe, **kw)


# ---- the exact bug: TP flipped to 2 but single-node load flags / no Ray env left behind ----
def test_multinode_with_singlenode_flags_is_flagged():
    recipe = {"name": "gpt-oss-120b", "services": [{
        "name": "agent", "engine": "vllm", "model": "openai/gpt-oss-120b",
        "parallel": {"tensor": 2}, "port": 8000,
        "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"],
    }]}
    msgs = _msgs(lint_recipe(recipe))
    assert any("fastsafetensors" in m for m in msgs)
    assert any("RAY_memory_monitor_refresh_ms" in m for m in msgs)


def test_eager_load_strategy_is_flagged_as_oom_risk():
    # eager loads the whole checkpoint into host RAM -> OOMs on unified memory (single OR multi-node)
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m", "parallel": {"tensor": 2}, "port": 8000,
        "extra_args": ["--load-format=fastsafetensors", "--safetensors-load-strategy=eager"],
        "env": {"RAY_memory_monitor_refresh_ms": "0"},
    }]}
    assert any("eager" in m and "host RAM" in m for m in _msgs(lint_recipe(recipe)))
    _apply_all_fixes(recipe)   # ...and --fix drops it
    assert "--safetensors-load-strategy=eager" not in recipe["services"][0]["extra_args"]


def test_correct_multinode_recipe_is_clean():
    recipe = {"name": "ok", "services": [{
        "name": "agent", "engine": "vllm", "model": "some/model",
        "parallel": {"tensor": 2}, "port": 8000,
        "extra_args": ["--load-format=fastsafetensors"],
        "env": {"RAY_memory_monitor_refresh_ms": "0"},
    }]}
    assert lint_recipe(recipe) == []


def test_multinode_pinned_node_is_flagged():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m", "node": "coach",
        "parallel": {"tensor": 2}, "extra_args": ["--load-format=fastsafetensors"],
        "env": {"RAY_memory_monitor_refresh_ms": "0"}, "port": 8000,
    }]}
    assert any("pinned 'node" in m for m in _msgs(lint_recipe(recipe)))


def test_singlenode_with_fastsafetensors_is_flagged():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
        "extra_args": ["--load-format=fastsafetensors"],
    }]}
    assert any("single-node should use" in m for m in _msgs(lint_recipe(recipe)))


def test_clean_singlenode_recipe_is_clean():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
        "extra_args": ["--load-format=safetensors"],   # streamed; NO eager (would OOM on unified mem)
    }]}
    assert lint_recipe(recipe) == []


# ---- provider / engine coupling ----
def test_vllm_only_fields_on_ollama_flagged():
    recipe = {"name": "x", "services": [{
        "name": "emb", "engine": "ollama", "model": "nomic-embed-text", "node": "ref",
        "port": 11434, "max_model_len": 8192, "parallel": {"tensor": 2},
    }]}
    msgs = _msgs(lint_recipe(recipe))
    assert any("max_model_len" in m and "ollama" in m for m in msgs)
    assert any("parallel" in m and "ollama" in m for m in msgs)


def test_bad_engine_is_error():
    recipe = {"name": "x", "services": [{"name": "a", "engine": "tgi", "model": "m"}]}
    findings = lint_recipe(recipe)
    assert "error" in _levels(findings)
    assert any("engine must be" in m for m in _msgs(findings, "error"))


# ---- structural + range checks ----
def test_nvfp4_without_flashinfer_env_is_clean():
    # NVFP4 no longer REQUIRES the (deprecated) FlashInfer MoE env — absence must not be flagged
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "RedHatAI/Qwen3-235B-A22B-NVFP4",
        "parallel": {"tensor": 2}, "port": 8000, "extra_args": ["--load-format=fastsafetensors"],
        "env": {"RAY_memory_monitor_refresh_ms": "0"},
    }]}
    assert not any("VLLM_USE_FLASHINFER_MOE_FP4" in m for m in _msgs(lint_recipe(recipe)))


def test_flashinfer_moe_fp4_env_is_flagged_deprecated():
    # ...and setting it IS flagged (deprecated + crashes some MoEs like Gemma-4), fixable by removal
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
        "node": "coach", "port": 8000, "env": {"VLLM_USE_FLASHINFER_MOE_FP4": "1"},
    }]}
    assert any("VLLM_USE_FLASHINFER_MOE_FP4" in m and "deprecated" in m
               for m in _msgs(lint_recipe(recipe)))
    _apply_all_fixes(recipe)   # --fix removes it
    assert "VLLM_USE_FLASHINFER_MOE_FP4" not in recipe["services"][0].get("env", {})


def test_gpu_util_out_of_range_is_error():
    recipe = {"name": "x", "services": [{
        "name": "a", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
        "gpu_memory_utilization": 1.5,
        "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"],
    }]}
    assert any("gpu_memory_utilization" in m for m in _msgs(lint_recipe(recipe), "error"))


def test_high_gpu_util_warns_oom_risk():
    recipe = {"name": "x", "services": [{
        "name": "a", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
        "gpu_memory_utilization": 0.95,
        "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"]}]}
    findings = lint_recipe(recipe)
    assert any("aggressive on unified memory" in m for m in _msgs(findings, "warn"))
    assert "error" not in _levels(findings)             # 0.95 is valid, just risky


def test_port_collision_same_node_is_error():
    recipe = {"name": "x", "services": [
        {"name": "a", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
         "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"]},
        {"name": "b", "engine": "vllm", "model": "m2", "node": "coach", "port": 8000,
         "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"]},
    ]}
    assert any("already used by" in m for m in _msgs(lint_recipe(recipe), "error"))


def test_unknown_node_is_error():
    recipe = {"name": "x", "services": [{
        "name": "a", "engine": "vllm", "model": "m", "node": "nope", "port": 8000,
        "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"]}]}
    findings = lint_recipe(recipe, known_nodes={"coach", "ref"}, head="coach")
    assert any("not in cluster.yaml" in m for m in _msgs(findings, "error"))


def test_missing_services_is_error():
    assert "error" in _levels(lint_recipe({"name": "x"}))
    assert "error" in _levels(lint_recipe({"name": "x", "services": []}))


# ---- suggestions + auto-fix ----
def test_every_finding_has_a_suggestion():
    # a representative recipe touching many rules
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "openai/gpt-oss-120b", "node": "coach",
        "parallel": {"tensor": 2}, "port": 8000,
        "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"],
    }]}
    for f in lint_recipe(recipe):
        assert f.suggestion, f"finding without suggestion: {f.message}"


def test_fix_repairs_the_exact_bug():
    recipe = {"name": "gpt-oss-120b", "services": [{
        "name": "agent", "engine": "vllm", "model": "openai/gpt-oss-120b",
        "parallel": {"tensor": 2}, "port": 8000,
        "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"],
    }]}
    remaining = _apply_all_fixes(recipe)
    assert remaining == []                       # fully repaired
    svc = recipe["services"][0]
    assert svc["extra_args"] == ["--load-format=fastsafetensors"]
    assert svc["env"]["RAY_memory_monitor_refresh_ms"] == "0"


def test_fix_preserves_unrelated_extra_args():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m", "parallel": {"tensor": 2}, "port": 8000,
        "extra_args": ["--trust-remote-code", "--load-format=safetensors",
                       "--safetensors-load-strategy=eager"],
        "env": {"RAY_memory_monitor_refresh_ms": "0"},
    }]}
    _apply_all_fixes(recipe)
    assert recipe["services"][0]["extra_args"] == ["--trust-remote-code",
                                                   "--load-format=fastsafetensors"]


def test_fix_singlenode_nvfp4_load_format():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "org/Model-NVFP4", "node": "coach", "port": 8000,
        "extra_args": ["--load-format=fastsafetensors"],
    }]}
    assert _apply_all_fixes(recipe) == []
    svc = recipe["services"][0]
    assert svc["extra_args"] == ["--load-format=safetensors"]   # streamed, no eager
    assert "env" not in svc                                     # no deprecated FlashInfer env forced


def test_fix_drops_vllm_only_fields_from_ollama():
    recipe = {"name": "x", "services": [{
        "name": "emb", "engine": "ollama", "model": "nomic-embed-text", "node": "ref",
        "port": 11434, "max_model_len": 8192, "parallel": {"tensor": 2},
    }]}
    assert _apply_all_fixes(recipe) == []
    svc = recipe["services"][0]
    assert "max_model_len" not in svc and "parallel" not in svc


def test_unfixable_findings_have_no_fix():
    recipe = {"name": "x", "services": [{
        "name": "a", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
        "gpu_memory_utilization": 1.5,
        "extra_args": ["--load-format=safetensors", "--safetensors-load-strategy=eager"]}]}
    gpu = [f for f in lint_recipe(recipe) if "gpu_memory_utilization" in f.message]
    assert gpu and gpu[0].fix is None
