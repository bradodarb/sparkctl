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


def test_mp_backend_multinode_needs_no_ray_env():
    # distributed_backend: mp -> no Ray, so RAY_memory_monitor_refresh_ms is NOT required (and absent = clean)
    recipe = {"name": "ok", "services": [{
        "name": "agent", "engine": "vllm", "model": "some/model",
        "parallel": {"tensor": 2}, "distributed_backend": "mp", "port": 8000,
        "extra_args": ["--load-format=fastsafetensors"],
    }]}
    assert lint_recipe(recipe) == []


def test_mp_backend_with_ray_env_is_info():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m",
        "parallel": {"tensor": 2}, "distributed_backend": "mp", "port": 8000,
        "extra_args": ["--load-format=fastsafetensors"],
        "env": {"RAY_memory_monitor_refresh_ms": "0"},
    }]}
    assert any("no effect with distributed_backend=mp" in m for m in _msgs(lint_recipe(recipe), "info"))
    _apply_all_fixes(recipe)
    assert "RAY_memory_monitor_refresh_ms" not in recipe["services"][0].get("env", {})


def test_mp_backend_on_singlenode_is_flagged():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
        "distributed_backend": "mp", "extra_args": ["--load-format=safetensors"],
    }]}
    assert any("only applies to multinode" in m for m in _msgs(lint_recipe(recipe), "warn"))
    _apply_all_fixes(recipe)
    assert "distributed_backend" not in recipe["services"][0]


def test_invalid_distributed_backend_is_error():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m",
        "parallel": {"tensor": 2}, "distributed_backend": "slurm", "port": 8000,
        "extra_args": ["--load-format=fastsafetensors"],
    }]}
    assert any("distributed_backend must be" in m for m in _msgs(lint_recipe(recipe), "error"))


def test_undeclared_pinned_image_is_flagged():
    recipe = {"name": "x", "services": [{
        "name": "agent", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
        "image": "csv-vllm:0.99-experimental", "extra_args": ["--load-format=safetensors"],
    }]}
    known = {"csv-vllm:26.06-ray"}
    assert any("pinned image" in m and "not the cluster default" in m
               for m in _msgs(lint_recipe(recipe, known_images=known), "warn"))
    # declared/default image -> no image warning; and no check at all when known_images is omitted
    ok = dict(recipe["services"][0], image="csv-vllm:26.06-ray")
    assert not any("pinned image" in m for m in _msgs(lint_recipe({"name": "x", "services": [ok]},
                                                                   known_images=known)))
    assert not any("pinned image" in m for m in _msgs(lint_recipe(recipe)))   # no known_images -> skip


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


def test_gemma4_tool_parser_without_reasoning_parser_is_flagged():
    # the silent bug: gemma4 tool parser handles tool calls but not the thought channel, which then
    # leaks into content. Flag it, and --fix pairs the reasoning parser.
    recipe = {"name": "x", "services": [{
        "name": "agent-coach", "engine": "vllm", "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
        "node": "coach", "port": 8000, "extra_args": ["--load-format=safetensors"],
        "tool_call_parser": "gemma4",
    }]}
    assert any("reasoning_parser" in m and "leak" in m for m in _msgs(lint_recipe(recipe), "warn"))
    _apply_all_fixes(recipe)
    assert recipe["services"][0]["reasoning_parser"] == "gemma4"


def test_gemma4_both_parsers_is_clean():
    recipe = {"name": "x", "services": [{
        "name": "agent-coach", "engine": "vllm", "model": "nvidia/Gemma-4-26B-A4B-NVFP4",
        "node": "coach", "port": 8000, "extra_args": ["--load-format=safetensors"],
        "tool_call_parser": "gemma4", "reasoning_parser": "gemma4",
    }]}
    assert lint_recipe(recipe) == []


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


# ---------------------------------------------------------------- nodes: fan-out lint rules
def _nodes_svc(**over):
    svc = {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "coder",
           "nodes": ["coach", "ref"], "port": 8000}
    svc.update(over)
    return {"name": "r", "services": [svc]}


def test_nodes_and_node_together_is_error():
    findings = lint_recipe(_nodes_svc(node="coach"))
    assert any(f.level == "error" and "not both" in f.message for f in findings)


def test_nodes_with_multinode_parallel_is_error():
    findings = lint_recipe(_nodes_svc(parallel={"tensor": 2}))
    assert any(f.level == "error" and "already spans nodes" in f.message for f in findings)


def test_nodes_unknown_node_is_error():
    findings = lint_recipe(_nodes_svc(nodes=["coach", "bogus"]), known_nodes={"coach", "ref"})
    assert any(f.level == "error" and "bogus" in f.message for f in findings)


def test_nodes_empty_list_is_error():
    findings = lint_recipe(_nodes_svc(nodes=[]))
    assert any(f.level == "error" and "non-empty list" in f.message for f in findings)


def test_nodes_duplicate_warns_and_fixes():
    recipe = _nodes_svc(nodes=["coach", "ref", "coach"])
    assert _apply_all_fixes(recipe, known_nodes={"coach", "ref"}) == []
    assert recipe["services"][0]["nodes"] == ["coach", "ref"]


def test_plain_nodes_recipe_is_clean():
    assert lint_recipe(_nodes_svc(), known_nodes={"coach", "ref"}) == []


# ---------------------------------------------------------------- multimodal / VLM lints
def _vlm_svc(**over):
    svc = {"name": "vlm", "engine": "vllm", "model": "org/VLM", "served_name": "vlm",
           "node": "coach", "port": 8000, "multimodal": True, "model_impl": "transformers",
           "limit_mm_per_prompt": {"image": 4}}
    svc.update(over)
    return {"name": "r", "services": [svc]}


def test_model_impl_invalid_is_error():
    assert any(f.level == "error" and "model_impl" in f.message
               for f in lint_recipe(_vlm_svc(model_impl="wrong")))


def test_limit_mm_must_be_mapping():
    assert any(f.level == "error" and "limit_mm_per_prompt" in f.message
               for f in lint_recipe(_vlm_svc(limit_mm_per_prompt=[4])))


def test_multimodal_must_be_bool():
    assert any(f.level == "error" and "multimodal" in f.message
               for f in lint_recipe(_vlm_svc(multimodal="yes")))


def test_multimodal_without_limit_is_info():
    fs = lint_recipe(_vlm_svc(limit_mm_per_prompt=None))
    assert any(f.level == "info" and "limit_mm_per_prompt" in f.message for f in fs)


def test_clean_vlm_recipe_passes():
    assert lint_recipe(_vlm_svc(), known_nodes={"coach", "ref"}) == []


def test_tool_parser_plugin_without_parser_warns():
    recipe = {"name": "r", "services": [{
        "name": "vlm", "engine": "vllm", "model": "m", "node": "coach", "port": 8000,
        "tool_parser_plugin": "/opt/p/atem.py"}]}   # no tool_call_parser
    assert any(f.level == "warn" and "tool_parser_plugin" in f.message for f in lint_recipe(recipe))


def test_speculative_needs_method_and_model():
    from sparkctl.cli.validate import lint_recipe as L
    base = {"name": "vlm", "engine": "vllm", "model": "m", "node": "coach", "port": 8000}
    assert any(f.level == "error" and "method" in f.message
               for f in L({"name": "r", "services": [dict(base, speculative={"num_speculative_tokens": 2})]}))
    assert any(f.level == "error" and "drafter 'model'" in f.message
               for f in L({"name": "r", "services": [dict(base, speculative={"method": "dflash"})]}))
    # ngram self-speculates -> no model required -> clean
    assert L({"name": "r", "services": [dict(base, speculative={"method": "ngram"})]},
             known_nodes={"coach"}) == []


# ---------------------------------------------------------------- sglang engine
def _sglang_svc(**over):
    svc = {"name": "agent", "engine": "sglang", "model": "org/M", "served_name": "muse",
           "node": "coach", "port": 8000, "max_model_len": 131072,
           "tool_call_parser": "muse", "reasoning_parser": "muse", "trust_remote_code": True,
           "speculative": {"method": "dflash", "model": "org/drafter", "draft_quantization": "fp8"},
           "extra_args": ["--mem-fraction-static 0.38", "--no-enable-multimodal"]}
    svc.update(over)
    return {"name": "r", "services": [svc]}


def test_sglang_engine_accepted_clean():
    assert lint_recipe(_sglang_svc(), known_nodes={"coach", "ref"}) == []


def test_sglang_rejects_multinode_parallel():
    fs = lint_recipe(_sglang_svc(parallel={"tensor": 2}), known_nodes={"coach"})
    assert any(f.level == "error" and "single-node" in f.message for f in fs)


def test_sglang_speculative_dflash_needs_model():
    fs = lint_recipe(_sglang_svc(speculative={"method": "dflash"}), known_nodes={"coach"})
    assert any(f.level == "error" and "drafter 'model'" in f.message for f in fs)


def test_sglang_warns_on_vllm_only_knob():
    fs = lint_recipe(_sglang_svc(gpu_memory_utilization=0.85), known_nodes={"coach"})
    assert any(f.level == "warn" and "vllm-only" in f.message for f in fs)


def test_dspark_method_needs_model():
    base = {"name": "svc", "engine": "vllm", "model": "m", "node": "coach", "port": 8000}
    assert any(f.level == "error" and "drafter 'model'" in f.message
               for f in lint_recipe({"name": "r", "services": [dict(base, speculative={"method": "dspark"})]}))
