"""SGLang engine — serve-flag construction, speculative translation, and (with `on` mocked) the
single docker command sglang_up emits. No live cluster. Run: pytest -q"""
from sparkctl import config, remote
from sparkctl.engines import sglang


def _svc(**over):
    base = {"name": "agent", "model": "org/M", "served_name": "muse",
            "max_model_len": 131072, "parallel": {"tensor": 1}}
    base.update(over)
    return base


# ---- serve-flag construction --------------------------------------------------
def test_serve_flags_core():
    f = sglang._sglang_serve_flags(_svc(port=8000))
    assert "--model-path org/M" in f
    assert "--served-model-name muse" in f
    assert "--host 0.0.0.0 --port 8000" in f
    assert "--tp-size 1" in f
    assert "--context-length 131072" in f            # max_model_len -> SGLang's name


def test_serve_flags_parsers_and_trc():
    f = sglang._sglang_serve_flags(_svc(tool_call_parser="muse", reasoning_parser="muse",
                                        trust_remote_code=True,
                                        extra_args=["--mem-fraction-static 0.38", "--enable-metrics"]))
    assert "--tool-call-parser muse" in f
    assert "--reasoning-parser muse" in f
    assert "--trust-remote-code" in f
    # SGLang-specific knobs pass through verbatim from extra_args
    assert "--mem-fraction-static 0.38" in f and "--enable-metrics" in f


def test_serve_flags_context_length_omitted_when_unset():
    f = sglang._sglang_serve_flags(_svc(max_model_len=None))
    assert "--context-length" not in f


# ---- speculative translation --------------------------------------------------
def test_speculative_dflash_translation():
    spec = {"method": "dflash", "model": "org/drafter", "draft_quantization": "fp8"}
    out = sglang._sglang_speculative_flags({"speculative": spec})
    j = " ".join(out)
    assert "--speculative-algorithm DFLASH" in j              # method uppercased to SGLang's convention
    assert "--speculative-draft-model-path org/drafter" in j
    assert "--speculative-draft-model-quantization fp8" in j
    # no num_speculative_tokens given -> let the image default it (flag omitted)
    assert "--speculative-num-draft-tokens" not in j


def test_speculative_num_tokens_when_present():
    out = sglang._sglang_speculative_flags(
        {"speculative": {"method": "eagle3", "model": "org/d", "num_speculative_tokens": 5}})
    assert "--speculative-num-draft-tokens 5" in " ".join(out)


def test_speculative_absent_is_empty():
    assert sglang._sglang_speculative_flags(_svc()) == []


# ---- sglang_up emits one single-node docker command ---------------------------
def test_sglang_up_single_node(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: calls.append((node, cmd)))
    sglang.sglang_up(_svc(node="coach", port=8000))
    assert len(calls) == 1
    node, cmd = calls[0]
    assert node == "coach"
    assert "docker run" in cmd
    assert "--entrypoint bash" in cmd                    # reuse vllm's shell-entrypoint plumbing
    assert f"{config.IMAGE} -lc " in cmd                 # command is `-lc <serve>`
    assert sglang.svc_cname("agent") in cmd
    assert "python3 -m sglang.launch_server --model-path org/M" in cmd
    assert "ray start" not in cmd                        # single-node: never Ray
    assert "--env-file" in cmd and "secrets.env" in cmd  # HF auth for gated weights
