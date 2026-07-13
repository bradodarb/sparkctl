"""vLLM engine — container naming, env construction, serve-flag construction, and (with `on`
mocked) the docker commands vllm_up emits for single- vs multi-node. No live cluster needed."""
import pytest

from sparkctl import config, remote
from sparkctl.engines import vllm


# ---- naming ---------------------------------------------------------------
def test_svc_cname():
    assert vllm.svc_cname("agent") == f"{config.PFX}-svc-agent"
    assert vllm.svc_cname("agent", "head") == f"{config.PFX}-svc-agent-head"
    assert vllm.svc_cname("agent", "worker") == f"{config.PFX}-svc-agent-worker"


# ---- env construction -----------------------------------------------------
def test_env_flags_quotes_values():
    assert vllm.env_flags({"VLLM_USE_FLASHINFER_MOE_FP4": 1}) == '-e VLLM_USE_FLASHINFER_MOE_FP4="1"'


def test_fabric_env_has_required_keys():
    e = vllm.fabric_env()
    for k in ("NCCL_IB_HCA", "NCCL_SOCKET_IFNAME", "NCCL_IB_GID_INDEX"):
        assert k in e
    # single-interface Gloo/TP vars derive from the first NCCL socket iface
    assert "," not in e["GLOO_SOCKET_IFNAME"]


# ---- serve-flag construction (the bit most likely to regress) -------------
def _svc(**over):
    base = {"name": "agent", "model": "org/M", "served_name": "M",
            "max_model_len": 4096, "gpu_memory_utilization": 0.8, "parallel": {"tensor": 1}}
    base.update(over)
    return base


def test_serve_flags_single_node():
    f = vllm._vllm_serve_flags(_svc(tool_call_parser=None))
    assert "--served-model-name M" in f
    assert "--max-model-len 4096" in f
    assert "--tensor-parallel-size 1" in f
    assert "--pipeline-parallel-size 1" in f
    assert "--enable-auto-tool-choice" not in f   # no parser -> no tool flags


def test_serve_flags_tool_and_reasoning():
    f = vllm._vllm_serve_flags(_svc(tool_call_parser="qwen3_coder", reasoning_parser="qwen3",
                                    extra_args=["--trust-remote-code"]))
    assert "--enable-auto-tool-choice" in f
    assert "--tool-call-parser qwen3_coder" in f
    assert "--reasoning-parser qwen3" in f
    assert "--trust-remote-code" in f


def test_serve_flags_tp2():
    assert "--tensor-parallel-size 2" in vllm._vllm_serve_flags(_svc(parallel={"tensor": 2}))


# ---- vllm_up emits the right docker commands (execution mocked) -----------
def test_vllm_up_single_node(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: calls.append((node, cmd)))
    vllm.vllm_up(_svc(node="coach", port=8000, gpu_memory_utilization=0.8))
    assert len(calls) == 1
    node, cmd = calls[0]
    assert node == "coach"
    assert "docker run" in cmd
    assert vllm.svc_cname("agent") in cmd
    assert "vllm serve org/M" in cmd
    assert "ray start" not in cmd                       # single-node path must not use Ray
    assert "distributed-executor-backend" not in cmd


def test_vllm_up_multinode_without_fabric_exits(monkeypatch):
    monkeypatch.setattr(config, "FABRIC", {})
    with pytest.raises(SystemExit, match="no.*fabric"):
        vllm.vllm_up(_svc(parallel={"tensor": 2}, port=8000))


def test_vllm_up_multinode_uses_ray(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: calls.append((node, cmd)))
    monkeypatch.setattr(vllm.time, "sleep", lambda *_: None)
    vllm.vllm_up(_svc(parallel={"tensor": 2}, port=8000))
    cmds = " ".join(c for _, c in calls)
    assert "ray start --head" in cmds
    assert "ray start --address=" in cmds
    assert "--distributed-executor-backend ray" in cmds
    assert "--tensor-parallel-size 2" in cmds
    # head + worker containers land on different nodes
    assert {n for n, _ in calls} >= {config.HEAD, config.WORKER}
