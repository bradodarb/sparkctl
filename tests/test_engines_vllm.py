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
    assert "--entrypoint bash" in cmd                   # works even if base image sets ENTRYPOINT ["vllm"]
    assert f"{config.IMAGE} -lc " in cmd                # command is `-lc <serve>`, not `bash -lc <serve>`
    assert vllm.svc_cname("agent") in cmd
    assert "vllm serve org/M" in cmd
    assert "ray start" not in cmd                       # single-node path must not use Ray
    assert "distributed-executor-backend" not in cmd
    assert "--env-file" in cmd and "secrets.env" in cmd  # serve container gets HF_TOKEN (auth)


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


# ---- mp (no-Ray) multinode ------------------------------------------------
def test_worker_engine_flags_are_engine_only():
    f = vllm._worker_engine_flags(_svc(parallel={"tensor": 2}, served_name="M", port=8000,
                                       tool_call_parser="deepseek_v4", reasoning_parser="deepseek_v4",
                                       extra_args=["--kv-cache-dtype=fp8", "--trust-remote-code"]))
    # engine config the worker must mirror
    assert "--tensor-parallel-size 2" in f
    assert "--max-model-len 4096" in f
    assert "--kv-cache-dtype=fp8" in f and "--trust-remote-code" in f
    # API-server flags are head-only — never on the headless worker
    assert "--served-model-name" not in f
    assert "--host" not in f and "--port" not in f
    assert "--tool-call-parser" not in f and "--reasoning-parser" not in f


def test_vllm_up_multinode_mp_no_ray(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: calls.append((node, cmd)))
    monkeypatch.setattr(vllm.time, "sleep", lambda *_: None)
    vllm.vllm_up(_svc(parallel={"tensor": 2}, port=8000, distributed_backend="mp"))
    by_node = {}
    for n, c in calls:
        by_node.setdefault(n, []).append(c)
    allcmds = " ".join(c for _, c in calls)
    # no Ray anywhere on the mp path
    assert "ray start" not in allcmds
    assert "--distributed-executor-backend mp" in allcmds
    assert "--nnodes 2" in allcmds
    assert "--master-addr" in allcmds
    # rank-0 API server on HEAD, headless rank-1 worker on WORKER
    head = " ".join(by_node[config.HEAD])
    worker = " ".join(by_node[config.WORKER])
    assert "--node-rank 0" in head and "--served-model-name M" in head
    assert "--node-rank 1" in worker and "--headless" in worker
    assert "vllm serve org/M" in head and "vllm serve org/M" in worker
    # both run vllm serve as PID 1 (no sleep-infinity + exec)
    assert "sleep infinity" not in allcmds


def test_vllm_up_multinode_defaults_to_ray_not_mp(monkeypatch):
    calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: calls.append((node, cmd)))
    monkeypatch.setattr(vllm.time, "sleep", lambda *_: None)
    vllm.vllm_up(_svc(parallel={"tensor": 2}, port=8000))   # no distributed_backend key
    allcmds = " ".join(c for _, c in calls)
    assert "ray start --head" in allcmds
    assert "--distributed-executor-backend mp" not in allcmds


# ---- multimodal / model-impl engine flags ---------------------------------
def test_engine_extra_flags_multimodal_and_impl():
    f = vllm._vllm_serve_flags(_svc(trust_remote_code=True, model_impl="transformers",
                                    limit_mm_per_prompt={"image": 4},
                                    mm_processor_kwargs={"foo": "bar"}))
    assert "--trust-remote-code" in f
    assert "--model-impl transformers" in f
    # JSON is single-quoted so it survives ssh->docker->bash as one literal arg
    assert "--limit-mm-per-prompt='{\"image\":4}'" in f
    assert "--mm-processor-kwargs='{\"foo\":\"bar\"}'" in f


def test_engine_extra_flags_absent_by_default():
    f = vllm._vllm_serve_flags(_svc(tool_call_parser=None))
    for x in ("--trust-remote-code", "--model-impl", "--limit-mm-per-prompt", "--mm-processor-kwargs"):
        assert x not in f


def test_worker_flags_mirror_engine_level_mm():
    # headless mp workers must get the SAME engine-shaping flags as the head (or ranks diverge)
    svc = _svc(trust_remote_code=True, model_impl="transformers", limit_mm_per_prompt={"image": 2})
    wf = vllm._worker_engine_flags(svc)
    assert "--model-impl transformers" in wf
    assert "--trust-remote-code" in wf
    assert "--limit-mm-per-prompt='{\"image\":2}'" in wf
    # but NOT the API-server-only flags
    assert "--served-model-name" not in wf and "--host" not in wf


def test_tool_parser_plugin_flag_emitted():
    f = vllm._vllm_serve_flags(_svc(tool_parser_plugin="/opt/p/atem.py", tool_call_parser="atem"))
    assert "--tool-parser-plugin /opt/p/atem.py" in f
    assert "--enable-auto-tool-choice" in f and "--tool-call-parser atem" in f
    # plugin must be loaded BEFORE the parser is selected
    assert f.index("--tool-parser-plugin") < f.index("--tool-call-parser")


def test_speculative_config_flag_shell_safe():
    f = vllm._vllm_serve_flags(_svc(speculative={"method": "dflash",
                                                  "model": "org/drafter", "num_speculative_tokens": 15}))
    # single-quoted JSON so it survives ssh->docker->bash as one arg
    assert "--speculative-config='{\"method\":\"dflash\",\"model\":\"org/drafter\",\"num_speculative_tokens\":15}'" in f


def test_docker_common_passes_rdma_devices_when_present():
    # without /dev/infiniband NCCL silently falls back to TCP sockets across the fabric (~2x slower
    # TP decode); the flag is gated on the node actually having the device ($() expands on-node)
    common = vllm._docker_common()
    assert "test -d /dev/infiniband && echo --device=/dev/infiniband --cap-add=IPC_LOCK" in common
