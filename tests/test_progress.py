"""apply's load telemetry — engine-log phase/fact parsing and the free-memory trend. Pure; the log
lines are real vLLM 0.31 / SGLang output (vLLM's from the 2026-10-11 Qwen3.8-Flash-Next TP=2 load)."""
from sparkctl import progress

VLLM_LOG = (
    "(Worker_TP0_EP0 pid=430) INFO 10-12 00:23:51 [weight_utils.py:1063] Checkpoint size: 123.57 GiB.\n"
    "Loading safetensors checkpoint shards:   0% Completed | 0/11 [00:00<?, ?it/s]\r"
    "Loading safetensors checkpoint shards:   9% Completed | 1/11 [00:13<02:14, 13.47s/it]\r"
    "Loading safetensors checkpoint shards:  27% Completed | 3/11 [01:31<04:27, 33.41s/it]\n"
)
VLLM_LOADED = VLLM_LOG + (
    "Loading safetensors checkpoint shards: 100% Completed | 11/11 [06:50<00:00, 37.3s/it]\n"
    "(Worker_TP0_EP0 pid=430) INFO 10-12 00:30:41 [default_loader.py:484] Loading weights took 410.76 seconds\n"
    "(Worker_TP0_EP0 pid=430) INFO 10-12 00:32:13 [model_runner.py:407] Model loading took 63.84 GiB memory "
    "and 507.122538 seconds\n"
    "Capturing decode CUDA graphs (FULL): 100%|##########| 4/4 [00:00<00:00, 68.03it/s]\n"
    "(Worker_TP0_EP0 pid=430) INFO 10-12 00:33:03 [model_runner.py:1087] Graph capturing finished in 12 secs, "
    "took 1.05 GiB\n"
    "(EngineCore pid=353) INFO 10-12 00:33:03 [kv_cache_utils.py:2464] GPU KV cache size: 1,938,465 tokens, "
    "Maximum concurrency for 262,144 tokens per request: 7.39x\n"
)


def test_phase_tracks_carriage_return_progress_bars():
    phase, facts = progress.parse_engine_log(VLLM_LOG)
    assert phase == "weights 3/11"          # last \r-rewritten bar wins, not the first
    assert facts == {}


def test_facts_after_load():
    phase, facts = progress.parse_engine_log(VLLM_LOADED)
    assert phase == "allocating KV cache"
    assert facts["weights"] == "model loaded — 63.8 GiB/rank in 507s"
    assert facts["kv"] == "KV cache 1,938,465 tokens"
    assert facts["concurrency"] == "max concurrency 7.39x at 262,144 tokens/request"
    assert facts["graphs"] == "CUDA graphs captured in 12s (1.05 GiB)"


def test_sglang_milestones():
    log = ("[2026-10-01 12:00:01] Load weight begin. avail mem=110.20 GB\n"
           "[2026-10-01 12:03:11] Load weight end. type=Qwen4ExpForCausalLM, dtype=torch.bfloat16, "
           "avail mem=29.10 GB, mem usage=81.10 GB.\n"
           "[2026-10-01 12:03:20] KV Cache is allocated. max_total_num_tokens=331776\n")
    phase, facts = progress.parse_engine_log(log)
    assert phase == "weights loaded"
    assert facts["sglang_weights"] == "model loaded — 81.1 GB"
    assert facts["sglang_kv"] == "KV cache 331,776 tokens"


def test_empty_log():
    assert progress.parse_engine_log("") == (None, {})


def test_mem_trend_gib_per_minute():
    hist = [(0, 44.0), (10, 40.0), (20, 36.0), (30, 32.0)]
    assert progress.mem_trend(hist, 32.0) == -24.0      # 12 GiB lost over 30s
    assert progress.mem_trend([(0, 44.0)], 44.0) is None
