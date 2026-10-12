"""Load-progress telemetry for `apply`'s readiness wait: what phase each service's engine is in,
the one-time facts worth knowing (weights per rank, KV capacity), per-node free memory, and memguard
kills. Everything an operator otherwise digs out of `docker logs` + `free -g` by hand while a big
model loads — on GB10 unified memory, watching MemAvailable is how you see an OOM hang coming.

Parsing is pure (`parse_engine_log`, `mem_trend`) so it's unit-testable; `probe_node` and
`engine_log` do the remote reads (one ssh round-trip each).
"""
import re

from sparkctl import config, remote

# (regex, phase-label builder) — the LAST matching line in the log tail wins, so order doesn't matter.
_PHASES = [
    (re.compile(r"Loading (?:\w+ )?checkpoint shards:\s+(\d+)% Completed \| (\d+)/(\d+)"),
     lambda m: f"weights {m.group(2)}/{m.group(3)}"),
    (re.compile(r"Loading weights took ([\d.]+) seconds"), lambda m: "weights loaded"),
    (re.compile(r"Model loading took"), lambda m: "profiling memory"),
    (re.compile(r"Capturing (?:\w+ )?CUDA graphs"), lambda m: "capturing CUDA graphs"),
    (re.compile(r"GPU KV cache size"), lambda m: "allocating KV cache"),
    (re.compile(r"Starting vLLM API server|Uvicorn running|Application startup complete"),
     lambda m: "starting API server"),
    # SGLang
    (re.compile(r"Load weight begin"), lambda m: "loading weights"),
    (re.compile(r"Load weight end"), lambda m: "weights loaded"),
    (re.compile(r"Capture cuda graph begin"), lambda m: "capturing CUDA graphs"),
    (re.compile(r"The server is fired up"), lambda m: "starting API server"),
]

# (key, regex, formatter) — printed once each, the first time they appear.
_FACTS = [
    ("weights", re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds"),
     lambda m: f"model loaded — {float(m.group(1)):.1f} GiB/rank in {float(m.group(2)):.0f}s"),
    ("kv", re.compile(r"GPU KV cache size: ([\d,]+) tokens"), lambda m: f"KV cache {m.group(1)} tokens"),
    ("concurrency", re.compile(r"Maximum concurrency for ([\d,]+) tokens per request: ([\d.]+)x"),
     lambda m: f"max concurrency {m.group(2)}x at {m.group(1)} tokens/request"),
    ("graphs", re.compile(r"Graph capturing finished in (\d+) secs, took ([\d.]+) GiB"),
     lambda m: f"CUDA graphs captured in {m.group(1)}s ({m.group(2)} GiB)"),
    ("sglang_kv", re.compile(r"max_total_num_tokens=(\d+)"), lambda m: f"KV cache {int(m.group(1)):,} tokens"),
    ("sglang_weights", re.compile(r"Load weight end\..*?mem usage=([\d.]+) GB"),
     lambda m: f"model loaded — {float(m.group(1)):.1f} GB"),
]


def parse_engine_log(text):
    """(phase, {fact_key: message}) from a vLLM/SGLang log tail. Progress bars rewrite one line
    with \\r, so split on both line endings. phase is None until a known milestone appears."""
    lines = re.split(r"[\r\n]+", text or "")
    phase, facts = None, {}
    for line in lines:
        for rx, label in _PHASES:
            m = rx.search(line)
            if m:
                phase = label(m)
        for key, rx, fmt in _FACTS:
            if key not in facts:
                m = rx.search(line)
                if m:
                    facts[key] = fmt(m)
    return phase, facts


def mem_trend(history, now_gib, window_s=30):
    """GiB/min change of MemAvailable over roughly the last `window_s` (negative = falling), from a
    [(t, gib), ...] history whose last entry is the current sample. None until there's a window."""
    if len(history) < 2:
        return None
    t_now = history[-1][0]
    older = [(t, g) for t, g in history if t_now - t >= window_s]
    t0, g0 = older[-1] if older else history[0]
    if t_now - t0 <= 0:
        return None
    return (now_gib - g0) / (t_now - t0) * 60


def memguard_name():
    return f"{config.PFX}-memguard"


def probe_node(node, since_unix):
    """(mem_available_gib | None, [memguard kill lines since `since_unix`]) — one round-trip."""
    cmd = ("awk '/^MemAvailable:/{print $2}' /proc/meminfo; "
           f"docker logs --since {int(since_unix)} {memguard_name()} 2>&1 | grep 'killing' | tail -3; true")
    r = remote.on(node, cmd, check=False, capture=True)
    out = (r.stdout or "").splitlines()
    if not out or not out[0].strip().isdigit():
        return None, []
    return int(out[0]) / 1048576, [ln.strip() for ln in out[1:] if ln.strip()]


def engine_log(node, cname, tail=400):
    return remote.on(node, f"docker logs --tail {tail} {cname} 2>&1",
                     check=False, capture=True).stdout or ""
