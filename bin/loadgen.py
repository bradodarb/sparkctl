#!/usr/bin/env python3
"""Synthetic load generator for pretty Grafana screenshots.

Hammers the sparkctl OpenAI-compatible endpoint with waves of traffic so the
vLLM dashboard panels (throughput, TTFT, KV-cache %, running/waiting queue)
all show movement. Not a benchmark — it's here to make the graphs interesting.

Stdlib only. Ctrl-C to stop.

Examples:
    ./bin/loadgen.py                        # localhost:8080, current model, forever
    ./bin/loadgen.py --model qwen3-coder-30b --peak 24
    ./bin/loadgen.py --url http://coach:8080 --key sk-local-dev --duration 600
"""
import argparse
import json
import math
import os
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

# Varied prompts -> varied prompt-token counts (moves "prompt throughput" + KV cache).
PROMPTS = [
    "Hi.",
    "Explain what a KV cache is in one sentence.",
    "Write a Python function that reverses a linked list.",
    "Summarize the tradeoffs of tensor vs pipeline parallelism.",
    "Refactor this loop for readability:\n" + "for i in range(10):\n    print(i*i)\n" * 6,
    "Here is a stack trace, tell me the likely cause:\n" + "  File 'x.py', line 42, in foo\n" * 20,
    "Write a detailed technical design doc for a distributed rate limiter, "
    "covering data model, failure modes, and rollout plan.",
]


class Stats:
    def __init__(self):
        self.lock = threading.Lock()
        self.ok = 0
        self.err = 0
        self.tokens = 0

    def record(self, ok, tokens=0):
        with self.lock:
            if ok:
                self.ok += 1
                self.tokens += tokens
            else:
                self.err += 1

    def snapshot_and_reset(self):
        with self.lock:
            ok, err, tokens = self.ok, self.err, self.tokens
            self.ok = self.err = self.tokens = 0
        return ok, err, tokens


def unique_filler(approx_tokens):
    """Random, non-repeating text ~approx_tokens long.

    Random so vLLM prefix-caching can't dedupe the KV blocks — each request
    must allocate its own, which is what actually fills the cache. ~0.75
    words/token is a rough English ratio.
    """
    n_words = max(1, int(approx_tokens * 0.75))
    return " ".join(str(random.getrandbits(20)) for _ in range(n_words))


def build_prompt(args):
    if args.prompt_tokens > 0:
        return ("Read this log dump and note anything unusual:\n"
                + unique_filler(args.prompt_tokens))
    return random.choice(PROMPTS)


def one_request(args, stats):
    """Fire a single chat completion with randomized shape."""
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": build_prompt(args)}],
        "max_tokens": random.randint(args.min_tokens, args.max_tokens),
        "temperature": round(random.uniform(0.2, 1.0), 2),
        "stream": False,
    }
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{args.url.rstrip('/')}/v1/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", **args.auth},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            payload = json.load(resp)
        toks = payload.get("usage", {}).get("completion_tokens", 0)
        stats.record(True, toks)
    except (urllib.error.URLError, TimeoutError, ValueError, OSError):
        stats.record(False)


def concurrency_at(t, period, peak, floor):
    """Sine wave (ramp -> peak -> cool-down) with occasional bursts."""
    base = floor + (peak - floor) * (0.5 - 0.5 * math.cos(2 * math.pi * t / period))
    # Every ~period/4 seconds, a short burst to build a visible queue.
    if int(t) % max(1, int(period / 4)) == 0 and int(t * 10) % 10 == 0:
        base = peak * 1.5
    return max(floor, int(round(base)))


def main():
    ap = argparse.ArgumentParser(description="Grafana load generator for sparkctl")
    ap.add_argument("--url", default=os.environ.get("SPARKCTL_URL", "http://localhost:8080"))
    ap.add_argument("--model", default=None, help="served model name (default: ./current)")
    ap.add_argument("--key", default=os.environ.get("SPARKCTL_KEY"), help="master_key if set")
    ap.add_argument("--peak", type=int, default=16, help="peak concurrent requests")
    ap.add_argument("--floor", type=int, default=1, help="minimum concurrent requests")
    ap.add_argument("--period", type=float, default=90.0, help="wave period in seconds")
    ap.add_argument("--min-tokens", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="pad each prompt with unique random text to ~N tokens "
                         "(fills KV cache; defeats prefix caching)")
    ap.add_argument("--kv-stress", action="store_true",
                    help="preset to exercise KV cache: long unique prompts, long "
                         "outputs, high sustained concurrency")
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--duration", type=float, default=0.0, help="seconds to run (0 = forever)")
    args = ap.parse_args()

    if args.kv_stress:
        # Only bump values the user didn't override from defaults.
        if args.prompt_tokens == 0:
            args.prompt_tokens = 8192
        if args.max_tokens == 512:
            args.max_tokens = 2048
        if args.min_tokens == 16:
            args.min_tokens = 1024
        if args.peak == 16:
            args.peak = 32
        if args.floor == 1:
            args.floor = 8

    if args.model is None:
        here = os.path.dirname(os.path.abspath(__file__))
        cur = os.path.join(os.path.dirname(here), "current")
        try:
            with open(cur) as f:
                args.model = f.read().strip()
        except OSError:
            print("no --model given and ./current not found", file=sys.stderr)
            sys.exit(1)

    args.auth = {"Authorization": f"Bearer {args.key}"} if args.key else {}

    print(f"load -> {args.url}  model={args.model}  peak={args.peak} floor={args.floor} "
          f"period={args.period}s  (Ctrl-C to stop)")

    stats = Stats()
    start = time.time()
    last_report = start
    # Big pool; we throttle by how many we submit per tick, not pool size.
    pool = ThreadPoolExecutor(max_workers=args.peak * 3 + 8)
    inflight = []
    try:
        while True:
            now = time.time()
            elapsed = now - start
            if args.duration and elapsed >= args.duration:
                break

            target = concurrency_at(elapsed, args.period, args.peak, args.floor)
            inflight = [f for f in inflight if not f.done()]
            deficit = target - len(inflight)
            for _ in range(max(0, deficit)):
                inflight.append(pool.submit(one_request, args, stats))

            if now - last_report >= 5.0:
                window = now - last_report
                ok, err, tokens = stats.snapshot_and_reset()
                rps = ok / window
                tps = tokens / window
                print(f"[{elapsed:6.0f}s] conc~{target:3d}  ok={ok:4d} err={err:3d}  "
                      f"{rps:5.1f} req/s  {tps:7.0f} tok/s", flush=True)
                last_report = now

            time.sleep(0.2)
    except KeyboardInterrupt:
        print("\nstopping…")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
