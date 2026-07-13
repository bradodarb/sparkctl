#!/usr/bin/env python3
"""Large-context / needle-in-haystack test for the vLLM OpenAI-compatible endpoint.

Builds a long prompt (~--tokens) of filler text with a secret "passcode" inserted at several
depths (start/middle/end), then asks the model to retrieve each one. Reports the ACTUAL prompt
token count (from the server's usage), latency, and pass/fail per depth — so you can see how the
model holds up as context grows and whether it can retrieve from anywhere in the window.

Dependency-free (urllib). Runs against any OpenAI-compatible server.
  ctx-test.py --base http://localhost:8000/v1 --tokens 30000
  ctx-test.py --base http://coach:8000/v1 --tokens 8000 --depths 0.0,0.25,0.5,0.75,1.0
"""
import argparse, json, time, urllib.request, urllib.error

FILLER = ("The archives of the northern library hold countless unremarkable records. "
          "Clerks copied ledgers by candlelight, noting weather, harvests, and tolls. ")

# deterministic passcodes per depth bucket, so results are reproducible run-to-run
CODES = ["ALPHA-7731", "BRAVO-4492", "CHARLIE-9016", "DELTA-2258", "ECHO-6604",
         "FOXTROT-3390", "GOLF-8125", "HOTEL-5047", "INDIA-1173", "JULIET-9982"]


def api(base, path, payload=None, timeout=600):
    url = base.rstrip("/") + path
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def detect_model(base):
    m = api(base, "/models")["data"][0]
    return m["id"], m.get("max_model_len")


OUTPUT_TOKENS = 512   # answer is just a passcode; leave headroom for a reasoning model to think
MARGIN = 600          # safety cushion since token estimation is approximate


def build_haystack(approx_tokens, needle, depth_frac):
    """~34 actual tokens per numbered filler line; drop the needle line at the requested depth."""
    total_lines = max(10, approx_tokens // 34)
    insert_at = min(total_lines - 1, int(total_lines * depth_frac))
    out = []
    for i in range(total_lines):
        if i == insert_at:
            out.append(f"Line {i}: IMPORTANT -> {needle}")
        else:
            out.append(f"Line {i}: {FILLER}")
    return "\n".join(out)


def run(base, model, tokens, depths, max_len):
    # clamp the prompt target so prompt + output + margin stays under the model's context window
    if max_len:
        cap = max_len - OUTPUT_TOKENS - MARGIN
        if tokens > cap:
            print(f"(clamped ~{tokens} -> ~{cap} to fit {max_len}-token window)")
            tokens = cap
    print(f"endpoint : {base}")
    print(f"model    : {model}  (max_model_len={max_len})")
    print(f"target   : ~{tokens} prompt tokens, {len(depths)} depths\n")
    passed = 0
    for idx, depth in enumerate(depths):
        code = CODES[idx % len(CODES)]
        needle = f"The secret passcode is {code}. Remember it."
        hay = build_haystack(tokens, needle, depth)
        prompt = (hay + "\n\nQuestion: What is the secret passcode mentioned above? "
                  "Answer with ONLY the passcode, nothing else.")
        payload = {"model": model,
                   "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": OUTPUT_TOKENS, "temperature": 0}
        t0 = time.time()
        try:
            d = api(base, "/chat/completions", payload)
        except urllib.error.HTTPError as e:
            print(f"  depth {depth:>4}: HTTP {e.code} {e.reason} -> {e.read()[:200]!r}")
            continue
        except Exception as e:
            print(f"  depth {depth:>4}: ERROR {type(e).__name__}: {str(e)[:120]}")
            continue
        dt = time.time() - t0
        msg = d["choices"][0]["message"]
        ans = (msg.get("content") or "") + (msg.get("reasoning") or "")
        u = d.get("usage", {})
        ok = code in ans
        passed += ok
        print(f"  depth {depth:>4} | prompt_tokens={u.get('prompt_tokens'):>6} "
              f"| {dt:5.1f}s | {'FOUND ' + code if ok else 'MISS (wanted ' + code + ')'}")
    print(f"\n{passed}/{len(depths)} depths retrieved "
          f"{'✅ PASS' if passed == len(depths) else '⚠️  PARTIAL' if passed else '❌ FAIL'}")
    return passed == len(depths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None, help="default: auto-detect from /models")
    ap.add_argument("--tokens", type=int, default=30000, help="approx prompt token target")
    ap.add_argument("--depths", default="0.0,0.5,1.0",
                    help="comma list of needle depths (0.0=start .. 1.0=end)")
    a = ap.parse_args()
    detected, max_len = detect_model(a.base)
    model = a.model or detected
    depths = [float(x) for x in a.depths.split(",")]
    ok = run(a.base, model, a.tokens, depths, max_len)
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
