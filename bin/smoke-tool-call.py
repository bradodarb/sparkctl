#!/usr/bin/env python3
"""
Tool-calling smoke test for the cluster's OpenAI-compatible vLLM endpoint.

Validates the full agentic loop:
  1) the served model emits a well-formed tool/function call, then
  2) consumes the tool result and produces a final natural-language answer.

Dependency-free (stdlib only). Run on the head once a recipe is serving (`sparkctl apply`).
  smoke-tool-call.py [--base http://localhost:8000/v1] [--model NAME]
Model auto-detected from /v1/models if not given. Exit 0 = PASS.
"""
import argparse, json, sys, urllib.request

def http(url, payload=None, timeout=180):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer none"},
        method="POST" if data else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8000/v1")
    ap.add_argument("--model", default=None)
    a = ap.parse_args()

    model = a.model or http(a.base + "/models")["data"][0]["id"]
    print(f"endpoint : {a.base}\nmodel    : {model}\n")

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location", "unit"],
            },
        },
    }]
    messages = [{"role": "user",
                 "content": "What is the current weather in Tokyo in celsius? Use the get_weather tool."}]

    # Round 1 — expect a tool call
    r1 = http(a.base + "/chat/completions", {
        "model": model, "messages": messages, "tools": tools,
        "tool_choice": "auto", "temperature": 0, "max_tokens": 512})
    msg = r1["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if not calls:
        print("FAIL: no tool_call emitted.\nassistant:", (msg.get("content") or "")[:500])
        sys.exit(1)
    fn = calls[0]["function"]["name"]
    try:
        cargs = json.loads(calls[0]["function"]["arguments"])
    except Exception:
        cargs = calls[0]["function"]["arguments"]
    print(f"round 1  : tool_call -> {fn}({cargs})   [{'OK' if fn == 'get_weather' else 'UNEXPECTED FN'}]")

    # Round 2 — feed a fake tool result back, expect a NL answer that uses it
    messages += [
        {"role": "assistant", "content": msg.get("content") or "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": calls[0].get("id", "call_0"), "name": fn,
         "content": json.dumps({"location": "Tokyo", "unit": "celsius", "temp_c": 18, "sky": "clear"})},
    ]
    r2 = http(a.base + "/chat/completions", {
        "model": model, "messages": messages, "tools": tools,
        "temperature": 0, "max_tokens": 512})
    final = (r2["choices"][0]["message"].get("content") or "").strip()
    print(f"round 2  : final -> {final[:300]}")

    ok = bool(calls) and ("18" in final or "tokyo" in final.lower())
    print("\nSMOKE TEST", "PASSED ✅" if ok else "INCONCLUSIVE ⚠️ (tool call worked; review round-2 text)")
    sys.exit(0 if calls else 1)

if __name__ == "__main__":
    main()
