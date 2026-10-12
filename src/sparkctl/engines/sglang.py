"""SGLang engine: single-node docker serve of an OpenAI-compatible SGLang server.

One GPU per node on Spark, so this engine is single-node only — use the vllm engine for tensor
parallel across the fabric. It reuses vllm.py's container plumbing (per-service naming, env flags,
and the shell-entrypoint docker options that survive ssh -> docker -> bash); the only real
difference is the launch command: `python3 -m sglang.launch_server` with SGLang's own flag names.

Cross-engine recipe fields (model / served_name / port / max_model_len / tool+reasoning parser /
trust_remote_code / speculative) are emitted as first-class flags; everything SGLang-specific
(--mem-fraction-static, --kv-cache-dtype, --max-running-requests, --enable-metrics,
--no-enable-multimodal, …) rides in extra_args verbatim so the recipe follows the SGLang docs 1:1.
"""
import json

from sparkctl import config, remote
from sparkctl.engines.vllm import _docker_common, env_flags, svc_cname


def _sglang_speculative_flags(svc):
    """Translate the engine-agnostic `speculative:` dict into SGLang's discrete flags (vLLM instead
    takes one JSON --speculative-config). `model` is the drafter repo — also what
    recipes.speculative_model downloads + mirrors to every node. Emit only the keys present so
    DFlash/DSpark can fall back to the image's built-in defaults (block size, draft-token count)."""
    spec = svc.get("speculative")
    if not isinstance(spec, dict):
        return []
    out = []
    if spec.get("method"):
        out.append(f"--speculative-algorithm {str(spec['method']).upper()}")
    if spec.get("model"):
        out.append(f"--speculative-draft-model-path {spec['model']}")
    if spec.get("num_speculative_tokens") is not None:
        out.append(f"--speculative-num-draft-tokens {spec['num_speculative_tokens']}")
    if spec.get("draft_quantization"):
        out.append(f"--speculative-draft-model-quantization {spec['draft_quantization']}")
    return out


def _sglang_serve_flags(svc):
    tp = svc.get("parallel", {}).get("tensor", 1)
    port = svc.get("port", config.CFG["cluster"]["api_port"])
    flags = [f"--model-path {svc['model']}",
             f"--served-model-name {svc.get('served_name', 'model')}",
             f"--host 0.0.0.0 --port {port}",
             f"--tp-size {tp}"]
    if svc.get("max_model_len"):
        flags.append(f"--context-length {svc['max_model_len']}")   # SGLang's name for max_model_len
    if svc.get("trust_remote_code"):
        flags.append("--trust-remote-code")
    if svc.get("tool_call_parser"):
        flags.append(f"--tool-call-parser {svc['tool_call_parser']}")
    if svc.get("reasoning_parser"):
        flags.append(f"--reasoning-parser {svc['reasoning_parser']}")
    flags += _sglang_speculative_flags(svc)
    flags += svc.get("extra_args", [])    # SGLang-only knobs: --mem-fraction-static, --kv-cache-dtype,
    return " ".join(flags)                #   --max-running-requests, --enable-metrics, --no-enable-multimodal…


def sglang_up(svc):
    node = svc.get("node", config.HEAD)
    img = svc.get("image", config.IMAGE)
    base_env = {k: str(v) for k, v in (svc.get("env") or {}).items()}
    serve = f"python3 -m sglang.launch_server {_sglang_serve_flags(svc)}"
    print(f"[sglang] single-node serve '{svc['name']}' on {node} (image {img})")
    remote.on(node, f"docker run {_docker_common()} --name {svc_cname(svc['name'])} "
                    f"{env_flags(base_env)} {img} -lc {json.dumps(serve)}")
    print(f"[sglang] launched; watch: sparkctl logs {svc['name']}")


def sglang_down():
    # serve containers share the {PFX}-svc- namespace with the vllm engine; remove them everywhere
    # (down() calls every engine's teardown, so this is idempotent with vllm_down).
    for node in config.NODES:
        remote.on(node, f"docker ps -aq --filter name={config.PFX}-svc- | xargs -r docker rm -f",
                  check=False)
