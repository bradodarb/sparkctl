"""`sparkctl validate [recipe] [--fix]` — a rule-based linter for recipes.

It catches the misconfigurations that don't surface until a service crashes minutes into loading —
most of them couplings a hand-edit can silently break: flags that only make sense single-node vs
multinode, env a multinode serve needs on unified memory, and vllm-only knobs set on an ollama
service. Each finding carries a plain-English `suggestion` and, when the fix is unambiguous, a
`fix` callable that mutates the recipe in place (used by `--fix`).

`lint_recipe` is pure (dict in, findings out) so it's unit-testable and reused by `edit`/`create`;
`cmd_validate` is the thin shell that loads recipes, prints, and (with --fix) writes back. Only
findings at level 'error' make `validate` exit non-zero.
"""
import sys

import yaml

ERROR, WARN, INFO = "error", "warn", "info"
_ICON = {ERROR: "❌", WARN: "⚠️ ", INFO: "· "}

# Fields that only the vllm engine honors — set on an ollama service they're silently ignored.
_VLLM_ONLY = ("parallel", "served_name", "max_model_len", "gpu_memory_utilization",
              "tool_call_parser", "reasoning_parser", "extra_args")


class Finding:
    """One lint result. `fix`, if present, is a zero-arg callable that repairs the recipe in place."""
    __slots__ = ("level", "message", "suggestion", "fix")

    def __init__(self, level, message, suggestion=None, fix=None):
        self.level, self.message, self.suggestion, self.fix = level, message, suggestion, fix


def _world(svc):
    p = svc.get("parallel") or {}
    return p.get("tensor", 1) * p.get("pipeline", 1)


# ---------------------------------------------------------------- in-place fix helpers
def _rewrite_load_format(svc, *, multinode):
    """Drop any existing --load-format / --safetensors-load-strategy args and set the Spark-correct
    one for the topology, preserving every other extra arg. Multinode uses fastsafetensors (GDS
    auto-disables when world>1); single-node uses plain streamed safetensors — NEVER eager, which
    loads the whole checkpoint into host RAM and OOMs on unified memory (GPU + host share one pool)."""
    extra = [a for a in (svc.get("extra_args") or [])
             if not a.startswith("--load-format") and not a.startswith("--safetensors-load-strategy")]
    extra += ["--load-format=fastsafetensors"] if multinode else ["--load-format=safetensors"]
    svc["extra_args"] = extra


def _set_env(svc, key, val):
    svc.setdefault("env", {})[key] = val


def _del_env(svc, key):
    env = svc.get("env")
    if env:
        env.pop(key, None)
        if not env:
            svc.pop("env", None)


def _drop_eager(svc):
    svc["extra_args"] = [a for a in (svc.get("extra_args") or [])
                         if "safetensors-load-strategy=eager" not in a]


# ---------------------------------------------------------------- the linter
def lint_recipe(recipe, *, known_nodes=None, head=None):
    """Return a list of Finding for one recipe dict. Pass known_nodes/head from config to enable
    node-existence and port-collision checks; omit them to skip those. Pure except that a Finding's
    `fix()` closure, if later invoked, mutates the passed-in recipe."""
    out = []

    def add(level, message, suggestion=None, fix=None):
        out.append(Finding(level, message, suggestion, fix))

    if not isinstance(recipe, dict):
        return [Finding(ERROR, "recipe is not a mapping")]
    if not recipe.get("name"):
        add(ERROR, "missing top-level 'name'", "add name: <recipe>")
    services = recipe.get("services")
    if not isinstance(services, list) or not services:
        add(ERROR, "recipe has no 'services' list", "add at least one service under services:")
        return out

    ports = {}  # (node, port) -> service label
    for i, svc in enumerate(services):
        label = svc.get("name") or f"services[{i}]"
        engine = svc.get("engine")
        if engine not in ("vllm", "ollama"):
            add(ERROR, f"{label}: engine must be 'vllm' or 'ollama' (got {engine!r})",
                "set engine: vllm  (or engine: ollama)")
            continue
        if not svc.get("model"):
            add(ERROR, f"{label}: missing 'model'", "add model: <hf-repo or ollama name>")

        world = _world(svc)
        provider = svc.get("provider")
        env = svc.get("env") or {}
        extra = svc.get("extra_args") or []
        node = svc.get("node")

        if engine == "ollama":
            for f in _VLLM_ONLY:
                if f in svc:
                    add(WARN, f"{label}: '{f}' has no effect on an ollama service (ignored)",
                        f"remove '{f}'", fix=lambda s=svc, k=f: s.pop(k, None))
            if provider not in (None, "ollama"):
                add(WARN, f"{label}: ollama engine expects provider 'ollama' (got {provider!r})",
                    "set provider: ollama", fix=lambda s=svc: s.__setitem__("provider", "ollama"))
            if not node:
                add(INFO, f"{label}: no 'node' set — ollama defaults to the head")
        else:  # vllm
            if provider not in (None, "hf"):
                add(WARN, f"{label}: vllm engine expects provider 'hf' (got {provider!r})",
                    "set provider: hf", fix=lambda s=svc: s.__setitem__("provider", "hf"))
            g = svc.get("gpu_memory_utilization")
            if g is not None and not (0 < g <= 1):
                add(ERROR, f"{label}: gpu_memory_utilization must be in (0, 1] (got {g})",
                    "use a fraction like 0.85 (leave unified-memory headroom)")
            elif g is not None and g > 0.9:
                # On GB10 the 128GB is UNIFIED: this fraction of it is pre-allocated for KV cache at
                # startup and shares silicon with the OS/Docker/page-cache reading the shards — too
                # high OOM-kills the node *during load* even for a small model.
                add(WARN, f"{label}: gpu_memory_utilization {g} is aggressive on unified memory — KV cache "
                    f"is pre-allocated from the shared 128GB pool and can OOM the node during load",
                    "lower to ~0.85 or below (unified memory is system+GPU, not spare VRAM)")
            mml = svc.get("max_model_len")
            if mml is not None and (not isinstance(mml, int) or mml <= 0):
                add(ERROR, f"{label}: max_model_len must be a positive integer (got {mml!r})",
                    "set a positive integer, e.g. 32768")

            has_fast = any("fastsafetensors" in a for a in extra)
            has_eager = any("safetensors-load-strategy=eager" in a for a in extra)
            ray_off = str(env.get("RAY_memory_monitor_refresh_ms")) == "0"

            # eager loads the ENTIRE checkpoint into host RAM before the GPU copy. On unified memory
            # that host buffer and the gpu_memory_utilization reservation draw from ONE 128GB pool, so
            # a large model OOM-kills the loader mid-load. vLLM streams shards by default — never pin
            # eager on Spark (single- OR multi-node).
            if has_eager:
                add(WARN, f"{label}: --safetensors-load-strategy=eager loads the whole checkpoint into host RAM "
                    f"and can OOM on unified memory (host load + GPU reservation share one 128GB pool)",
                    "remove --safetensors-load-strategy=eager (vLLM streams shards by default)",
                    fix=lambda s=svc: _drop_eager(s))

            if world >= 2:  # multinode: runs on the Ray head, tensor-parallel over the fabric
                if node:
                    add(WARN, f"{label}: parallel.tensor>=2 runs on the Ray head; pinned 'node: {node}' is ignored",
                        "remove 'node:' for a multinode service",
                        fix=lambda s=svc: s.pop("node", None))
                if not has_fast:
                    add(WARN, f"{label}: multinode (tensor>=2) should use --load-format=fastsafetensors — "
                        f"single-node load flags don't auto-disable GDS/cuFile and can crash on load",
                        "set extra_args load-format to --load-format=fastsafetensors",
                        fix=lambda s=svc: _rewrite_load_format(s, multinode=True))
                if not ray_off:
                    add(WARN, f"{label}: multinode on unified memory needs env RAY_memory_monitor_refresh_ms: \"0\" "
                        f"or Ray's memory monitor will OOM-kill the TP workers during load",
                        'add env RAY_memory_monitor_refresh_ms: "0"',
                        fix=lambda s=svc: _set_env(s, "RAY_memory_monitor_refresh_ms", "0"))
            else:  # single-node
                if has_fast:
                    add(WARN, f"{label}: fastsafetensors is the multinode/GDS path (auto-disables only when world>1); "
                        f"single-node should use plain --load-format=safetensors (streamed)",
                        "set extra_args load-format to --load-format=safetensors",
                        fix=lambda s=svc: _rewrite_load_format(s, multinode=False))
                if "RAY_memory_monitor_refresh_ms" in env:
                    add(INFO, f"{label}: RAY_memory_monitor_refresh_ms only matters for multinode (tensor>=2)",
                        "remove env RAY_memory_monitor_refresh_ms",
                        fix=lambda s=svc: _del_env(s, "RAY_memory_monitor_refresh_ms"))

            if "nvfp4" in (svc.get("model") or "").lower() and env.get("VLLM_USE_FLASHINFER_MOE_FP4") != "1":
                add(WARN, f"{label}: NVFP4 model without env VLLM_USE_FLASHINFER_MOE_FP4: \"1\" (FlashInfer MoE kernels off)",
                    'add env VLLM_USE_FLASHINFER_MOE_FP4: "1"',
                    fix=lambda s=svc: _set_env(s, "VLLM_USE_FLASHINFER_MOE_FP4", "1"))

        # cross-service / topology checks (both engines)
        if node and known_nodes and node not in known_nodes:
            add(ERROR, f"{label}: node '{node}' is not in cluster.yaml nodes ({', '.join(sorted(known_nodes))})",
                f"use one of: {', '.join(sorted(known_nodes))}")
        eff_node = head if (engine == "vllm" and world >= 2) else (node or head)
        port = svc.get("port")
        if port is not None and eff_node is not None:
            key = (eff_node, port)
            if key in ports:
                add(ERROR, f"{label}: port {port} on '{eff_node}' already used by '{ports[key]}'",
                    "give each service on a node a unique port")
            else:
                ports[key] = label

    return out


def format_findings(findings):
    """Render findings as indented lines. Empty list -> a single 'looks good' line. Manual-only
    findings (no auto-fix) are marked so `--fix` won't be expected to handle them."""
    if not findings:
        return ["  ✅ looks good"]
    lines = []
    for f in findings:
        lines.append(f"  {_ICON[f.level]} {f.message}")
        if f.suggestion:
            tag = "" if f.fix else " (manual)"
            lines.append(f"       ↳ {f.suggestion}{tag}")
    return lines


# ---------------------------------------------------------------- YAML write-back (--fix)
def _dump_recipe(recipe, original_text=""):
    """Serialize a recipe, preserving the leading comment/header block from the original file.
    (PyYAML can't round-trip inline comments; the header is the part worth keeping.)"""
    header = []
    for line in original_text.splitlines():
        if line.startswith("#") or not line.strip():
            header.append(line)
        else:
            break
    prefix = ("\n".join(header).rstrip() + "\n\n") if any(h.strip() for h in header) else ""
    return prefix + yaml.safe_dump(recipe, sort_keys=False, default_flow_style=False,
                                   width=100, allow_unicode=True)


def cmd_validate(args):
    """Lint one recipe (by name) or every recipe in recipes/. With --fix, apply the auto-fixable
    findings and write the file back. Exits 1 if any error remains."""
    from sparkctl import config
    from sparkctl.recipes import load_recipe

    if args.recipe:
        names = [args.recipe]
    else:
        names = sorted(p.stem for p in (config.ROOT / "recipes").glob("*.yaml"))
        if not names:
            print("no recipes to validate.")
            return

    kn, head = set(config.NODES), config.HEAD
    total_err = total_warn = total_fixed = 0
    for name in names:
        path = config.ROOT / "recipes" / f"{name}.yaml"
        recipe = load_recipe(name)   # exits with a clear message if a named recipe is missing
        findings = lint_recipe(recipe, known_nodes=kn, head=head)
        print(f"recipe: {name}")
        for line in format_findings(findings):
            print(line)

        if args.fix:
            fixable = [f for f in findings if f.fix]
            for f in fixable:
                f.fix()
            if fixable:
                path.write_text(_dump_recipe(recipe, path.read_text()))
                findings = lint_recipe(recipe, known_nodes=kn, head=head)   # re-lint the fixed dict
                total_fixed += len(fixable)
                print(f"  🔧 applied {len(fixable)} fix(es) -> {path}")
                manual = [f for f in findings if not f.fix]
                if manual:
                    print(f"  {len(manual)} finding(s) still need a manual fix:")
                    for line in format_findings(manual):
                        print(line)

        total_err += sum(1 for f in findings if f.level == ERROR)
        total_warn += sum(1 for f in findings if f.level == WARN)

    tail = f" — {total_fixed} auto-fixed" if args.fix else ""
    if len(names) > 1 or args.fix:
        print(f"\n{len(names)} recipe(s): {total_err} error(s), {total_warn} warning(s){tail}")
    if total_err:
        sys.exit(1)
