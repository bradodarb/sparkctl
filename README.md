# sparkctl

Config-driven model serving for **NVIDIA DGX Spark** — one node or a cluster (e.g. 2× GB10 +
200G CX7 fabric).

> Valuable even on a **single** Spark: stable model names, a `localhost` OpenAI endpoint decoupled
> from topology, sha256-verified downloads, one-command model switching, and built-in metrics with
> optionally zero extra infrastructure.

## CLI —  one tool, multiple contexts

`sparkctl` auto-detects where it runs (by hostname): on your **control machine** (Mac/laptop),
read-only verbs run locally and mutating verbs auto-deploy the repo then forward to the head over
SSH; on a **cluster node** it is the orchestrator the boot daemon and forwarded commands use.

```bash
sparkctl get nodes|services|recipes|models [-o wide|json]
sparkctl describe node|service|recipe <name>
sparkctl create recipe                     # interactive wizard: writes recipes/<name>.yaml (sane defaults)
sparkctl apply [recipe | -f recipe.yaml] [--wait]  # ensure weights -> (re)start -> update current
                                           #   --wait blocks until every endpoint answers (loaded)
sparkctl delete services --all             # tear everything down (delete service <name> for one)
sparkctl logs <service> [-f] [--tail N]
sparkctl top nodes|services               # live terminal metrics (engine + node baseline)
sparkctl status                            # one-glance summary: drift check + API health
sparkctl serve [stop|status|config|tunnel] [--wait]  # the unified server (gateway+metrics+dash)
sparkctl pull [recipe] | pull-queue <r>... # pre-warm weights without deploying
sparkctl secret set|unset|list [NAME]      # HF_TOKEN etc. -> ~/.sparkctl/secrets.env, synced to nodes
sparkctl deploy [--init] | build           # push repo to nodes | build the serving image
sparkctl test | ctx-test | current | manifest | mirror
```

Install: `pip install -e .` (or just run `./bin/sparkctl` from a checkout — nodes run it straight
from the rsynced repo, needing only `python3` + PyYAML).

## Quickstart

```bash
cp cluster.yaml.example cluster.yaml       # 2-node template; single Spark: cluster.yaml.single-node.example
./bin/sparkctl deploy --init               # one-time: provision nodes + install the boot daemon (sudo)
./bin/sparkctl build                       # build the vLLM+Ray serving image on the nodes
./bin/sparkctl apply qwen3-coder-30b       # pull-if-missing (verified) + serve + set current
./bin/sparkctl serve                       # unified server on localhost:8080 (no Docker needed)
curl localhost:8080/v1/models              # OpenAI-compatible, routed across the cluster
open http://localhost:8080/dash            # live status page
```

## The unified server — gateway + metrics + dashboard, one port

`sparkctl serve` runs ONE process (declared in `cluster.yaml server:`, not separate commands):

| Route | What |
| --- | --- |
| `/v1/*` | OpenAI-compatible API — **LiteLLM** routing/load-balancing generated from the current deployment (replicas of a `served_name` become a pool; chat + embeddings unified) |
| `/metrics` | Prometheus exposition for the **whole cluster** — every node's vLLM metrics (labeled `node=`/`service=`) plus per-node baseline gauges. Point any Prometheus at this one target |
| `/dash` | Zero-dependency HTML status page (services, per-node memory/GPU/disk) |
| `/healthz` | Control-plane health |

LiteLLM runs as a supervised subprocess — if it crashes, `/metrics` and `/dash` stay up (and vice
versa), and the server auto-restarts it with capped backoff, so `/v1` heals on its own. Run modes via `server.mode`: **local** (plain process, no Docker — default; deps
auto-installed into `~/.sparkctl/venv` on first run), **docker** (server container + LiteLLM
sidecar), **k8s** (roadmap). `server.host` places it on your dev machine (`local`) or a node
(`serve tunnel` port-forwards to it). The server restarts itself with fresh routes after `apply`.

**Node baseline metrics** (works even for engines with no metrics endpoint, e.g. Ollama): unified
memory used/total from `/proc/meminfo` (the honest number on GB10 — `nvidia-smi` memory reads
`N/A`), GPU utilization, and model-cache disk, sampled over SSH — no agents to install. Exposed as
`sparkctl_node_*` gauges, on `/dash`, and in `sparkctl top nodes`.

**Optional Grafana** (`server.grafana.enabled: true`, needs Docker): Prometheus + Grafana
containers auto-provisioned with the vLLM cluster dashboard, scraping the unified server's
`/metrics` as a single target.

## Recipes & serving modes

```yaml
name: <recipe>
services:
  - name: agent
    engine: vllm
    provider: hf                 # hf | ollama  (defaults from engine)
    image: nvcr.io/...           # OPTIONAL per-recipe pin; omit to inherit cluster default
    model: <hf-repo>             # NVFP4 preferred (Blackwell-native)
    served_name: <api-name>
    node: coach                  # OPTIONAL: pin single-node serve (omit for TP=2 multinode)
    parallel: { tensor: 1 }      # 1 = single-node; 2 = across both nodes (Ray over the fabric)
    port: 8000
    max_model_len: 65536
    gpu_memory_utilization: 0.85
    tool_call_parser: qwen3_coder     # + reasoning_parser, extra_args: [...], env: {...}
  - name: embeddings
    engine: ollama
    node: ref
    model: nomic-embed-text
    port: 11434
```

- **Single-node** (`tensor: 1` + `node:`) — one `vllm serve`, no Ray. Replicate the same
  `served_name` on both nodes for two concurrent agents behind one endpoint.
- **Multi-node** (`tensor: 2`, no `node:`) — Ray head+worker, tensor-parallel over the fabric, one
  endpoint on the head. For models too big for one node.

`sparkctl apply -f my-recipe.yaml` copies a manifest into `recipes/` and deploys it.

## Backends

`backend:` in `cluster.yaml` selects where services run: **docker** (default — containers over
SSH, boot persistence via systemd on the head) or **k8s** (roadmap stub; the seam is
`src/sparkctl/backends/base.py` — the gateway and metrics consume only `endpoints()` /
`metrics_targets()`, so a new backend needs nothing else).

## Model distribution — verified, NAS-aware

`apply` ensures weights before serving: **nodes → NAS → download**.

- Default (no NAS): download **once on the head** (single stream, resumable, stall-watchdog),
  **verify every shard** (`sha256 == blob name`, corrupt shards auto-deleted + re-fetched), then
  rsync-mirror over the fabric and re-verify — a corrupt shard can never reach a serve.
- With a `nas:` block: weights already on the NAS **replicate instead of re-downloading** (mounted
  path or rsync-over-ssh endpoint). A mounted NAS is always the download target for fresh pulls;
  an ssh-mode NAS gets a copy archived back after the head downloads.
- `sparkctl get models` inventories **everything installed** — each node's HF cache, each node's
  ollama store, and the NAS — unioned with the current recipe's requirements. Columns: SOURCE
  (`hf`/`ollama`), SIZE + PRECISION (read from the weights themselves: quantization config /
  dtype / GGUF file type), SERVICES (current-recipe services using the model), plus per-node and
  NAS presence. What's actively serving lives in `get services` / `status`.

**Secrets** (`HF_TOKEN` for gated models, LiteLLM upstream keys, ...): `sparkctl secret set
HF_TOKEN` prompts without echo, writes `~/.sparkctl/secrets.env` (0600), and syncs it to every
node — never in cluster.yaml or git. Downloads, the ollama container, and the LiteLLM child all
pick it up automatically. Weights live in `cluster.model_cache`, never in git. `pull-queue a b c`
queues several overnight (detached on the head, survives your laptop sleeping).

## Deployment verification

On `apply`, each node gets `state/active.json` (recipe + `recipe_sha256` + services + time).
`status` / `get services` show whether the *running* deployment matches the intended recipe
version (`✅ matches` / `⚠️ DRIFT` / `nothing active`) — not just "something is up".

## Layout

```
cluster.yaml.example         # topology, backend, server, nas, download, fabric — the one config
cluster.yaml.single-node.example   # same, for a single Spark (no fabric block)
cluster.yaml, current        # YOUR config + active-recipe pointer (gitignored — local per cluster)
recipes/*.yaml               # one file per model config; each may run MULTIPLE services
pyproject.toml, src/sparkctl # the Python package (cli/, backends/, engines/, server/, ...)
bin/sparkctl                 # shim: run the package straight from a checkout (what nodes use)
bin/verify-model.py          # sha256 integrity gate (blob name == sha256(content))
bin/ctx-test.py              # large-context needle-in-haystack test  (sparkctl ctx-test)
bin/smoke-tool-call.py       # tool-calling smoke test                (sparkctl test)
docker/Dockerfile            # derived vLLM + Ray serving image (built by `sparkctl build`)
docker/server.Dockerfile     # the unified server image (server.mode: docker)
docker/grafana/dashboards/   # Grafana dashboards (optional extra)
systemd/                     # boot daemon + optional server unit (installed by deploy --init)
tests/                       # pytest suite — no live cluster needed (pip install -e '.[dev]')
```

## Tests

```bash
pytest -q        # config/routing/manifest/distribution/server logic — no live cluster needed
```

## Notes / gotchas (hard-won)

- **Fabric:** the 200G port is two bonded 100G lanes — NCCL must use **both** or you get ~half
  bandwidth (MTU 9000, GID 3; see the `fabric:` block).
- **vLLM version:** NVFP4 compressed-tensors MoE checkpoints need vLLM ≥ 0.22 (the `26.06` image);
  older images fail to load them. **Ray** isn't in `26.06` — the derived image adds it.
- **Unified memory:** the 128GB is shared system+GPU; keep `gpu_memory_utilization` low enough to
  leave headroom (0.85), and don't run big downloads *while loading* a big model. `nvidia-smi`
  memory reads `N/A` — sparkctl's node baseline uses `/proc/meminfo` instead.
- **Mirror** is owner/group-agnostic and excludes `.locks/`/`*.incomplete`; never leave an orphaned
  root download container running (it holds the HF flock and blocks later downloads).
