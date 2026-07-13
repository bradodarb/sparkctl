# Runtime backends

**Status:** implemented — `src/sparkctl/backends/` (this file is the design rationale + roadmap).

`cluster.yaml`'s `backend:` key selects *where/how* a recipe's services actually run. It is the
third pluggability axis, mirroring the two per-service registries that already existed:

- **engine** (`vllm` | `ollama`) — *how* a model is served
- **provider** (`hf` | `ollama`) — *where* weights come from
- **backend** (`docker` | `k8s`) — *where processes run*

## The interface

`backends/base.py` defines the ABC: `up(recipe)`, `down()`, `status()`, `logs(svc)`,
`pull(recipe)`, `endpoints(recipe, served_from)`, `metrics_targets(recipe, served_from)`, and the
generic `run_workload(...)` primitive. The unified server consumes **only**
`endpoints()`/`metrics_targets()` — the gateway route table and metrics aggregation are fully
backend-agnostic, so a new backend needs nothing beyond this class.

## docker (default)

Today's flagship zero-infra path: containers over SSH (`remote.on`), verified download + fabric
rsync for weights, boot persistence via `systemd/sparkctl.service` on the head. Ideal for 1–2
Sparks / homelab.

## k8s (roadmap)

`backends/k8s.py` is a stub (`NotImplementedError`). The sketch: services become Deployments (one
per service; multinode TP via a Ray operator or StatefulSet), `pull` becomes a Job writing to a
shared PVC or per-node hostPath cache, `endpoints()` reads Service addresses, `metrics_targets()`
reads pod endpoints, and `server.mode: k8s` runs the unified server as a Deployment+Service.
Per-backend config lives under a `k8s:` block (`context`, `namespace`, `storage_class`).
