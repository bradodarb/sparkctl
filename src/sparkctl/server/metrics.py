"""Metrics aggregation: the unified server scrapes every node's vLLM /metrics and re-exposes the
whole cluster on its own /metrics — one scrape target for any external Prometheus (and the data
source for /dash). All transformation functions are pure; the Scraper holds the live cache."""
import asyncio
import time

from sparkctl import remote
from sparkctl.backends import get_backend

# Headline metrics surfaced on /dash and `sparkctl top` (vLLM's native exposition names).
HEADLINE = ["vllm:num_requests_running", "vllm:num_requests_waiting", "vllm:gpu_cache_usage_perc"]


def scrape_targets(recipe, served_from):
    """The backend's /metrics endpoints for a recipe (only engines that expose Prometheus
    metrics — Ollama doesn't, so it's never a target)."""
    return get_backend().metrics_targets(recipe, served_from)


def inject_labels(body, labels):
    """Pure: add node=/service= labels to every sample in a Prometheus exposition body so
    multi-node output stays distinguishable after concatenation. Comment lines (# HELP/# TYPE)
    are dropped — they'd be duplicated across nodes; untyped samples are valid exposition."""
    lab = ",".join(f'{k}="{v}"' for k, v in labels.items())
    out = []
    for line in (body or "").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        head = line.split(" ", 1)[0]
        if "{" in head and line.rstrip().find("}") != -1:
            name, rest = line.split("{", 1)
            existing, _, tail = rest.partition("}")
            merged = f"{lab},{existing}" if existing else lab
            out.append(f"{name}{{{merged}}}{tail}")
        elif " " in line:
            name, _, tail = line.partition(" ")
            out.append(f"{name}{{{lab}}} {tail}")
    return "\n".join(out)


def parse_gauges(body, names):
    """Pure: sum each named metric across its label sets in an exposition body."""
    out = {}
    for line in (body or "").splitlines():
        if line.startswith("#"):
            continue
        for n in names:
            if line.startswith(n + "{") or line.startswith(n + " "):
                try:
                    out[n] = out.get(n, 0.0) + float(line.rsplit(" ", 1)[1])
                except (IndexError, ValueError):
                    pass
    return out


def grafana_prometheus_config(server_addr, interval_s=10):
    """Pure: Prometheus scrape config for the optional Grafana stack — the unified server's
    /metrics is the single target (it already aggregates the whole cluster)."""
    return {"global": {"scrape_interval": f"{interval_s}s"},
            "scrape_configs": [{"job_name": "sparkctl", "metrics_path": "/metrics",
                                "static_configs": [{"targets": [server_addr]}]}]}


# ---------------------------------------------------------------- node baseline stats
# Engine metrics only exist where the engine exposes them (Ollama doesn't). These node-level
# gauges are the baseline every DGX Spark can report: unified memory from /proc/meminfo (the
# real signal — nvidia-smi reads N/A for memory on GB10 unified memory), GPU utilization from
# nvidia-smi (which does work), and model-cache disk usage.
NODE_GAUGES = {  # key -> (exposition name, multiplier to base unit)
    "mem_total_kb": ("sparkctl_node_memory_total_bytes", 1024),
    "mem_used_kb": ("sparkctl_node_memory_used_bytes", 1024),
    "gpu_util_pct": ("sparkctl_node_gpu_utilization_percent", 1),
    "disk_total_kb": ("sparkctl_node_disk_total_bytes", 1024),
    "disk_used_kb": ("sparkctl_node_disk_used_bytes", 1024),
}


def node_stats_cmd(cache):
    """Pure: one shell command emitting key=value baseline stats for a node. No $() command
    substitution — remote.on ships commands inside double quotes, so $() would expand on the
    CALLING machine's shell before ssh ever runs."""
    return ("awk '/MemTotal/{print \"mem_total_kb=\"$2} /MemAvailable/{print \"mem_avail_kb=\"$2}' "
            "/proc/meminfo; "
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null "
            "| head -1 | sed 's/^/gpu_util_pct=/'; "
            f"df -k {cache} 2>/dev/null "
            "| awk 'NR==2{print \"disk_total_kb=\"$2; print \"disk_used_kb=\"$3}'")


def parse_node_stats(text):
    """Pure: key=value lines -> float dict; non-numeric values (nvidia-smi '[N/A]', blanks) are
    dropped rather than reported wrong. Derives mem_used_kb from total - available."""
    out = {}
    for line in (text or "").splitlines():
        k, sep, v = line.strip().partition("=")
        if not sep:
            continue
        try:
            out[k] = float(v)
        except ValueError:
            pass
    if "mem_total_kb" in out and "mem_avail_kb" in out:
        out["mem_used_kb"] = out["mem_total_kb"] - out.pop("mem_avail_kb")
    return out


class NodeSampler:
    """Baseline node stats sampled over SSH on the scrape interval — no node-side installs."""

    def __init__(self, nodes, cache, interval_s=10):
        self.nodes = list(nodes)
        self.cmd = node_stats_cmd(cache)
        self.interval = interval_s
        self.results = {}   # node -> {key: float}

    def sample_node(self, node):
        r = remote.on(node, self.cmd, capture=True, check=False)
        self.results[node] = parse_node_stats(r.stdout) if r.returncode == 0 else {}

    async def run(self):
        while True:
            await asyncio.gather(*(asyncio.to_thread(self.sample_node, n) for n in self.nodes))
            await asyncio.sleep(self.interval)

    def exposition(self):
        parts = []
        for node, stats in self.results.items():
            for key, (name, mult) in NODE_GAUGES.items():
                if key in stats:
                    parts.append(f'{name}{{node="{node}"}} {stats[key] * mult:g}')
        return "\n".join(parts) + ("\n" if parts else "")

    def summaries(self):
        rows = []
        for node in self.nodes:
            s = self.results.get(node, {})
            rows.append({"node": node,
                         "mem_used_gib": s.get("mem_used_kb", 0) / 2**20,
                         "mem_total_gib": s.get("mem_total_kb", 0) / 2**20,
                         "gpu_util": s.get("gpu_util_pct"),
                         "disk_used_gib": s.get("disk_used_kb", 0) / 2**20,
                         "disk_total_gib": s.get("disk_total_kb", 0) / 2**20})
        return rows


class Scraper:
    """Async cache of every target's latest exposition body, refreshed on an interval."""

    def __init__(self, targets, interval_s=10):
        self.targets = targets
        self.interval = interval_s
        self.results = {}   # (node, service) -> {"ok": bool, "body": str, "ts": float}

    async def _one(self, client, t):
        key = (t["node"], t["service"])
        try:
            r = await client.get(t["url"])
            r.raise_for_status()
            self.results[key] = {"ok": True, "body": r.text, "ts": time.time()}
        except Exception:
            self.results[key] = {"ok": False, "body": "",
                                 "ts": self.results.get(key, {}).get("ts", 0.0)}

    async def scrape_once(self):
        import httpx
        async with httpx.AsyncClient(timeout=5) as client:
            await asyncio.gather(*(self._one(client, t) for t in self.targets))

    async def run(self):
        while True:
            await self.scrape_once()
            await asyncio.sleep(self.interval)

    def exposition(self):
        """The aggregated /metrics body: every target's samples with node/service labels injected,
        plus sparkctl's own per-target up/down gauge."""
        parts = []
        for t in self.targets:
            res = self.results.get((t["node"], t["service"]), {"ok": False, "body": ""})
            parts.append(f'sparkctl_target_up{{node="{t["node"]}",service="{t["service"]}"}} '
                         f'{1 if res["ok"] else 0}')
            if res["ok"]:
                parts.append(inject_labels(res["body"], {"node": t["node"], "service": t["service"]}))
        return "\n".join(p for p in parts if p) + "\n"

    def summaries(self):
        """Per-service headline gauges for /dash."""
        rows = []
        for t in self.targets:
            res = self.results.get((t["node"], t["service"]), {"ok": False, "body": ""})
            g = parse_gauges(res["body"], HEADLINE) if res["ok"] else {}
            rows.append({"service": t["service"], "node": t["node"], "up": res["ok"],
                         "running": g.get("vllm:num_requests_running", 0.0),
                         "waiting": g.get("vllm:num_requests_waiting", 0.0),
                         "kv_cache": g.get("vllm:gpu_cache_usage_perc", 0.0)})
        return rows
