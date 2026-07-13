"""Unified-server metrics — scrape-target generation, label injection, aggregation, and the
single-target Prometheus config for the optional Grafana stack. Pure logic, no live cluster."""
from sparkctl import config
from sparkctl.server.metrics import (NodeSampler, Scraper, grafana_prometheus_config,
                                     inject_labels, node_stats_cmd, parse_gauges,
                                     parse_node_stats, scrape_targets)

REPLICATED = {"services": [
    {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "coder",
     "node": config.HEAD, "parallel": {"tensor": 1}, "port": 8000},
    {"name": "agent-ref", "engine": "vllm", "model": "org/M", "served_name": "coder",
     "node": config.WORKER, "parallel": {"tensor": 1}, "port": 8000},
    {"name": "embeddings", "engine": "ollama", "model": "nomic-embed-text",
     "node": config.WORKER, "port": 11434},
]}


def test_scrape_targets_vllm_only():
    ts = scrape_targets(REPLICATED, "local")
    assert len(ts) == 2                                   # ollama is NOT a scrape target
    assert all(t["url"].endswith(":8000/metrics") for t in ts)
    assert {t["node"] for t in ts} == {config.HEAD, config.WORKER}


def test_scrape_targets_host_mode_changes_addresses():
    local = scrape_targets(REPLICATED, "local")
    onnode = scrape_targets(REPLICATED, config.HEAD)
    fabric = config.NODES[config.HEAD]["fabric_ip"]
    assert any(fabric in t["url"] for t in onnode)        # node-hosted -> fabric IP
    assert not any(fabric in t["url"] for t in local)     # dev-machine -> lan_ip/host


def test_inject_labels():
    body = ('# HELP vllm:x help text\n'
            'vllm:x{model_name="m"} 1.0\n'
            'vllm:y 2.0\n')
    out = inject_labels(body, {"node": "coach", "service": "agent"})
    assert 'vllm:x{node="coach",service="agent",model_name="m"} 1.0' in out
    assert 'vllm:y{node="coach",service="agent"} 2.0' in out
    assert "# HELP" not in out                            # comments dropped (would duplicate per node)


def test_parse_gauges_sums_label_sets():
    body = 'vllm:num_requests_running{a="1"} 2.0\nvllm:num_requests_running{a="2"} 3.0\n'
    assert parse_gauges(body, ["vllm:num_requests_running"])["vllm:num_requests_running"] == 5.0


def test_scraper_exposition_and_summaries():
    s = Scraper(scrape_targets(REPLICATED, "local"))
    s.results[(config.HEAD, "agent")] = {"ok": True, "ts": 1.0, "body":
        'vllm:num_requests_running{m="x"} 4.0\nvllm:gpu_cache_usage_perc{m="x"} 0.5\n'}
    s.results[(config.WORKER, "agent-ref")] = {"ok": False, "ts": 0.0, "body": ""}
    exp = s.exposition()
    assert f'sparkctl_target_up{{node="{config.HEAD}",service="agent"}} 1' in exp
    assert f'sparkctl_target_up{{node="{config.WORKER}",service="agent-ref"}} 0' in exp
    assert f'vllm:num_requests_running{{node="{config.HEAD}",service="agent",m="x"}} 4.0' in exp
    rows = {r["service"]: r for r in s.summaries()}
    assert rows["agent"]["up"] and rows["agent"]["running"] == 4.0
    assert not rows["agent-ref"]["up"]


def test_parse_node_stats_derives_used_and_drops_na():
    out = parse_node_stats("mem_total_kb=125000000\nmem_avail_kb=60000000\n"
                           "gpu_util_pct=[N/A]\ndisk_total_kb=1000\ndisk_used_kb=400\n")
    assert out["mem_used_kb"] == 65000000            # total - available
    assert "gpu_util_pct" not in out                 # N/A dropped, never reported wrong
    assert out["disk_used_kb"] == 400


def test_node_stats_cmd_targets_cache_path():
    cmd = node_stats_cmd("/home/x/models")
    assert "df -k /home/x/models" in cmd and "/proc/meminfo" in cmd and "nvidia-smi" in cmd


def test_node_sampler_exposition():
    s = NodeSampler(["coach"], "/m")
    s.results["coach"] = {"mem_total_kb": 100, "mem_used_kb": 50, "gpu_util_pct": 42.0}
    exp = s.exposition()
    assert 'sparkctl_node_memory_total_bytes{node="coach"} 102400' in exp
    assert 'sparkctl_node_gpu_utilization_percent{node="coach"} 42' in exp
    rows = s.summaries()
    assert rows[0]["gpu_util"] == 42.0


def test_grafana_prometheus_config_single_target():
    cfg = grafana_prometheus_config("localhost:8080", 10)
    assert cfg["scrape_configs"][0]["static_configs"][0]["targets"] == ["localhost:8080"]
    assert cfg["scrape_configs"][0]["metrics_path"] == "/metrics"
    assert cfg["global"]["scrape_interval"] == "10s"
