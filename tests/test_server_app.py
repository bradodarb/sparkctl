"""Unified-server app — /healthz, /dash, /metrics routes and the /v1 proxy error path, exercised
via Starlette's TestClient (no network, no litellm: SPARKCTL_LITELLM_URL short-circuits the child).
Skipped automatically when the [server] extra isn't installed."""

import pytest
import yaml

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from sparkctl import config  # noqa: E402
from sparkctl.server.app import create_app  # noqa: E402


@pytest.fixture
def app_env(tmp_path, monkeypatch):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "r1.yaml").write_text(yaml.safe_dump({
        "name": "r1", "services": [
            {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "m",
             "node": config.HEAD, "max_model_len": 1024, "gpu_memory_utilization": 0.5},
        ]}))
    (tmp_path / "current").write_text("r1\n")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    # no managed litellm child, no background scraping — hermetic app
    monkeypatch.setenv("SPARKCTL_LITELLM_URL", "http://127.0.0.1:9")   # nothing listens here
    monkeypatch.setattr(config, "SERVER", {"host": "local", "port": 8080,
                                           "metrics": {"enabled": False}})
    return tmp_path


def test_healthz(app_env):
    with TestClient(create_app()) as c:
        r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["recipe"] == "r1"


def test_dash_renders(app_env):
    with TestClient(create_app()) as c:
        r = c.get("/dash")
    assert r.status_code == 200
    assert "r1" in r.text and "<code>m</code>" in r.text     # recipe + routed model shown


def test_root_redirects_to_dash(app_env):
    with TestClient(create_app()) as c:
        r = c.get("/", follow_redirects=False)
    assert r.status_code in (302, 307) and r.headers["location"] == "/dash"


def test_metrics_endpoint_serves_exposition(app_env):
    with TestClient(create_app()) as c:
        app_scraper = c.app.state.scraper
        app_scraper.targets = [{"node": "coach", "service": "agent", "url": "http://x/metrics"}]
        app_scraper.results[("coach", "agent")] = {"ok": True, "ts": 1.0,
                                                   "body": "vllm:num_requests_running 2.0\n"}
        r = c.get("/metrics")
    assert r.status_code == 200
    assert 'sparkctl_target_up{node="coach",service="agent"} 1' in r.text
    assert 'vllm:num_requests_running{node="coach",service="agent"} 2.0' in r.text


def test_v1_proxy_returns_502_when_upstream_down(app_env):
    with TestClient(create_app()) as c:
        r = c.get("/v1/models")
    assert r.status_code == 502
    assert "upstream unavailable" in r.text
