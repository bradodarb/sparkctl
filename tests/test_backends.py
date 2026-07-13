"""Backend registry + DockerBackend interface behavior + the k8s roadmap stub."""
import pytest

from sparkctl import config
from sparkctl.backends import BACKENDS, get_backend
from sparkctl.backends.docker import DockerBackend
from sparkctl.backends.k8s import K8sBackend

RECIPE = {"services": [
    {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "coder",
     "node": config.HEAD, "parallel": {"tensor": 1}, "port": 8000},
    {"name": "agent-ref", "engine": "vllm", "model": "org/M", "served_name": "coder",
     "node": config.WORKER, "parallel": {"tensor": 1}, "port": 8000},
    {"name": "embeddings", "engine": "ollama", "model": "nomic-embed-text",
     "node": config.WORKER, "port": 11434},
]}


def test_default_backend_is_docker(monkeypatch):
    monkeypatch.delitem(config.CFG, "backend", raising=False)
    assert isinstance(get_backend(), DockerBackend)


def test_backend_selected_from_config(monkeypatch):
    monkeypatch.setitem(config.CFG, "backend", "k8s")
    assert isinstance(get_backend(), K8sBackend)


def test_unknown_backend_exits(monkeypatch):
    monkeypatch.setitem(config.CFG, "backend", "nomad")
    with pytest.raises(SystemExit):
        get_backend()


def test_docker_endpoints_pool_replicas():
    eps = DockerBackend().endpoints(RECIPE, "local")
    assert len(eps["coder"]) == 2                       # replicated served_name -> LB pool
    assert len(set(eps["coder"])) == 2                  # ...on two distinct addresses
    assert eps["nomic-embed-text"][0].endswith(":11434/v1")


def test_docker_metrics_targets_vllm_only():
    ts = DockerBackend().metrics_targets(RECIPE, "local")
    assert len(ts) == 2 and all(t["url"].endswith(":8000/metrics") for t in ts)


def test_k8s_stub_raises_clear_message():
    with pytest.raises(NotImplementedError, match="backend: docker"):
        K8sBackend().up(RECIPE)
    with pytest.raises(NotImplementedError):
        K8sBackend().endpoints(RECIPE, "local")
    assert set(BACKENDS) == {"docker", "k8s"}
