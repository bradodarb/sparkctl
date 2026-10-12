"""Backend registry + DockerBackend interface behavior + the k8s roadmap stub."""
import pytest

from sparkctl import config
from sparkctl.backends import BACKENDS, get_backend
from sparkctl.backends import docker as docker_backend
from sparkctl.backends.docker import DockerBackend, launch_waves
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


def test_docker_scrapes_sglang_metrics_too():
    # sglang is a SERVED engine: it exposes /metrics and must be scraped like vllm (ollama isn't)
    recipe = {"services": [
        {"name": "v", "engine": "vllm", "model": "org/M", "served_name": "m",
         "node": config.HEAD, "parallel": {"tensor": 1}, "port": 8000},
        {"name": "s", "engine": "sglang", "model": "org/S", "served_name": "s",
         "node": config.WORKER, "port": 8001},
        {"name": "e", "engine": "ollama", "model": "nomic-embed-text", "node": config.WORKER, "port": 11434},
    ]}
    ts = DockerBackend().metrics_targets(recipe, "local")
    svcs = {t["service"] for t in ts}
    assert svcs == {"v", "s"}                            # both served engines, not ollama
    assert any(t["url"].endswith(":8001/metrics") for t in ts)


def test_compact_nodes_form_expands_into_pool():
    # the whole point: one `nodes:` service -> expand_services -> a 2-address LB pool in endpoints()
    from sparkctl.recipes import expand_services
    compact = {"services": [{"name": "agent", "engine": "vllm", "model": "org/M",
                             "served_name": "coder", "nodes": [config.HEAD, config.WORKER],
                             "port": 8000}]}
    eps = DockerBackend().endpoints(expand_services(compact), "local")
    assert len(eps["coder"]) == 2 and len(set(eps["coder"])) == 2


def _svc(name, engine="vllm", node=None, **kw):
    return {"name": name, "engine": engine, "model": "org/M", "node": node or config.HEAD, **kw}


def test_launch_waves_serialise_served_services_per_node():
    # tres-amigos shape: one on HEAD, two served + ollama on WORKER -> the 2nd WORKER model waits
    flash, a = _svc("flash"), _svc("a", node=config.WORKER)
    b, emb = _svc("b", node=config.WORKER), _svc("emb", engine="ollama", node=config.WORKER)
    assert launch_waves([flash, a, b, emb]) == [[flash, a, emb], [b]]


def test_launch_waves_one_per_node_is_a_single_wave():
    a, b = _svc("a"), _svc("b", engine="sglang", node=config.WORKER)
    assert launch_waves([a, b]) == [[a, b]]


def test_launch_waves_multinode_blocks_both_nodes():
    tp2 = _svc("big", parallel={"tensor": 2})
    solo = _svc("solo", node=config.WORKER)
    assert launch_waves([tp2, solo]) == [[tp2], [solo]]


def test_up_waits_only_on_node_mates(monkeypatch):
    events = []
    monkeypatch.setitem(docker_backend.ENGINES, "vllm",
                        (lambda s: events.append(("up", s["name"])), lambda: None))
    monkeypatch.setattr(docker_backend, "_wait_loaded",
                        lambda svcs, timeout: events.append(("wait", [s["name"] for s in svcs])))
    DockerBackend().up({"services": [_svc("flash"), _svc("a", node=config.WORKER),
                                     _svc("b", node=config.WORKER)]})
    # b waits for a (same node) but NOT for flash, which may still be compiling on the other node
    assert events == [("up", "flash"), ("up", "a"), ("wait", ["a"]), ("up", "b")]


def test_k8s_stub_raises_clear_message():
    with pytest.raises(NotImplementedError, match="backend: docker"):
        K8sBackend().up(RECIPE)
    with pytest.raises(NotImplementedError):
        K8sBackend().endpoints(RECIPE, "local")
    assert set(BACKENDS) == {"docker", "k8s"}
