"""CLI grammar — removed-verb hints, control-machine dispatch (apply deploys then forwards),
and the resource verbs against a hermetic tmp repo root. No live cluster."""
import json
from types import SimpleNamespace

import pytest
import yaml

from sparkctl import config, remote
from sparkctl.cli import main, resource


@pytest.fixture
def tmp_root(tmp_path, monkeypatch):
    (tmp_path / "recipes").mkdir()
    (tmp_path / "recipes" / "r1.yaml").write_text(yaml.safe_dump({
        "name": "r1", "services": [
            {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "m",
             "node": config.HEAD, "max_model_len": 1024, "gpu_memory_utilization": 0.5},
            {"name": "emb", "engine": "ollama", "model": "nomic", "node": config.WORKER},
        ]}))
    (tmp_path / "recipes" / "r2.yaml").write_text(yaml.safe_dump(
        {"name": "r2", "services": []}))
    (tmp_path / "current").write_text("r1\n")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    return tmp_path


# ---- removed verbs -----------------------------------------------------------
@pytest.mark.parametrize("verb,hint", [
    ("up", "apply"), ("down", "delete services --all"), ("switch", "apply"), ("list", "get recipes"),
])
def test_removed_verbs_hint_and_exit_2(verb, hint, capsys):
    with pytest.raises(SystemExit) as e:
        main.check_removed([verb])
    assert e.value.code == 2
    assert hint in capsys.readouterr().err


# ---- control-machine dispatch -------------------------------------------------
def test_apply_deploys_then_forwards_to_head(tmp_root, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "deploy", lambda: calls.append(("deploy",)))
    monkeypatch.setattr(main, "run_head", lambda argv: calls.append(("run_head", argv)))
    monkeypatch.setattr(main, "refresh_server_if_running", lambda: calls.append(("refresh",)))
    main.control_main(["apply", "r2"])
    assert ("deploy",) in calls                       # auto-deploy-before-forward must survive
    assert ("run_head", ["apply", "r2"]) in calls
    assert (tmp_root / "current").read_text().strip() == "r2"   # repo pointer kept in sync
    assert calls.index(("deploy",)) < calls.index(("run_head", ["apply", "r2"]))


def test_apply_dash_f_copies_into_recipes(tmp_root, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "deploy", lambda: calls.append("deploy"))
    monkeypatch.setattr(main, "run_head", lambda argv: calls.append(("run_head", argv)))
    monkeypatch.setattr(main, "refresh_server_if_running", lambda: None)
    src = tmp_path / "elsewhere.yaml"
    src.write_text(yaml.safe_dump({"name": "elsewhere", "services": []}))
    main.control_main(["apply", "-f", str(src)])
    assert (tmp_root / "recipes" / "elsewhere.yaml").exists()   # manifest became a repo recipe
    assert ("run_head", ["apply", "elsewhere"]) in calls        # forwarded by name, not path


def test_readonly_verbs_do_not_deploy(tmp_root, monkeypatch, capsys):
    monkeypatch.setattr(main, "deploy", lambda: pytest.fail("get must not deploy"))
    monkeypatch.setattr(main, "run_head", lambda argv: pytest.fail("get must not forward"))
    main.control_main(["get", "recipes"])
    out = capsys.readouterr().out
    assert "r1" in out and "r2" in out


# ---- resource verbs ------------------------------------------------------------
def test_get_recipes_marks_current(tmp_root, capsys):
    resource.cmd_get(SimpleNamespace(resource="recipes", output="table"))
    lines = capsys.readouterr().out.splitlines()
    assert any("*" in ln and "r1" in ln for ln in lines)
    assert not any("*" in ln and "r2" in ln for ln in lines)


def test_get_services_json(tmp_root, monkeypatch, capsys):
    fake = SimpleNamespace(stdout=f"{config.PFX}-svc-agent\tUp 3 hours\n", returncode=0)
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: fake)
    resource.cmd_get(SimpleNamespace(resource="services", output="json"))
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)}
    assert rows["agent"]["status"] == "Up 3 hours"
    assert rows["agent"]["node"] == config.HEAD
    assert rows["emb"]["status"] == "not running"     # no ollama container in fake docker ps


def test_delete_all_tears_everything_down(tmp_root, monkeypatch):
    cmds = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: cmds.append((node, cmd)) or
                        SimpleNamespace(stdout="", returncode=0))
    resource.cmd_delete(SimpleNamespace(kind="services", all=True, name=None))
    joined = " ".join(c for _, c in cmds)
    assert f"name={config.PFX}-svc-" in joined                  # vllm containers removed
    assert f"{config.PFX}-ollama" in joined                     # ollama removed
    assert "rm -f" in joined and "active.json" in joined        # manifests cleared


def test_delete_services_requires_all_flag(tmp_root):
    with pytest.raises(SystemExit):
        resource.cmd_delete(SimpleNamespace(kind="services", all=False, name=None))


# ---- top: metric parsing --------------------------------------------------------
def test_parse_metrics_sums_label_sets():
    text = """# HELP vllm:num_requests_running ...
vllm:num_requests_running{model_name="a"} 2.0
vllm:num_requests_running{model_name="b"} 3.0
vllm:gpu_cache_usage_perc{model_name="a"} 0.42
vllm:generation_tokens_total{model_name="a"} 1000.0
other_metric 99
"""
    m = resource.parse_metrics(text, ["vllm:num_requests_running", "vllm:gpu_cache_usage_perc",
                                      "vllm:generation_tokens_total"])
    assert m["vllm:num_requests_running"] == 5.0
    assert m["vllm:gpu_cache_usage_perc"] == 0.42
    assert m["vllm:generation_tokens_total"] == 1000.0
    assert "other_metric" not in m


def test_parse_metrics_handles_garbage():
    assert resource.parse_metrics(None, ["vllm:x"]) == {}
    assert resource.parse_metrics("vllm:x{bad", ["vllm:x"]) == {}
