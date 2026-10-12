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
    monkeypatch.setattr(main, "refresh_or_start_server", lambda: calls.append(("refresh",)))
    main.control_main(["apply", "r2"])
    assert ("deploy",) in calls                       # auto-deploy-before-forward must survive
    fwd = ("run_head", ["apply", "r2", "--timeout", "1800"])
    assert fwd in calls                               # waits by default (timeout forwarded, no --detach)
    assert (tmp_root / "current").read_text().strip() == "r2"   # repo pointer kept in sync
    assert calls.index(("deploy",)) < calls.index(fwd)


def test_apply_waits_by_default_and_detach_opts_out(tmp_root, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "deploy", lambda: None)
    monkeypatch.setattr(main, "run_head", lambda argv: calls.append(argv))
    monkeypatch.setattr(main, "refresh_or_start_server", lambda: None)
    main.control_main(["apply", "r2", "--timeout", "600"])
    main.control_main(["apply", "r2", "--detach"])
    assert calls[0] == ["apply", "r2", "--timeout", "600"]   # default: wait, no --detach forwarded
    assert "--detach" in calls[1]                            # opt-out is forwarded to the head


# ---- unified server: apply starts it when down (on its host) -----------------
def test_refresh_or_start_server_restarts_when_running(monkeypatch):
    from sparkctl.server import lifecycle
    calls = []
    monkeypatch.setattr(lifecycle, "server_running", lambda: True)
    monkeypatch.setattr(lifecycle, "serve_start", lambda: calls.append("start"))
    lifecycle.refresh_or_start_server()
    assert calls == ["start"]          # already up -> restart to pick up the new deployment


def test_refresh_or_start_server_starts_when_down_on_host(monkeypatch):
    from sparkctl.server import lifecycle
    calls = []
    monkeypatch.setattr(lifecycle, "server_running", lambda: False)
    monkeypatch.setattr(lifecycle, "_is_server_host", lambda: True)
    monkeypatch.setattr(lifecycle, "serve_start", lambda: calls.append("start"))
    lifecycle.refresh_or_start_server()
    assert calls == ["start"]          # down but this IS the server host -> start it (the fix)


def test_refresh_or_start_server_noop_when_down_off_host(monkeypatch):
    from sparkctl.server import lifecycle
    calls = []
    monkeypatch.setattr(lifecycle, "server_running", lambda: False)
    monkeypatch.setattr(lifecycle, "_is_server_host", lambda: False)
    monkeypatch.setattr(lifecycle, "serve_start", lambda: calls.append("start"))
    lifecycle.refresh_or_start_server()
    assert calls == []                 # not the server host (e.g. head when host=local) -> don't start


# ---- shell completion --------------------------------------------------------
def test_completion_render_zsh_and_bash():
    from sparkctl.cli import completion
    verbs = ["apply", "validate", "deploy", "build"]
    zsh = completion.render("zsh", "/repo", verbs)
    assert "compdef _sparkctl sparkctl" in zsh
    assert 'root="/repo"' in zsh and "${root}/recipes/*.yaml" in zsh
    assert "apply|validate|pull|pull-queue) compadd" in zsh   # RECIPE_VERBS injected into the case
    assert "edit) compadd -- $recipes cluster" in zsh
    bash = completion.render("bash", "/repo", verbs)
    assert "complete -F _sparkctl sparkctl" in bash
    assert "apply validate deploy build" in bash             # verbs space-joined into compgen -W


def test_all_verbs_covers_parser_and_control_verbs():
    v = main._all_verbs()
    # subcommands + the pre-argparse control verbs both show up, so completion never drifts
    assert {"apply", "validate", "pull", "completion", "deploy", "build"} <= set(v)


def test_completion_command_defaults_shell_from_env(monkeypatch, capsys):
    monkeypatch.setenv("SHELL", "/usr/bin/zsh")
    main.cmd_completion(SimpleNamespace(shell=None))
    assert "compdef _sparkctl sparkctl" in capsys.readouterr().out


# ---- multi-image build + preflight -------------------------------------------
def test_build_default_image_on_all_nodes(monkeypatch):
    cmds = []
    monkeypatch.setattr(remote, "sh", lambda c, **k: cmds.append(c))
    monkeypatch.setattr(remote, "node_addr", lambda n: f"{n}.addr")
    main.build()
    # build() emits the default image on every node (plus any declared cluster.images variants)
    default_builds = [c for c in cmds if f"docker build -t {config.IMAGE} " in c]
    assert len(default_builds) == len(config.NODES)
    assert all("-f " in c and "/docker/Dockerfile " in c for c in default_builds)


def test_image_targets_includes_declared_variants(monkeypatch):
    monkeypatch.setitem(config.CFG["cluster"], "images",
                        {"csv-vllm:0.25": {"dockerfile": "Dockerfile.v025"},
                         "csv-vllm:muse": {"pull": "vllm/vllm-openai:muse-glimmer-arm64-cu130"}})
    targets = {t: (df, pull) for t, df, pull in main._image_targets()}
    assert targets[config.IMAGE] == ("Dockerfile", None)          # default always first, built
    assert targets["csv-vllm:0.25"] == ("Dockerfile.v025", None)  # built variant
    assert targets["csv-vllm:muse"] == (None, "vllm/vllm-openai:muse-glimmer-arm64-cu130")  # pulled


def test_build_pull_variant_pulls_and_retags(monkeypatch):
    monkeypatch.setitem(config.CFG["cluster"], "images",
                        {"csv-vllm:muse": {"pull": "vllm/vllm-openai:muse-glimmer-arm64-cu130"}})
    cmds = []
    monkeypatch.setattr(remote, "sh", lambda c, **k: cmds.append(c))
    monkeypatch.setattr(remote, "node_addr", lambda n: f"{n}.addr")
    main.build("csv-vllm:muse")
    pulls = [c for c in cmds if "docker pull vllm/vllm-openai:muse-glimmer-arm64-cu130" in c]
    assert len(pulls) == len(config.NODES)
    assert all("docker tag vllm/vllm-openai:muse-glimmer-arm64-cu130 csv-vllm:muse" in c for c in pulls)
    assert not any("docker build" in c for c in cmds)   # pull variant is never built


def test_build_unknown_image_exits(monkeypatch):
    monkeypatch.setattr(remote, "sh", lambda c, **k: None)
    monkeypatch.setattr(remote, "node_addr", lambda n: "a")
    with pytest.raises(SystemExit, match="unknown image"):
        main.build("csv-vllm:does-not-exist")


def test_ensure_images_undeclared_missing_exits(monkeypatch):
    # missing AND not declared under cluster.images -> can't auto-provision, hard stop
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: SimpleNamespace(returncode=1))
    recipe = {"name": "r", "services": [{"name": "a", "engine": "vllm", "model": "m",
              "node": config.HEAD, "image": "csv-vllm:nope",
              "max_model_len": 1, "gpu_memory_utilization": 0.5}]}
    with pytest.raises(SystemExit, match="not declared"):
        resource._ensure_images(recipe)


def _provision_recipe(image):
    return {"name": "r", "services": [{"name": "a", "engine": "vllm", "model": "m",
            "node": config.HEAD, "image": image,
            "max_model_len": 1, "gpu_memory_utilization": 0.5}]}


def test_ensure_images_pulls_declared_missing(monkeypatch):
    # declared as a pull image + missing -> apply pulls & retags it (no separate build step needed)
    monkeypatch.setitem(config.CFG["cluster"], "images",
                        {"csv-vllm:muse": {"pull": "vllm/vllm-openai:muse-glimmer-arm64-cu130"}})
    calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: calls.append(cmd) or
                        SimpleNamespace(returncode=1 if "docker image inspect" in cmd else 0, stdout=""))
    resource._ensure_images(_provision_recipe("csv-vllm:muse"))
    assert any("docker pull vllm/vllm-openai:muse-glimmer-arm64-cu130" in c and "docker tag" in c
               for c in calls)
    assert not any("docker build" in c for c in calls)


def test_ensure_images_builds_declared_missing(monkeypatch):
    # declared with a dockerfile + missing -> apply builds it
    monkeypatch.setitem(config.CFG["cluster"], "images", {"csv-vllm:x": {"dockerfile": "Dockerfile.x"}})
    calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: calls.append(cmd) or
                        SimpleNamespace(returncode=1 if "docker image inspect" in cmd else 0, stdout=""))
    resource._ensure_images(_provision_recipe("csv-vllm:x"))
    assert any("docker build -t csv-vllm:x" in c and "Dockerfile.x" in c for c in calls)


def test_ensure_images_present_skips_provision(monkeypatch):
    # already present -> no build/pull attempted
    monkeypatch.setitem(config.CFG["cluster"], "images", {"csv-vllm:x": {"dockerfile": "Dockerfile.x"}})
    calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: calls.append(cmd) or
                        SimpleNamespace(returncode=0, stdout=""))
    resource._ensure_images(_provision_recipe("csv-vllm:x"))
    assert not any("docker build" in c or "docker pull" in c for c in calls)


def test_ensure_images_default_image_skips_inspect(monkeypatch):
    called = []
    monkeypatch.setattr(remote, "on",
                        lambda node, cmd, **k: called.append(cmd) or SimpleNamespace(returncode=0))
    recipe = {"name": "r", "services": [{"name": "a", "engine": "vllm", "model": "m",
              "node": config.HEAD, "max_model_len": 1, "gpu_memory_utilization": 0.5}]}
    resource._ensure_images(recipe)   # cluster-default image is a build/deploy invariant -> no probe
    assert called == []


class WaitBackend:
    def __init__(self, bases=("http://n1:8000/v1", "http://n2:8000/v1"), running=True):
        self.bases, self.running = list(bases), running

    def endpoints(self, recipe, served_from):
        return {"m": self.bases}

    def status(self):
        if not self.running:
            return {n: {} for n in config.NODES}
        return {n: {f"{config.PFX}-svc-agent": "Up 2 minutes",
                    f"{config.PFX}-ollama": "Up 2 minutes"} for n in config.NODES}


def test_wait_ready_polls_until_every_endpoint_answers(tmp_root, monkeypatch):
    rcs = iter([1, 0, 0])                       # round 1: n1 down, n2 up; round 2: n1 up
    curls = []
    monkeypatch.setattr(resource, "get_backend", lambda: WaitBackend())
    monkeypatch.setattr(resource.remote, "sh",
                        lambda cmd, **kw: curls.append(cmd) or SimpleNamespace(
                            returncode=next(rcs), stdout="", stderr=""))
    monkeypatch.setattr(resource.time, "sleep", lambda s: None)
    resource._wait_ready({"services": []}, timeout=60)
    assert len(curls) == 3
    assert all("curl -sf" in c and "/models" in c for c in curls)


def test_wait_ready_times_out(tmp_root, monkeypatch):
    monkeypatch.setattr(resource, "get_backend", lambda: WaitBackend(bases=["http://n1:8000/v1"]))
    monkeypatch.setattr(resource.remote, "sh",
                        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr=""))
    with pytest.raises(SystemExit, match="TIMEOUT"):
        resource._wait_ready({"services": []}, timeout=-1)   # first failed round exceeds the limit


def test_wait_ready_fails_fast_on_dead_multinode_serve(tmp_root, monkeypatch):
    # the Ray head container is `sleep infinity` — Up while the exec'd vllm serve is long gone
    recipe = {"services": [{"name": "agent", "engine": "vllm", "model": "org/M",
                            "parallel": {"tensor": 2}}]}
    class Up(WaitBackend):
        def status(self):
            return {n: {f"{config.PFX}-svc-agent-head": "Up 16 minutes"} for n in config.NODES}
    monkeypatch.setattr(resource, "get_backend", lambda: Up(bases=["http://n1:8000/v1"]))
    monkeypatch.setattr(resource, "multinode_serve_alive", lambda cname: False)
    monkeypatch.setattr(resource.remote, "sh",
                        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr=""))
    # crash report probes the container over remote.on — return an exit code + OOM flag + logs
    monkeypatch.setattr(resource.remote, "on",
                        lambda node, cmd, **kw: SimpleNamespace(returncode=0, stdout="1|true", stderr=""))
    monkeypatch.setattr(resource.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="not serving"):
        resource._wait_ready(recipe, timeout=60)


def test_wait_ready_fails_fast_when_a_service_dies(tmp_root, monkeypatch):
    # a crashed container never becomes ready — bail with a logs hint, don't sit out the timeout
    recipe = {"services": [{"name": "agent", "engine": "vllm", "model": "org/M",
                            "node": config.HEAD, "parallel": {"tensor": 1}}]}
    monkeypatch.setattr(resource, "get_backend",
                        lambda: WaitBackend(bases=["http://n1:8000/v1"], running=False))
    monkeypatch.setattr(resource.remote, "sh",
                        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr=""))
    onc = []
    monkeypatch.setattr(resource.remote, "on",
                        lambda node, cmd, **kw: onc.append(cmd) or
                        SimpleNamespace(returncode=0, stdout="1|false", stderr="crash log line"))
    monkeypatch.setattr(resource.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="not serving"):
        resource._wait_ready(recipe, timeout=60)
    assert any("docker inspect" in c for c in onc)   # auto crash report pulled the exit/OOM state
    assert any("docker logs" in c for c in onc)      # ...and the log tail


def test_apply_dash_f_copies_into_recipes(tmp_root, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(main, "deploy", lambda: calls.append("deploy"))
    monkeypatch.setattr(main, "run_head", lambda argv: calls.append(("run_head", argv)))
    monkeypatch.setattr(main, "refresh_or_start_server", lambda: None)
    src = tmp_path / "elsewhere.yaml"
    src.write_text(yaml.safe_dump({"name": "elsewhere", "services": []}))
    main.control_main(["apply", "-f", str(src)])
    assert (tmp_root / "recipes" / "elsewhere.yaml").exists()   # manifest became a repo recipe
    assert any(c[0] == "run_head" and c[1][:2] == ["apply", "elsewhere"]
               for c in calls)                                  # forwarded by name, not path


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
vllm:kv_cache_usage_perc{model_name="a"} 0.42
vllm:generation_tokens_total{model_name="a"} 1000.0
other_metric 99
"""
    m = resource.parse_metrics(text, ["vllm:num_requests_running", "vllm:kv_cache_usage_perc",
                                      "vllm:generation_tokens_total"])
    assert m["vllm:num_requests_running"] == 5.0
    assert m["vllm:kv_cache_usage_perc"] == 0.42
    assert m["vllm:generation_tokens_total"] == 1000.0
    assert "other_metric" not in m


def test_parse_metrics_handles_garbage():
    assert resource.parse_metrics(None, ["vllm:x"]) == {}
    assert resource.parse_metrics("vllm:x{bad", ["vllm:x"]) == {}


def test_free_page_cache_drops_on_every_node(tmp_root, monkeypatch):
    cmds = []
    monkeypatch.setattr(resource.remote, "on", lambda node, cmd, **k: cmds.append((node, cmd)))
    resource._free_page_cache()
    assert len(cmds) == len(config.NODES)
    assert all("drop_caches" in c for _, c in cmds)   # page-cache flush issued per node


def _telemetry_remote(meminfo_kb, kills="", engine_log=""):
    """remote.on fake for the wait loop's telemetry: node probe, engine log tail, crash inspect."""
    def on(node, cmd, **kw):
        if "MemAvailable" in cmd:
            return SimpleNamespace(returncode=0, stdout=f"{meminfo_kb}\n{kills}", stderr="")
        if "docker logs --tail 400" in cmd:
            return SimpleNamespace(returncode=0, stdout=engine_log, stderr="")
        return SimpleNamespace(returncode=0, stdout="137|false", stderr="")
    return on


def test_wait_ready_crash_report_names_memguard_kill(tmp_root, monkeypatch, capsys):
    recipe = {"services": [{"name": "agent", "engine": "vllm", "model": "org/M",
                            "node": config.HEAD, "served_name": "m"}]}
    monkeypatch.setattr(resource, "get_backend",
                        lambda: WaitBackend(bases=["http://n1:8000/v1"], running=False))
    monkeypatch.setattr(resource.remote, "sh",
                        lambda cmd, **kw: SimpleNamespace(returncode=1, stdout="", stderr=""))
    kill = "2026-10-12T00:49:00Z memguard: MemAvailable 6000000kB < 6291456kB — killing: csv-svc-agent"
    monkeypatch.setattr(resource.remote, "on", _telemetry_remote(6000000, kill))
    monkeypatch.setattr(resource.time, "sleep", lambda s: None)
    with pytest.raises(SystemExit, match="not serving"):
        resource._wait_ready(recipe, timeout=60)
    out = capsys.readouterr().out
    assert "🛑" in out and "memguard" in out                       # surfaced the moment it's seen
    assert f"memguard on {config.HEAD} killed it" in out           # ...and in the crash report
    assert "⚠️" in out                                             # 5.7 GiB free < critical floor


def test_wait_ready_heartbeat_shows_phase_facts_and_memory(tmp_root, monkeypatch, capsys):
    recipe = {"services": [{"name": "agent", "engine": "vllm", "model": "org/M",
                            "node": config.HEAD, "served_name": "m"}]}
    monkeypatch.setattr(resource, "get_backend", lambda: WaitBackend(bases=["http://n1:8000/v1"]))
    rcs = iter([1, 1, 0])                                          # ready on the 3rd poll
    monkeypatch.setattr(resource.remote, "sh",
                        lambda cmd, **kw: SimpleNamespace(returncode=next(rcs), stdout="", stderr=""))
    log = ("Loading safetensors checkpoint shards:  27% Completed | 3/11 [01:31<04:27, 33.41s/it]\n"
           "INFO Model loading took 63.84 GiB memory and 507.1 seconds\n")
    monkeypatch.setattr(resource.remote, "on", _telemetry_remote(44 * 1048576, engine_log=log))
    clock = iter(range(0, 10_000, 40))                             # every poll is a heartbeat
    monkeypatch.setattr(resource.time, "time", lambda: next(clock))
    monkeypatch.setattr(resource.time, "sleep", lambda s: None)
    resource._wait_ready(recipe, timeout=600)
    out = capsys.readouterr().out
    assert "agent: model loaded — 63.8 GiB/rank in 507s" in out    # one-time fact
    assert out.count("model loaded") == 1                          # ...printed once, not per beat
    assert "agent: profiling memory" in out and f"{config.HEAD} 44 GiB free" in out
