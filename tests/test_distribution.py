"""Model distribution — download naming/env (pure) and the ensure_models nodes->NAS->download
flow with every remote command mocked. No live cluster, no network."""
import pytest

from sparkctl import config, distribution, remote

RECIPE = {"services": [
    {"name": "agent", "engine": "vllm", "model": "org/M", "served_name": "m",
     "max_model_len": 1024, "gpu_memory_utilization": 0.5},
]}


class FakeRemote:
    """Record every on() call; answer presence probes from configurable sets."""

    def __init__(self, present_nodes=(), on_nas=False, ollama_nodes=()):
        self.present_nodes = set(present_nodes)
        self.on_nas = on_nas
        self.ollama_nodes = set(ollama_nodes)
        self.calls = []

    def __call__(self, node, cmd, **kw):
        self.calls.append((node, cmd))
        rc, stdout = 0, ""
        nas_target = "ssh -o BatchMode" in cmd or "/mnt/nas" in cmd or "/export/" in cmd
        if "models--*" in cmd:                          # HF inventory listing (node cache or NAS)
            if self.on_nas if nas_target else node in self.present_nodes:
                stdout = "models--org--M\n"
        elif "manifests" in cmd:                        # ollama store listing
            if node in self.ollama_nodes:
                stdout = "registry.ollama.ai/library/nomic-embed-text/latest\n"
        elif "snapshots" in cmd and "rsync" not in cmd:  # single-model presence probe
            if nas_target:
                rc = 0 if self.on_nas else 1
            else:
                rc = 0 if node in self.present_nodes else 1
        import types
        return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr="")

    def cmds(self, node=None):
        return [c for n, c in self.calls if node is None or n == node]


@pytest.fixture
def fake(monkeypatch):
    def make(present_nodes=(), on_nas=False, nas=None, ollama_nodes=()):
        fr = FakeRemote(present_nodes, on_nas, ollama_nodes)
        monkeypatch.setattr(remote, "on", fr)
        monkeypatch.setattr(distribution, "verify_model",
                            lambda node, model, delete_bad=False, cache=None: True)
        if nas is not None:
            monkeypatch.setitem(config.CFG, "nas", nas)
        else:
            monkeypatch.delitem(config.CFG, "nas", raising=False)
        return fr
    return make


def test_ensure_noop_when_present_everywhere(fake):
    fr = fake(present_nodes=set(config.NODES))
    distribution.ensure_models(RECIPE)
    assert not [c for c in fr.cmds() if "rsync" in c or "docker run" in c]


def test_ensure_no_nas_downloads_on_head_and_replicates(fake, monkeypatch):
    fr = fake(present_nodes=set())
    pulled = []
    monkeypatch.setattr(distribution, "_pull_hf",
                        lambda svc, cache=None: pulled.append((svc["model"], cache)) or True)
    distribution.ensure_models(RECIPE)
    assert pulled == [("org/M", None)]                       # downloaded to the head cache
    worker_pulls = [c for c in fr.cmds(config.HEAD) if "rsync" in c and "fabric" not in c]
    assert any(config.NODES[config.WORKER]["fabric_ip"] in c for c in worker_pulls)


def test_ensure_replicates_from_nas_path_mode(fake):
    fr = fake(present_nodes={config.HEAD}, on_nas=True,
              nas={"mode": "path", "path": "/mnt/nas/models"})
    distribution.ensure_models(RECIPE)
    # worker was missing -> rsync FROM the NAS mount, pushed from the head over the fabric
    pushes = [c for c in fr.cmds(config.HEAD) if "rsync" in c and "/mnt/nas/models" in c]
    assert pushes and config.NODES[config.WORKER]["fabric_ip"] in pushes[0]
    assert not any("docker run" in c for c in fr.cmds())     # no download happened


def test_ensure_replicates_from_nas_ssh_mode(fake):
    fr = fake(present_nodes={config.HEAD}, on_nas=True,
              nas={"mode": "ssh", "host": "nas.lan", "user": "u", "remote_path": "/export/models"})
    distribution.ensure_models(RECIPE)
    # ssh mode: the DESTINATION node pulls from the NAS endpoint
    pulls = [c for c in fr.cmds(config.WORKER) if "rsync" in c]
    assert pulls and "u@nas.lan:/export/models" in pulls[0]


def test_ensure_downloads_to_nas_when_mounted(fake, monkeypatch):
    # a mounted NAS is always the download target — no opt-in attribute
    fr = fake(present_nodes=set(), on_nas=False,
              nas={"mode": "path", "path": "/mnt/nas/models"})
    pulled = []
    monkeypatch.setattr(distribution, "_pull_hf",
                        lambda svc, cache=None: pulled.append((svc["model"], cache)) or True)
    distribution.ensure_models(RECIPE)
    assert pulled == [("org/M", "/mnt/nas/models")]          # download landed on the NAS
    assert any("rsync" in c and "/mnt/nas/models" in c for c in fr.cmds(config.HEAD))


def test_ensure_archives_fresh_download_to_nas(fake, monkeypatch):
    fr = fake(present_nodes=set(), on_nas=False,
              nas={"mode": "ssh", "host": "nas.lan", "remote_path": "/export/models"})
    monkeypatch.setattr(distribution, "_pull_hf", lambda svc, cache=None: True)
    distribution.ensure_models(RECIPE)
    archives = [c for c in fr.cmds(config.HEAD) if "rsync" in c and "nas.lan:/export/models" in c]
    assert archives                                           # head copy pushed back to the NAS


def test_inventory_shape(fake):
    fake(present_nodes={config.HEAD}, on_nas=True, nas={"mode": "path", "path": "/mnt/nas/models"})
    m = distribution.inventory(RECIPE)
    assert m["org/M"]["source"] == "hf"
    assert m["org/M"]["served"] == ["m"]                      # the gateway alias
    assert m["org/M"]["nodes"][config.HEAD] is True
    assert m["org/M"]["nodes"][config.WORKER] is False
    assert m["org/M"]["nas"] is True
    assert m["org/M"]["services"] == ["agent"]


def test_inventory_covers_every_store(fake):
    # installed-but-unreferenced HF models AND ollama models show up, not just recipe services
    fake(present_nodes=set(config.NODES), ollama_nodes={config.WORKER})
    recipe = {"services": [
        {"name": "embeddings", "engine": "ollama", "model": "nomic-embed-text",
         "node": config.WORKER, "port": 11434}]}
    m = distribution.inventory(recipe)
    assert m["org/M"]["services"] == []                       # on disk, no service uses it
    assert m["org/M"]["nodes"] == {config.HEAD: True, config.WORKER: True}
    emb = m["nomic-embed-text"]
    assert emb["source"] == "ollama"
    assert emb["served"] == ["nomic-embed-text"]              # ollama alias == model ref
    assert emb["services"] == ["embeddings"]
    assert emb["nodes"][config.WORKER] is True
    assert emb["nodes"][config.HEAD] is False
    assert emb["nas"] is None                                 # no NAS flow for ollama pulls
    # recipe-referenced models sort ahead of the unreferenced library
    assert list(m) == ["nomic-embed-text", "org/M"]


def test_inventory_lists_nas_only_models(fake):
    fake(present_nodes=set(), on_nas=True, nas={"mode": "path", "path": "/mnt/nas/models"})
    m = distribution.inventory({"services": []})
    assert m["org/M"]["nodes"] == {config.HEAD: False, config.WORKER: False}
    assert m["org/M"]["nas"] is True


def test_hf_precision_tags():
    ct_nvfp4 = {"quantization_config": {"quant_method": "compressed-tensors",
                                        "format": "nvfp4-pack-quantized"}}
    assert distribution._hf_precision(ct_nvfp4) == "NVFP4"
    assert distribution._hf_precision({"quantization_config": {"quant_method": "fp8"}}) == "FP8"
    assert distribution._hf_precision({"torch_dtype": "bfloat16"}) == "BF16"   # unquantized
    assert distribution._hf_precision({}) == "-"


def test_hub_dir_model_roundtrip():
    assert distribution._hub_dir_to_model("models--RedHatAI--Qwen3-235B-A22B-NVFP4") == \
        "RedHatAI/Qwen3-235B-A22B-NVFP4"


def test_ollama_manifest_ref_roundtrip():
    for ref, path in [("nomic-embed-text",
                       "models/manifests/registry.ollama.ai/library/nomic-embed-text/latest"),
                      ("gemma3:4b", "models/manifests/registry.ollama.ai/library/gemma3/4b"),
                      # ollama can serve HF repos directly — the registry path carries through
                      ("hf.co/org/repo:Q4_K_M", "models/manifests/hf.co/org/repo/Q4_K_M")]:
        assert distribution._ollama_manifest(ref) == path
        assert distribution._ollama_ref(path[len("models/manifests/"):]) == ref


def test_corrupt_replica_triggers_checksum_repair(fake, monkeypatch):
    fr = fake(present_nodes={config.HEAD}, on_nas=True,
              nas={"mode": "path", "path": "/mnt/nas/models"})
    verdicts = iter([False, True])                            # first verify fails -> repair pass
    monkeypatch.setattr(distribution, "verify_model",
                        lambda node, model, delete_bad=False, cache=None: next(verdicts))
    distribution.ensure_models(RECIPE)
    rsyncs = [c for c in fr.cmds(config.HEAD) if "rsync" in c and "/mnt/nas/models" in c]
    assert len(rsyncs) == 2 and "-ac" in rsyncs[1]            # second pass uses checksum mode


def test_dl_cname_flattens_slashes():
    assert distribution.dl_cname("RedHatAI/Qwen3-235B-A22B-NVFP4") == \
        f"{config.PFX}-dl-RedHatAI_Qwen3-235B-A22B-NVFP4"


def test_dl_env_robustness(monkeypatch):
    monkeypatch.setattr(config, "DL", {"request_timeout": 45, "etag_timeout": 20, "use_xet": False})
    e = distribution._dl_env()
    assert "HF_HUB_DOWNLOAD_TIMEOUT=45" in e
    assert "HF_HUB_ETAG_TIMEOUT=20" in e
    assert "HF_HUB_DISABLE_XET=1" in e                       # xet off by default (known stall bug)
    monkeypatch.setattr(config, "DL", {"use_xet": True})
    assert "HF_HUB_DISABLE_XET" not in distribution._dl_env()  # opt back into xet


# ---- progress display: human sizes, repo total, and the watchdog progress line ----
def test_human_and_dur_formatting():
    assert distribution._human(0) == "0B"
    assert distribution._human(1536) == "1.5KB"
    assert distribution._human(2 * 1024**3) == "2.0GB"
    assert distribution._fmt_dur(0) == "0s"
    assert distribution._fmt_dur(65) == "1m05s"
    assert distribution._fmt_dur(3661) == "1h01m"


def test_hf_repo_size_sums_lfs_then_size(monkeypatch):
    import json as _json
    import types as _t
    listing = _json.dumps([
        {"path": "model.safetensors", "size": 42, "lfs": {"size": 1000}},  # lfs wins over size
        {"path": "config.json", "size": 500},                              # plain size
        {"path": "extra.bin", "lfs": {"size": 2000}},
    ])
    monkeypatch.setattr(distribution.remote, "on",
                        lambda node, cmd, **kw: _t.SimpleNamespace(stdout=listing, returncode=0, stderr=""))
    assert distribution.hf_repo_size("org/M") == 1000 + 500 + 2000


def test_hf_repo_size_none_on_garbage(monkeypatch):
    import types as _t
    monkeypatch.setattr(distribution.remote, "on",
                        lambda node, cmd, **kw: _t.SimpleNamespace(stdout="404 not found", returncode=0, stderr=""))
    assert distribution.hf_repo_size("org/M") is None


def test_watchdog_prints_percent_progress_then_exits(monkeypatch, capsys):
    import types as _t
    states = iter(["running", "running", "exited"])
    sizes = iter(["1073741824", "2147483648"])              # 1GB then 2GB downloaded

    def fake_on(node, cmd, **kw):
        if "State.Status" in cmd:
            return _t.SimpleNamespace(stdout=next(states) + "\n", returncode=0, stderr="")
        if "du -sb" in cmd:
            return _t.SimpleNamespace(stdout=next(sizes, "2147483648") + "\n", returncode=0, stderr="")
        return _t.SimpleNamespace(stdout="", returncode=0, stderr="")

    monkeypatch.setattr(distribution.remote, "on", fake_on)
    monkeypatch.setattr(distribution.time, "sleep", lambda s: None)
    rc = distribution._wait_with_watchdog("dl", "org/M", total=4 * 1024**3)   # 4GB total
    assert rc == "exited"
    out = capsys.readouterr().out
    assert "1.0GB/4.0GB (25%)" in out
    assert "2.0GB/4.0GB (50%)" in out


def test_watchdog_size_only_when_total_unknown(monkeypatch, capsys):
    import types as _t
    states = iter(["running", "exited"])

    def fake_on(node, cmd, **kw):
        if "State.Status" in cmd:
            return _t.SimpleNamespace(stdout=next(states) + "\n", returncode=0, stderr="")
        if "du -sb" in cmd:
            return _t.SimpleNamespace(stdout="524288000\n", returncode=0, stderr="")   # 500MB
        return _t.SimpleNamespace(stdout="", returncode=0, stderr="")

    monkeypatch.setattr(distribution.remote, "on", fake_on)
    monkeypatch.setattr(distribution.time, "sleep", lambda s: None)
    distribution._wait_with_watchdog("dl", "org/M", total=None)
    out = capsys.readouterr().out
    assert "500.0MB downloaded" in out and "%" not in out


def test_replicate_model_uses_progress2(monkeypatch):
    calls = []
    monkeypatch.setattr(distribution.remote, "on",
                        lambda node, cmd, **kw: calls.append(cmd) or __import__("types").SimpleNamespace(
                            stdout="", returncode=0, stderr=""))
    monkeypatch.delitem(config.CFG, "nas", raising=False)
    distribution.replicate_model("org/M", config.WORKER, source="head")
    assert any("--info=progress2" in c for c in calls)       # node replication shows progress


def test_dl_env_xet_high_performance(monkeypatch):
    monkeypatch.setattr(config, "DL", {"use_xet": True})
    e = distribution._dl_env()
    assert "HF_XET_HIGH_PERFORMANCE=1" in e          # fast path when xet is opted into
    assert "HF_HUB_DISABLE_XET" not in e


def test_download_start_logs_auth_and_opt_in_workers(monkeypatch):
    import types as _t
    monkeypatch.setattr(config, "DL", {"use_xet": False, "max_workers": 8})
    sent = {}
    monkeypatch.setattr(distribution.remote, "on",
                        lambda node, cmd, **kw: sent.update(cmd=cmd) or
                        _t.SimpleNamespace(stdout="", returncode=0, stderr=""))
    name = distribution.hf_download_start("org/M")
    assert name == distribution.dl_cname("org/M")
    assert "HF auth:" in sent["cmd"]                 # auth visibility echo
    assert "${HF_TOKEN:+-e HF_TOKEN}" in sent["cmd"]  # token passed by name only when present
    assert "--max-workers 8" in sent["cmd"]


def test_download_start_no_workers_flag_by_default(monkeypatch):
    import types as _t
    monkeypatch.setattr(config, "DL", {"use_xet": False})
    sent = {}
    monkeypatch.setattr(distribution.remote, "on",
                        lambda node, cmd, **kw: sent.update(cmd=cmd) or
                        _t.SimpleNamespace(stdout="", returncode=0, stderr=""))
    distribution.hf_download_start("org/M")
    assert "--max-workers" not in sent["cmd"]        # opt-in only


def test_prune_model_cache_is_safe_and_targeted(monkeypatch):
    import base64 as _b64, types as _t
    cmds = []
    monkeypatch.setattr(distribution.remote, "on",
                        lambda node, cmd, **k: cmds.append((node, cmd)) or
                        _t.SimpleNamespace(returncode=0, stdout="", stderr=""))
    distribution.prune_model_cache("org/M")
    assert len(cmds) == len(config.NODES)                       # runs on every node
    script = _b64.b64decode(cmds[0][1].split("echo ", 1)[1].split(" |", 1)[0]).decode()
    assert "models--org--M" in script                          # targets the right hub dir
    assert '*.incomplete' in script                            # clears stale partials
    assert '[ -s "$ref" ]' in script                           # refuses to delete blobs w/o live refs
    assert 'blobs/' in script and 'snapshots' in script


def test_ensure_auto_prunes_recipe_models(fake, monkeypatch):
    fr = fake(present_nodes=set(config.NODES))                 # present everywhere -> still prunes
    pruned = []
    monkeypatch.setattr(distribution, "prune_model_cache", lambda m, **k: pruned.append(m))
    distribution.ensure_models(RECIPE)
    assert pruned == ["org/M"]                                 # auto-prune ran for the hf model
