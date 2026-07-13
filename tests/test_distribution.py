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

    def __init__(self, present_nodes=(), on_nas=False):
        self.present_nodes = set(present_nodes)
        self.on_nas = on_nas
        self.calls = []

    def __call__(self, node, cmd, **kw):
        self.calls.append((node, cmd))
        rc = 0
        if "snapshots" in cmd and "rsync" not in cmd:   # presence probe
            if "ssh -o BatchMode" in cmd or "/mnt/nas" in cmd or "/export/" in cmd:
                rc = 0 if self.on_nas else 1
            else:
                rc = 0 if node in self.present_nodes else 1
        import types
        return types.SimpleNamespace(returncode=rc, stdout="", stderr="")

    def cmds(self, node=None):
        return [c for n, c in self.calls if node is None or n == node]


@pytest.fixture
def fake(monkeypatch):
    def make(present_nodes=(), on_nas=False, nas=None):
        fr = FakeRemote(present_nodes, on_nas)
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


def test_ensure_downloads_to_nas_when_configured(fake, monkeypatch):
    fr = fake(present_nodes=set(), on_nas=False,
              nas={"mode": "path", "path": "/mnt/nas/models", "download_to": "nas"})
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


def test_presence_matrix_shape(fake):
    fake(present_nodes={config.HEAD}, on_nas=True, nas={"mode": "path", "path": "/mnt/nas/models"})
    m = distribution.presence_matrix(RECIPE)
    assert m["org/M"]["nodes"][config.HEAD] is True
    assert m["org/M"]["nodes"][config.WORKER] is False
    assert m["org/M"]["nas"] is True
    assert m["org/M"]["services"] == ["agent"]


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
