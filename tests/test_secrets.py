"""Secrets: env-file round-trip, permissions, parse edge cases, and node sync emission."""
from sparkctl import config, remote, secrets


def test_save_load_roundtrip_and_mode(tmp_path):
    p = tmp_path / "secrets.env"
    secrets.save({"HF_TOKEN": "hf_abc", "OLLAMA_KEY": "ok"}, path=p)
    assert secrets.load(p) == {"HF_TOKEN": "hf_abc", "OLLAMA_KEY": "ok"}
    assert (p.stat().st_mode & 0o777) == 0o600


def test_load_ignores_comments_blanks_and_junk(tmp_path):
    p = tmp_path / "secrets.env"
    p.write_text("# a comment\n\nHF_TOKEN = x \nnot-a-pair\n")
    assert secrets.load(p) == {"HF_TOKEN": "x"}


def test_load_missing_file_is_empty(tmp_path):
    assert secrets.load(tmp_path / "nope.env") == {}


def test_sync_rsyncs_to_every_node(monkeypatch, tmp_path):
    p = tmp_path / "secrets.env"
    secrets.save({"HF_TOKEN": "x"}, path=p)
    monkeypatch.setattr(secrets, "PATH", p)
    on_calls = []
    monkeypatch.setattr(remote, "on", lambda node, cmd, **k: on_calls.append((node, cmd)))
    calls = []
    monkeypatch.setattr(remote, "sh", lambda cmd, **k: calls.append(cmd))
    secrets.sync_to_nodes()
    assert len(calls) == len(config.NODES)
    # a file transfer — the value itself never rides a shell command line. No --chmod flag (older
    # rsync, e.g. macOS 2.6.9, rejects it); perms are enforced with an explicit chmod on the node.
    assert all("rsync" in c and "--chmod" not in c for c in calls)
    assert all(any(f"chmod 600 {secrets.NODE_PATH}" in cmd for n, cmd in on_calls if n == node)
               for node in config.NODES)


def test_sync_noop_without_file(monkeypatch, tmp_path):
    monkeypatch.setattr(secrets, "PATH", tmp_path / "nope.env")
    calls = []
    monkeypatch.setattr(remote, "sh", lambda cmd, **k: calls.append(cmd))
    secrets.sync_to_nodes()
    assert not calls
