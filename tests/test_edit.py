"""`sparkctl edit` target resolution, YAML validation, and editor precedence — pure, no cluster."""
from pathlib import Path

import pytest

from sparkctl.cli.edit import resolve_editor, resolve_target, validate_yaml


def test_resolve_target_cluster_and_recipe():
    root = Path("/repo")
    assert resolve_target(root, "cluster", None) == root / "cluster.yaml"
    assert resolve_target(root, "recipe", "qwen3-235b") == root / "recipes" / "qwen3-235b.yaml"


def test_resolve_target_recipe_without_name_exits():
    with pytest.raises(SystemExit):
        resolve_target(Path("/repo"), "recipe", None)


def test_resolve_target_bad_kind_exits():
    with pytest.raises(SystemExit):
        resolve_target(Path("/repo"), "widget", "x")


def test_validate_yaml_accepts_mapping():
    ok, err = validate_yaml("name: x\nservices: []\n")
    assert ok and err is None


def test_validate_yaml_rejects_broken_and_non_mapping():
    ok, err = validate_yaml("name: [unclosed\n")
    assert not ok and "did not parse" in err
    ok, err = validate_yaml("- just\n- a\n- list\n")
    assert not ok and "mapping" in err
    ok, err = validate_yaml("")
    assert not ok and "empty" in err


def test_resolve_editor_precedence(monkeypatch):
    for v in ("SPARKCTL_EDITOR", "VISUAL", "EDITOR"):
        monkeypatch.delenv(v, raising=False)
    assert resolve_editor() == "vi"                       # nothing set -> default
    assert resolve_editor("nano") == "nano"               # cluster.yaml editor: key
    monkeypatch.setenv("EDITOR", "emacs")
    assert resolve_editor("nano") == "emacs"              # $EDITOR beats config
    monkeypatch.setenv("SPARKCTL_EDITOR", "code -w")
    assert resolve_editor("nano") == "code -w"            # SPARKCTL_EDITOR wins
