"""Boot arm/disarm + self-heal: `sparkctl stop` and `apply --boot` break the OOM/crash-on-boot loop.

The decision logic (skip if paused / auto-disarm on a stale boot_attempt / re-arm on a human apply)
is exercised with an in-memory stand-in for the head-local state markers, and the heavy apply steps
(resolve/pull/teardown/up/wait) stubbed — no cluster, no SSH.
"""
from types import SimpleNamespace

import pytest

from sparkctl import config
from sparkctl.cli import resource


@pytest.fixture
def env(tmp_path, monkeypatch):
    """In-memory state markers + stubbed apply steps. Returns (store, calls)."""
    store, calls = {}, []
    monkeypatch.setattr(config, "ROOT", tmp_path)
    monkeypatch.setattr(resource, "_state_write",
                        lambda name, content="": store.__setitem__(name, content))
    monkeypatch.setattr(resource, "_state_read", lambda name: store.get(name))
    monkeypatch.setattr(resource, "_state_clear", lambda name: store.pop(name, None))
    monkeypatch.setattr(resource, "resolve_apply_target", lambda a: a.recipe or "cur")
    monkeypatch.setattr(resource, "load_recipe", lambda name: {"name": name, "services": []})
    monkeypatch.setattr(resource, "ensure_models", lambda recipe: calls.append("ensure"))
    monkeypatch.setattr(resource, "_services_down", lambda: calls.append("down"))
    monkeypatch.setattr(resource, "_free_page_cache", lambda: calls.append("freecache"))
    monkeypatch.setattr(resource, "_recipe_up", lambda name: calls.append(("up", name)))
    monkeypatch.setattr(resource, "_wait_ready", lambda recipe, timeout: calls.append("wait"))
    return SimpleNamespace(store=store, calls=calls, root=tmp_path)


def _apply(**kw):
    return SimpleNamespace(**{"recipe": None, "filename": None, "wait": False,
                              "timeout": 1800, "boot": False, **kw})


# ---- stop --------------------------------------------------------------------
def test_stop_disarms_and_tears_down(env):
    resource.cmd_stop(SimpleNamespace())
    assert "down" in env.calls
    assert resource.PAUSED in env.store                 # boot now serves nothing
    assert resource.BOOT_ATTEMPT not in env.store


# ---- boot apply: guards -------------------------------------------------------
def test_boot_apply_skips_when_paused(env):
    env.store[resource.PAUSED] = "stopped"
    resource.cmd_apply(_apply(boot=True))
    assert env.calls == []                              # nothing brought up
    assert resource.PAUSED in env.store                 # stays disarmed


def test_boot_apply_auto_disarms_on_stale_attempt(env):
    env.store[resource.BOOT_ATTEMPT] = "gpt-oss-120b"   # last boot started it, never confirmed
    resource.cmd_apply(_apply(boot=True))
    assert env.calls == []                              # did NOT retry the killer recipe
    assert resource.PAUSED in env.store                 # loop broken
    assert "gpt-oss-120b" in env.store[resource.PAUSED]
    assert resource.BOOT_ATTEMPT not in env.store


# ---- boot apply: healthy paths ------------------------------------------------
def test_boot_apply_arms_then_clears_on_success(env):
    resource.cmd_apply(_apply(boot=True, recipe="r1"))
    assert ("up", "r1") in env.calls
    assert resource.BOOT_ATTEMPT not in env.store       # cleared once brought up (non-wait)
    assert resource.PAUSED not in env.store


def test_boot_apply_wait_failure_disarms_and_reraises(env, monkeypatch):
    def boom(recipe, timeout):
        raise SystemExit("[wait] TIMEOUT")
    monkeypatch.setattr(resource, "_wait_ready", boom)
    with pytest.raises(SystemExit):
        resource.cmd_apply(_apply(boot=True, recipe="r1", wait=True))
    assert resource.PAUSED in env.store                 # next boot won't retry the OOM
    assert resource.BOOT_ATTEMPT not in env.store


# ---- human apply: intent always wins -----------------------------------------
def test_human_apply_ignores_pause_and_rearms(env):
    env.store[resource.PAUSED] = "stopped"
    resource.cmd_apply(_apply(recipe="r1"))             # boot=False
    assert ("up", "r1") in env.calls                    # proceeds despite pause
    assert resource.PAUSED not in env.store             # re-armed


def test_human_apply_does_not_write_boot_attempt(env):
    resource.cmd_apply(_apply(recipe="r1"))
    assert resource.BOOT_ATTEMPT not in env.store       # dirty flag is a boot-only concern
