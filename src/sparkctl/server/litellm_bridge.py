"""LiteLLM bridge: config generation (pure) + supervision of the LiteLLM proxy as a managed
subprocess. LiteLLM's proxy is architected to own its process, so we run it as a child bound to
loopback and reverse-proxy /v1/* to it — a LiteLLM crash never takes down /metrics or /dash."""
import subprocess
import sys
from pathlib import Path

import yaml

from sparkctl import config
from sparkctl.backends import get_backend

LITELLM_IMAGE = "ghcr.io/berriai/litellm:main-stable"
LITELLM_LOG = Path.home() / ".sparkctl" / "litellm.log"


def litellm_config(recipe, served_from, settings):
    """Turn the backend's endpoint map into a LiteLLM config dict. Services sharing a served_name
    on different nodes become multiple entries with the same model_name -> a load-balance pool."""
    model_list = []
    for served_name, api_bases in get_backend().endpoints(recipe, served_from).items():
        for base in api_bases:
            model_list.append({
                "model_name": served_name,
                "litellm_params": {"model": f"openai/{served_name}",
                                   "api_base": base,
                                   "api_key": "none"}})
    cfg = {"model_list": model_list,
           "router_settings": {"routing_strategy": settings.get("routing_strategy", "simple-shuffle")}}
    if settings.get("master_key"):
        cfg["general_settings"] = {"master_key": settings["master_key"]}
    return cfg


def internal_port(settings):
    """Loopback port the managed LiteLLM child listens on (the public port serves the unified app)."""
    return settings.get("litellm_port", settings.get("port", 8080) + 1)


def config_path():
    d = (Path.home() / ".sparkctl") if config.SELF is None else Path(config.STATE)
    d.mkdir(parents=True, exist_ok=True)
    return d / "litellm-config.yaml"


def write_config(recipe, settings):
    cfg = litellm_config(recipe, settings.get("host", "local"), settings)
    if not cfg["model_list"]:
        sys.exit("[server] current recipe exposes no routable models")
    p = config_path()
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p, cfg


def _litellm_bin():
    """The litellm CLI — prefer the sparkctl-managed venv, fall back to PATH."""
    venv_bin = Path.home() / ".sparkctl" / "venv" / "bin" / "litellm"
    return str(venv_bin) if venv_bin.exists() else "litellm"


def start_child(cfg_file, settings):
    """Launch the LiteLLM proxy as a supervised child on loopback. Returns the Popen handle."""
    LITELLM_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = open(LITELLM_LOG, "a")
    return subprocess.Popen(
        [_litellm_bin(), "--config", str(cfg_file),
         "--host", "127.0.0.1", "--port", str(internal_port(settings))],
        stdout=log, stderr=subprocess.STDOUT)
