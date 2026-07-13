"""Ollama engine: one shared container per node, models pulled into it."""
import time

from sparkctl import config, remote


def ollama_ensure(node):
    remote.on(node, f"mkdir -p {config.CACHE}/ollama", check=False)
    remote.on(node, f"docker inspect {config.PFX}-ollama >/dev/null 2>&1 || "
                    f"docker run -d --rm --gpus all --network host --name {config.PFX}-ollama "
                    f"-v {config.CACHE}/ollama:/root/.ollama docker.io/ollama/ollama", check=False)
    time.sleep(3)


def ollama_up(svc):
    node = svc.get("node", config.HEAD)
    print(f"[ollama] serving {svc['model']} on {node}")
    ollama_ensure(node)
    remote.on(node, f"docker exec {config.PFX}-ollama ollama pull {svc['model']}", check=False)


def ollama_down():
    for node in config.NODES:
        remote.on(node, f"docker rm -f {config.PFX}-ollama 2>/dev/null || true", check=False)
