"""Backend registry — selected by the cluster.yaml `backend:` key (default: docker)."""
import sys

from sparkctl import config
from sparkctl.backends.docker import DockerBackend
from sparkctl.backends.k8s import K8sBackend

BACKENDS = {"docker": DockerBackend, "k8s": K8sBackend}


def get_backend():
    name = config.CFG.get("backend", "docker")
    cls = BACKENDS.get(name)
    if cls is None:
        sys.exit(f"unknown backend: {name} (choose from: {', '.join(BACKENDS)})")
    return cls()
