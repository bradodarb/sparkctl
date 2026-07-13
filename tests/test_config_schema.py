"""Config loading rejects the removed pre-1.0 gateway:/metrics: schema with a clear message.
Runs the import in a subprocess (config loads at import time) pointed at a tmp repo root."""
import subprocess
import sys
from pathlib import Path

SRC = str(Path(__file__).resolve().parent.parent / "src")


def _load_config(root):
    return subprocess.run([sys.executable, "-c", "import sparkctl.config"],
                          env={"SPARKCTL_ROOT": str(root), "PYTHONPATH": SRC, "PATH": "/usr/bin"},
                          capture_output=True, text=True)


def _base_yaml():
    return ("cluster: {name: t, head: a, container_image: i, model_cache: /m, api_port: 8000}\n"
            "nodes: {a: {host: a.local}}\n"
            "fabric: {nccl_ib_hca: x, nccl_socket_ifname: y, nccl_ib_gid_index: 3}\n")


def test_legacy_gateway_key_rejected(tmp_path):
    (tmp_path / "cluster.yaml").write_text(_base_yaml() + "gateway: {port: 8080}\n")
    r = _load_config(tmp_path)
    assert r.returncode != 0
    assert "server:" in r.stderr and "cluster.yaml.example" in r.stderr


def test_legacy_metrics_key_rejected(tmp_path):
    (tmp_path / "cluster.yaml").write_text(_base_yaml() + "metrics: {enable: true}\n")
    assert _load_config(tmp_path).returncode != 0


def test_new_schema_loads(tmp_path):
    (tmp_path / "cluster.yaml").write_text(_base_yaml() + "server: {mode: local, port: 8080}\n")
    r = _load_config(tmp_path)
    assert r.returncode == 0, r.stderr


def test_single_node_no_fabric_loads(tmp_path):
    # single-Spark clusters have no inter-node fabric — the block is optional
    (tmp_path / "cluster.yaml").write_text(
        "cluster: {name: t, head: a, container_image: i, model_cache: /m, api_port: 8000}\n"
        "nodes: {a: {host: a.local}}\n")
    r = _load_config(tmp_path)
    assert r.returncode == 0, r.stderr
