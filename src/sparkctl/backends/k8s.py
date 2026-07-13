"""K8sBackend — roadmap stub. The seam is designed (see backends/base.py and docs/backends.md);
services would run as Deployments/Jobs with weights on a shared PVC or per-node cache."""
from sparkctl.backends.base import Backend

_MSG = ("the k8s backend is not yet implemented — set `backend: docker` in cluster.yaml "
        "(roadmap: docs/backends.md)")


class K8sBackend(Backend):
    def up(self, recipe):
        raise NotImplementedError(_MSG)

    def down(self):
        raise NotImplementedError(_MSG)

    def status(self):
        raise NotImplementedError(_MSG)

    def logs(self, svc, follow=False, tail=80):
        raise NotImplementedError(_MSG)

    def pull(self, recipe):
        raise NotImplementedError(_MSG)

    def endpoints(self, recipe, served_from):
        raise NotImplementedError(_MSG)

    def metrics_targets(self, recipe, served_from):
        raise NotImplementedError(_MSG)

    def run_workload(self, name, image, command="", node=None, env=None, volumes=None):
        raise NotImplementedError(_MSG)
