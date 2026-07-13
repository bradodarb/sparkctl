"""Backend ABC: WHERE and HOW a recipe's model services actually run.

The rest of sparkctl (CLI resource verbs, the unified server) talks to this interface only —
`endpoints()` feeds the LiteLLM route table, `metrics_targets()` feeds the metrics aggregator —
so adding an execution substrate means implementing this class, nothing else."""
from abc import ABC, abstractmethod


class Backend(ABC):
    @abstractmethod
    def up(self, recipe):
        """Launch every service in the recipe."""

    @abstractmethod
    def down(self):
        """Tear down all managed services everywhere."""

    @abstractmethod
    def status(self):
        """node -> {container/workload name -> status string} for everything managed."""

    @abstractmethod
    def logs(self, svc, follow=False, tail=80):
        """Stream/print a service's logs."""

    @abstractmethod
    def pull(self, recipe):
        """Ensure the recipe's model weights are present on every node that needs them."""

    @abstractmethod
    def endpoints(self, recipe, served_from):
        """served_name -> [api_base, ...] as reachable from `served_from` ('local' | node name).
        Multiple entries under one name form a gateway load-balance pool."""

    @abstractmethod
    def metrics_targets(self, recipe, served_from):
        """[{node, service, url}, ...] — the Prometheus endpoints the unified server scrapes."""

    @abstractmethod
    def run_workload(self, name, image, command="", node=None, env=None, volumes=None):
        """Generic primitive: run a named auxiliary workload (download job, sidecar, ...)."""
