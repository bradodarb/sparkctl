"""sparkctl CLI — one tool, two contexts (auto-detected by hostname):

  • control machine (your Mac): `deploy`/`build` run locally; `pull-queue` launches detached on the
    head (survives laptop sleep); read-only resource verbs (get/describe/top) run locally; mutating
    verbs are deployed-then-forwarded to the head over SSH.
  • a cluster node (the boot daemon / forwarded commands): acts as the orchestrator — reads
    cluster.yaml + the `current` recipe pointer and brings each service up/down.

Config is the single source of truth: cluster.yaml (topology/deploy) + recipes/*.yaml + `current`.

  sparkctl get nodes|services|recipes [-o wide|json]
  sparkctl describe node|service|recipe <name>
  sparkctl apply [recipe | -f recipe.yaml]      # ensure + (re)start a deployment
  sparkctl delete services --all | delete service <name>
  sparkctl logs <service> [-f] [--tail N] | top nodes|services | status
  sparkctl pull [recipe] | pull-queue <recipe>... | mirror
  sparkctl deploy | build | test | ctx-test | current | manifest
"""
import argparse
import json
import sys

from sparkctl import config, remote
from sparkctl.backends import get_backend
from sparkctl.cli import resource
from sparkctl.deploy import deploy, deploy_init
from sparkctl.distribution import mirror_to_others
from sparkctl.manifest import verify_deployment, write_active_manifest
from sparkctl.recipes import current_recipe, load_recipe, recipe_hash
from sparkctl.server.lifecycle import cmd_serve, refresh_server_if_running

# Removed verbs -> where their behavior lives now. Clean cutover: hint + exit 2, no forwarding.
REMOVED = {
    "up": "sparkctl apply [recipe]",
    "down": "sparkctl delete services --all",
    "switch": "sparkctl apply <recipe>",
    "list": "sparkctl get recipes",
    "gateway": "sparkctl serve [stop|status|config|tunnel] (configured via cluster.yaml server:)",
    "metrics": "configure cluster.yaml server.metrics/server.grafana — served by `sparkctl serve`",
}


def check_removed(argv):
    """If argv starts with a removed verb, print a hint and exit 2."""
    if argv and argv[0] in REMOVED:
        print(f"sparkctl: '{argv[0]}' was removed — use: {REMOVED[argv[0]]}", file=sys.stderr)
        sys.exit(2)


# ---------------------------------------------------------------- node-mode commands
def cmd_pull(args):
    get_backend().pull(load_recipe(args.recipe or current_recipe()))
    print("[pull] complete")


def cmd_pull_queue(args):
    for r in args.recipes:
        print(f"=== {r} ===")
        try:
            cmd_pull(argparse.Namespace(recipe=r))
        except SystemExit as e:
            print(f"PULL FAILED: {r} ({e}) — continuing")
    print("QUEUE COMPLETE")


def cmd_mirror(args):
    mirror_to_others()
    print("[mirror] complete")


def cmd_status(args):
    cur = current_recipe()
    have = (config.ROOT / "recipes" / f"{cur}.yaml").exists()
    print(f"current recipe: {cur}" + (f"  (sha {recipe_hash(cur)[:12]})" if have else "  (recipe file missing!)"),
          flush=True)
    for node in config.NODES:
        rec, ok = verify_deployment(node)
        badge = "✅ matches current" if ok else ("⚠️  DRIFT" if rec else "· nothing active")
        print(f"--- {node} --- active: {rec or '(none)'}  [{badge}]", flush=True)
        remote.on(node, f"docker ps --filter name={config.PFX}- "
                        f"--format 'table {{{{.Names}}}}\\t{{{{.Status}}}}'", check=False)
    print("--- API health (head) ---", flush=True)
    remote.on(config.HEAD, f"curl -s --max-time 5 http://localhost:{config.CFG['cluster']['api_port']}/v1/models "
                           f"|| echo 'API not ready'", check=False)


def cmd_logs(args):
    svc = next((s for s in load_recipe(current_recipe())["services"]
                if s["name"] == args.service), None)
    if not svc:
        sys.exit(f"no service '{args.service}' in current recipe")
    get_backend().logs(svc, follow=args.follow, tail=args.tail)


def cmd_test(args):
    port = config.CFG["cluster"]["api_port"]
    remote.sh(f"python3 {config.ROOT}/bin/smoke-tool-call.py --base http://localhost:{port}/v1",
              check=False)


def cmd_ctxtest(args):
    port = config.CFG["cluster"]["api_port"]
    remote.sh(f"python3 {config.ROOT}/bin/ctx-test.py --base http://localhost:{port}/v1 "
              f"--tokens {args.tokens} --depths {args.depths}", check=False)


def cmd_manifest(args):
    name = current_recipe()
    write_active_manifest(name)
    print(f"[manifest] active.json written for current recipe: {name} @ {recipe_hash(name)[:12]}")


def cmd_current(args):
    print(current_recipe())


def build_parser():
    ap = argparse.ArgumentParser(prog="sparkctl")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # resource verbs
    p = sub.add_parser("get", help="list resources")
    p.add_argument("resource", choices=["nodes", "services", "recipes", "models"])
    p.add_argument("-o", "--output", choices=["table", "wide", "json"], default="table")
    p.set_defaults(fn=resource.cmd_get)
    p = sub.add_parser("describe", help="show detail for one resource")
    p.add_argument("kind", choices=["node", "service", "recipe"])
    p.add_argument("name")
    p.set_defaults(fn=resource.cmd_describe)
    p = sub.add_parser("apply", help="deploy a recipe (validate, restart services, update current)")
    p.add_argument("recipe", nargs="?")
    p.add_argument("-f", "--filename", help="recipe manifest file (copied into recipes/)")
    p.set_defaults(fn=resource.cmd_apply)
    p = sub.add_parser("delete", help="tear down services")
    p.add_argument("kind", choices=["service", "services"])
    p.add_argument("name", nargs="?")
    p.add_argument("--all", action="store_true")
    p.set_defaults(fn=resource.cmd_delete)
    p = sub.add_parser("top", help="live vLLM metrics in the terminal")
    p.add_argument("resource", choices=["nodes", "services"])
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--once", action="store_true")
    p.set_defaults(fn=resource.cmd_top)

    # summaries + logs
    sub.add_parser("status").set_defaults(fn=cmd_status)
    p = sub.add_parser("logs"); p.add_argument("service")
    p.add_argument("-f", "--follow", action="store_true"); p.add_argument("--tail", type=int, default=80)
    p.set_defaults(fn=cmd_logs)

    # model distribution
    p = sub.add_parser("pull"); p.add_argument("recipe", nargs="?"); p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("pull-queue"); p.add_argument("recipes", nargs="+"); p.set_defaults(fn=cmd_pull_queue)
    sub.add_parser("mirror").set_defaults(fn=cmd_mirror)

    # utilities
    sub.add_parser("current").set_defaults(fn=cmd_current)
    sub.add_parser("manifest").set_defaults(fn=cmd_manifest)
    sub.add_parser("test").set_defaults(fn=cmd_test)
    p = sub.add_parser("ctx-test"); p.add_argument("--tokens", type=int, default=30000)
    p.add_argument("--depths", default="0.0,0.5,1.0"); p.set_defaults(fn=cmd_ctxtest)

    # unified server (gateway + metrics + dash)
    p = sub.add_parser("serve", help="run the unified server (gateway + metrics + dash)")
    p.add_argument("action", nargs="?", choices=["start", "stop", "status", "config", "tunnel"],
                   default="start")
    p.add_argument("--foreground", action="store_true",
                   help="stay attached (systemd/launchd units, containers)")
    p.set_defaults(fn=cmd_serve)
    return ap


# ---------------------------------------------------------------- control-machine (Mac) mode
def build():
    for node in config.NODES:
        a = remote.node_addr(node)
        print(f"[build] {node} -> {config.IMAGE}")
        remote.sh(f"ssh {config.USER}@{a} 'docker build -t {config.IMAGE} "
                  f"-f {config.REMOTE}/docker/Dockerfile {config.REMOTE}/docker'")


def run_node(node, argv):
    remote.sh(f"ssh {config.USER}@{remote.node_addr(node)} "
              f"{json.dumps(f'{config.REMOTE}/bin/sparkctl ' + ' '.join(argv))}", check=False)


def run_head(argv):
    run_node(config.HEAD, argv)


def launch_pull_queue_on_head(recipes):
    a = remote.node_addr(config.HEAD)
    inner = (f"setsid bash -c 'nohup {config.REMOTE}/bin/sparkctl pull-queue {' '.join(recipes)} "
             f"> {config.CACHE.rsplit('/', 1)[0]}/pull-queue.log 2>&1' >/dev/null 2>&1 &")
    remote.sh(f"ssh {config.USER}@{a} {json.dumps(inner)}")
    print(f"[pull-queue] launched detached on {config.HEAD}: {' '.join(recipes)}")
    print(f"[pull-queue] follow: tail ~/pull-queue.log on {config.HEAD}")


# Read-only verbs run directly on the control machine (they SSH per node as needed) — no deploy.
LOCAL_VERBS = ("get", "describe", "top", "current")
# Mutating verbs push the repo to the nodes first, then run on the head.
DEPLOY_VERBS = ("apply", "delete", "pull", "mirror", "build")


def control_main(argv):
    """Runs on the control machine (not a cluster node). deploy/build local; pull-queue detached on
    head; read-only verbs local; mutating verbs deploy-then-forward to the head over SSH."""
    check_removed(argv)
    cmd = argv[0] if argv else "help"
    if cmd == "deploy":
        return deploy_init() if "--init" in argv else deploy()
    if cmd == "build":
        deploy(); return build()
    if cmd == "pull-queue":
        deploy(); return launch_pull_queue_on_head(argv[1:])
    if cmd in LOCAL_VERBS:
        a = build_parser().parse_args(argv); return a.fn(a)
    if cmd == "serve":
        srv_host = config.SERVER.get("host", "local")
        if srv_host == "local" or (len(argv) > 1 and argv[1] in ("config", "tunnel")):
            a = build_parser().parse_args(argv); return a.fn(a)   # run on the dev machine
        deploy(); return run_node(srv_host, argv)                 # server lives on a node
    if cmd == "apply":
        # resolve -f / name locally (repo is the source of truth), then forward by name
        a = build_parser().parse_args(argv)
        name = resource.resolve_apply_target(a)
        (config.ROOT / "current").write_text(name + "\n")
        argv = ["apply", name]
    if cmd in DEPLOY_VERBS:
        deploy()
    run_head(argv)
    # keep a dev-machine-local server in sync after a deployment change
    if cmd == "apply" and config.SERVER.get("host", "local") == "local":
        refresh_server_if_running()


def main():
    argv = sys.argv[1:]
    try:
        if config.SELF is None:     # control machine (e.g. the Mac)
            control_main(argv)
        else:                       # a cluster node — act as the orchestrator
            check_removed(argv)
            args = build_parser().parse_args(argv)
            args.fn(args)
    except NotImplementedError as e:  # roadmap stubs (k8s) exit cleanly, not with a traceback
        sys.exit(f"sparkctl: {e}")


if __name__ == "__main__":
    main()
