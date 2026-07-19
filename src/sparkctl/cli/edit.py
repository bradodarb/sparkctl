"""`sparkctl edit cluster|recipe <name>` — open a config file in $EDITOR and re-validate on save.

Purely local: it only touches files in the repo (cluster.yaml / recipes/<name>.yaml), never the
nodes. Deploying an edited recipe is still an explicit `apply`. `resolve_target` and `validate_yaml`
are pure so they're unit-testable; `cmd_edit` is the thin interactive shell over them.
"""
import os
import shlex
import subprocess
import sys

import yaml


def resolve_target(root, kind, name):
    """Map (kind, name) to the file to edit. Returns a Path. Exits on a bad kind / missing name."""
    if kind == "cluster":
        return root / "cluster.yaml"
    if kind == "recipe":
        if not name:
            sys.exit("edit: 'edit recipe' requires a recipe name (e.g. sparkctl edit recipe qwen3-235b)")
        return root / "recipes" / f"{name}.yaml"
    sys.exit(f"edit: unknown kind '{kind}' (expected 'cluster' or 'recipe')")


def validate_yaml(text):
    """Return (ok, error). ok=True when text parses as a YAML mapping; error is a message otherwise."""
    try:
        doc = yaml.safe_load(text)
    except yaml.YAMLError as e:
        return False, f"YAML did not parse: {e}"
    if doc is None:
        return False, "file is empty"
    if not isinstance(doc, dict):
        return False, f"expected a mapping at the top level, got {type(doc).__name__}"
    return True, None


def resolve_editor(configured=None):
    """Editor command, kubectl-style precedence: $SPARKCTL_EDITOR (our KUBE_EDITOR analog), then
    $VISUAL/$EDITOR, then the `editor:` key in cluster.yaml, then `vi`. Env always wins so you can
    override the persistent default for one session."""
    return (os.environ.get("SPARKCTL_EDITOR") or os.environ.get("VISUAL")
            or os.environ.get("EDITOR") or configured or "vi")


def cmd_edit(args):
    """Open cluster.yaml or a recipe in $EDITOR, then re-parse it and warn on invalid YAML."""
    from sparkctl import config   # lazy: keeps resolve_target/validate_yaml config-free (testable)

    path = resolve_target(config.ROOT, args.kind, args.name)
    if not path.exists():
        hint = ("  (create one with: sparkctl create recipe)" if args.kind == "recipe"
                else "  (copy cluster.yaml.example)")
        sys.exit(f"edit: {path} does not exist\n{hint}")

    editor = resolve_editor(config.CFG.get("editor"))
    before = path.read_text()
    rc = subprocess.call(shlex.split(editor) + [str(path)])
    if rc != 0:
        print(f"[edit] editor exited with status {rc}", file=sys.stderr)

    after = path.read_text()
    if after == before:
        print(f"[edit] no changes to {path}")
        return

    ok, err = validate_yaml(after)
    if not ok:
        print(f"[edit] ⚠️  {path} is not valid: {err}", file=sys.stderr)
        print("[edit] the file was saved as-is — re-run `sparkctl edit` to fix it.", file=sys.stderr)
        sys.exit(1)

    print(f"[edit] saved {path}")
    if args.kind == "recipe":
        # a hand-edit is exactly where flag/parallel couplings get out of sync — lint before serving
        from sparkctl.cli.validate import format_findings, lint_recipe
        findings = lint_recipe(yaml.safe_load(after),
                               known_nodes=set(config.NODES), head=config.HEAD)
        lint_warnings = [f for f in findings if f.level != "info"]
        if lint_warnings:
            print("[edit] validation flagged issues:")
            for line in format_findings(lint_warnings):
                print(line)
            print(f"[edit] auto-fix them with: sparkctl validate {args.name} --fix")
        print(f"[edit] apply it with: sparkctl apply {args.name} --wait")
    else:
        print("[edit] cluster.yaml changed — re-run mutating commands to push it to the nodes.")
