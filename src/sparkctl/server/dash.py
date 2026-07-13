"""The /dash page: a zero-dependency, server-rendered HTML status view — the always-available,
no-Docker way to see what the cluster is doing. Auto-refreshes via <meta>; no client JS."""
import html


def _badge(up):
    color, text = ("#2da44e", "up") if up else ("#cf222e", "down")
    return f'<span style="color:{color};font-weight:600">{text}</span>'


def _pct(used, total):
    return f"{used / total * 100:.0f}%" if total else "–"


def render(recipe_name, recipe_sha, rows, node_rows, models, litellm_ok, port):
    body_rows = "\n".join(
        f"<tr><td>{html.escape(r['service'])}</td><td>{html.escape(r['node'])}</td>"
        f"<td>{_badge(r['up'])}</td><td>{r['running']:.0f}</td><td>{r['waiting']:.0f}</td>"
        f"<td>{r['kv_cache'] * 100:.1f}%</td></tr>"
        for r in rows) or '<tr><td colspan="6">no vLLM services in the current recipe</td></tr>'
    node_cells = []
    for n in node_rows:
        gpu = f"{n['gpu_util']:.0f}%" if n["gpu_util"] is not None else "–"
        node_cells.append(
            f"<tr><td>{html.escape(n['node'])}</td>"
            f"<td>{n['mem_used_gib']:.0f} / {n['mem_total_gib']:.0f} GiB"
            f" ({_pct(n['mem_used_gib'], n['mem_total_gib'])})</td>"
            f"<td>{gpu}</td>"
            f"<td>{n['disk_used_gib']:.0f} / {n['disk_total_gib']:.0f} GiB"
            f" ({_pct(n['disk_used_gib'], n['disk_total_gib'])})</td></tr>")
    node_body = "\n".join(node_cells) or '<tr><td colspan="4">sampling…</td></tr>'
    model_list = "".join(f"<li><code>{html.escape(m)}</code></li>" for m in models)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>sparkctl — {html.escape(recipe_name)}</title>
<style>
 body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem auto; max-width: 46rem;
        color: #1f2328; }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
 th, td {{ text-align: left; padding: .35rem .6rem; border-bottom: 1px solid #d0d7de; }}
 th {{ font-weight: 600; color: #57606a; }}
 code {{ background: #f6f8fa; padding: .1rem .3rem; border-radius: 4px; }}
 .muted {{ color: #57606a; }}
</style></head><body>
<h2>sparkctl</h2>
<p>recipe <code>{html.escape(recipe_name)}</code> <span class="muted">(sha {recipe_sha[:12]})</span>
 &nbsp;·&nbsp; gateway {_badge(litellm_ok)}</p>
<table>
<tr><th>SERVICE</th><th>NODE</th><th>STATE</th><th>RUNNING</th><th>WAITING</th><th>KV-CACHE</th></tr>
{body_rows}
</table>
<table>
<tr><th>NODE</th><th>MEMORY (unified)</th><th>GPU</th><th>MODEL CACHE</th></tr>
{node_body}
</table>
<p class="muted">routed models:</p><ul>{model_list}</ul>
<p class="muted">endpoints: <code>/v1</code> (OpenAI-compatible) · <code>/metrics</code> (Prometheus)
 · <code>/healthz</code> — port {port}</p>
</body></html>"""
