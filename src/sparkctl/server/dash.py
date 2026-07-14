"""The /dash page: a zero-dependency HTML status view — the always-available, no-Docker way to
see what the cluster is doing. Server-renders the first paint (so curl/tests see real numbers
with no JS), then a small inline script polls /dash/data every second and repaints the same
markup client-side — no reload flicker, no build step, no framework."""
import html

# Gauge color thresholds (percent) shared by the server-rendered first paint and the client
# repaint script below — keep both in sync if these change.
_WARN_PCT = 70
_CRIT_PCT = 90


def _gauge_color(pct):
    if pct >= _CRIT_PCT:
        return "#cf222e"
    if pct >= _WARN_PCT:
        return "#bf8700"
    return "#2da44e"


def _gauge(pct, label=None):
    """A small radial gauge (conic-gradient ring) for a 0-100 percent value; `pct=None` renders
    a neutral empty ring (metric not available on this engine, e.g. GPU util on some backends)."""
    if pct is None:
        color, text, deg = "#8c959f", "–", 0.0
    else:
        p = max(0.0, min(100.0, pct))
        color, text, deg = _gauge_color(p), f"{p:.0f}%", p * 3.6
    return (f'<div class="gauge" style="background:conic-gradient({color} {deg:.1f}deg, var(--track) 0)">'
            f'<div class="hole">{label or text}</div></div>')


def _meter(value, floor=8):
    """A native <meter> bar for an open-ended count (requests running/waiting) — scaled so the
    bar has headroom above the current value instead of pinning at max."""
    top = max(floor, value * 1.25, 1)
    return (f'<meter min="0" max="{top:.2f}" value="{value:.2f}"></meter>'
            f'<span class="val">{value:.0f}</span>')


def _cell(widget, text=None):
    """Wrap a gauge/meter (and optional trailing text) in a flex box INSIDE the <td> — the flex
    layout must live on a child, not the cell itself, or the cell drops out of the table grid and
    columns stop aligning."""
    tail = f'<span class="val">{text}</span>' if text else ""
    return f'<div class="cell">{widget}{tail}</div>'


def _badge(up):
    color, text = ("#2da44e", "up") if up else ("#cf222e", "down")
    return f'<span style="color:{color};font-weight:600">{text}</span>'


def render(recipe_name, recipe_sha, rows, node_rows, models, litellm_ok, port):
    body_rows = "\n".join(
        f"<tr><td>{html.escape(r['service'])}</td><td>{html.escape(r['node'])}</td>"
        f"<td>{_badge(r['up'])}</td><td>{_cell(_meter(r['running']))}</td>"
        f"<td>{_cell(_meter(r['waiting']))}</td>"
        f"<td>{_cell(_gauge(r['kv_cache'] * 100))}</td></tr>"
        for r in rows) or '<tr><td colspan="6">no vLLM services in the current recipe</td></tr>'
    node_cells = []
    for n in node_rows:
        mem_pct = n["mem_used_gib"] / n["mem_total_gib"] * 100 if n["mem_total_gib"] else None
        disk_pct = n["disk_used_gib"] / n["disk_total_gib"] * 100 if n["disk_total_gib"] else None
        mem_txt = f"{n['mem_used_gib']:.0f} / {n['mem_total_gib']:.0f} GiB"
        disk_txt = f"{n['disk_used_gib']:.0f} / {n['disk_total_gib']:.0f} GiB"
        node_cells.append(
            f"<tr><td>{html.escape(n['node'])}</td>"
            f"<td>{_cell(_gauge(mem_pct), mem_txt)}</td>"
            f"<td>{_cell(_gauge(n['gpu_util']))}</td>"
            f"<td>{_cell(_gauge(disk_pct), disk_txt)}</td></tr>")
    node_body = "\n".join(node_cells) or '<tr><td colspan="4">sampling…</td></tr>'
    model_list = "".join(f"<li><code>{html.escape(m)}</code></li>" for m in models)
    return f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>sparkctl — {html.escape(recipe_name)}</title>
<style>
 :root {{ color-scheme: light dark;
         --bg: #ffffff; --fg: #1f2328; --muted: #57606a;
         --border: #d0d7de; --track: #d0d7de; --code-bg: #f6f8fa; }}
 @media (prefers-color-scheme: dark) {{
   :root {{ --bg: #0d1117; --fg: #e6edf3; --muted: #9198a1;
            --border: #30363d; --track: #30363d; --code-bg: #161b22; }}
 }}
 body {{ font: 14px/1.5 -apple-system, system-ui, sans-serif; margin: 2rem auto; max-width: 50rem;
        background: var(--bg); color: var(--fg); }}
 table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
 th, td {{ text-align: left; padding: .45rem .6rem; border-bottom: 1px solid var(--border);
          vertical-align: middle; }}
 th {{ font-weight: 600; color: var(--muted); }}
 code {{ background: var(--code-bg); padding: .1rem .3rem; border-radius: 4px; }}
 .muted {{ color: var(--muted); }}
 .cell {{ display: flex; align-items: center; gap: .5rem; white-space: nowrap; }}
 .cell .val {{ font-variant-numeric: tabular-nums; }}
 .gauge {{ position: relative; width: 42px; height: 42px; border-radius: 50%; flex: none; }}
 .gauge .hole {{ position: absolute; inset: 5px; border-radius: 50%; background: var(--bg);
                display: flex; align-items: center; justify-content: center;
                font-size: .68rem; font-weight: 700; color: var(--fg); }}
 meter {{ width: 6rem; height: 1rem; vertical-align: middle; flex: none; }}
 .stale {{ opacity: .5; }}
</style></head><body>
<h2>sparkctl</h2>
<p id="summary">recipe <code>{html.escape(recipe_name)}</code>
 <span class="muted">(sha {recipe_sha[:12]})</span>
 &nbsp;·&nbsp; gateway <span id="gw">{_badge(litellm_ok)}</span></p>
<table>
<tr><th>SERVICE</th><th>NODE</th><th>STATE</th><th>RUNNING</th><th>WAITING</th><th>KV-CACHE</th></tr>
<tbody id="svc-rows">
{body_rows}
</tbody>
</table>
<table>
<tr><th>NODE</th><th>MEMORY (unified)</th><th>GPU</th><th>MODEL CACHE</th></tr>
<tbody id="node-rows">
{node_body}
</tbody>
</table>
<p class="muted">routed models:</p><ul id="model-list">{model_list}</ul>
<p class="muted">endpoints: <code>/v1</code> (OpenAI-compatible) · <code>/metrics</code> (Prometheus)
 · <code>/healthz</code> — port {port}</p>
<script>
const POLL_MS = 1000;
const WARN_PCT = {_WARN_PCT}, CRIT_PCT = {_CRIT_PCT};

function esc(s) {{
  return String(s).replace(/[&<>"']/g, c => ({{"&":"&amp;","<":"&lt;",">":"&gt;",
                                               '"':"&quot;","'":"&#39;"}}[c]));
}}

function badge(up) {{
  return up ? '<span style="color:#2da44e;font-weight:600">up</span>'
            : '<span style="color:#cf222e;font-weight:600">down</span>';
}}

function gauge(pct, label) {{
  if (pct == null) {{
    return `<div class="gauge" style="background:conic-gradient(var(--track) 0deg, var(--track) 0)">`
         + `<div class="hole">${{label || "–"}}</div></div>`;
  }}
  const p = Math.max(0, Math.min(100, pct));
  const color = p >= CRIT_PCT ? "#cf222e" : p >= WARN_PCT ? "#bf8700" : "#2da44e";
  const text = label || `${{p.toFixed(0)}}%`;
  return `<div class="gauge" style="background:conic-gradient(${{color}} ${{p * 3.6}}deg, var(--track) 0)">`
       + `<div class="hole">${{text}}</div></div>`;
}}

function meter(value, floor) {{
  const top = Math.max(floor || 8, value * 1.25, 1);
  return `<meter min="0" max="${{top}}" value="${{value}}"></meter>`
       + `<span class="val">${{value.toFixed(0)}}</span>`;
}}

function cell(widget, text) {{
  const tail = text ? `<span class="val">${{text}}</span>` : "";
  return `<div class="cell">${{widget}}${{tail}}</div>`;
}}

function paintServices(rows) {{
  const body = document.getElementById("svc-rows");
  if (!rows.length) {{
    body.innerHTML = '<tr><td colspan="6">no vLLM services in the current recipe</td></tr>';
    return;
  }}
  body.innerHTML = rows.map(r => `<tr><td>${{esc(r.service)}}</td><td>${{esc(r.node)}}</td>`
    + `<td>${{badge(r.up)}}</td><td>${{cell(meter(r.running))}}</td>`
    + `<td>${{cell(meter(r.waiting))}}</td>`
    + `<td>${{cell(gauge(r.kv_cache * 100))}}</td></tr>`).join("\\n");
}}

function paintNodes(rows) {{
  const body = document.getElementById("node-rows");
  if (!rows.length) {{
    body.innerHTML = '<tr><td colspan="4">sampling…</td></tr>';
    return;
  }}
  body.innerHTML = rows.map(n => {{
    const memPct = n.mem_total_gib ? n.mem_used_gib / n.mem_total_gib * 100 : null;
    const diskPct = n.disk_total_gib ? n.disk_used_gib / n.disk_total_gib * 100 : null;
    const memTxt = `${{n.mem_used_gib.toFixed(0)}} / ${{n.mem_total_gib.toFixed(0)}} GiB`;
    const diskTxt = `${{n.disk_used_gib.toFixed(0)}} / ${{n.disk_total_gib.toFixed(0)}} GiB`;
    return `<tr><td>${{esc(n.node)}}</td>`
      + `<td>${{cell(gauge(memPct), memTxt)}}</td>`
      + `<td>${{cell(gauge(n.gpu_util))}}</td>`
      + `<td>${{cell(gauge(diskPct), diskTxt)}}</td></tr>`;
  }}).join("\\n");
}}

async function tick() {{
  try {{
    const r = await fetch("/dash/data", {{cache: "no-store"}});
    if (!r.ok) throw new Error(r.status);
    const d = await r.json();
    document.title = `sparkctl — ${{d.recipe_name}}`;
    document.getElementById("summary").innerHTML =
      `recipe <code>${{esc(d.recipe_name)}}</code> <span class="muted">(sha `
      + `${{esc(d.recipe_sha.slice(0, 12))}})</span> &nbsp;·&nbsp; gateway ${{badge(d.litellm_ok)}}`;
    paintServices(d.rows);
    paintNodes(d.node_rows);
    document.getElementById("model-list").innerHTML =
      d.models.map(m => `<li><code>${{esc(m)}}</code></li>`).join("");
    document.body.classList.remove("stale");
  }} catch (e) {{
    document.body.classList.add("stale");   // server unreachable — keep last good paint, dim it
  }}
}}

setInterval(tick, POLL_MS);
</script>
</body></html>"""