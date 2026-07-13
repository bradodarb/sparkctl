"""The unified server: one FastAPI app on one port.

  /v1/*     reverse proxy (streaming-aware) to the managed LiteLLM child / sidecar
  /metrics  Prometheus aggregation of every node's vLLM /metrics (single scrape target)
  /healthz  control-plane health (app + LiteLLM + per-target scrape state)
  /dash     zero-dependency HTML status page

LiteLLM runs as a supervised subprocess by default; set SPARKCTL_LITELLM_URL to an existing
upstream (the docker sidecar sets this) to skip child management."""
import asyncio
import contextlib
import os

from sparkctl import config
from sparkctl.recipes import current_recipe, load_recipe, recipe_hash, services_by_node
from sparkctl.server import dash, litellm_bridge
from sparkctl.server.metrics import NodeSampler, Scraper, scrape_targets

HOP_BY_HOP = {"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
              "te", "trailers", "transfer-encoding", "upgrade", "content-length", "host"}


def create_app():
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import HTMLResponse, PlainTextResponse, RedirectResponse, Response, StreamingResponse
    from starlette.background import BackgroundTask

    settings = config.SERVER
    recipe_name = current_recipe()
    recipe = load_recipe(recipe_name)
    served_from = settings.get("host", "local") if config.SELF is None else config.SELF

    upstream = os.environ.get("SPARKCTL_LITELLM_URL")
    manage_child = upstream is None
    if manage_child:
        upstream = f"http://127.0.0.1:{litellm_bridge.internal_port(settings)}"

    mx = settings.get("metrics", {})
    metrics_on = mx.get("enabled", True)
    interval = mx.get("scrape_interval_s", 10)
    scraper = Scraper(scrape_targets(recipe, served_from) if metrics_on else [], interval)
    sampler = NodeSampler(config.NODES if metrics_on else [], config.CACHE, interval)

    state = {"litellm_proc": None, "scrape_task": None, "sample_task": None}

    @contextlib.asynccontextmanager
    async def lifespan(app):
        if manage_child:
            cfg_file, _ = litellm_bridge.write_config(recipe, settings)
            state["litellm_proc"] = litellm_bridge.start_child(cfg_file, settings)
        if scraper.targets:
            state["scrape_task"] = asyncio.create_task(scraper.run())
        if sampler.nodes:
            state["sample_task"] = asyncio.create_task(sampler.run())
        yield
        for k in ("scrape_task", "sample_task"):
            if state[k]:
                state[k].cancel()
        if state["litellm_proc"]:
            state["litellm_proc"].terminate()
            with contextlib.suppress(Exception):
                state["litellm_proc"].wait(timeout=10)

    app = FastAPI(lifespan=lifespan)
    app.state.scraper = scraper
    app.state.sampler = sampler
    client = httpx.AsyncClient(base_url=upstream, timeout=httpx.Timeout(600, connect=10))

    def _litellm_ok():
        proc = state["litellm_proc"]
        return proc.poll() is None if proc else True   # sidecar mode: trust /v1 proxying to surface errors

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok",
                "recipe": recipe_name,
                "litellm": "up" if _litellm_ok() else "DOWN",
                "targets": {f"{n}/{s}": ("up" if r["ok"] else "down")
                            for (n, s), r in scraper.results.items()}}

    @app.get("/metrics")
    async def metrics():
        return PlainTextResponse(scraper.exposition() + sampler.exposition())

    @app.get("/dash")
    async def dashboard():
        models = sorted({s["served_name"]
                         for svcs in services_by_node(recipe).values() for s in svcs})
        return HTMLResponse(dash.render(recipe_name, recipe_hash(recipe_name), scraper.summaries(),
                                        sampler.summaries(), models, _litellm_ok(),
                                        settings.get("port", 8080)))

    @app.get("/")
    async def index():
        return RedirectResponse("/dash")

    @app.api_route("/v1/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
    async def proxy(request: Request, path: str):
        headers = {k: v for k, v in request.headers.items() if k.lower() not in HOP_BY_HOP}
        req = client.build_request(request.method, f"/v1/{path}",
                                   headers=headers, params=request.query_params,
                                   content=request.stream())
        try:
            r = await client.send(req, stream=True)
        except httpx.HTTPError as e:
            return Response(f"gateway upstream unavailable: {e}", status_code=502)
        resp_headers = {k: v for k, v in r.headers.items() if k.lower() not in HOP_BY_HOP}
        return StreamingResponse(r.aiter_raw(), status_code=r.status_code,
                                 headers=resp_headers, background=BackgroundTask(r.aclose))

    return app
