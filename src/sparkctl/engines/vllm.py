"""vLLM engine: single-node docker serve, or multi-node Ray head+worker over the CX7 fabric."""
import json
import sys
import time

from sparkctl import config, remote


def fabric_env():
    fabric = config.FABRIC
    return {
        "NCCL_IB_HCA": fabric["nccl_ib_hca"],
        "NCCL_SOCKET_IFNAME": fabric["nccl_socket_ifname"],
        "NCCL_IB_GID_INDEX": str(fabric["nccl_ib_gid_index"]),
        "NCCL_IB_TIMEOUT": "22",
        "NCCL_DMABUF_ENABLE": "1",
        "GLOO_SOCKET_IFNAME": fabric["nccl_socket_ifname"].split(",")[0],
        "TP_SOCKET_IFNAME": fabric["nccl_socket_ifname"].split(",")[0],
    }


def env_flags(d):
    return " ".join(f"-e {k}={json.dumps(str(v))}" for k, v in d.items())


def svc_cname(name, role=None):
    """Per-service container name so multiple agents can run concurrently across nodes."""
    return f"{config.PFX}-svc-{name}" + (f"-{role}" if role else "")


def _engine_extra_flags(svc):
    """Engine-level flags that shape the VllmConfig and so must match on EVERY rank (the API head
    AND every headless mp worker): trust-remote-code, model implementation, multimodal limits, then
    the free-form extra_args passthrough. Kept separate from the API-server-only flags
    (served-name/host/port/tool+reasoning parsers) so head and worker stay in lockstep."""
    flags = []
    if svc.get("trust_remote_code"):
        flags.append("--trust-remote-code")
    if svc.get("model_impl"):                                  # auto | vllm | transformers
        flags.append(f"--model-impl {svc['model_impl']}")
    for field, cli in (("limit_mm_per_prompt", "--limit-mm-per-prompt"),
                       ("mm_processor_kwargs", "--mm-processor-kwargs"),
                       ("speculative", "--speculative-config")):
        val = svc.get(field)
        if val is not None:
            # JSON must survive ssh -> docker -> bash -lc: wrap in single quotes so bash passes it as
            # one literal arg instead of stripping quotes and brace-expanding {a,b}. We hold these as
            # dicts in YAML, so no hand-escaping in the recipe.
            flags.append(f"{cli}='{json.dumps(val, separators=(',', ':'))}'")
    flags += svc.get("extra_args", [])
    return flags


def _vllm_serve_flags(svc):
    tp = svc.get("parallel", {}).get("tensor", 1)
    pp = svc.get("parallel", {}).get("pipeline", 1)
    flags = [f"--served-model-name {svc.get('served_name', 'model')}",
             f"--host 0.0.0.0 --port {svc.get('port', config.CFG['cluster']['api_port'])}",
             f"--max-model-len {svc['max_model_len']}",
             f"--gpu-memory-utilization {svc['gpu_memory_utilization']}",
             f"--tensor-parallel-size {tp}", f"--pipeline-parallel-size {pp}"]
    if svc.get("tool_parser_plugin"):        # load a custom out-of-tree ToolParser (before selecting it)
        flags += [f"--tool-parser-plugin {svc['tool_parser_plugin']}"]
    if svc.get("tool_call_parser"):
        flags += ["--enable-auto-tool-choice", f"--tool-call-parser {svc['tool_call_parser']}"]
    if svc.get("reasoning_parser"):
        flags += [f"--reasoning-parser {svc['reasoning_parser']}"]
    return " ".join(flags + _engine_extra_flags(svc))


def _worker_engine_flags(svc):
    """Engine-side flags a headless mp worker needs — they must produce the SAME VllmConfig as the
    head (model/parallel/memory/quant/multimodal all have to match across ranks). The API-server
    flags (served-name/host/port/tool+reasoning parsers) are head-only and unused under --headless."""
    tp = svc.get("parallel", {}).get("tensor", 1)
    pp = svc.get("parallel", {}).get("pipeline", 1)
    flags = [f"--max-model-len {svc['max_model_len']}",
             f"--gpu-memory-utilization {svc['gpu_memory_utilization']}",
             f"--tensor-parallel-size {tp}", f"--pipeline-parallel-size {pp}"]
    return " ".join(flags + _engine_extra_flags(svc))


def _docker_common():
    # secrets (sparkctl secret set ...) flow in via --env-file so vLLM authenticates to the HF Hub
    # (gated tokenizers/config fetched at serve time); $() expands on the node. Mirrors ollama.py.
    # --entrypoint bash: our launch commands are `-lc <cmd>`, so force a shell entrypoint regardless
    # of the base image's own ENTRYPOINT — official images (vllm/vllm-openai) set ENTRYPOINT ["vllm"],
    # which would otherwise swallow our command as `vllm bash -lc ...` -> "unrecognized arguments".
    # RDMA: --network host exposes the RoCE netdevs but NOT the verbs devices; without
    # /dev/infiniband NCCL can't open IB and SILENTLY falls back to TCP sockets (found 2026-10-11:
    # RoCE port_xmit_data stayed 0 while TP=2 decode ran ~2x slow over eth). Passed only when the
    # node has it ($() expands on the node, like the secrets file); IPC_LOCK lets verbs pin memory.
    return (f"--network host --gpus all --ipc host --ulimit memlock=-1 "
            f"--ulimit stack=67108864 -d --entrypoint bash "
            f'$(test -d /dev/infiniband && echo --device=/dev/infiniband --cap-add=IPC_LOCK) '
            f'$(test -f "$HOME/.sparkctl/secrets.env" && echo --env-file "$HOME/.sparkctl/secrets.env") '
            f"-v {config.CACHE}:/root/.cache/huggingface")


def vllm_up(svc):
    model = svc["model"]
    tp, pp = svc.get("parallel", {}).get("tensor", 1), svc.get("parallel", {}).get("pipeline", 1)
    world = tp * pp                                    # total ranks == GPUs (1 per Spark node)
    img = svc.get("image", config.IMAGE)               # per-recipe pin; else cluster default
    base_env = {k: str(v) for k, v in (svc.get("env") or {}).items()}
    serve_flags = _vllm_serve_flags(svc)

    if world <= 1:   # single-node: one container runs `vllm serve` directly, no Ray
        node = svc.get("node", config.HEAD)
        serve = f"vllm serve {model} {serve_flags}"
        print(f"[vllm] single-node serve '{svc['name']}' on {node} (image {img})")
        remote.on(node, f"docker run {_docker_common()} --name {svc_cname(svc['name'])} "
                        f"{env_flags(base_env)} {img} -lc {json.dumps(serve)}")
        print(f"[vllm] launched; watch: sparkctl logs {svc['name']}")
        return

    # multinode: TP/PP across HEAD + WORKER over the CX7 fabric.
    #   distributed_backend: ray (default, unchanged) | mp (no-Ray, vLLM-native nnodes/node-rank).
    if not config.FABRIC or config.WORKER is None:
        sys.exit(f"service '{svc['name']}' wants {world} ranks, but this cluster has no "
                 f"fabric:/second node configured — multi-node serving needs both (cluster.yaml)")
    fenv = fabric_env()
    fenv.update(base_env)
    head_ip = config.NODES[config.HEAD]["fabric_ip"]
    worker_ip = config.NODES[config.WORKER]["fabric_ip"]
    backend = svc.get("distributed_backend", "ray")
    up = _mp_multinode_up if backend == "mp" else _ray_multinode_up
    up(svc, model, img, serve_flags, tp, pp, world, fenv, head_ip, worker_ip)


def _ray_multinode_up(svc, model, img, serve_flags, tp, pp, world, fenv, head_ip, worker_ip):
    """Ray head on HEAD + worker on WORKER; the serve is a detached `docker exec` inside a
    sleep-infinity head container (so a crashed serve doesn't tear the container down)."""
    hn, wn = svc_cname(svc['name'], 'head'), svc_cname(svc['name'], 'worker')
    print(f"[vllm] Ray containers for '{svc['name']}' (image {img}, TP={tp} PP={pp}, world={world})")
    remote.on(config.HEAD, f"docker run {_docker_common()} --rm --name {hn} "
                           f"{env_flags({**fenv, 'VLLM_HOST_IP': head_ip})} {img} -lc {json.dumps('sleep infinity')}")
    remote.on(config.WORKER, f"docker run {_docker_common()} --rm --name {wn} "
                             f"{env_flags({**fenv, 'VLLM_HOST_IP': worker_ip})} {img} -lc {json.dumps('sleep infinity')}")
    obj = "--object-store-memory=1073741824"          # Spark OOM fix: shrink Ray plasma store
    ray_head = (f"ray start --head --node-ip-address={head_ip} --port=6379 "
                f"--num-gpus=1 --dashboard-host=0.0.0.0 {obj}")
    ray_worker = f"ray start --address={head_ip}:6379 --node-ip-address={worker_ip} --num-gpus=1 {obj}"
    # NOT detached: a non-zero `ray start` (missing ray / no cluster) surfaces instead of failing
    # silently under the serve. `bash -lc` so the login-shell PATH resolves the ray CLI.
    remote.on(config.HEAD, f"docker exec {hn} bash -lc {json.dumps(ray_head)}")
    time.sleep(5)
    remote.on(config.WORKER, f"docker exec {wn} bash -lc {json.dumps(ray_worker)}")
    time.sleep(5)
    serve = f"vllm serve {model} --distributed-executor-backend ray {serve_flags}"
    print(f"[vllm] launching: {serve}")
    # detached exec output is lost to `docker logs`; tee to a file so `logs` can follow it
    remote.on(config.HEAD, f"docker exec -d {hn} bash -lc {json.dumps(serve + ' > /tmp/vllm-serve.log 2>&1')}")
    print(f"[vllm] launched; watch: sparkctl logs {svc['name']}")


def _mp_multinode_up(svc, model, img, serve_flags, tp, pp, world, fenv, head_ip, worker_ip):
    """No-Ray multinode: vLLM's native mp backend spans the fabric via --nnodes/--node-rank/
    --master-addr. rank-0 (HEAD) runs the API server + its local worker; rank-1 (WORKER) runs
    `--headless` (serve.py routes node_rank>0 to a workers-only MultiprocExecutor). Both containers
    run `vllm serve` as PID 1 — so `docker logs` and container exit codes are accurate, and there's
    no Ray daemon / plasma store / memory-monitor (the unified-memory OOM source the Ray path tunes
    away). 1 GPU/node ⇒ nnodes == world. Weights are already mirrored to both nodes' HF cache."""
    hn, wn = svc_cname(svc['name'], 'head'), svc_cname(svc['name'], 'worker')
    mp = (f"--distributed-executor-backend mp --nnodes {world} "
          f"--master-addr {head_ip} --master-port 29501")   # torch-distributed TCPStore on the fabric
    head_serve = f"vllm serve {model} {mp} --node-rank 0 {serve_flags}"
    worker_serve = f"vllm serve {model} {mp} --node-rank 1 --headless {_worker_engine_flags(svc)}"
    print(f"[vllm] mp (no-Ray) multinode for '{svc['name']}' (image {img}, TP={tp} PP={pp}, world={world})")
    # HEAD first: rank 0 binds the rendezvous store the worker connects back to. No --rm, so a crashed
    # container survives for `docker logs` (the crash report reads it); clear any stale one first.
    remote.on(config.HEAD,
              f"docker rm -f {hn} >/dev/null 2>&1 || true; "
              f"docker run {_docker_common()} --name {hn} "
              f"{env_flags({**fenv, 'VLLM_HOST_IP': head_ip})} {img} -lc {json.dumps(head_serve)}")
    time.sleep(3)
    remote.on(config.WORKER,
              f"docker rm -f {wn} >/dev/null 2>&1 || true; "
              f"docker run {_docker_common()} --name {wn} "
              f"{env_flags({**fenv, 'VLLM_HOST_IP': worker_ip})} {img} -lc {json.dumps(worker_serve)}")
    print(f"[vllm] launched; watch: sparkctl logs {svc['name']}")


def multinode_serve_alive(cname):
    """True while the exec'd `vllm serve` is running inside a Ray head container. The container
    itself runs `sleep infinity`, so docker's Up status can't distinguish a live serve from a
    crashed one — this can."""
    return remote.on(config.HEAD, f"docker exec {cname} pgrep -f 'vllm serve' >/dev/null 2>&1",
                     check=False, capture=True).returncode == 0


def vllm_down():
    for node in config.NODES:  # remove serving containers everywhere; leave downloads (dl-*) alone
        remote.on(node, f"docker ps -aq --filter name={config.PFX}-svc- | xargs -r docker rm -f",
                  check=False)
