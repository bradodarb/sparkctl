#!/bin/sh
# memguard — kill sparkctl serve containers before unified-memory exhaustion hard-locks a GB10 node.
#
# On DGX Spark, GPU allocations and host RAM share one pool, and running it dry does NOT end in a
# clean kernel OOM-kill: the NVIDIA driver fails allocations (NVRM NV_ERR_NO_MEMORY), tasks wedge in
# D-state, and the node needs a power cycle (2026-10-11, Qwen3.8-Flash-Next Engram cpu_offload).
# This polls the host's MemAvailable (a container's /proc/meminfo is the host's) and `docker kill`s
# every {PFX}-svc-* container once it drops below the floor — a crashed deployment instead of a dead
# node. Only serve containers are touched (never ollama, downloads, or this guard).
#
# Runs as a tiny always-restarting container (no sudo needed), launched on each node with:
#   docker run -d --name csv-memguard --restart unless-stopped --oom-score-adj -1000 \
#     -v /var/run/docker.sock:/var/run/docker.sock -v /opt/sparkctl/docker:/g:ro \
#     -e MEMGUARD_MIN_AVAIL_MB=6144 -e MEMGUARD_PREFIX=csv-svc- docker:cli sh /g/memguard.sh
# Kills are logged: `docker logs csv-memguard`.
MIN_KB=$(( ${MEMGUARD_MIN_AVAIL_MB:-6144} * 1024 ))
PREFIX=${MEMGUARD_PREFIX:-csv-svc-}
INTERVAL=${MEMGUARD_INTERVAL_S:-1}

echo "$(date -u +%FT%TZ) memguard: floor ${MIN_KB}kB MemAvailable, target ${PREFIX}*, every ${INTERVAL}s"
while true; do
  avail=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  if [ -n "$avail" ] && [ "$avail" -lt "$MIN_KB" ]; then
    names=$(docker ps --filter "name=${PREFIX}" --format '{{.Names}}' | tr '\n' ' ')
    if [ -n "$names" ]; then
      echo "$(date -u +%FT%TZ) memguard: MemAvailable ${avail}kB < ${MIN_KB}kB — killing: ${names}"
      docker kill $names >/dev/null 2>&1
    fi
  fi
  sleep "$INTERVAL"
done
