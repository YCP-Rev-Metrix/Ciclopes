#!/usr/bin/env bash
set -euo pipefail

container_name="${1:-ciclopes-tunnel}"

for i in $(seq 1 30); do
  url=$(docker logs "$container_name" 2>&1 | grep -o 'https://[a-zA-Z0-9-]*\.trycloudflare\.com' | tail -1)
  if [ -n "$url" ]; then
    echo "$url"
    exit 0
  fi
  sleep 1
done

echo "Tunnel URL not found in logs after 30 seconds." >&2
exit 1
