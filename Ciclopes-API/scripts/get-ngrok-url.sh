#!/usr/bin/env bash
set -euo pipefail

container_name="${1:-ciclopes-ngrok}"

body="$(
  docker exec "$container_name" bash -lc '
    exec 3<>/dev/tcp/127.0.0.1/4040
    printf "GET /api/tunnels HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n" >&3
    cat <&3
  ' | awk 'BEGIN { body = 0 } body { print } /^\r?$/ { body = 1 }'
)"

python - <<'PY' "$body"
import json
import sys

payload = json.loads(sys.argv[1])
tunnels = payload.get("tunnels", [])

https_url = next((t["public_url"] for t in tunnels if t.get("proto") == "https"), None)
if https_url:
    print(https_url)
    raise SystemExit(0)

if tunnels:
    print(tunnels[0]["public_url"])
    raise SystemExit(0)

raise SystemExit("No ngrok tunnels found.")
PY
