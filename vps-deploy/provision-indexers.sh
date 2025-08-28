#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ $# -lt 1 ]; then
  echo "Usage: $0 path/to/indexers.json"
  exit 1
fi

JSON_FILE=$1
if [ ! -f "$JSON_FILE" ]; then
  echo "File not found: $JSON_FILE"
  exit 1
fi

PROWLARR_KEY=${PROWLARR_API_KEY:-}
if [ -z "$PROWLARR_KEY" ]; then
  # try to read from container config
  if docker ps --format '{{.Names}}' | grep -q '^prowlarr$'; then
    PROWLARR_KEY=$(docker exec prowlarr sh -c 'grep -Eo "<ApiKey>[^<]+</ApiKey>" /config/config.xml 2>/dev/null | sed -E "s/<\/?ApiKey>//g"')
  elif docker ps --format '{{.Names}}' | grep -q '^prowlarr-vpn$'; then
    PROWLARR_KEY=$(docker exec prowlarr-vpn sh -c 'grep -Eo "<ApiKey>[^<]+</ApiKey>" /config/config.xml 2>/dev/null | sed -E "s/<\/?ApiKey>//g"')
  fi
fi

if [ -z "$PROWLARR_KEY" ]; then
  echo "Could not determine Prowlarr API key. Set PROWLARR_API_KEY env or complete initial setup."
  exit 1
fi

PROWLARR_HOST=${PROWLARR_HOST:-http://localhost:9696}

echo "Using Prowlarr: $PROWLARR_HOST"

# Post each indexer in the JSON array
count=0
jq -c '.[]' "$JSON_FILE" | while read -r item; do
  name=$(echo "$item" | jq -r '.name // .implementation')
  echo "Adding indexer: $name"
  curl -sS -X POST \
    -H "X-Api-Key: $PROWLARR_KEY" \
    -H 'Content-Type: application/json' \
    "$PROWLARR_HOST/api/v1/indexer" \
    -d "$item" >/dev/null
  echo "  -> added"
  count=$((count+1)) || true
done

echo "Done. Indexers processed."
