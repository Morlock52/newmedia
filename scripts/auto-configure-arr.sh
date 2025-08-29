#!/usr/bin/env bash
set -euo pipefail

echo "🔧 Auto-configuring *arr and Prowlarr integrations..."

curl_json() {
  local method=$1 url=$2 body=${3:-}
  if [ -n "$body" ]; then
    curl -sS -X "$method" "$url" -H 'Content-Type: application/json' -d "$body"
  else
    curl -sS -X "$method" "$url"
  fi
}

get_api_key() {
  local svc=$1
  docker exec "$svc" sh -c "grep -o '<ApiKey>[^<]*</ApiKey>' /config/config.xml | sed -e 's#<ApiKey>##' -e 's#</ApiKey>##'" 2>/dev/null || true
}

attempts=30
while [ $attempts -gt 0 ]; do
  SONARR_KEY=$(get_api_key sonarr)
  RADARR_KEY=$(get_api_key radarr)
  PROWLARR_KEY=$(get_api_key prowlarr)
  if [ -n "$SONARR_KEY" ] && [ -n "$RADARR_KEY" ] && [ -n "$PROWLARR_KEY" ]; then
    break
  fi
  attempts=$((attempts-1))
  sleep 2
done

echo "Sonarr key: ${SONARR_KEY:-missing}"
echo "Radarr key: ${RADARR_KEY:-missing}"
echo "Prowlarr key: ${PROWLARR_KEY:-missing}"

if [ -z "$SONARR_KEY" ] || [ -z "$RADARR_KEY" ] || [ -z "$PROWLARR_KEY" ]; then
  echo "⚠️ Could not detect one or more API keys yet. Complete initial setup in the web UIs, then rerun scripts/auto-configure-arr.sh"
  exit 0
fi

SONARR=http://localhost:8989
RADARR=http://localhost:7878
PROWLARR=http://localhost:9696

echo "➡️ Setting Sonarr root folder to /media (if not present)"
EXISTING_SONARR_ROOT=$(curl -sS -H "X-Api-Key: $SONARR_KEY" "$SONARR/api/v3/rootfolder" | jq -r '.[].path' | rg -n "^/media$" -n || true)
if [ -z "$EXISTING_SONARR_ROOT" ]; then
  curl_json POST "$SONARR/api/v3/rootfolder" "{\"path\":\"/media\",\"name\":\"media\"}" -H "X-Api-Key: $SONARR_KEY" >/dev/null || true
fi

echo "➡️ Setting Radarr root folder to /media (if not present)"
EXISTING_RADARR_ROOT=$(curl -sS -H "X-Api-Key: $RADARR_KEY" "$RADARR/api/v3/rootfolder" | jq -r '.[].path' | rg -n "^/media$" -n || true)
if [ -z "$EXISTING_RADARR_ROOT" ]; then
  curl_json POST "$RADARR/api/v3/rootfolder" "{\"path\":\"/media\",\"name\":\"media\"}" -H "X-Api-Key: $RADARR_KEY" >/dev/null || true
fi

echo "➡️ Adding qBittorrent download client to Sonarr"
curl -sS -H "X-Api-Key: $SONARR_KEY" "$SONARR/api/v3/downloadclient" | jq -r '.[].name' | rg -q "qBittorrent" || \
curl -sS -X POST "$SONARR/api/v3/downloadclient" \
  -H 'Content-Type: application/json' -H "X-Api-Key: $SONARR_KEY" \
  -d '{
    "enable": true,
    "protocol": "torrent",
    "priority": 1,
    "name": "qBittorrent",
    "implementation": "QBittorrent",
    "configContract": "QBittorrentSettings",
    "fields": [
      {"name":"host","value":"qbittorrent"},
      {"name":"port","value":8080},
      {"name":"useSsl","value":false},
      {"name":"urlBase","value":""},
      {"name":"username","value":"admin"},
      {"name":"password","value":"adminadmin"},
      {"name":"tvCategory","value":"tv"}
    ]
  }' >/dev/null || true

echo "➡️ Adding qBittorrent download client to Radarr"
curl -sS -H "X-Api-Key: $RADARR_KEY" "$RADARR/api/v3/downloadclient" | jq -r '.[].name' | rg -q "qBittorrent" || \
curl -sS -X POST "$RADARR/api/v3/downloadclient" \
  -H 'Content-Type: application/json' -H "X-Api-Key: $RADARR_KEY" \
  -d '{
    "enable": true,
    "protocol": "torrent",
    "priority": 1,
    "name": "qBittorrent",
    "implementation": "QBittorrent",
    "configContract": "QBittorrentSettings",
    "fields": [
      {"name":"host","value":"qbittorrent"},
      {"name":"port","value":8080},
      {"name":"useSsl","value":false},
      {"name":"urlBase","value":""},
      {"name":"username","value":"admin"},
      {"name":"password","value":"adminadmin"},
      {"name":"movieCategory","value":"movies"}
    ]
  }' >/dev/null || true

echo "➡️ Registering Sonarr & Radarr in Prowlarr"

register_in_prowlarr() {
  local name=$1 url=$2 api_key=$3 impl=$4 contract=$5
  local exists=$(curl -sS -H "X-Api-Key: $PROWLARR_KEY" "$PROWLARR/api/v1/app" | jq -r '.[].name' | rg -n "^'$name'$" -n || true)
  if [ -z "$exists" ]; then
    curl -sS -X POST "$PROWLARR/api/v1/app" \
      -H 'Content-Type: application/json' -H "X-Api-Key: $PROWLARR_KEY" \
      -d "{
        \"name\": \"$name\",
        \"implementation\": \"$impl\",
        \"configContract\": \"$contract\",
        \"syncLevel\": \"fullSync\",
        \"fields\": [
          {\"name\": \"apiKey\", \"value\": \"$api_key\"},
          {\"name\": \"baseUrl\", \"value\": \"\"},
          {\"name\": \"url\", \"value\": \"$url\"}
        ]
      }" >/dev/null || true
  fi
}

register_in_prowlarr "Sonarr" "http://sonarr:8989" "$SONARR_KEY" "Sonarr" "SonarrSettings"
register_in_prowlarr "Radarr" "http://radarr:7878" "$RADARR_KEY" "Radarr" "RadarrSettings"

echo "➡️ Adding public indexers (Nyaa, YTS) to Prowlarr if available"
# Try to add some public indexers if present in schema
SCHEMA=$(curl -sS -H "X-Api-Key: $PROWLARR_KEY" "$PROWLARR/api/v1/indexer/schema" || echo "[]")
add_indexer() {
  local impl=$1 name=$2
  if echo "$SCHEMA" | jq -r '.[].implementation' | rg -q "^$impl$"; then
    curl -sS -X POST "$PROWLARR/api/v1/indexer" -H 'Content-Type: application/json' -H "X-Api-Key: $PROWLARR_KEY" \
      -d "{\"enable\":true,\"name\":\"$name\",\"implementation\":\"$impl\",\"configContract\":\"${name}Settings\",\"priority\":25,\"fields\":[]}" >/dev/null || true
  fi
}

add_indexer "Nyaa" "Nyaa"
add_indexer "Yts" "YTS"

echo "✅ Auto-configuration complete. Review settings in Sonarr, Radarr, and Prowlarr UIs."
