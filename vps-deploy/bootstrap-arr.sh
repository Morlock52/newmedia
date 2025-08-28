#!/usr/bin/env bash
set -euo pipefail

RED="\033[0;31m"; GREEN="\033[0;32m"; YELLOW="\033[1;33m"; NC="\033[0m"

info() { echo -e "${YELLOW}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[OK]${NC} $*"; }
err()  { echo -e "${RED}[ERR]${NC} $*"; }

command -v docker >/dev/null 2>&1 || { err "Missing command: docker"; exit 1; }

get_api_key() {
  local cname=$1
  docker exec "$cname" sh -c 'grep -Eo "<ApiKey>[^<]+</ApiKey>" /config/config.xml 2>/dev/null | sed -E "s/<\/?ApiKey>//g"' || true
}

get_prowlarr_key() {
  docker exec prowlarr sh -c 'grep -Eo "<ApiKey>[^<]+</ApiKey>" /config/config.xml 2>/dev/null | sed -E "s/<\/?ApiKey>//g"' || true
}

SONARR_KEY=$(get_api_key sonarr)
RADARR_KEY=$(get_api_key radarr)
PROWLARR_KEY=$(get_prowlarr_key)

if [[ -z "$SONARR_KEY" || -z "$RADARR_KEY" || -z "$PROWLARR_KEY" ]]; then
  err "Could not read one or more API keys. Ensure containers are started and initial setup is complete."
  echo "Sonarr key: ${SONARR_KEY:-<missing>}"
  echo "Radarr key: ${RADARR_KEY:-<missing>}"
  echo "Prowlarr key: ${PROWLARR_KEY:-<missing>}"
  exit 1
fi

ok "API keys acquired"

PROWLARR_URL="http://prowlarr:9696"
HDR=( -H "X-Api-Key: $PROWLARR_KEY" -H 'Content-Type: application/json' )

add_app() {
  local name=$1 baseurl=$2 apikey=$3
  local payload
  payload=$(cat <<JSON
{
  "name": "$name",
  "implementation": "$name",
  "configContract": "${name}Settings",
  "syncLevel": "full",
  "fields": [
    {"name": "baseUrl", "value": "$baseurl"},
    {"name": "apiKey", "value": "$apikey"}
  ]
}
JSON
)

  curl -sS -X POST "$PROWLARR_URL/api/v1/applications" "${HDR[@]}" -d "$payload" | sed 's/.*/&/' >/dev/null 2>&1 || return 1
}

info "Registering Sonarr and Radarr in Prowlarr..."

if add_app Sonarr "http://sonarr:8989" "$SONARR_KEY"; then
  ok "Sonarr added to Prowlarr"
else
  err "Failed to add Sonarr to Prowlarr. Check API compatibility."
fi

if add_app Radarr "http://radarr:7878" "$RADARR_KEY"; then
  ok "Radarr added to Prowlarr"
else
  err "Failed to add Radarr to Prowlarr. Check API compatibility."
fi

info "Done. If any step failed, configure manually with these keys:"
printf "Sonarr API Key: %s\n" "$SONARR_KEY"
printf "Radarr API Key: %s\n" "$RADARR_KEY"
printf "Prowlarr API Key: %s\n" "$PROWLARR_KEY"
