#!/usr/bin/env bash
set -euo pipefail

# Configure download clients for Sonarr/Radarr and prepare categories
# - Adds qBittorrent and SABnzbd to Sonarr/Radarr
# - Creates qBittorrent categories and sets save paths
# - Creates SABnzbd categories

QBIT_HOST=${QBIT_HOST:-http://qbittorrent:8080}
QBIT_USER=${QBITTORRENT_USERNAME:-admin}
QBIT_PASS=${QBITTORRENT_PASSWORD:-adminadmin}
SONARR_CAT=${SONARR_CATEGORY:-tv}
RADARR_CAT=${RADARR_CATEGORY:-movies}

SAB_HOST=${SAB_HOST:-http://sabnzbd:8080}
SAB_CAT_TV=${SAB_CATEGORY_TV:-tv}
SAB_CAT_MOVIES=${SAB_CATEGORY_MOVIES:-movies}

get_api_key() {
  local cname=$1
  docker exec "$cname" sh -c 'grep -Eo "<ApiKey>[^<]+</ApiKey>" /config/config.xml 2>/dev/null | sed -E "s/<\/?ApiKey>//g"' || true
}

# Extract API keys
SONARR_KEY=$(get_api_key sonarr)
RADARR_KEY=$(get_api_key radarr)
SAB_API_KEY=$(docker exec sabnzbd sh -c "grep -E 'api_key *= *' /config/sabnzbd.ini 2>/dev/null | tail -n1 | awk -F'= ' '{print $2}'" || true)

if [[ -z "$SONARR_KEY" || -z "$RADARR_KEY" ]]; then
  echo "Could not read Sonarr/Radarr API keys. Start containers and complete initial setup first." >&2
  exit 1
fi

add_qbittorrent_sonarr() {
  curl -sS -X POST \
    -H "X-Api-Key: $SONARR_KEY" \
    -H 'Content-Type: application/json' \
    http://sonarr:8989/api/v3/downloadclient \
    -d "{\
      \"name\": \"qBittorrent\",\
      \"enable\": true,\
      \"protocol\": \"torrent\",\
      \"priority\": 1,\
      \"removeCompletedDownloads\": true,\
      \"removeFailedDownloads\": true,\
      \"configContract\": \"QBittorrentSettings\",\
      \"implementation\": \"QBittorrent\",\
      \"fields\": [\
        {\"name\": \"host\", \"value\": \"$QBIT_HOST\"},\
        {\"name\": \"username\", \"value\": \"$QBIT_USER\"},\
        {\"name\": \"password\", \"value\": \"$QBIT_PASS\"},\
        {\"name\": \"category\", \"value\": \"$SONARR_CAT\"},\
        {\"name\": \"ssl\", \"value\": false}\
      ]\
    }" >/dev/null || true
}

add_qbittorrent_radarr() {
  curl -sS -X POST \
    -H "X-Api-Key: $RADARR_KEY" \
    -H 'Content-Type: application/json' \
    http://radarr:7878/api/v3/downloadclient \
    -d "{\
      \"name\": \"qBittorrent\",\
      \"enable\": true,\
      \"protocol\": \"torrent\",\
      \"priority\": 1,\
      \"removeCompletedDownloads\": true,\
      \"removeFailedDownloads\": true,\
      \"configContract\": \"QBittorrentSettings\",\
      \"implementation\": \"QBittorrent\",\
      \"fields\": [\
        {\"name\": \"host\", \"value\": \"$QBIT_HOST\"},\
        {\"name\": \"username\", \"value\": \"$QBIT_USER\"},\
        {\"name\": \"password\", \"value\": \"$QBIT_PASS\"},\
        {\"name\": \"category\", \"value\": \"$RADARR_CAT\"},\
        {\"name\": \"ssl\", \"value\": false}\
      ]\
    }" >/dev/null || true
}

# qBittorrent category creation and save paths
qb_login() {
  curl -sS -c /tmp/qb_cookies.txt -d "username=$QBIT_USER&password=$QBIT_PASS" "$QBIT_HOST/api/v2/auth/login" >/dev/null || true
}

qb_add_category() {
  local name=$1 path=$2
  curl -sS -b /tmp/qb_cookies.txt -d "category=$name&savePath=$path" "$QBIT_HOST/api/v2/torrents/createCategory" >/dev/null || true
}

qb_set_preferences() {
  local json=$1
  curl -sS -b /tmp/qb_cookies.txt -d "json=$json" "$QBIT_HOST/api/v2/app/setPreferences" >/dev/null || true
}

# SABnzbd
sab_add_category() {
  local name=$1 pp=${2:-0}
  curl -sS "$SAB_HOST/api?mode=add_category&name=$name&pp=$pp&apikey=$SAB_API_KEY" >/dev/null || true
}

add_sab_sonarr() {
  curl -sS -X POST \
    -H "X-Api-Key: $SONARR_KEY" \
    -H 'Content-Type: application/json' \
    http://sonarr:8989/api/v3/downloadclient \
    -d "{\
      \"name\": \"SABnzbd\",\
      \"enable\": true,\
      \"protocol\": \"usenet\",\
      \"priority\": 2,\
      \"configContract\": \"SABnzbdSettings\",\
      \"implementation\": \"SABnzbd\",\
      \"fields\": [\
        {\"name\": \"host\", \"value\": \"$SAB_HOST\"},\
        {\"name\": \"apiKey\", \"value\": \"$SAB_API_KEY\"},\
        {\"name\": \"category\", \"value\": \"$SONARR_CAT\"}\
      ]\
    }" >/dev/null || true
}

add_sab_radarr() {
  curl -sS -X POST \
    -H "X-Api-Key: $RADARR_KEY" \
    -H 'Content-Type: application/json' \
    http://radarr:7878/api/v3/downloadclient \
    -d "{\
      \"name\": \"SABnzbd\",\
      \"enable\": true,\
      \"protocol\": \"usenet\",\
      \"priority\": 2,\
      \"configContract\": \"SABnzbdSettings\",\
      \"implementation\": \"SABnzbd\",\
      \"fields\": [\
        {\"name\": \"host\", \"value\": \"$SAB_HOST\"},\
        {\"name\": \"apiKey\", \"value\": \"$SAB_API_KEY\"},\
        {\"name\": \"category\", \"value\": \"$RADARR_CAT\"}\
      ]\
    }" >/dev/null || true
}

# Begin
echo "Configuring qBittorrent in Sonarr (category=$SONARR_CAT)..."
add_qbittorrent_sonarr || echo "Sonarr configuration may require manual adjustment."

echo "Configuring qBittorrent in Radarr (category=$RADARR_CAT)..."
add_qbittorrent_radarr || echo "Radarr configuration may require manual adjustment."

echo "Creating qBittorrent categories and save paths..."
qb_login || true
qb_add_category "$SONARR_CAT" "/downloads/complete/$SONARR_CAT" || true
qb_add_category "$RADARR_CAT" "/downloads/complete/$RADARR_CAT" || true
qb_set_preferences "{\"save_path\": \"/downloads/complete\", \"create_subfolder_enabled\": true}" || true

if [[ -n "$SAB_API_KEY" ]]; then
  echo "Configuring SABnzbd categories..."
  sab_add_category "$SONARR_CAT" 0 || true
  sab_add_category "$RADARR_CAT" 0 || true
  echo "Setting SABnzbd download directories..."
  curl -sS "$SAB_HOST/api?mode=set_config&section=misc&keyword=download_dir&value=/incomplete-downloads&apikey=$SAB_API_KEY" >/dev/null || true
  curl -sS "$SAB_HOST/api?mode=set_config&section=misc&keyword=complete_dir&value=/downloads/complete&apikey=$SAB_API_KEY" >/dev/null || true
  echo "Adding SABnzbd to Sonarr/Radarr..."
  add_sab_sonarr || true
  add_sab_radarr || true
else
  echo "SABnzbd API key not found; skipping SAB configuration (complete initial setup first)."
fi

echo "Done."
