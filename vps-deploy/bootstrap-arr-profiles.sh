#!/usr/bin/env bash
set -euo pipefail

# Auto-create Sonarr/Radarr root folders and clone a quality profile
# Requires jq installed on the VPS. If not present, prints guidance.

command -v jq >/dev/null 2>&1 || { echo "jq is required for this script. Install with: sudo apt-get install -y jq || brew install jq"; exit 1; }

get_api_key() {
  local cname=$1
  docker exec "$cname" sh -c 'grep -Eo "<ApiKey>[^<]+</ApiKey>" /config/config.xml 2>/dev/null | sed -E "s/<\/?ApiKey>//g"' || true
}

SONARR_KEY=$(get_api_key sonarr)
RADARR_KEY=$(get_api_key radarr)

[ -n "$SONARR_KEY" ] || { echo "Missing Sonarr API key. Start containers and complete setup first."; exit 1; }
[ -n "$RADARR_KEY" ] || { echo "Missing Radarr API key. Start containers and complete setup first."; exit 1; }

# Create root folders if missing
create_root_folder() {
  local app=$1 host=$2 key=$3 path=$4
  echo "Ensuring root folder $path exists in $app..."
  existing=$(curl -sS -H "X-Api-Key: $key" "$host/api/v3/rootfolder" | jq -r '.[].path')
  if echo "$existing" | grep -qx "$path"; then
    echo "  -> already exists"
  else
    curl -sS -X POST -H "X-Api-Key: $key" -H 'Content-Type: application/json' \
      "$host/api/v3/rootfolder" \
      -d "{\"path\": \"$path\"}" >/dev/null || echo "  -> failed to create root folder (try manually in UI)"
  fi
}

create_root_folder "Sonarr" http://sonarr:8989 "$SONARR_KEY" "/tv"
create_root_folder "Radarr" http://radarr:7878 "$RADARR_KEY" "/movies"

# Clone default quality profile to HD-1080p (best-effort)
clone_profile() {
  local app=$1 host=$2 key=$3 newname=$4 cutoff=$5
  echo "Cloning $app quality profile to '$newname' with cutoff '$cutoff' (best-effort)..."
  profiles=$(curl -sS -H "X-Api-Key: $key" "$host/api/v3/qualityprofile")
  if [ -z "$profiles" ] || [ "$profiles" = "null" ]; then
    echo "  -> failed to fetch profiles"
    return 0
  fi
  tmpl=$(echo "$profiles" | jq '.[0] // empty')
  if [ -z "$tmpl" ]; then
    echo "  -> no template profile found"
    return 0
  fi
  # Set name and cutoff if present
  body=$(echo "$tmpl" | jq --arg name "$newname" '.name=$name')
  # If desired cutoff quality exists, set it; else keep original
  if echo "$body" | jq -e --arg c "$cutoff" '.items[].quality.name == $c' >/dev/null 2>&1; then
    body=$(echo "$body" | jq --arg c "$cutoff" '.cutoff = (.items[] | select(.quality.name==$c) | .quality.id)')
  fi
  # Remove id to create new profile
  body=$(echo "$body" | jq 'del(.id)')
  curl -sS -X POST -H "X-Api-Key: $key" -H 'Content-Type: application/json' \
    "$host/api/v3/qualityprofile" -d "$body" >/dev/null || echo "  -> failed to create profile (API may differ)."
}

clone_profile "Sonarr" http://sonarr:8989 "$SONARR_KEY" "HD-1080p" "HD-1080p"
clone_profile "Radarr" http://radarr:7878 "$RADARR_KEY" "HD-1080p" "HD-1080p"


# Also create 720p and 2160p profiles (best-effort)
clone_profile "Sonarr" http://sonarr:8989 "$SONARR_KEY" "HD-720p" "HD-720p"
clone_profile "Radarr" http://radarr:7878 "$RADARR_KEY" "HD-720p" "HD-720p"
clone_profile "Sonarr" http://sonarr:8989 "$SONARR_KEY" "UHD-2160p" "UHD-2160p"
clone_profile "Radarr" http://radarr:7878 "$RADARR_KEY" "UHD-2160p" "UHD-2160p"

# Attempt to set root folder default quality profile to 1080p if supported
set_rootfolder_default_profile() {
  local app=$1 host=$2 key=$3 path=$4 profname=$5
  profs=$(curl -sS -H "X-Api-Key: $key" "$host/api/v3/qualityprofile")
  prof_id=$(echo "$profs" | jq -r ".[] | select(.name=="$profname").id" | head -n1)
  [ -n "$prof_id" ] || return 0
  rf=$(curl -sS -H "X-Api-Key: $key" "$host/api/v3/rootfolder")
  rf_id=$(echo "$rf" | jq -r ".[] | select(.path=="$path").id")
  if [ -n "$rf_id" ] && echo "$rf" | jq -e ".[] | select(.id==$rf_id) | has("defaultQualityProfileId")" >/dev/null; then
    body=$(echo "$rf" | jq ".[] | select(.id==$rf_id) | .defaultQualityProfileId=$prof_id")
    curl -sS -X PUT -H "X-Api-Key: $key" -H 'Content-Type: application/json'       "$host/api/v3/rootfolder/$rf_id" -d "$body" >/dev/null || true
  fi
}

set_rootfolder_default_profile "Sonarr" http://sonarr:8989 "$SONARR_KEY" "/tv" "HD-1080p"
set_rootfolder_default_profile "Radarr" http://radarr:7878 "$RADARR_KEY" "/movies" "HD-1080p"

echo "Profiles/root folders provisioning done."
