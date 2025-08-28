#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/vps-deploy"

if [ ! -f .env ]; then
  echo "No .env found in vps-deploy. Copying .env.example..."
  cp .env.example .env
  echo "Edit vps-deploy/.env with your DOMAIN, ACME_EMAIL, MEDIA_PATH, DOWNLOADS_PATH, then rerun."
  exit 1
fi

# Ensure acme.json exists inside the named volume by starting traefik once
echo "Starting stack..."
docker compose up -d

echo "Done. Visit Traefik dashboard at: https://traefik.$(grep -E '^DOMAIN=' .env | cut -d= -f2)"
