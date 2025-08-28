#!/usr/bin/env bash
set -euo pipefail

PORT="${1:-8080}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

echo "Serving ${ROOT_DIR} at http://localhost:${PORT}/"
cd "${ROOT_DIR}"

if command -v python3 >/dev/null 2>&1; then
  exec python3 -m http.server "${PORT}"
elif command -v python >/dev/null 2>&1; then
  exec python -m SimpleHTTPServer "${PORT}"
else
  echo "Python is required to run a simple HTTP server." >&2
  echo "Install Python 3 and try again." >&2
  exit 1
fi

