#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   BASE_URL=http://localhost ./run-jest-in-docker.sh
#   or just ./run-jest-in-docker.sh (defaults BASE_URL to http://host.docker.internal)

export BASE_URL="${BASE_URL:-http://host.docker.internal}"

echo "Running Jest tests in Docker with BASE_URL=$BASE_URL"

docker compose -f docker-compose.jest.yml build jest-runner
docker compose -f docker-compose.jest.yml up --remove-orphans --abort-on-container-exit --exit-code-from jest-runner
