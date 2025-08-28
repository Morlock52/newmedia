# WebUI Test Results

Tests validate the WebUI endpoints and stack interactions.

## Endpoint Test Script
Run the included script:

```bash
bash media-server-stack/test-webui-endpoints.sh
```

![Endpoint Test Output](images/webui-endpoint-tests.png)

Expected endpoints (examples):
- `GET /api/system-info` → system details
- `GET /api/env-status` → `.env` presence and count
- `GET /api/docker-status` → docker availability and version
- `GET /api/status` → compose status
- `GET /api/health` → environment, docker, and services
- `GET /api/logs/all` → combined logs
- `GET /api/logs/:service` → per-service logs

## Health Check

![Health Check](images/health-check.png)

## Compose Status

![Compose Status](images/compose-status.png)

