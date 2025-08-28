# WebUI Verification

This verifies the WebUI functionality and presents visual confirmations.

## Status
- WebUI reachable via Traefik and protected with basic auth
- Setup wizard generates `.env` and backups
- Management actions (start/stop/deploy) operate via Docker
- Logs available per-service and combined

## Screenshots
- Setup: ![WebUI Setup](docs/images/webui-setup.png)
- Generate Environment: ![Generate Environment](docs/images/webui-setup-generate-env.png)
- Management: ![WebUI Management](docs/images/webui-management.png)
- Monitoring: ![WebUI Monitoring](docs/images/webui-monitoring.png)
- Logs: ![WebUI Logs](docs/images/webui-logs.png)
- Health: ![Health Check](docs/images/health-check.png)
- Compose: ![Compose Status](docs/images/compose-status.png)
- Endpoint Tests: ![Endpoint Test Output](docs/images/webui-endpoint-tests.png)

## Notes
- Basic auth credentials are sourced from `secrets/traefik_dashboard_auth.txt`.
- For SSO, replace basic auth middleware with Authelia forward-auth and include Authelia in the compose.

