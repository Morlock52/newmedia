# Media Server Stack

Modern self-hosted media stack with Traefik, Jellyfin, Sonarr, Radarr, Prowlarr, Overseerr, qBittorrent (behind VPN), and a WebUI for setup and management.

This README includes visual walkthroughs to accelerate setup.

## Quick Links
- WebUI Guide: `media-server-stack/docs/WEBUI.md`
- Environment Setup Guide: `media-server-stack/docs/ENV-SETUP-GUIDE.md`
- WebUI Test Results: `media-server-stack/docs/WEBUI-TEST-RESULTS.md`
- CUI (CLI Setup) Summary: `media-server-stack/docs/CUI-IMPLEMENTATION-SUMMARY.md`
- CUI Test Results: `media-server-stack/docs/CUI-TEST-RESULTS.md`

## Architecture Overview
![Architecture Overview](docs/images/architecture.png)

Key points:
- Traefik reverse proxy fronts all services and the WebUI
- Cloudflare Tunnel optional for external access
- qBittorrent isolated behind a Gluetun VPN gateway
- WebUI manages `docker compose` lifecycle and `.env` generation

## WebUI Screenshots
The WebUI provides four primary tabs covering end-to-end setup.

- Setup Wizard: ![WebUI Setup](docs/images/webui-setup.png)
- Management: ![WebUI Management](docs/images/webui-management.png)
- Monitoring: ![WebUI Monitoring](docs/images/webui-monitoring.png)
- Logs: ![WebUI Logs](docs/images/webui-logs.png)

## Getting Started
1. Copy or generate `.env` (via WebUI or `node media-server-stack/setup-env.js`).
2. Start the WebUI: `docker compose -f media-server-stack/docker-compose.yml up -d webui`.
3. Access the WebUI via Traefik (recommended): `https://setup.<your-domain>`.
4. Complete the Setup wizard and Deploy.

## Security Notes
- The WebUI is now only exposed via Traefik (no direct host port) and protected with basic auth by default. Update `secrets/traefik_dashboard_auth.txt` accordingly.
- Do not commit real secrets; keep them in `secrets/` and/or Docker secrets.

