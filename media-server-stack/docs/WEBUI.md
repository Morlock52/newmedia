# WebUI Guide

This guide shows the WebUI screens and explains how to use them to configure and manage the stack.

> Tip: The WebUI is exposed via Traefik at `https://setup.<your-domain>` and protected with basic auth. Update `secrets/traefik_dashboard_auth.txt` to change credentials.

## Setup Wizard
The Setup tab detects sensible defaults (PUID/PGID/TZ) and lets you enter domain, email, and VPN settings. It can generate a secure `.env` with one click.

![WebUI Setup](images/webui-setup.png)

Actions:
- Validate: Checks domain/email format and required fields
- Generate Environment: Writes `.env` and a timestamped backup in the stack directory

## Management
Start/stop/restart/deploy the entire stack and inspect status.

![WebUI Management](images/webui-management.png)

Features:
- Start: `docker compose up -d`
- Stop: `docker compose down`
- Deploy: Clean, pull, and bring up services
- Status: Shows current compose state and counts running services

## Monitoring
Lightweight, live status and quick links to the primary apps.

![WebUI Monitoring](images/webui-monitoring.png)

## Logs
Tail per-service logs or download combined logs for quick diagnosis.

![WebUI Logs](images/webui-logs.png)

Endpoints used:
- Per-service: `GET /api/logs/:service?lines=200`
- Combined: `GET /api/logs/all?lines=500&download=true`

## Traefik Dashboard
You can optionally access the Traefik dashboard via `https://traefik.<your-domain>` (requires basic auth).

![Traefik Dashboard](images/traefik-dashboard.png)

## Troubleshooting
- WebUI unreachable: confirm Traefik is up and DNS points to your proxy/tunnel.
- Docker not detected: the WebUI container must mount `/var/run/docker.sock` and include docker + compose (already configured).
- Permissions: verify PUID/PGID and data/config directories.

