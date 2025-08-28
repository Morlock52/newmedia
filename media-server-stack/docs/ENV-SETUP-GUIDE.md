# Environment Setup Guide

This guide walks through generating a secure `.env` using either the WebUI or the CLI tool, with screenshots.

## Option A: WebUI
1. Open the Setup tab.
2. Fill in domain, email, and select your VPN provider and type.
3. Click Validate, then Generate Environment.

![WebUI Generate Environment](images/webui-setup-generate-env.png)

What happens:
- `.env` is created in the stack directory
- A timestamped backup is created (e.g., `.env.backup.2025-07-20T18-10-00`)
- Sensitive fields are left as placeholders or routed to Docker secrets

## Option B: CLI (CUI)
Run the included CLI tool for an interactive terminal experience.

```bash
node media-server-stack/setup-env.js         # Interactive
node media-server-stack/setup-env.js -n      # Non-interactive (defaults)
node media-server-stack/setup-env.js -v      # Validate-only
node media-server-stack/setup-env.js -d      # Dry run
```

![CLI Setup](images/cli-setup.png)

## Validating Configuration
Use either:
- WebUI Management → Status/Health
- CLI validate: `node media-server-stack/setup-env.js --validate-only`

![Health Check](images/health-check.png)

## Deploying Services
From the WebUI Management tab:

![WebUI Deploy](images/webui-deploy.png)

Or via CLI:

```bash
cd media-server-stack
docker compose up -d
```

## Secrets
Do not commit real secrets. Place secrets in `secrets/` as files (e.g., `traefik_dashboard_auth.txt`, `wg_private_key.txt`).

![Secrets Directory](images/secrets-directory.png)

