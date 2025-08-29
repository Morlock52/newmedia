# AGENTS.md — Project guide for AI agents and contributors

This repository contains a comprehensive, Docker‑Compose driven media server stack with AI add‑ons and supporting scripts. This document gives AI coding agents and human contributors the essential context to work safely and effectively in this codebase.

Scope
- Orchestrates 30+ services: media servers (Jellyfin/Plex/Emby), the *arr suite (Sonarr/Radarr/etc.), download clients (qBittorrent/SABnzbd), dashboards, monitoring, and more.
- Includes automation and deployment scripts, health checks, and demo dashboards.
- Provides optional AI/assistant components and MCP integrations.

Quick map of important areas
- Docker Compose: docker-compose*.yml (multiple variants for different scenarios)
- Env templates: .env.example, .env.template, .env.production.template, .env.fixed.template
- API and Node services: api/, middleware/, performance-optimized-api.js, social-media-integration.js
- Dashboards and demos: dashboard/, ai-dashboard.html, dashboard*.html, AI_CENTRAL_INDEX.html
- AI/MCP: ai-services/, ai-models/, archon-*/ and mcp configs (.mcp.json*)
- Voice client demo: voice-ai-system/
- Health and testing scripts: comprehensive-test-suite.sh, demo-health-tests.sh, check-status.sh, test-production-system.sh
- Deployment entry points: deploy-*.sh (multiple presets), deploy-now.sh (quick add‑on)
- Configuration and data roots: config/, data/, logs/, media/, downloads/ (paths controlled via .env)

Profiles and default workflow
- Default (recommended for agents): Fixed profile
  - Compose: docker-compose.fixed.yml
  - Env template: .env.fixed.template
  - Why: referenced by comprehensive-test-suite.sh and other automation; safest baseline for CI/QA.
  - Setup
    1) cp .env.fixed.template .env
    2) docker-compose -f docker-compose.fixed.yml config
    3) docker-compose -f docker-compose.fixed.yml up -d

- Ultimate Single Container (2025)
  - Compose: docker-compose.ultimate-single-container-2025.yml
  - Fixed variant: docker-compose.ultimate-single-container-2025-fixed.yml (with matching deploy script)
  - Env templates: .env.ultimate-single-container-2025.template or .env.ultimate-single-container-2025-fixed.template
  - Setup (example)
    1) cp .env.ultimate-single-container-2025.template .env
    2) docker-compose -f docker-compose.ultimate-single-container-2025.yml pull
    3) docker-compose -f docker-compose.ultimate-single-container-2025.yml up -d

- Ultimate/Multi-service variants
  - Compose: docker-compose.ultimate.yml, docker-compose.ultimate-2025.yml, docker-compose.yml (baseline)
  - Choose based on README instructions for your target stack; keep ports consistent with dashboard/docs.

- macOS optimized path
  - Script: ./deploy-macos-optimized.sh (handles common macOS Docker quirks)

Minimal run (generic)
- Copy env: cp .env.example .env (or a profile-specific template above), then adjust to your system. Do not commit secrets.
- Start core services (example): docker-compose up -d jellyfin sonarr radarr prowlarr qbittorrent
- Full stack: docker-compose up -d (or choose a specific compose variant)
- Quick enhancement (dashboard + AI): ./deploy-now.sh

Validate and test
- Compose validation: docker-compose -f docker-compose.fixed.yml config
- Health checks: ./check-status.sh or ./demo-health-tests.sh
- Comprehensive tests: ./comprehensive-test-suite.sh (generates a timestamped report)
- API quick check (if relevant): see api/ and run node-based scripts or docker-compose.jest.yml if present

Conventions and safety rules for agents
- Use rg for repository search instead of grep/ls -R (respects .gitignore and is faster): rg -n "pattern"
- Prefer minimal, targeted changes aligned with existing patterns; avoid refactors unless requested.
- Validate Docker changes with docker-compose config before suggesting a full up -d.
- Never commit secrets. Use .env.* templates; mention required variables in docs, not values.
- Don’t modify backup or snapshot directories unless explicitly asked (e.g., .backup-*, homarr-configs.backup/, backups/).
- Keep OS‑specific assumptions out of scripts unless the file is already OS‑scoped. Favor POSIX sh/bash patterns already used here.
- When adding services: update the appropriate docker-compose.*.yml variant, associated scripts, and documentation if user‑facing.
- When touching API/Node code: match file/module style; ensure server starts locally; check middleware/error handling paths in middleware/ and api/middleware/.
- If you change ports, reflect them in docs that list access points (e.g., README.md, dashboards) if they become outdated.

Common tasks for agents
- Add an environment variable to a service
  1) Edit the correct compose file (choose the flavor the user is working with).
  2) Validate with docker-compose config.
  3) Update .env.template if it’s a new variable users must provide.
- Add a lightweight API route
  1) Locate api/server.js (or relevant service file) and follow adjacent patterns.
  2) Run locally (node or docker) and smoke test with curl.
- Tweak download paths across services
  1) Update volume mounts and envs in compose files to keep /downloads consistent.
  2) Re-run configure-download-clients*.sh if applicable.
- Update dashboards/demos
  1) Edit the specific html or dashboard project; keep URLs/ports consistent with compose.

Notable entry points and references
- README.md (Quick Start, service matrix)
- DEPLOYMENT_GUIDE.md and various DEPLOYMENT_* summaries
- ARR_INTEGRATION_* docs for *arr suite behavior
- UNIFIED_MCP_SERVER_COMPLETE.md and MCP_CONNECTION_GUIDE.md for MCP interfaces

Style and tooling
- Shell: follow the existing echo/color patterns and safety flags set -e when appropriate.
- JS/Node: adhere to existing module structure; prefer existing logger patterns if present.
- Pre-commit hooks: if available, run on edited files: pre-commit run --files <changed_files>
- Formatting: project has .prettierrc.json and .markdownlint.json; keep markdown headings and lists tidy.

Acceptance checklist for changes
- Search impact: rg your change to ensure all references are updated.
- Compose validation passes; services still build/start where relevant.
- Scripts are executable when intended (chmod +x) and don’t echo secrets.
- Docs updated if behavior/ports/envs changed.

Troubleshooting tips
- If docker-compose uses hyphen vs. docker compose: both are used in docs/scripts; prefer docker-compose for parity with tests, unless the user environment dictates docker compose.
- If pre-commit fails on untouched lines, it may be pre-existing; only fix issues on lines you modify.

Notes for Codex CLI users
- Codex automatically includes this AGENTS.md in context. You can add another doc with --project-doc <file> or disable with --no-project-doc.
- Provide clear, minimal diffs; validate with local commands before finalizing.
