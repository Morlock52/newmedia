# 🚀 Ultimate Media Server 2025

A comprehensive, seedbox-style media server ecosystem with automated management for movies, TV shows, music, audiobooks, e-books, and comics. Features a fun, gamified interface suitable for both newbies and tech enthusiasts.

![Media Server](https://img.shields.io/badge/Media%20Server-2025-brightgreen)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)
![ARM64](https://img.shields.io/badge/ARM64-Compatible-orange)
![License](https://img.shields.io/badge/License-MIT-purple)

## 🎬 Features

### Core Services
- **Jellyfin** - Media streaming server (Plex alternative)
- **Sonarr** - TV show automation
- **Radarr** - Movie automation
- **Lidarr** - Music automation
- **Readarr** - Book/audiobook automation
- **Prowlarr** - Indexer management
- **Overseerr** - Request management

### Additional Services
- **AudioBookshelf** - Audiobook & podcast server
- **Navidrome** - Music streaming (Spotify alternative)
- **Bazarr** - Subtitle management
- **Calibre-Web** - E-book server
- **Tautulli** - Media analytics
- **Uptime Kuma** - Service monitoring
- **qBittorrent** - Download client
- **FlareSolverr** - Cloudflare bypass

## 🎮 Gamified Dashboard

Access the fun dashboard with three modes:
- **👶 Newbie Mode** - Helpful tips and simple interface
- **🤓 Techie Mode** - Technical details and advanced features
- **🎮 Gamer Mode** - Achievements, XP system, and easter eggs

## 🚀 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ultimate-media-server-2025.git
cd ultimate-media-server-2025
```

2. Run the deployment script:
```bash
# For ARM64/Apple Silicon
chmod +x deploy-arm64-media-apps.sh
./deploy-arm64-media-apps.sh
```

3. Access the services:
- **Fun Dashboard**: Open `ultimate-fun-dashboard.html`
- **Homepage**: http://localhost:3001
- **Jellyfin**: http://localhost:8096

## 📊 Service Ports

| Service | Port | Description |
|---------|------|-------------|
| Jellyfin | 8096 | Media Server |
| Sonarr | 8989 | TV Management |
| Radarr | 7878 | Movie Management |
| Lidarr | 8686 | Music Management |
| Readarr | 8787 | Book Management |
| Prowlarr | 9696 | Indexer Manager |
| Overseerr | 5055 | Request Portal |
| qBittorrent | 8080 | Downloads |
| AudioBookshelf | 13378 | Audiobooks |
| Navidrome | 4533 | Music Streaming |
| Homepage | 3001 | Dashboard |

## 📚 Documentation

- [25 Game-Changing Improvements](./ULTIMATE_MEDIA_SERVER_2025_REVIEW_AND_IMPROVEMENTS.md)
- [Deployment Summary](./DEPLOYMENT_COMPLETE_SUMMARY.md)

---

**Created by**: Multi-Agent AI Consensus  
**Last Updated**: August 27, 2025

## 🌐 Publish the Static Site (GitHub Pages)

This repository includes a GitHub Actions workflow that publishes the static site (index.html and other HTML files) to GitHub Pages.

Steps:
1. Push this repository to GitHub.
2. In GitHub, go to Settings -> Pages, and set Source to "GitHub Actions" (if not already).
3. Ensure your default branch is `main` or `master` (the workflow supports both).
4. Push any change to trigger the deployment. After it completes, your site will be available at the repository's Pages URL.

Notes:
- The workflow deploys the repository root as the site. If you prefer a `docs/` folder or a subfolder, update `.github/workflows/deploy-pages.yml` `path` accordingly.
- All links should be relative (they are, in most cases). If you use absolute paths, ensure they resolve on Pages.

## ✅ Markdown Formatting & Pre-commit Hooks

Consistent Markdown formatting is configured via Prettier and markdownlint.

Quick setup:
```bash
pip install pre-commit  # or brew install pre-commit
pre-commit install
# On first run, pre-commit will download hooks
pre-commit run --all-files
```

What’s configured:
- `.pre-commit-config.yaml` with:
  - Trailing whitespace and EOF fixes
  - Prettier for `.md` and `.yml` files
  - markdownlint rules (configured in `.markdownlint.json`)
- `.prettierrc.json` for formatting settings
- CI workflow `.github/workflows/markdown-quality.yml` to enforce checks on PRs


## 🖥️ VPS Docker Deployment

A production-ready Docker Compose setup for a VPS with Traefik + HTTPS is included in `vps-deploy/`.

Quick start on your VPS:
```bash
cd vps-deploy
cp .env.example .env  # edit DOMAIN, ACME_EMAIL, MEDIA_PATH, DOWNLOADS_PATH
docker compose up -d
```

See `vps-deploy/README.md` for details.
