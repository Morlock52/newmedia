# VPS Deployment (Docker + Traefik)

This setup runs the core media stack on your VPS behind Traefik with automatic HTTPS.

Included services:
- Traefik (reverse proxy + TLS, dashboard protected by basic auth)
- Jellyfin
- Prowlarr
- Sonarr
- Radarr
- Bazarr
- Lidarr
- Readarr
- qBittorrent routed through Gluetun VPN (peer ports 6881 TCP/UDP exposed on host)
 - Overseerr (request management)
 - Calibre-Web (ebooks)
 - Navidrome (music streaming)
 - Audiobookshelf (audiobooks)
 - Jellyseerr (request portal) [optional]
 - Tautulli (monitoring)
 - SABnzbd (Usenet)
 - Uptime Kuma (service monitoring)
 - Prometheus (metrics)
 - Grafana (dashboards)
 - Jellyseerr (request portal)
 - Tautulli (monitoring)

## Prerequisites
- A VPS with Docker and Docker Compose installed
- A domain with DNS A/AAAA records pointing to your VPS IP
- Open ports: 80 and 443 (HTTP/HTTPS)

Optional:
- Cloudflare DNS API token for wildcard certs (set TRAEFIK_CERT_RESOLVER=cloudflare)
- VPN credentials for your provider (for Gluetun)

## Setup

1) Copy env file and edit values
```bash
cd vps-deploy
cp .env.example .env
# edit .env: DOMAIN, ACME_EMAIL, MEDIA_PATH, DOWNLOADS_PATH, etc.
```

2) Create media paths on the VPS
```bash
sudo mkdir -p /srv/media/{movies,tv,music,books,audiobooks}
sudo mkdir -p /srv/downloads
sudo chown -R $UID:$GID /srv/media /srv/downloads
```

3) Start the stack
```bash
docker compose up -d
```

4) Access services (replace example.com)
- Jellyfin: https://jellyfin.example.com
- Sonarr: https://sonarr.example.com
- Radarr
- Bazarr
- Lidarr
- Readarr: https://radarr.example.com
- Prowlarr: https://prowlarr.example.com
- qBittorrent: https://qbittorrent.example.com
- Bazarr: https://bazarr.example.com
- Lidarr: https://lidarr.example.com
- Readarr: https://readarr.example.com
- Overseerr: https://overseerr.example.com
- Calibre-Web: https://calibre.example.com
- Navidrome: https://navidrome.example.com
- Audiobookshelf: https://audiobooks.example.com
- Jellyseerr: https://jellyseerr.example.com
- Tautulli: https://tautulli.example.com
- Jellyseerr: https://jellyseerr.example.com
- Tautulli: https://tautulli.example.com
- Traefik dashboard: https://traefik.example.com (protected by basic auth)
- SABnzbd: https://sabnzbd.example.com
- Uptime Kuma: https://uptime.example.com
- Prometheus: https://prometheus.example.com
- Grafana: https://grafana.example.com

## Notes
- Certificates: By default uses HTTP challenge (TRAEFIK_CERT_RESOLVER=http). For wildcard and subdomain certs via DNS challenge, set TRAEFIK_CERT_RESOLVER=cloudflare and provide CF_API_EMAIL and CF_DNS_API_TOKEN, then `docker compose up -d` again.
- Hardware acceleration: Intel/AMD VAAPI is enabled by mapping /dev/dri. If not needed, remove `devices: - /dev/dri:/dev/dri` from jellyfin.
- qBittorrent is routed through Gluetun VPN. Peer ports 6881 TCP/UDP are exposed on the host by the gluetun container. Ensure your firewall allows them or adjust as needed.
- Data: Container configs are stored in Docker named volumes; media/downloads bind to paths you set in .env.

## Stop / update
```bash
docker compose pull
docker compose up -d
```

## Troubleshooting
- `docker logs traefik -f` to see certificate or routing issues
- Confirm DNS: `dig jellyfin.example.com +short` should return your VPS IP
- Ensure ports 80/443 are reachable from the internet
- For 502 errors: wait for first-time setup or check the service logs


### Gluetun VPN
- Set VPN_PROVIDER, VPN_TYPE, VPN_USER, VPN_PASSWORD in `vps-deploy/.env`. You can also narrow by `VPN_COUNTRY` and/or `VPN_CITY`.
- qBittorrent shares the network namespace with Gluetun (`network_mode: service:gluetun`). Traefik routes the Web UI (`https://qbittorrent.example.com`) to port 8080 on the gluetun container.
 - Optional: You may also route Prowlarr via VPN by changing its service to `network_mode: service:gluetun` and removing its Traefik labels onto the `gluetun` service accordingly. This is advanced and not enabled by default.

### Traefik dashboard basic auth
- Set `TRAEFIK_DASHBOARD_AUTH` in `vps-deploy/.env` to `username:hashed_password`.
- Generate a bcrypt hash:
  - With Apache utils: `htpasswd -nbBC 10 admin 'yourpassword'`
  - Or with openssl/docker as preferred
- The compose file applies the middleware to the Traefik dashboard router.


## Bootstrap: Wire Sonarr/Radarr to Prowlarr
Once initial setup is complete and containers have generated API keys, you can auto-register Sonarr and Radarr in Prowlarr:

```bash
cd vps-deploy
./bootstrap-arr.sh
```

This script:
- Reads API keys from /config/config.xml inside the containers
- Calls the Prowlarr API to add Sonarr and Radarr as applications
- Prints keys for manual configuration if the API format differs


## Toggle Prowlarr VPN routing
Set `PROWLARR_VPN=true` in `vps-deploy/.env` to route Prowlarr via the Gluetun VPN. Then run:

```bash
cd vps-deploy
./up.sh
```

- When enabled, access Prowlarr at: `https://prowlarrvpn.${DOMAIN}`.
- When disabled, access Prowlarr at: `https://prowlarr.${DOMAIN}`.


## Provision Prowlarr indexers
Prepare a JSON file describing indexers (see `vps-deploy/indexers.sample.json`) and run:

```bash
cd vps-deploy
./provision-indexers.sh indexers.json
```

The script will use the Prowlarr API key from the running container or the `PROWLARR_API_KEY` environment variable.

For a head start, use the provided defaults:
```bash
./provision-indexers.sh prowlarr-indexers-default.json
```


## One-shot first boot automation
Run a single command to bring up the stack, wire Prowlarr/Sonarr/Radarr, configure qBittorrent, and optionally provision indexers:

```bash
cd vps-deploy
./first-boot.sh               # basic
./first-boot.sh indexers.json # with indexer provisioning
```

First-run order:
- up.sh (respects REQUEST_PORTAL and PROWLARR_VPN)
- bootstrap-arr.sh (register Sonarr/Radarr in Prowlarr)
- bootstrap-download-clients.sh (add qBittorrent to Sonarr/Radarr)
- provision-indexers.sh (optional JSON-based provision)
- bootstrap-arr-profiles.sh (creates /tv and /movies root folders and clones a basic 1080p quality profile)

You can also run the profiles bootstrap separately:
```bash
./bootstrap-arr-profiles.sh
```

### Grafana dashboards
Two minimal dashboards are pre-provisioned:
- Traefik Overview (Minimal)
- Container Overview (Minimal)

Find them in Grafana under Dashboards after first startup. Datasource is preconfigured to Prometheus.

### Blackbox exporter (HTTP checks)
- We added a Blackbox exporter to probe external HTTP endpoints.
- Edit `monitoring/prometheus/blackbox-targets.yml` with the URLs you want to probe.
- Prometheus scrapes `blackbox-exporter:9115` with the `http_2xx` module.
