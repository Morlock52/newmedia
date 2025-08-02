# Unified Media Server Architecture 2025
## Comprehensive Integration Design with Enable/Disable Functionality

### Executive Summary

This document provides a comprehensive design for unifying your media server infrastructure using Docker Compose profiles, modern 2025 best practices, and intelligent service management. The architecture enables selective service activation while maintaining seamless integration.

## Current State Analysis

### Existing Infrastructure
Based on analysis of your codebase:

1. **Multiple Docker Compose Files**: 27+ compose files with overlapping services
2. **Service Categories**:
   - Core Media: Jellyfin, Plex (missing), Emby (missing)
   - Arr Suite: Sonarr, Radarr, Lidarr, Bazarr, Prowlarr, Readarr (partial)
   - Download: qBittorrent, SABnzbd, VPN (Gluetun)
   - Management: Overseerr, Portainer, Homepage, Homarr
   - Advanced: AI/ML Nexus, AR/VR, Blockchain, Voice AI
   - Monitoring: Prometheus, Grafana, Tautulli

3. **Key Issues Identified**:
   - Service fragmentation across multiple compose files
   - Lack of unified profile management
   - Missing enable/disable functionality
   - Inconsistent network segmentation
   - No centralized configuration management

## Unified Architecture Design

### 1. Docker Compose Profile Structure

```yaml
# docker-compose.unified.yml
name: ultimate-media-server

# Extension fields for reusability
x-common-variables: &common-variables
  PUID: ${PUID:-1000}
  PGID: ${PGID:-1000}
  TZ: ${TZ:-America/New_York}

x-security-opts: &security-opts
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  restart: unless-stopped

x-healthcheck-defaults: &healthcheck-defaults
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s

# Unified network architecture
networks:
  # Core infrastructure network
  core:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/24
    driver_opts:
      com.docker.network.bridge.name: media_core

  # Frontend services network
  frontend:
    driver: bridge
    internal: false
    ipam:
      config:
        - subnet: 172.20.1.0/24

  # Backend services network
  backend:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.2.0/24

  # Download network (VPN isolated)
  downloads:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.3.0/24

  # Monitoring network
  monitoring:
    driver: bridge
    internal: true
    ipam:
      config:
        - subnet: 172.20.4.0/24

services:
  # =================
  # CORE PROFILE - Always enabled
  # =================
  
  traefik:
    profiles: ["core", "all"]
    image: traefik:v3.1
    container_name: traefik
    <<: *security-opts
    cap_add:
      - NET_BIND_SERVICE
    networks:
      - core
      - frontend
    ports:
      - "80:80"
      - "443:443"
      - "443:443/udp" # HTTP/3
    environment:
      <<: *common-variables
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - ./config/traefik:/etc/traefik:ro
      - ./data/traefik:/data
    command:
      - --api.dashboard=true
      - --providers.docker=true
      - --providers.docker.exposedbydefault=false
      - --providers.docker.network=media_frontend
      - --entrypoints.web.address=:80
      - --entrypoints.websecure.address=:443
      - --certificatesresolvers.cloudflare.acme.dnschallenge=true
      - --certificatesresolvers.cloudflare.acme.dnschallenge.provider=cloudflare
      - --experimental.http3=true
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "traefik", "healthcheck", "--ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.dashboard.rule=Host(`traefik.${DOMAIN}`)"
      - "traefik.http.routers.dashboard.entrypoints=websecure"
      - "traefik.http.routers.dashboard.tls.certresolver=cloudflare"
      - "traefik.http.services.dashboard.loadbalancer.server.port=8080"

  authelia:
    profiles: ["core", "all"]
    image: authelia/authelia:latest
    container_name: authelia
    <<: *security-opts
    networks:
      - core
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/authelia:/config
      - ./data/authelia:/data
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:9091/api/health"]
    depends_on:
      - redis
      - postgres
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.authelia.rule=Host(`auth.${DOMAIN}`)"
      - "traefik.http.routers.authelia.entrypoints=websecure"
      - "traefik.http.routers.authelia.tls.certresolver=cloudflare"

  postgres:
    profiles: ["core", "database", "all"]
    image: postgres:15-alpine
    container_name: postgres
    <<: *security-opts
    networks:
      - backend
    environment:
      <<: *common-variables
      POSTGRES_DB: ${POSTGRES_DB:-mediaserver}
      POSTGRES_USER: ${POSTGRES_USER:-mediauser}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - ./data/postgres:/var/lib/postgresql/data
      - ./config/postgres/init.sql:/docker-entrypoint-initdb.d/init.sql:ro
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER}"]

  redis:
    profiles: ["core", "cache", "all"]
    image: redis:7-alpine
    container_name: redis
    <<: *security-opts
    networks:
      - backend
    command: redis-server --appendonly yes --requirepass ${REDIS_PASSWORD}
    volumes:
      - ./data/redis:/data
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]

  # =================
  # MEDIA PROFILE - Primary media servers
  # =================
  
  jellyfin:
    profiles: ["media", "streaming", "all"]
    image: jellyfin/jellyfin:latest
    container_name: jellyfin
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
      JELLYFIN_PublishedServerUrl: https://jellyfin.${DOMAIN}
    volumes:
      - ./config/jellyfin:/config
      - ${MEDIA_PATH:-./media}:/media:ro
      - ./cache/jellyfin:/cache
      - /dev/shm:/transcoding # RAM transcoding
    devices:
      - /dev/dri:/dev/dri # Intel GPU
      # - /dev/nvidia0:/dev/nvidia0 # NVIDIA GPU (if available)
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8096/health"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.jellyfin.rule=Host(`jellyfin.${DOMAIN}`)"
      - "traefik.http.routers.jellyfin.entrypoints=websecure"
      - "traefik.http.routers.jellyfin.tls.certresolver=cloudflare"
      - "traefik.http.routers.jellyfin.middlewares=authelia@docker"
      - "traefik.http.services.jellyfin.loadbalancer.server.port=8096"

  plex:
    profiles: ["media", "streaming", "plex", "all"]
    image: plexinc/pms-docker:latest
    container_name: plex
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
      PLEX_CLAIM: ${PLEX_CLAIM}
      ADVERTISE_IP: https://plex.${DOMAIN}
    volumes:
      - ./config/plex:/config
      - ${MEDIA_PATH:-./media}:/media:ro
      - /dev/shm:/transcode
    devices:
      - /dev/dri:/dev/dri
    ports:
      - "32400:32400/tcp" # Plex Media Server
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:32400/identity"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.plex.rule=Host(`plex.${DOMAIN}`)"
      - "traefik.http.routers.plex.entrypoints=websecure"
      - "traefik.http.routers.plex.tls.certresolver=cloudflare"
      - "traefik.http.services.plex.loadbalancer.server.port=32400"

  emby:
    profiles: ["media", "streaming", "emby", "all"]
    image: emby/embyserver:latest
    container_name: emby
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/emby:/config
      - ${MEDIA_PATH:-./media}:/media:ro
      - /dev/shm:/transcode
    devices:
      - /dev/dri:/dev/dri
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8096/emby/System/Ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.emby.rule=Host(`emby.${DOMAIN}`)"
      - "traefik.http.routers.emby.entrypoints=websecure"
      - "traefik.http.routers.emby.tls.certresolver=cloudflare"
      - "traefik.http.routers.emby.middlewares=authelia@docker"

  # =================
  # MUSIC PROFILE - Music streaming
  # =================
  
  navidrome:
    profiles: ["music", "media", "all"]
    image: deluan/navidrome:latest
    container_name: navidrome
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
      ND_SCANSCHEDULE: 1h
      ND_LOGLEVEL: info
      ND_BASEURL: https://music.${DOMAIN}
    volumes:
      - ./config/navidrome:/data
      - ${MUSIC_PATH:-./media/music}:/music:ro
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "wget", "-O", "-", "http://localhost:4533/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.navidrome.rule=Host(`music.${DOMAIN}`)"
      - "traefik.http.routers.navidrome.entrypoints=websecure"
      - "traefik.http.routers.navidrome.tls.certresolver=cloudflare"
      - "traefik.http.routers.navidrome.middlewares=authelia@docker"

  # =================
  # BOOKS PROFILE - E-books and audiobooks
  # =================
  
  audiobookshelf:
    profiles: ["books", "media", "all"]
    image: ghcr.io/advplyr/audiobookshelf:latest
    container_name: audiobookshelf
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/audiobookshelf:/config
      - ${AUDIOBOOKS_PATH:-./media/audiobooks}:/audiobooks
      - ${PODCASTS_PATH:-./media/podcasts}:/podcasts
      - ${BOOKS_PATH:-./media/books}:/books
      - ./data/audiobookshelf/metadata:/metadata
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:80/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.audiobookshelf.rule=Host(`audiobooks.${DOMAIN}`)"
      - "traefik.http.routers.audiobookshelf.entrypoints=websecure"
      - "traefik.http.routers.audiobookshelf.tls.certresolver=cloudflare"
      - "traefik.http.routers.audiobookshelf.middlewares=authelia@docker"

  calibre-web:
    profiles: ["books", "media", "all"]
    image: lscr.io/linuxserver/calibre-web:latest
    container_name: calibre-web
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
      DOCKER_MODS: linuxserver/mods:universal-calibre
    volumes:
      - ./config/calibre-web:/config
      - ${BOOKS_PATH:-./media/books}:/books
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8083"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.calibre.rule=Host(`books.${DOMAIN}`)"
      - "traefik.http.routers.calibre.entrypoints=websecure"
      - "traefik.http.routers.calibre.tls.certresolver=cloudflare"
      - "traefik.http.routers.calibre.middlewares=authelia@docker"

  kavita:
    profiles: ["books", "comics", "media", "all"]
    image: kizaing/kavita:latest
    container_name: kavita
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/kavita:/kavita/config
      - ${BOOKS_PATH:-./media/books}:/books
      - ${COMICS_PATH:-./media/comics}:/comics
      - ${MANGA_PATH:-./media/manga}:/manga
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:5000/api/health"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.kavita.rule=Host(`comics.${DOMAIN}`)"
      - "traefik.http.routers.kavita.entrypoints=websecure"
      - "traefik.http.routers.kavita.tls.certresolver=cloudflare"
      - "traefik.http.routers.kavita.middlewares=authelia@docker"

  # =================
  # PHOTOS PROFILE - Photo management
  # =================
  
  immich-server:
    profiles: ["photos", "media", "all"]
    image: ghcr.io/immich-app/immich-server:release
    container_name: immich-server
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
      DB_HOSTNAME: postgres
      DB_USERNAME: ${POSTGRES_USER:-mediauser}
      DB_PASSWORD: ${POSTGRES_PASSWORD}
      DB_DATABASE_NAME: immich
      REDIS_HOSTNAME: redis
      REDIS_PASSWORD: ${REDIS_PASSWORD}
      UPLOAD_LOCATION: /usr/src/app/upload
    volumes:
      - ${PHOTOS_PATH:-./media/photos}:/usr/src/app/upload
      - ./data/immich:/usr/src/app/.reverse-geocoding-dump
    depends_on:
      - postgres
      - redis
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:3001/server-info/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.immich.rule=Host(`photos.${DOMAIN}`)"
      - "traefik.http.routers.immich.entrypoints=websecure"
      - "traefik.http.routers.immich.tls.certresolver=cloudflare"
      - "traefik.http.routers.immich.middlewares=authelia@docker"

  immich-microservices:
    profiles: ["photos", "media", "all"]
    image: ghcr.io/immich-app/immich-server:release
    container_name: immich-microservices
    <<: *security-opts
    networks:
      - backend
    command: ["start.sh", "microservices"]
    environment:
      <<: *common-variables
      DB_HOSTNAME: postgres
      DB_USERNAME: ${POSTGRES_USER:-mediauser}
      DB_PASSWORD: ${POSTGRES_PASSWORD}
      DB_DATABASE_NAME: immich
      REDIS_HOSTNAME: redis
      REDIS_PASSWORD: ${REDIS_PASSWORD}
    volumes:
      - ${PHOTOS_PATH:-./media/photos}:/usr/src/app/upload
      - ./data/immich:/usr/src/app/.reverse-geocoding-dump
    devices:
      - /dev/dri:/dev/dri # For hardware acceleration
    depends_on:
      - postgres
      - redis
      - immich-server

  immich-machine-learning:
    profiles: ["photos", "media", "ml", "all"]
    image: ghcr.io/immich-app/immich-machine-learning:release
    container_name: immich-machine-learning
    <<: *security-opts
    networks:
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./data/immich/model-cache:/cache
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  # =================
  # AUTOMATION PROFILE - Content automation (*arr stack)
  # =================
  
  prowlarr:
    profiles: ["automation", "indexers", "all"]
    image: lscr.io/linuxserver/prowlarr:latest
    container_name: prowlarr
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/prowlarr:/config
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:9696/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.prowlarr.rule=Host(`prowlarr.${DOMAIN}`)"
      - "traefik.http.routers.prowlarr.entrypoints=websecure"
      - "traefik.http.routers.prowlarr.tls.certresolver=cloudflare"
      - "traefik.http.routers.prowlarr.middlewares=authelia@docker"

  sonarr:
    profiles: ["automation", "tv", "all"]
    image: lscr.io/linuxserver/sonarr:latest
    container_name: sonarr
    <<: *security-opts
    networks:
      - frontend
      - backend
      - downloads
    environment:
      <<: *common-variables
    volumes:
      - ./config/sonarr:/config
      - ${MEDIA_PATH:-./media}:/media
      - ${DOWNLOADS_PATH:-./downloads}:/downloads
    depends_on:
      - prowlarr
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8989/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.sonarr.rule=Host(`sonarr.${DOMAIN}`)"
      - "traefik.http.routers.sonarr.entrypoints=websecure"
      - "traefik.http.routers.sonarr.tls.certresolver=cloudflare"
      - "traefik.http.routers.sonarr.middlewares=authelia@docker"

  radarr:
    profiles: ["automation", "movies", "all"]
    image: lscr.io/linuxserver/radarr:latest
    container_name: radarr
    <<: *security-opts
    networks:
      - frontend
      - backend
      - downloads
    environment:
      <<: *common-variables
    volumes:
      - ./config/radarr:/config
      - ${MEDIA_PATH:-./media}:/media
      - ${DOWNLOADS_PATH:-./downloads}:/downloads
    depends_on:
      - prowlarr
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:7878/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.radarr.rule=Host(`radarr.${DOMAIN}`)"
      - "traefik.http.routers.radarr.entrypoints=websecure"
      - "traefik.http.routers.radarr.tls.certresolver=cloudflare"
      - "traefik.http.routers.radarr.middlewares=authelia@docker"

  lidarr:
    profiles: ["automation", "music", "all"]
    image: lscr.io/linuxserver/lidarr:latest
    container_name: lidarr
    <<: *security-opts
    networks:
      - frontend
      - backend
      - downloads
    environment:
      <<: *common-variables
    volumes:
      - ./config/lidarr:/config
      - ${MEDIA_PATH:-./media}:/media
      - ${DOWNLOADS_PATH:-./downloads}:/downloads
    depends_on:
      - prowlarr
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8686/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.lidarr.rule=Host(`lidarr.${DOMAIN}`)"
      - "traefik.http.routers.lidarr.entrypoints=websecure"
      - "traefik.http.routers.lidarr.tls.certresolver=cloudflare"
      - "traefik.http.routers.lidarr.middlewares=authelia@docker"

  readarr:
    profiles: ["automation", "books", "all"]
    image: lscr.io/linuxserver/readarr:develop
    container_name: readarr
    <<: *security-opts
    networks:
      - frontend
      - backend
      - downloads
    environment:
      <<: *common-variables
    volumes:
      - ./config/readarr:/config
      - ${MEDIA_PATH:-./media}:/media
      - ${DOWNLOADS_PATH:-./downloads}:/downloads
    depends_on:
      - prowlarr
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8787/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.readarr.rule=Host(`readarr.${DOMAIN}`)"
      - "traefik.http.routers.readarr.entrypoints=websecure"
      - "traefik.http.routers.readarr.tls.certresolver=cloudflare"
      - "traefik.http.routers.readarr.middlewares=authelia@docker"

  bazarr:
    profiles: ["automation", "subtitles", "all"]
    image: lscr.io/linuxserver/bazarr:latest
    container_name: bazarr
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/bazarr:/config
      - ${MEDIA_PATH:-./media}:/media
    depends_on:
      - sonarr
      - radarr
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:6767/ping"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.bazarr.rule=Host(`bazarr.${DOMAIN}`)"
      - "traefik.http.routers.bazarr.entrypoints=websecure"
      - "traefik.http.routers.bazarr.tls.certresolver=cloudflare"
      - "traefik.http.routers.bazarr.middlewares=authelia@docker"

  # =================
  # DOWNLOADS PROFILE - Download clients
  # =================
  
  vpn:
    profiles: ["downloads", "vpn", "all"]
    image: qmcgaw/gluetun:latest
    container_name: vpn
    <<: *security-opts
    cap_add:
      - NET_ADMIN
    networks:
      - downloads
    environment:
      <<: *common-variables
      VPN_SERVICE_PROVIDER: ${VPN_PROVIDER:-mullvad}
      VPN_TYPE: ${VPN_TYPE:-wireguard}
      WIREGUARD_PRIVATE_KEY: ${VPN_PRIVATE_KEY}
      WIREGUARD_ADDRESSES: ${VPN_ADDRESSES}
      FIREWALL_OUTBOUND_SUBNETS: 172.20.0.0/16
      FIREWALL_KILL_SWITCH: "on"
      DOT: "on"
      HEALTH_VPN_DURATION_INITIAL: 30s
    volumes:
      - ./config/gluetun:/gluetun
    ports:
      - "8080:8080" # qBittorrent WebUI
      - "6881:6881" # qBittorrent listening port
      - "6881:6881/udp"
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8000"]

  qbittorrent:
    profiles: ["downloads", "torrent", "all"]
    image: lscr.io/linuxserver/qbittorrent:latest
    container_name: qbittorrent
    <<: *security-opts
    network_mode: "service:vpn"
    environment:
      <<: *common-variables
      WEBUI_PORT: 8080
    volumes:
      - ./config/qbittorrent:/config
      - ${DOWNLOADS_PATH:-./downloads}:/downloads
    depends_on:
      - vpn

  sabnzbd:
    profiles: ["downloads", "usenet", "all"]
    image: lscr.io/linuxserver/sabnzbd:latest
    container_name: sabnzbd
    <<: *security-opts
    networks:
      - frontend
      - backend
      - downloads
    environment:
      <<: *common-variables
    volumes:
      - ./config/sabnzbd:/config
      - ${DOWNLOADS_PATH:-./downloads}:/downloads
      - ${USENET_PATH:-./downloads/usenet}:/incomplete-downloads
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8080/sabnzbd/api?mode=version"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.sabnzbd.rule=Host(`sabnzbd.${DOMAIN}`)"
      - "traefik.http.routers.sabnzbd.entrypoints=websecure"
      - "traefik.http.routers.sabnzbd.tls.certresolver=cloudflare"
      - "traefik.http.routers.sabnzbd.middlewares=authelia@docker"

  # =================
  # REQUESTS PROFILE - Content request management
  # =================
  
  overseerr:
    profiles: ["requests", "management", "all"]
    image: lscr.io/linuxserver/overseerr:latest
    container_name: overseerr
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/overseerr:/config
    depends_on:
      - sonarr
      - radarr
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:5055/api/v1/status"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.overseerr.rule=Host(`requests.${DOMAIN}`)"
      - "traefik.http.routers.overseerr.entrypoints=websecure"
      - "traefik.http.routers.overseerr.tls.certresolver=cloudflare"
      # No auth middleware - Overseerr has its own authentication

  jellyseerr:
    profiles: ["requests", "jellyfin", "all"]
    image: fallenbagel/jellyseerr:latest
    container_name: jellyseerr
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/jellyseerr:/app/config
    depends_on:
      - jellyfin
      - sonarr
      - radarr
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:5055/api/v1/status"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.jellyseerr.rule=Host(`jellyseerr.${DOMAIN}`)"
      - "traefik.http.routers.jellyseerr.entrypoints=websecure"
      - "traefik.http.routers.jellyseerr.tls.certresolver=cloudflare"

  # =================
  # MONITORING PROFILE - System monitoring
  # =================
  
  prometheus:
    profiles: ["monitoring", "metrics", "all"]
    image: prom/prometheus:latest
    container_name: prometheus
    <<: *security-opts
    networks:
      - monitoring
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/prometheus:/etc/prometheus:ro
      - ./data/prometheus:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/usr/share/prometheus/console_libraries'
      - '--web.console.templates=/usr/share/prometheus/consoles'
      - '--storage.tsdb.retention.time=90d'
      - '--web.enable-lifecycle'
      - '--web.enable-admin-api'
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "wget", "--no-verbose", "--tries=1", "--spider", "http://localhost:9090/-/healthy"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.prometheus.rule=Host(`prometheus.${DOMAIN}`)"
      - "traefik.http.routers.prometheus.entrypoints=websecure"
      - "traefik.http.routers.prometheus.tls.certresolver=cloudflare"
      - "traefik.http.routers.prometheus.middlewares=authelia@docker"

  grafana:
    profiles: ["monitoring", "visualization", "all"]
    image: grafana/grafana:latest
    container_name: grafana
    <<: *security-opts
    networks:
      - monitoring
      - frontend
    environment:
      <<: *common-variables
      GF_SECURITY_ADMIN_USER: ${GRAFANA_USER:-admin}
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: grafana-clock-panel,grafana-simple-json-datasource,grafana-piechart-panel
    volumes:
      - ./data/grafana:/var/lib/grafana
      - ./config/grafana/provisioning:/etc/grafana/provisioning:ro
    depends_on:
      - prometheus
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD-SHELL", "curl -f http://localhost:3000/api/health || exit 1"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.grafana.rule=Host(`grafana.${DOMAIN}`)"
      - "traefik.http.routers.grafana.entrypoints=websecure"
      - "traefik.http.routers.grafana.tls.certresolver=cloudflare"
      - "traefik.http.routers.grafana.middlewares=authelia@docker"

  tautulli:
    profiles: ["monitoring", "analytics", "all"]
    image: lscr.io/linuxserver/tautulli:latest
    container_name: tautulli
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/tautulli:/config
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8181/status"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.tautulli.rule=Host(`tautulli.${DOMAIN}`)"
      - "traefik.http.routers.tautulli.entrypoints=websecure"
      - "traefik.http.routers.tautulli.tls.certresolver=cloudflare"
      - "traefik.http.routers.tautulli.middlewares=authelia@docker"

  # =================
  # MANAGEMENT PROFILE - Administration tools
  # =================
  
  portainer:
    profiles: ["management", "admin", "all"]
    image: portainer/portainer-ce:latest
    container_name: portainer
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./data/portainer:/data
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "wget", "-O", "-", "http://localhost:9000/api/system/status"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.portainer.rule=Host(`portainer.${DOMAIN}`)"
      - "traefik.http.routers.portainer.entrypoints=websecure"
      - "traefik.http.routers.portainer.tls.certresolver=cloudflare"
      - "traefik.http.routers.portainer.middlewares=authelia@docker"
      - "traefik.http.services.portainer.loadbalancer.server.port=9000"

  homepage:
    profiles: ["management", "dashboard", "all"]
    image: ghcr.io/gethomepage/homepage:latest
    container_name: homepage
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/homepage:/app/config
      - /var/run/docker.sock:/var/run/docker.sock:ro
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "wget", "-O", "-", "http://localhost:3000"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.homepage.rule=Host(`home.${DOMAIN}`)"
      - "traefik.http.routers.homepage.entrypoints=websecure"
      - "traefik.http.routers.homepage.tls.certresolver=cloudflare"
      - "traefik.http.services.homepage.loadbalancer.server.port=3000"

  homarr:
    profiles: ["management", "dashboard", "all"]
    image: ghcr.io/ajnart/homarr:latest
    container_name: homarr
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
      BASE_URL: https://dashboard.${DOMAIN}
    volumes:
      - ./config/homarr/configs:/app/data/configs
      - ./config/homarr/icons:/app/public/icons
      - ./data/homarr:/data
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "wget", "-O", "-", "http://localhost:7575"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.homarr.rule=Host(`dashboard.${DOMAIN}`)"
      - "traefik.http.routers.homarr.entrypoints=websecure"
      - "traefik.http.routers.homarr.tls.certresolver=cloudflare"

  # =================
  # PROCESSING PROFILE - Media processing
  # =================
  
  tdarr:
    profiles: ["processing", "transcoding", "all"]
    image: ghcr.io/haveagitgat/tdarr:latest
    container_name: tdarr
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
      serverIP: 0.0.0.0
      serverPort: 8266
      webUIPort: 8265
      internalNode: true
      nodeID: InternalNode
    volumes:
      - ./config/tdarr/server:/app/server
      - ./config/tdarr/configs:/app/configs
      - ./config/tdarr/logs:/app/logs
      - ${MEDIA_PATH:-./media}:/media
      - ./cache/tdarr:/temp
    devices:
      - /dev/dri:/dev/dri
    ports:
      - "8265:8265" # WebUI
      - "8266:8266" # Server
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8265/api/v2/status"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.tdarr.rule=Host(`tdarr.${DOMAIN}`)"
      - "traefik.http.routers.tdarr.entrypoints=websecure"
      - "traefik.http.routers.tdarr.tls.certresolver=cloudflare"
      - "traefik.http.routers.tdarr.middlewares=authelia@docker"
      - "traefik.http.services.tdarr.loadbalancer.server.port=8265"

  handbrake:
    profiles: ["processing", "encoding", "all"]
    image: jlesage/handbrake:latest
    container_name: handbrake
    <<: *security-opts
    networks:
      - frontend
      - backend
    environment:
      <<: *common-variables
      AUTOMATED_CONVERSION: 0
      HANDBRAKE_GUI: 1
    volumes:
      - ./config/handbrake:/config
      - ${MEDIA_PATH:-./media}:/storage:ro
      - ./data/handbrake/watch:/watch
      - ./data/handbrake/output:/output
    devices:
      - /dev/dri:/dev/dri
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.handbrake.rule=Host(`handbrake.${DOMAIN}`)"
      - "traefik.http.routers.handbrake.entrypoints=websecure"
      - "traefik.http.routers.handbrake.tls.certresolver=cloudflare"
      - "traefik.http.routers.handbrake.middlewares=authelia@docker"

  # =================
  # BACKUP PROFILE - Backup services
  # =================
  
  duplicati:
    profiles: ["backup", "storage", "all"]
    image: lscr.io/linuxserver/duplicati:latest
    container_name: duplicati
    <<: *security-opts
    networks:
      - backend
    environment:
      <<: *common-variables
    volumes:
      - ./config/duplicati:/config
      - ./backups:/backups
      - ${MEDIA_PATH:-./media}:/source/media:ro
      - ./config:/source/config:ro
      - ./data:/source/data:ro
    healthcheck:
      <<: *healthcheck-defaults
      test: ["CMD", "curl", "-f", "http://localhost:8200"]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.duplicati.rule=Host(`backup.${DOMAIN}`)"
      - "traefik.http.routers.duplicati.entrypoints=websecure"
      - "traefik.http.routers.duplicati.tls.certresolver=cloudflare"
      - "traefik.http.routers.duplicati.middlewares=authelia@docker"

  # =================
  # ADVANCED PROFILE - AI/ML and experimental features
  # =================
  
  ai-ml-nexus:
    profiles: ["advanced", "ai", "experimental"]
    build: ./ai-ml-nexus
    container_name: ai-ml-nexus
    <<: *security-opts
    networks:
      - backend
    environment:
      <<: *common-variables
      NODE_ENV: production
      CUDA_VISIBLE_DEVICES: 0
    volumes:
      - ./data/ai-models:/app/models
      - ${MEDIA_PATH:-./media}:/app/media:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.ai.rule=Host(`ai.${DOMAIN}`)"
      - "traefik.http.routers.ai.entrypoints=websecure"
      - "traefik.http.routers.ai.tls.certresolver=cloudflare"
      - "traefik.http.routers.ai.middlewares=authelia@docker"

# Volume definitions
volumes:
  # Data volumes
  postgres_data:
    driver: local
  redis_data:
    driver: local
  
  # Config volumes (using bind mounts in service definitions)
  # Media volumes (using bind mounts from environment variables)
```

### 2. Service Management API

```javascript
// service-manager.js
const express = require('express');
const { Docker } = require('node-docker-api');
const yaml = require('js-yaml');
const fs = require('fs').promises;
const path = require('path');

class UnifiedServiceManager {
  constructor() {
    this.docker = new Docker({ socketPath: '/var/run/docker.sock' });
    this.configPath = './config/service-state.json';
    this.composePath = './docker-compose.unified.yml';
    
    // Profile definitions with dependencies
    this.profiles = {
      core: {
        name: 'Core Infrastructure',
        services: ['traefik', 'authelia', 'postgres', 'redis'],
        required: true,
        dependencies: []
      },
      media: {
        name: 'Media Streaming',
        services: ['jellyfin', 'plex', 'emby'],
        dependencies: ['core']
      },
      music: {
        name: 'Music Services',
        services: ['navidrome'],
        dependencies: ['core']
      },
      books: {
        name: 'Books & Comics',
        services: ['audiobookshelf', 'calibre-web', 'kavita'],
        dependencies: ['core']
      },
      photos: {
        name: 'Photo Management',
        services: ['immich-server', 'immich-microservices', 'immich-machine-learning'],
        dependencies: ['core', 'database']
      },
      automation: {
        name: 'Content Automation',
        services: ['prowlarr', 'sonarr', 'radarr', 'lidarr', 'readarr', 'bazarr'],
        dependencies: ['core', 'downloads']
      },
      downloads: {
        name: 'Download Clients',
        services: ['vpn', 'qbittorrent', 'sabnzbd'],
        dependencies: ['core']
      },
      requests: {
        name: 'Request Management',
        services: ['overseerr', 'jellyseerr'],
        dependencies: ['core', 'media', 'automation']
      },
      monitoring: {
        name: 'Monitoring & Analytics',
        services: ['prometheus', 'grafana', 'tautulli'],
        dependencies: ['core']
      },
      management: {
        name: 'Administration',
        services: ['portainer', 'homepage', 'homarr'],
        dependencies: ['core']
      },
      processing: {
        name: 'Media Processing',
        services: ['tdarr', 'handbrake'],
        dependencies: ['core']
      },
      backup: {
        name: 'Backup Services',
        services: ['duplicati'],
        dependencies: ['core']
      },
      advanced: {
        name: 'Advanced Features',
        services: ['ai-ml-nexus'],
        dependencies: ['core', 'database']
      }
    };

    // Service-specific dependencies
    this.serviceDependencies = {
      'authelia': ['redis', 'postgres'],
      'sonarr': ['prowlarr'],
      'radarr': ['prowlarr'],
      'lidarr': ['prowlarr'],
      'readarr': ['prowlarr'],
      'bazarr': ['sonarr', 'radarr'],
      'qbittorrent': ['vpn'],
      'overseerr': ['sonarr', 'radarr'],
      'jellyseerr': ['jellyfin', 'sonarr', 'radarr'],
      'grafana': ['prometheus'],
      'immich-microservices': ['immich-server', 'postgres', 'redis'],
      'immich-machine-learning': ['immich-server']
    };
  }

  async initialize() {
    // Ensure config directory exists
    await fs.mkdir(path.dirname(this.configPath), { recursive: true });
    
    // Load or create initial state
    const state = await this.loadState();
    if (!state.profiles || state.profiles.length === 0) {
      state.profiles = ['core'];
      state.services = {};
      await this.saveState(state);
    }
  }

  async loadState() {
    try {
      const data = await fs.readFile(this.configPath, 'utf8');
      return JSON.parse(data);
    } catch (error) {
      return { profiles: [], services: {} };
    }
  }

  async saveState(state) {
    await fs.writeFile(this.configPath, JSON.stringify(state, null, 2));
  }

  async getServiceStatus() {
    try {
      const containers = await this.docker.container.list({ all: true });
      const status = {};
      
      for (const container of containers) {
        const info = await container.status();
        const name = info.data.Names[0].replace('/', '');
        
        status[name] = {
          state: info.data.State.Status,
          health: info.data.State.Health?.Status || 'none',
          created: info.data.Created,
          started: info.data.State.StartedAt,
          ports: info.data.Ports || [],
          image: info.data.Image
        };
      }
      
      return status;
    } catch (error) {
      console.error('Error getting service status:', error);
      return {};
    }
  }

  async enableProfile(profileName) {
    if (!this.profiles[profileName]) {
      throw new Error(`Unknown profile: ${profileName}`);
    }

    const state = await this.loadState();
    
    // Check dependencies
    const profile = this.profiles[profileName];
    for (const dep of profile.dependencies) {
      if (!state.profiles.includes(dep)) {
        throw new Error(`Profile ${profileName} requires ${dep} to be enabled first`);
      }
    }

    // Add profile if not already enabled
    if (!state.profiles.includes(profileName)) {
      state.profiles.push(profileName);
      await this.saveState(state);
      await this.applyConfiguration();
    }

    return { success: true, message: `Profile ${profileName} enabled` };
  }

  async disableProfile(profileName) {
    const profile = this.profiles[profileName];
    if (profile.required) {
      throw new Error(`Profile ${profileName} is required and cannot be disabled`);
    }

    const state = await this.loadState();
    
    // Check if other profiles depend on this one
    const dependents = Object.entries(this.profiles)
      .filter(([name, p]) => p.dependencies.includes(profileName) && state.profiles.includes(name))
      .map(([name]) => name);

    if (dependents.length > 0) {
      throw new Error(`Cannot disable ${profileName}: required by ${dependents.join(', ')}`);
    }

    // Remove profile
    state.profiles = state.profiles.filter(p => p !== profileName);
    await this.saveState(state);
    await this.applyConfiguration();

    return { success: true, message: `Profile ${profileName} disabled` };
  }

  async applyConfiguration() {
    const state = await this.loadState();
    const profileArgs = state.profiles.flatMap(p => ['--profile', p]);
    
    // Run docker-compose with selected profiles
    const { exec } = require('child_process');
    const command = `docker-compose -f ${this.composePath} ${profileArgs.join(' ')} up -d --remove-orphans`;
    
    return new Promise((resolve, reject) => {
      exec(command, (error, stdout, stderr) => {
        if (error) {
          reject(error);
        } else {
          resolve({ stdout, stderr });
        }
      });
    });
  }

  async getRecommendedProfiles(userNeeds) {
    const recommendations = {
      basic: ['core', 'media', 'downloads', 'management'],
      movies_tv: ['core', 'media', 'automation', 'downloads', 'requests', 'management'],
      music_lover: ['core', 'music', 'automation', 'downloads', 'management'],
      photographer: ['core', 'photos', 'backup', 'management'],
      bookworm: ['core', 'books', 'automation', 'downloads', 'management'],
      power_user: ['core', 'media', 'music', 'books', 'photos', 'automation', 'downloads', 'requests', 'monitoring', 'management', 'processing'],
      developer: ['core', 'monitoring', 'management', 'advanced']
    };

    return recommendations[userNeeds] || recommendations.basic;
  }

  async healthCheck() {
    const status = await this.getServiceStatus();
    const state = await this.loadState();
    const health = {
      profiles: {},
      services: {},
      overall: 'healthy'
    };

    // Check each enabled profile
    for (const profileName of state.profiles) {
      const profile = this.profiles[profileName];
      const profileHealth = {
        services: {},
        status: 'healthy'
      };

      for (const service of profile.services) {
        const serviceStatus = status[service];
        if (!serviceStatus) {
          profileHealth.services[service] = 'missing';
          profileHealth.status = 'degraded';
        } else if (serviceStatus.state !== 'running') {
          profileHealth.services[service] = serviceStatus.state;
          profileHealth.status = 'degraded';
        } else if (serviceStatus.health && serviceStatus.health !== 'healthy') {
          profileHealth.services[service] = serviceStatus.health;
          profileHealth.status = 'degraded';
        } else {
          profileHealth.services[service] = 'healthy';
        }
      }

      health.profiles[profileName] = profileHealth;
      if (profileHealth.status !== 'healthy') {
        health.overall = 'degraded';
      }
    }

    return health;
  }
}

// Express API
const app = express();
app.use(express.json());

const manager = new UnifiedServiceManager();

// Initialize manager
manager.initialize().then(() => {
  console.log('Service manager initialized');
});

// API Routes
app.get('/api/status', async (req, res) => {
  try {
    const status = await manager.getServiceStatus();
    const state = await manager.loadState();
    const health = await manager.healthCheck();
    
    res.json({ status, state, health });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/profiles/:name/enable', async (req, res) => {
  try {
    const result = await manager.enableProfile(req.params.name);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.post('/api/profiles/:name/disable', async (req, res) => {
  try {
    const result = await manager.disableProfile(req.params.name);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

app.get('/api/recommendations/:userType', async (req, res) => {
  try {
    const recommendations = await manager.getRecommendedProfiles(req.params.userType);
    res.json({ recommendations });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/health', async (req, res) => {
  try {
    const health = await manager.healthCheck();
    res.json(health);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

const PORT = process.env.SERVICE_MANAGER_PORT || 3010;
app.listen(PORT, () => {
  console.log(`Unified Service Manager API running on port ${PORT}`);
});

module.exports = UnifiedServiceManager;
```

### 3. Web Dashboard

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Unified Media Server Dashboard</title>
    <style>
        :root {
            --primary: #667eea;
            --secondary: #764ba2;
            --success: #48bb78;
            --warning: #f6ad55;
            --danger: #fc8181;
            --dark: #2d3748;
            --light: #f7fafc;
            --shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        }

        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1600px;
            margin: 0 auto;
        }

        .dashboard {
            background: rgba(255, 255, 255, 0.98);
            border-radius: 20px;
            padding: 30px;
            box-shadow: var(--shadow);
        }

        .header {
            text-align: center;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 2.5em;
            color: var(--dark);
            margin-bottom: 10px;
        }

        .header p {
            color: #718096;
            font-size: 1.1em;
        }

        .quick-setup {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 40px;
        }

        .quick-setup h3 {
            margin-bottom: 20px;
            color: var(--dark);
        }

        .preset-buttons {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }

        .preset-btn {
            padding: 15px 20px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            background: white;
            cursor: pointer;
            transition: all 0.3s;
            text-align: center;
        }

        .preset-btn:hover {
            border-color: var(--primary);
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }

        .preset-btn h4 {
            color: var(--dark);
            margin-bottom: 5px;
        }

        .preset-btn p {
            color: #718096;
            font-size: 0.9em;
        }

        .profiles-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }

        .profile-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: var(--shadow);
            border-left: 5px solid #e2e8f0;
            transition: all 0.3s;
        }

        .profile-card.enabled {
            border-left-color: var(--success);
        }

        .profile-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }

        .profile-name {
            font-size: 1.3em;
            font-weight: 600;
            color: var(--dark);
        }

        .profile-description {
            color: #718096;
            margin-bottom: 15px;
        }

        .toggle-switch {
            position: relative;
            width: 60px;
            height: 30px;
            background: #cbd5e0;
            border-radius: 15px;
            cursor: pointer;
            transition: all 0.3s;
        }

        .toggle-switch.active {
            background: var(--success);
        }

        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 26px;
            height: 26px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: all 0.3s;
        }

        .toggle-switch.active::after {
            transform: translateX(30px);
        }

        .services-list {
            margin-top: 15px;
        }

        .service-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            margin: 5px 0;
            background: #f8f9fa;
            border-radius: 8px;
        }

        .service-name {
            font-weight: 500;
            color: var(--dark);
        }

        .service-status {
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 0.8em;
            font-weight: 500;
        }

        .status-running {
            background: #d4f1e4;
            color: #22543d;
        }

        .status-stopped {
            background: #fed7d7;
            color: #742a2a;
        }

        .status-starting {
            background: #feebc8;
            color: #744210;
        }

        .health-overview {
            background: #f8f9fa;
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 40px;
        }

        .health-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }

        .health-item {
            text-align: center;
            padding: 20px;
            background: white;
            border-radius: 10px;
            border: 2px solid #e2e8f0;
        }

        .health-item.healthy {
            border-color: var(--success);
        }

        .health-item.degraded {
            border-color: var(--warning);
        }

        .health-item.critical {
            border-color: var(--danger);
        }

        .health-icon {
            font-size: 2em;
            margin-bottom: 10px;
        }

        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 1em;
            cursor: pointer;
            transition: all 0.3s;
            display: inline-flex;
            align-items: center;
            gap: 8px;
        }

        .btn-primary {
            background: var(--primary);
            color: white;
        }

        .btn-primary:hover {
            background: #5a67d8;
            transform: translateY(-2px);
            box-shadow: var(--shadow);
        }

        .actions {
            text-align: center;
            margin-top: 30px;
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }

        .notification {
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 15px 25px;
            border-radius: 10px;
            color: white;
            font-weight: 500;
            transform: translateX(400px);
            transition: all 0.3s;
            z-index: 1000;
        }

        .notification.show {
            transform: translateX(0);
        }

        .notification.success {
            background: var(--success);
        }

        .notification.error {
            background: var(--danger);
        }

        .notification.warning {
            background: var(--warning);
        }

        .loading-overlay {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            display: none;
            justify-content: center;
            align-items: center;
            z-index: 999;
        }

        .loading-overlay.show {
            display: flex;
        }

        .spinner {
            width: 50px;
            height: 50px;
            border: 5px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        @keyframes spin {
            to { transform: rotate(360deg); }
        }

        .modal {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 0, 0, 0.5);
            z-index: 1001;
        }

        .modal.show {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .modal-content {
            background: white;
            border-radius: 15px;
            padding: 30px;
            max-width: 600px;
            width: 90%;
            max-height: 80vh;
            overflow-y: auto;
        }

        .modal-header {
            margin-bottom: 20px;
        }

        .modal-header h3 {
            color: var(--dark);
        }

        .dependency-tree {
            margin: 20px 0;
            padding: 20px;
            background: #f8f9fa;
            border-radius: 10px;
        }

        .dependency-item {
            margin-left: 20px;
            color: #718096;
        }

        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }

        .stat-card {
            background: white;
            border-radius: 10px;
            padding: 20px;
            text-align: center;
            box-shadow: var(--shadow);
        }

        .stat-value {
            font-size: 2em;
            font-weight: 700;
            color: var(--primary);
        }

        .stat-label {
            color: #718096;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="dashboard">
            <div class="header">
                <h1>🎬 Unified Media Server Dashboard</h1>
                <p>Seamlessly manage your entire media ecosystem</p>
            </div>

            <!-- Stats Overview -->
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-value" id="totalServices">0</div>
                    <div class="stat-label">Total Services</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="runningServices">0</div>
                    <div class="stat-label">Running</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="enabledProfiles">0</div>
                    <div class="stat-label">Active Profiles</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value" id="systemHealth">--</div>
                    <div class="stat-label">System Health</div>
                </div>
            </div>

            <!-- Quick Setup -->
            <div class="quick-setup">
                <h3>🚀 Quick Setup Presets</h3>
                <div class="preset-buttons">
                    <div class="preset-btn" onclick="applyPreset('basic')">
                        <h4>Basic Setup</h4>
                        <p>Essential media streaming</p>
                    </div>
                    <div class="preset-btn" onclick="applyPreset('movies_tv')">
                        <h4>Movies & TV</h4>
                        <p>Full automation stack</p>
                    </div>
                    <div class="preset-btn" onclick="applyPreset('music_lover')">
                        <h4>Music Lover</h4>
                        <p>Music streaming & management</p>
                    </div>
                    <div class="preset-btn" onclick="applyPreset('photographer')">
                        <h4>Photographer</h4>
                        <p>Photo management & AI</p>
                    </div>
                    <div class="preset-btn" onclick="applyPreset('bookworm')">
                        <h4>Bookworm</h4>
                        <p>E-books & audiobooks</p>
                    </div>
                    <div class="preset-btn" onclick="applyPreset('power_user')">
                        <h4>Power User</h4>
                        <p>Everything enabled</p>
                    </div>
                </div>
            </div>

            <!-- Health Overview -->
            <div class="health-overview">
                <h3>🏥 System Health</h3>
                <div class="health-grid" id="healthGrid">
                    <!-- Dynamically populated -->
                </div>
            </div>

            <!-- Profiles Grid -->
            <div class="profiles-grid" id="profilesGrid">
                <!-- Dynamically populated -->
            </div>

            <!-- Actions -->
            <div class="actions">
                <button class="btn btn-primary" onclick="refreshStatus()">
                    🔄 Refresh Status
                </button>
                <button class="btn btn-primary" onclick="showDependencies()">
                    🔗 View Dependencies
                </button>
                <button class="btn btn-primary" onclick="exportConfiguration()">
                    💾 Export Config
                </button>
                <button class="btn btn-primary" onclick="showLogs()">
                    📋 View Logs
                </button>
            </div>
        </div>
    </div>

    <!-- Loading Overlay -->
    <div class="loading-overlay" id="loadingOverlay">
        <div class="spinner"></div>
    </div>

    <!-- Notification -->
    <div class="notification" id="notification"></div>

    <!-- Modal -->
    <div class="modal" id="modal">
        <div class="modal-content">
            <div class="modal-header">
                <h3 id="modalTitle">Modal Title</h3>
            </div>
            <div id="modalBody">
                <!-- Dynamic content -->
            </div>
            <div style="text-align: right; margin-top: 20px;">
                <button class="btn btn-primary" onclick="closeModal()">Close</button>
            </div>
        </div>
    </div>

    <script>
        class MediaServerDashboard {
            constructor() {
                this.apiUrl = '/api';
                this.profiles = {};
                this.services = {};
                this.health = {};
            }

            async init() {
                await this.loadProfiles();
                await this.refreshStatus();
                
                // Auto-refresh every 30 seconds
                setInterval(() => this.refreshStatus(), 30000);
            }

            async loadProfiles() {
                this.profiles = {
                    core: {
                        name: 'Core Infrastructure',
                        description: 'Essential services (Traefik, Auth, Database)',
                        services: ['traefik', 'authelia', 'postgres', 'redis'],
                        color: '#dc3545',
                        required: true
                    },
                    media: {
                        name: 'Media Streaming',
                        description: 'Jellyfin, Plex, Emby for streaming',
                        services: ['jellyfin', 'plex', 'emby'],
                        color: '#28a745'
                    },
                    music: {
                        name: 'Music Services',
                        description: 'Music streaming with Navidrome',
                        services: ['navidrome'],
                        color: '#17a2b8'
                    },
                    books: {
                        name: 'Books & Comics',
                        description: 'E-books, audiobooks, and comics',
                        services: ['audiobookshelf', 'calibre-web', 'kavita'],
                        color: '#6610f2'
                    },
                    photos: {
                        name: 'Photo Management',
                        description: 'AI-powered photo organization',
                        services: ['immich-server', 'immich-microservices', 'immich-machine-learning'],
                        color: '#e83e8c'
                    },
                    automation: {
                        name: 'Content Automation',
                        description: 'Automated content management',
                        services: ['prowlarr', 'sonarr', 'radarr', 'lidarr', 'readarr', 'bazarr'],
                        color: '#007bff'
                    },
                    downloads: {
                        name: 'Download Clients',
                        description: 'Secure downloading with VPN',
                        services: ['vpn', 'qbittorrent', 'sabnzbd'],
                        color: '#fd7e14'
                    },
                    requests: {
                        name: 'Request Management',
                        description: 'User request handling',
                        services: ['overseerr', 'jellyseerr'],
                        color: '#20c997'
                    },
                    monitoring: {
                        name: 'Monitoring',
                        description: 'System monitoring and analytics',
                        services: ['prometheus', 'grafana', 'tautulli'],
                        color: '#6f42c1'
                    },
                    management: {
                        name: 'Management',
                        description: 'Admin dashboards',
                        services: ['portainer', 'homepage', 'homarr'],
                        color: '#795548'
                    },
                    processing: {
                        name: 'Media Processing',
                        description: 'Transcoding and optimization',
                        services: ['tdarr', 'handbrake'],
                        color: '#ff6b6b'
                    },
                    backup: {
                        name: 'Backup Services',
                        description: 'Automated backups',
                        services: ['duplicati'],
                        color: '#4ecdc4'
                    },
                    advanced: {
                        name: 'Advanced Features',
                        description: 'AI/ML and experimental',
                        services: ['ai-ml-nexus'],
                        color: '#a8e6cf'
                    }
                };
            }

            async refreshStatus() {
                this.showLoading(true);
                
                try {
                    const response = await fetch(`${this.apiUrl}/status`);
                    const data = await response.json();
                    
                    this.services = data.status;
                    this.state = data.state;
                    this.health = data.health;
                    
                    this.updateUI();
                } catch (error) {
                    this.showNotification('Failed to refresh status', 'error');
                } finally {
                    this.showLoading(false);
                }
            }

            updateUI() {
                // Update stats
                const totalServices = Object.keys(this.services).length;
                const runningServices = Object.values(this.services).filter(s => s.state === 'running').length;
                const enabledProfiles = this.state.profiles.length;
                
                document.getElementById('totalServices').textContent = totalServices;
                document.getElementById('runningServices').textContent = runningServices;
                document.getElementById('enabledProfiles').textContent = enabledProfiles;
                document.getElementById('systemHealth').textContent = this.health.overall || 'N/A';

                // Update health grid
                this.updateHealthGrid();
                
                // Update profiles grid
                this.updateProfilesGrid();
            }

            updateHealthGrid() {
                const healthGrid = document.getElementById('healthGrid');
                healthGrid.innerHTML = '';

                Object.entries(this.health.profiles || {}).forEach(([profileName, profileHealth]) => {
                    const healthItem = document.createElement('div');
                    healthItem.className = `health-item ${profileHealth.status}`;
                    
                    let icon = '✅';
                    if (profileHealth.status === 'degraded') icon = '⚠️';
                    if (profileHealth.status === 'critical') icon = '❌';
                    
                    healthItem.innerHTML = `
                        <div class="health-icon">${icon}</div>
                        <div>${this.profiles[profileName]?.name || profileName}</div>
                        <div style="font-size: 0.9em; color: #718096; margin-top: 5px;">
                            ${Object.values(profileHealth.services).filter(s => s === 'healthy').length} / 
                            ${Object.keys(profileHealth.services).length} healthy
                        </div>
                    `;
                    
                    healthGrid.appendChild(healthItem);
                });
            }

            updateProfilesGrid() {
                const profilesGrid = document.getElementById('profilesGrid');
                profilesGrid.innerHTML = '';

                Object.entries(this.profiles).forEach(([profileKey, profile]) => {
                    const isEnabled = this.state.profiles.includes(profileKey);
                    
                    const card = document.createElement('div');
                    card.className = `profile-card ${isEnabled ? 'enabled' : ''}`;
                    card.style.borderLeftColor = profile.color;
                    
                    const servicesHtml = profile.services.map(service => {
                        const status = this.services[service];
                        const statusClass = status?.state === 'running' ? 'status-running' : 
                                          status?.state === 'starting' ? 'status-starting' : 'status-stopped';
                        const statusText = status?.state || 'not deployed';
                        
                        return `
                            <div class="service-item">
                                <span class="service-name">${service}</span>
                                <span class="service-status ${statusClass}">${statusText}</span>
                            </div>
                        `;
                    }).join('');
                    
                    card.innerHTML = `
                        <div class="profile-header">
                            <div>
                                <div class="profile-name">${profile.name}</div>
                            </div>
                            <div class="toggle-switch ${isEnabled ? 'active' : ''}" 
                                 onclick="dashboard.toggleProfile('${profileKey}')"
                                 ${profile.required ? 'style="opacity: 0.5; cursor: not-allowed;"' : ''}></div>
                        </div>
                        <div class="profile-description">${profile.description}</div>
                        <div class="services-list">
                            ${servicesHtml}
                        </div>
                    `;
                    
                    profilesGrid.appendChild(card);
                });
            }

            async toggleProfile(profileKey) {
                const profile = this.profiles[profileKey];
                if (profile.required) {
                    this.showNotification('Core profile cannot be disabled', 'warning');
                    return;
                }

                const isEnabled = this.state.profiles.includes(profileKey);
                const action = isEnabled ? 'disable' : 'enable';
                
                this.showLoading(true);
                
                try {
                    const response = await fetch(`${this.apiUrl}/profiles/${profileKey}/${action}`, {
                        method: 'POST'
                    });
                    
                    if (response.ok) {
                        this.showNotification(`Profile ${profile.name} ${action}d successfully`, 'success');
                        await this.refreshStatus();
                    } else {
                        const error = await response.json();
                        this.showNotification(error.error, 'error');
                    }
                } catch (error) {
                    this.showNotification(`Failed to ${action} profile`, 'error');
                } finally {
                    this.showLoading(false);
                }
            }

            async applyPreset(presetName) {
                const presets = {
                    basic: ['core', 'media', 'downloads', 'management'],
                    movies_tv: ['core', 'media', 'automation', 'downloads', 'requests', 'management'],
                    music_lover: ['core', 'music', 'automation', 'downloads', 'management'],
                    photographer: ['core', 'photos', 'backup', 'management'],
                    bookworm: ['core', 'books', 'automation', 'downloads', 'management'],
                    power_user: Object.keys(this.profiles).filter(p => p !== 'advanced')
                };

                const targetProfiles = presets[presetName];
                if (!targetProfiles) return;

                this.showLoading(true);

                try {
                    // Disable profiles not in preset
                    for (const profile of this.state.profiles) {
                        if (!targetProfiles.includes(profile) && profile !== 'core') {
                            await fetch(`${this.apiUrl}/profiles/${profile}/disable`, { method: 'POST' });
                        }
                    }

                    // Enable profiles in preset
                    for (const profile of targetProfiles) {
                        if (!this.state.profiles.includes(profile)) {
                            await fetch(`${this.apiUrl}/profiles/${profile}/enable`, { method: 'POST' });
                        }
                    }

                    this.showNotification(`Applied ${presetName} preset successfully`, 'success');
                    await this.refreshStatus();
                } catch (error) {
                    this.showNotification('Failed to apply preset', 'error');
                } finally {
                    this.showLoading(false);
                }
            }

            showDependencies() {
                const modalTitle = document.getElementById('modalTitle');
                const modalBody = document.getElementById('modalBody');
                
                modalTitle.textContent = 'Service Dependencies';
                
                let html = '<div class="dependency-tree">';
                Object.entries(this.profiles).forEach(([profileKey, profile]) => {
                    html += `<h4 style="color: ${profile.color}; margin-bottom: 10px;">${profile.name}</h4>`;
                    html += '<ul>';
                    profile.services.forEach(service => {
                        html += `<li>${service}`;
                        const deps = this.getServiceDependencies(service);
                        if (deps.length > 0) {
                            html += '<ul>';
                            deps.forEach(dep => {
                                html += `<li class="dependency-item">→ ${dep}</li>`;
                            });
                            html += '</ul>';
                        }
                        html += '</li>';
                    });
                    html += '</ul>';
                });
                html += '</div>';
                
                modalBody.innerHTML = html;
                this.showModal();
            }

            getServiceDependencies(service) {
                const dependencies = {
                    'authelia': ['redis', 'postgres'],
                    'sonarr': ['prowlarr'],
                    'radarr': ['prowlarr'],
                    'lidarr': ['prowlarr'],
                    'readarr': ['prowlarr'],
                    'bazarr': ['sonarr', 'radarr'],
                    'qbittorrent': ['vpn'],
                    'overseerr': ['sonarr', 'radarr'],
                    'jellyseerr': ['jellyfin', 'sonarr', 'radarr'],
                    'grafana': ['prometheus'],
                    'immich-microservices': ['immich-server', 'postgres', 'redis'],
                    'immich-machine-learning': ['immich-server']
                };
                
                return dependencies[service] || [];
            }

            exportConfiguration() {
                const config = {
                    profiles: this.state.profiles,
                    services: this.services,
                    health: this.health,
                    timestamp: new Date().toISOString()
                };
                
                const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `media-server-config-${new Date().toISOString().split('T')[0]}.json`;
                a.click();
                
                this.showNotification('Configuration exported successfully', 'success');
            }

            showLogs() {
                // This would typically open a logs viewer
                this.showNotification('Logs viewer coming soon', 'warning');
            }

            showModal() {
                document.getElementById('modal').classList.add('show');
            }

            closeModal() {
                document.getElementById('modal').classList.remove('show');
            }

            showLoading(show) {
                document.getElementById('loadingOverlay').classList.toggle('show', show);
            }

            showNotification(message, type = 'success') {
                const notification = document.getElementById('notification');
                notification.textContent = message;
                notification.className = `notification ${type} show`;
                
                setTimeout(() => {
                    notification.classList.remove('show');
                }, 5000);
            }
        }

        // Initialize dashboard
        const dashboard = new MediaServerDashboard();
        dashboard.init();

        // Global functions for HTML onclick handlers
        function applyPreset(preset) {
            dashboard.applyPreset(preset);
        }

        function refreshStatus() {
            dashboard.refreshStatus();
        }

        function showDependencies() {
            dashboard.showDependencies();
        }

        function exportConfiguration() {
            dashboard.exportConfiguration();
        }

        function showLogs() {
            dashboard.showLogs();
        }

        function closeModal() {
            dashboard.closeModal();
        }
    </script>
</body>
</html>
```

### 4. CLI Management Tool

```bash
#!/bin/bash
# unified-media-manager.sh

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.unified.yml"
CONFIG_FILE="$SCRIPT_DIR/config/service-state.json"
ENV_FILE="$SCRIPT_DIR/.env"

# Profile definitions
declare -A PROFILES=(
    ["core"]="Core Infrastructure (Required)"
    ["media"]="Media Streaming (Jellyfin, Plex, Emby)"
    ["music"]="Music Services (Navidrome)"
    ["books"]="Books & Comics (Audiobookshelf, Calibre, Kavita)"
    ["photos"]="Photo Management (Immich)"
    ["automation"]="Content Automation (Arr Stack)"
    ["downloads"]="Download Clients (VPN, qBittorrent, SABnzbd)"
    ["requests"]="Request Management (Overseerr, Jellyseerr)"
    ["monitoring"]="Monitoring & Analytics"
    ["management"]="Admin Dashboards"
    ["processing"]="Media Processing (Transcoding)"
    ["backup"]="Backup Services"
    ["advanced"]="Advanced Features (AI/ML)"
)

# Presets
declare -A PRESETS=(
    ["basic"]="core,media,downloads,management"
    ["movies_tv"]="core,media,automation,downloads,requests,management"
    ["music_lover"]="core,music,automation,downloads,management"
    ["photographer"]="core,photos,backup,management"
    ["bookworm"]="core,books,automation,downloads,management"
    ["power_user"]="core,media,music,books,photos,automation,downloads,requests,monitoring,management,processing"
)

# Logging
log() {
    local level=$1
    shift
    local message="$*"
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    case $level in
        "INFO")  echo -e "${GREEN}[INFO]${NC} ${timestamp} - $message" ;;
        "WARN")  echo -e "${YELLOW}[WARN]${NC} ${timestamp} - $message" ;;
        "ERROR") echo -e "${RED}[ERROR]${NC} ${timestamp} - $message" ;;
        "DEBUG") echo -e "${BLUE}[DEBUG]${NC} ${timestamp} - $message" ;;
    esac
}

# Check dependencies
check_dependencies() {
    local missing_deps=()
    
    for cmd in docker docker-compose jq curl; do
        if ! command -v $cmd &> /dev/null; then
            missing_deps+=($cmd)
        fi
    done
    
    if [[ ${#missing_deps[@]} -gt 0 ]]; then
        log ERROR "Missing dependencies: ${missing_deps[*]}"
        log INFO "Install missing dependencies and try again"
        exit 1
    fi
}

# Initialize configuration
initialize() {
    mkdir -p "$SCRIPT_DIR/config" "$SCRIPT_DIR/data" "$SCRIPT_DIR/backups"
    
    if [[ ! -f "$CONFIG_FILE" ]]; then
        echo '{"profiles": ["core"], "services": {}}' > "$CONFIG_FILE"
        log INFO "Created initial configuration"
    fi
    
    if [[ ! -f "$ENV_FILE" ]]; then
        log WARN ".env file not found. Creating from template..."
        if [[ -f "$SCRIPT_DIR/.env.template" ]]; then
            cp "$SCRIPT_DIR/.env.template" "$ENV_FILE"
            log INFO "Created .env file from template. Please edit it with your values."
        else
            log ERROR ".env.template not found. Please create .env file manually."
            exit 1
        fi
    fi
}

# Load current state
load_state() {
    if [[ -f "$CONFIG_FILE" ]]; then
        cat "$CONFIG_FILE"
    else
        echo '{"profiles": [], "services": {}}'
    fi
}

# Save state
save_state() {
    local state="$1"
    echo "$state" | jq '.' > "$CONFIG_FILE"
}

# Get enabled profiles
get_enabled_profiles() {
    load_state | jq -r '.profiles[]' 2>/dev/null || echo ""
}

# Enable profile
enable_profile() {
    local profile="$1"
    
    if [[ -z "${PROFILES[$profile]}" ]]; then
        log ERROR "Unknown profile: $profile"
        return 1
    fi
    
    local current_state=$(load_state)
    
    # Check if already enabled
    if echo "$current_state" | jq -e --arg p "$profile" '.profiles | contains([$p])' > /dev/null; then
        log INFO "Profile '$profile' is already enabled"
        return 0
    fi
    
    log INFO "Enabling profile: $profile - ${PROFILES[$profile]}"
    
    # Add profile to state
    local new_state=$(echo "$current_state" | jq --arg p "$profile" '.profiles += [$p] | .profiles |= unique')
    save_state "$new_state"
    
    # Apply configuration
    apply_configuration
}

# Disable profile
disable_profile() {
    local profile="$1"
    
    if [[ "$profile" == "core" ]]; then
        log ERROR "Core profile cannot be disabled"
        return 1
    fi
    
    local current_state=$(load_state)
    
    # Check if enabled
    if ! echo "$current_state" | jq -e --arg p "$profile" '.profiles | contains([$p])' > /dev/null; then
        log INFO "Profile '$profile' is not enabled"
        return 0
    fi
    
    log INFO "Disabling profile: $profile"
    
    # Remove profile from state
    local new_state=$(echo "$current_state" | jq --arg p "$profile" '.profiles -= [$p]')
    save_state "$new_state"
    
    # Apply configuration
    apply_configuration
}

# Apply configuration
apply_configuration() {
    local profiles=$(get_enabled_profiles | tr '\n' ' ')
    local profile_args=""
    
    for profile in $profiles; do
        profile_args="$profile_args --profile $profile"
    done
    
    log INFO "Applying configuration with profiles: $profiles"
    
    docker-compose -f "$COMPOSE_FILE" $profile_args up -d --remove-orphans
}

# Apply preset
apply_preset() {
    local preset="$1"
    
    if [[ -z "${PRESETS[$preset]}" ]]; then
        log ERROR "Unknown preset: $preset"
        return 1
    fi
    
    log INFO "Applying preset: $preset"
    
    # Parse preset profiles
    IFS=',' read -ra preset_profiles <<< "${PRESETS[$preset]}"
    
    # Disable all non-core profiles first
    local current_profiles=$(get_enabled_profiles)
    for profile in $current_profiles; do
        if [[ "$profile" != "core" ]] && [[ ! " ${preset_profiles[@]} " =~ " ${profile} " ]]; then
            disable_profile "$profile"
        fi
    done
    
    # Enable preset profiles
    for profile in "${preset_profiles[@]}"; do
        enable_profile "$profile"
    done
    
    log INFO "Preset '$preset' applied successfully"
}

# Show status
show_status() {
    echo -e "\n${CYAN}=== Unified Media Server Status ===${NC}\n"
    
    # Show enabled profiles
    echo -e "${YELLOW}Enabled Profiles:${NC}"
    local profiles=$(get_enabled_profiles)
    if [[ -z "$profiles" ]]; then
        echo "  None"
    else
        for profile in $profiles; do
            echo -e "  ${GREEN}✓${NC} $profile - ${PROFILES[$profile]}"
        done
    fi
    
    echo -e "\n${YELLOW}Service Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps --format table
}

# List profiles
list_profiles() {
    echo -e "\n${CYAN}=== Available Profiles ===${NC}\n"
    
    local enabled_profiles=$(get_enabled_profiles)
    
    for profile in "${!PROFILES[@]}"; do
        local status="${RED}✗${NC}"
        if echo "$enabled_profiles" | grep -q "^$profile$"; then
            status="${GREEN}✓${NC}"
        fi
        echo -e "  $status $profile - ${PROFILES[$profile]}"
    done
}

# List presets
list_presets() {
    echo -e "\n${CYAN}=== Available Presets ===${NC}\n"
    
    for preset in "${!PRESETS[@]}"; do
        echo -e "  ${PURPLE}$preset${NC}:"
        IFS=',' read -ra profiles <<< "${PRESETS[$preset]}"
        for profile in "${profiles[@]}"; do
            echo -e "    - $profile"
        done
        echo
    done
}

# Create backup
create_backup() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local backup_dir="$SCRIPT_DIR/backups/$timestamp"
    
    log INFO "Creating backup..."
    
    mkdir -p "$backup_dir"
    
    # Backup configuration
    cp -r "$SCRIPT_DIR/config" "$backup_dir/"
    cp "$CONFIG_FILE" "$backup_dir/service-state.json"
    cp "$ENV_FILE" "$backup_dir/.env" 2>/dev/null || true
    
    # Backup docker-compose files
    cp "$COMPOSE_FILE" "$backup_dir/"
    
    # Create backup metadata
    cat > "$backup_dir/metadata.json" << EOF
{
  "timestamp": "$(date -Iseconds)",
  "profiles": $(load_state | jq -c '.profiles'),
  "version": "1.0.0"
}
EOF
    
    log INFO "Backup created: $backup_dir"
    echo "$backup_dir"
}

# Restore backup
restore_backup() {
    local backup_path="$1"
    
    if [[ ! -d "$backup_path" ]]; then
        log ERROR "Backup directory not found: $backup_path"
        return 1
    fi
    
    log INFO "Restoring from backup: $backup_path"
    
    # Stop all services
    docker-compose -f "$COMPOSE_FILE" down
    
    # Restore configuration
    cp -r "$backup_path/config/"* "$SCRIPT_DIR/config/" 2>/dev/null || true
    cp "$backup_path/service-state.json" "$CONFIG_FILE"
    
    # Apply restored configuration
    apply_configuration
    
    log INFO "Backup restored successfully"
}

# Show health check
health_check() {
    echo -e "\n${CYAN}=== System Health Check ===${NC}\n"
    
    local profiles=$(get_enabled_profiles)
    local all_healthy=true
    
    for profile in $profiles; do
        echo -e "${YELLOW}Profile: $profile${NC}"
        
        # Get services for profile (simplified for this example)
        local services=""
        case $profile in
            "core") services="traefik authelia postgres redis" ;;
            "media") services="jellyfin plex emby" ;;
            "automation") services="prowlarr sonarr radarr" ;;
            "downloads") services="vpn qbittorrent" ;;
            "monitoring") services="prometheus grafana" ;;
            "management") services="portainer homepage" ;;
        esac
        
        for service in $services; do
            if docker ps --format "table {{.Names}}" | grep -q "^$service$"; then
                echo -e "  ${GREEN}✓${NC} $service: Running"
            else
                echo -e "  ${RED}✗${NC} $service: Not running"
                all_healthy=false
            fi
        done
        echo
    done
    
    if $all_healthy; then
        echo -e "${GREEN}Overall health: HEALTHY${NC}"
    else
        echo -e "${RED}Overall health: DEGRADED${NC}"
    fi
}

# Show help
show_help() {
    cat << EOF
${CYAN}Unified Media Server Manager${NC}

Usage: $0 [COMMAND] [OPTIONS]

${YELLOW}Commands:${NC}
  ${GREEN}Profile Management:${NC}
    status                    Show current status and enabled profiles
    list-profiles            List all available profiles
    enable <profile>         Enable a service profile
    disable <profile>        Disable a service profile
    
  ${GREEN}Preset Management:${NC}
    list-presets            List available presets
    preset <name>           Apply a preset configuration
    
  ${GREEN}Service Control:${NC}
    start                   Start all enabled services
    stop                    Stop all services
    restart                 Restart all services
    logs [service]          Show logs (all services or specific)
    
  ${GREEN}Maintenance:${NC}
    health                  Run health check
    backup                  Create configuration backup
    restore <path>          Restore from backup
    update                  Update all container images
    
  ${GREEN}General:${NC}
    help                    Show this help

${YELLOW}Available Profiles:${NC}
  core          - Core infrastructure (required)
  media         - Media streaming services
  music         - Music streaming
  books         - E-books and audiobooks
  photos        - Photo management
  automation    - Content automation (arr stack)
  downloads     - Download clients
  requests      - Request management
  monitoring    - System monitoring
  management    - Admin dashboards
  processing    - Media processing
  backup        - Backup services
  advanced      - AI/ML features

${YELLOW}Available Presets:${NC}
  basic         - Essential media setup
  movies_tv     - Movies and TV automation
  music_lover   - Music focused setup
  photographer  - Photo management setup
  bookworm      - Books and audiobooks
  power_user    - Full featured setup

${YELLOW}Examples:${NC}
  $0 status
  $0 enable media
  $0 preset movies_tv
  $0 logs jellyfin
  $0 backup

EOF
}

# Main command handler
main() {
    check_dependencies
    initialize
    
    case "${1:-help}" in
        "status")
            show_status
            ;;
        "list-profiles")
            list_profiles
            ;;
        "enable")
            if [[ -z "$2" ]]; then
                log ERROR "Profile name required"
                exit 1
            fi
            enable_profile "$2"
            ;;
        "disable")
            if [[ -z "$2" ]]; then
                log ERROR "Profile name required"
                exit 1
            fi
            disable_profile "$2"
            ;;
        "list-presets")
            list_presets
            ;;
        "preset")
            if [[ -z "$2" ]]; then
                log ERROR "Preset name required"
                exit 1
            fi
            apply_preset "$2"
            ;;
        "start")
            apply_configuration
            ;;
        "stop")
            docker-compose -f "$COMPOSE_FILE" down
            ;;
        "restart")
            docker-compose -f "$COMPOSE_FILE" restart
            ;;
        "logs")
            if [[ -n "$2" ]]; then
                docker-compose -f "$COMPOSE_FILE" logs -f "$2"
            else
                docker-compose -f "$COMPOSE_FILE" logs -f
            fi
            ;;
        "health")
            health_check
            ;;
        "backup")
            create_backup
            ;;
        "restore")
            if [[ -z "$2" ]]; then
                log ERROR "Backup path required"
                exit 1
            fi
            restore_backup "$2"
            ;;
        "update")
            log INFO "Pulling latest images..."
            docker-compose -f "$COMPOSE_FILE" pull
            apply_configuration
            ;;
        "help"|"-h"|"--help")
            show_help
            ;;
        *)
            log ERROR "Unknown command: $1"
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"
```

### 5. Environment Template

```bash
# .env.unified
# Unified Media Server Environment Configuration
# ==============================================

# Basic Configuration
TZ=America/New_York
PUID=1000
PGID=1000

# Domain Configuration
DOMAIN=example.com
ACME_EMAIL=admin@example.com

# Security
POSTGRES_PASSWORD=secure_postgres_password_here
REDIS_PASSWORD=secure_redis_password_here
AUTHELIA_JWT_SECRET=secure_jwt_secret_here
AUTHELIA_SESSION_SECRET=secure_session_secret_here
AUTHELIA_STORAGE_ENCRYPTION_KEY=secure_encryption_key_here

# VPN Configuration (for downloads)
VPN_PROVIDER=mullvad
VPN_TYPE=wireguard
VPN_PRIVATE_KEY=your_vpn_private_key_here
VPN_ADDRESSES=10.64.0.0/32

# Media Paths
MEDIA_PATH=/path/to/media
DOWNLOADS_PATH=/path/to/downloads
MUSIC_PATH=/path/to/media/music
BOOKS_PATH=/path/to/media/books
AUDIOBOOKS_PATH=/path/to/media/audiobooks
PODCASTS_PATH=/path/to/media/podcasts
COMICS_PATH=/path/to/media/comics
MANGA_PATH=/path/to/media/manga
PHOTOS_PATH=/path/to/media/photos
USENET_PATH=/path/to/downloads/usenet

# Service Specific
PLEX_CLAIM=claim-token-from-plex.tv
GRAFANA_USER=admin
GRAFANA_PASSWORD=secure_grafana_password

# Performance
JELLYFIN_HARDWARE_ACCELERATION=intel # Options: none, intel, nvidia, amd

# Optional Features
ENABLE_PLEX=false
ENABLE_EMBY=false
ENABLE_ADVANCED_FEATURES=false
```

## Implementation Guide

### Step 1: Backup Current Setup
```bash
# Create complete backup
mkdir -p /backup/media-server-$(date +%Y%m%d)
cd /backup/media-server-$(date +%Y%m%d)

# Backup all compose files
cp -r /path/to/newmedia/*.yml .

# Backup configurations
cp -r /path/to/newmedia/config .

# Backup environment files
cp /path/to/newmedia/.env* .
```

### Step 2: Deploy Unified Architecture
```bash
# Clone or create new directory
mkdir -p /opt/unified-media-server
cd /opt/unified-media-server

# Copy unified files
cp docker-compose.unified.yml docker-compose.yml
cp .env.unified .env
cp service-manager.js .
cp unified-media-manager.sh .
chmod +x unified-media-manager.sh

# Copy existing configurations
cp -r /path/to/newmedia/config/* ./config/

# Initialize
./unified-media-manager.sh status
```

### Step 3: Migrate Services
```bash
# Start with core
./unified-media-manager.sh enable core

# Enable desired profiles based on needs
./unified-media-manager.sh preset movies_tv  # For movie/TV focus
# OR
./unified-media-manager.sh preset power_user  # For everything

# Check health
./unified-media-manager.sh health
```

### Step 4: Access Services
```bash
# View all service URLs
./unified-media-manager.sh status

# Access dashboard
open https://dashboard.example.com
```

## Key Benefits

1. **Unified Management**
   - Single compose file with profile-based organization
   - Consistent naming and networking
   - Centralized configuration

2. **Selective Activation**
   - Enable only needed services
   - Resource optimization
   - Cost control for cloud deployments

3. **Modern Architecture**
   - Docker Compose profiles (2025 standard)
   - Health checks on all services
   - Proper security implementation
   - Hardware acceleration support

4. **Easy Maintenance**
   - Single point of control
   - Automated dependency management
   - Built-in backup/restore
   - Health monitoring

5. **Scalability**
   - Add new services easily
   - Horizontal scaling ready
   - Cloud-native design

## Security Considerations

1. **Network Segmentation**
   - Isolated networks for different service types
   - VPN isolation for downloads
   - Internal-only networks for databases

2. **Authentication**
   - Authelia SSO for all services
   - MFA support
   - LDAP/AD integration ready

3. **Secrets Management**
   - Docker secrets for sensitive data
   - Environment variable separation
   - Encrypted storage

4. **Access Control**
   - Traefik with TLS termination
   - Cloudflare integration
   - Rate limiting and DDoS protection

## Performance Optimization

1. **Hardware Acceleration**
   - GPU support for transcoding
   - Intel QuickSync
   - NVIDIA/AMD GPU passthrough

2. **Caching**
   - Redis for session/metadata
   - RAM disk for transcoding
   - CDN integration ready

3. **Resource Limits**
   - CPU/Memory limits per service
   - Automatic scaling rules
   - Priority-based allocation

## Monitoring & Alerting

1. **Metrics Collection**
   - Prometheus for all services
   - Custom exporters included
   - Long-term storage

2. **Visualization**
   - Grafana dashboards
   - Service-specific views
   - Mobile-friendly design

3. **Alerting**
   - Health check failures
   - Resource exhaustion
   - Service degradation

## Conclusion

This unified architecture provides:
- Complete integration of all media services
- Modern Docker Compose profile management
- Enable/disable functionality per service group
- Production-ready security and monitoring
- Scalable and maintainable design

The system is ready for 2025 and beyond, incorporating the latest best practices and technologies while maintaining flexibility for future additions.