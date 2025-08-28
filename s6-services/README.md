# S6-Overlay Service Configuration for Ultimate Media Server 2025

This directory contains the complete s6-overlay v3 service configuration for the Ultimate Media Server with 30+ services organized in a 6-tier dependency architecture.

## 🏗️ Service Architecture Overview

```
Tier 6: UI & Frontend
├── react-dashboard (React 18 + WebGL UI)
├── mcp-server (AI Assistant API)
└── websocket-service (Real-time updates)

Tier 5: Application Services  
├── ai-services (Safety, Recommendations, Content Analysis)
├── request-services (Overseerr, Jellyseerr, Ombi)
├── management-tools (Tautulli, Heimdall, Organizr)
├── content-libraries (Calibre, AudioBookshelf, Komga)
└── notification-services (Gotify, Diun)

Tier 4: Media Management & Downloads
├── arr-stack (Sonarr, Radarr, Prowlarr, Lidarr, Readarr, Bazarr)
└── download-clients (qBittorrent, Transmission, SABnzbd, NZBGet)

Tier 3: Media Servers
├── jellyfin (Primary FOSS media server)
├── plex (Optional premium media server) 
├── emby (Optional alternative media server)
├── photo-services (Immich, PhotoPrism)
├── document-services (Paperless-ngx)
└── cloud-services (Nextcloud)

Tier 2: Platform Services
├── traefik (Reverse proxy & SSL termination)
├── authelia (Single sign-on & authentication)
├── security-services (ClamAV, Fail2ban)
├── vpn-services (Gluetun, WireGuard)
├── dns-services (PiHole, AdGuard Home)
└── backup-services (Rsync, Duplicati)

Tier 1: Infrastructure Services
├── postgres (Primary database)
├── redis (Caching & sessions)
├── rabbitmq (Message queue)
├── elasticsearch (Search & logging)
├── prometheus (Metrics collection)
└── grafana (Metrics visualization)
```

## 📁 Directory Structure

```
s6-services/
├── infrastructure/
│   ├── postgres/
│   │   ├── type (longrun)
│   │   ├── run (service script)
│   │   └── finish (cleanup script)
│   ├── redis/
│   ├── rabbitmq/
│   ├── elasticsearch/
│   ├── prometheus/
│   └── grafana/
├── platform/
│   ├── traefik/
│   ├── authelia/
│   ├── security-monitor/
│   ├── gluetun/
│   ├── pihole/
│   └── backup-service/
├── media-core/
│   ├── jellyfin/
│   ├── plex/ (optional)
│   ├── emby/ (optional)
│   ├── immich/
│   ├── paperless-ngx/
│   └── nextcloud/
├── media-management/
│   ├── prowlarr/
│   ├── sonarr/
│   ├── radarr/
│   ├── lidarr/
│   ├── readarr/
│   └── bazarr/
├── download-clients/
│   ├── qbittorrent/
│   ├── transmission/
│   ├── sabnzbd/
│   └── nzbget/
├── application-services/
│   ├── ai-safety/
│   ├── recommendation-engine/
│   ├── content-analysis/
│   ├── overseerr/
│   ├── jellyseerr/
│   ├── ombi/
│   ├── tautulli/
│   ├── heimdall/
│   └── organizr/
├── frontend/
│   ├── react-dashboard/
│   ├── mcp-server/
│   └── websocket-service/
└── bundles/
    ├── infrastructure/
    ├── platform/
    ├── media-core/
    ├── media-management/
    ├── download-clients/
    ├── application-services/
    ├── frontend/
    └── user/
```

## 🔧 Service Configuration Details

### Infrastructure Services (Tier 1)

**PostgreSQL Database**
```bash
# /etc/s6-overlay/s6-rc.d/postgres/run
#!/command/with-contenv bash
exec postgres -D /opt/media-server/data/databases/postgres \
  -c config_file=/opt/media-server/config/infrastructure/postgresql.conf
```

**Redis Cache**
```bash  
# /etc/s6-overlay/s6-rc.d/redis/run
#!/command/with-contenv bash
exec redis-server /opt/media-server/config/infrastructure/redis.conf \
  --daemonize no
```

### Platform Services (Tier 2)

**Traefik Reverse Proxy**
```bash
# /etc/s6-overlay/s6-rc.d/traefik/run
#!/command/with-contenv bash
exec traefik \
  --configfile=/opt/media-server/config/platform/traefik.yml \
  --log.level=INFO
```

**Authelia SSO**
```bash
# /etc/s6-overlay/s6-rc.d/authelia/run  
#!/command/with-contenv bash
exec authelia \
  --config=/opt/media-server/config/platform/authelia/configuration.yml
```

### Media Server Services (Tier 3)

**Jellyfin**
```bash
# /etc/s6-overlay/s6-rc.d/jellyfin/run
#!/command/with-contenv bash
export JELLYFIN_DATA_DIR=/opt/media-server/config/media/jellyfin
export JELLYFIN_CACHE_DIR=/opt/media-server/cache/jellyfin

exec jellyfin \
  --datadir $JELLYFIN_DATA_DIR \
  --cachedir $JELLYFIN_CACHE_DIR \
  --webdir /usr/share/jellyfin/web
```

### *arr Stack Services (Tier 4)

**Sonarr**
```bash
# /etc/s6-overlay/s6-rc.d/sonarr/run
#!/command/with-contenv bash
export HOME=/opt/media-server/config/management/sonarr
exec /opt/arr-apps/Sonarr/Sonarr \
  -nobrowser \
  -data=/opt/media-server/config/management/sonarr
```

**Prowlarr**
```bash
# /etc/s6-overlay/s6-rc.d/prowlarr/run
#!/command/with-contenv bash
export HOME=/opt/media-server/config/management/prowlarr
exec /opt/arr-apps/Prowlarr/Prowlarr \
  -nobrowser \
  -data=/opt/media-server/config/management/prowlarr
```

### AI Services (Tier 5)

**AI Safety System**
```bash
# /etc/s6-overlay/s6-rc.d/ai-safety/run
#!/command/with-contenv bash
cd /opt/ai-services

export PYTHONPATH=/opt/ai-services/src
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/ai_services
export REDIS_URL=redis://localhost:6379/4

exec /opt/ai-services/venv/bin/python src/safety_system.py
```

### Frontend Services (Tier 6)

**React Dashboard**
```bash
# /etc/s6-overlay/s6-rc.d/react-dashboard/run
#!/command/with-contenv bash
cd /opt/media-server/ui/dashboard

export NODE_ENV=production
export PORT=3000
export API_BASE_URL=http://localhost:8090

exec node server.js
```

**MCP Server**
```bash
# /etc/s6-overlay/s6-rc.d/mcp-server/run
#!/command/with-contenv bash
cd /opt/media-server/mcp

export NODE_ENV=production
export PORT=8090

# Service URLs for integration
export JELLYFIN_URL=http://localhost:8096
export SONARR_URL=http://localhost:8989
export RADARR_URL=http://localhost:7878
export PROWLARR_URL=http://localhost:9696
export QBITTORRENT_URL=http://localhost:8080

exec node src/index.js
```

## 🔗 Service Dependencies

### Dependency Chain Rules

1. **Infrastructure First**: PostgreSQL → Redis → RabbitMQ → Elasticsearch
2. **Platform Layer**: Traefik → Authelia → Security services (depend on infrastructure)
3. **Media Core**: Media servers (depend on platform + databases)
4. **Management Layer**: *arr services (depend on media core + download clients)
5. **Application Layer**: Request/AI services (depend on media management)
6. **Frontend Layer**: UI components (depend on all backend services)

### Critical Dependencies

```yaml
# Essential startup order
startup_sequence:
  phase_1: [postgres, redis]
  phase_2: [traefik, authelia] 
  phase_3: [jellyfin, prowlarr]
  phase_4: [sonarr, radarr, qbittorrent]
  phase_5: [overseerr, ai-services]
  phase_6: [react-dashboard, mcp-server]
```

## 🚀 Service Bundles

### Infrastructure Bundle
```bash
# /etc/s6-overlay/s6-rc.d/infrastructure/contents.d/
postgres
redis  
rabbitmq
elasticsearch
prometheus
grafana
```

### Platform Bundle
```bash
# /etc/s6-overlay/s6-rc.d/platform/contents.d/
traefik
authelia
security-monitor
gluetun
pihole
backup-service
```

### Media Core Bundle
```bash
# /etc/s6-overlay/s6-rc.d/media-core/contents.d/
jellyfin
plex
emby
immich
paperless-ngx
nextcloud
```

## 📊 Service Management Commands

### Manual Service Control
```bash
# Start/stop individual services
s6-rc -u change sonarr        # Start Sonarr
s6-rc -d change sonarr        # Stop Sonarr

# Start/stop service bundles
s6-rc -u change media-core    # Start all media servers
s6-rc -d change media-core    # Stop all media servers

# Check service status
s6-rc -l                      # List all services
s6-rc -a list                 # List active services
```

### Service Logs
```bash
# View service logs
s6-rc-oneshot-log jellyfin    # View Jellyfin logs
s6-rc-oneshot-log sonarr      # View Sonarr logs

# Tail logs in real-time
tail -f /var/log/services/jellyfin.log
tail -f /var/log/services/sonarr.log
```

## 🔧 Configuration Management

### Service Configuration Locations
```bash
/opt/media-server/config/
├── infrastructure/
│   ├── postgresql.conf
│   ├── redis.conf
│   └── elasticsearch.yml
├── platform/
│   ├── traefik.yml
│   ├── authelia/configuration.yml
│   └── dynamic/
├── media/
│   ├── jellyfin/
│   ├── plex/
│   └── emby/
├── management/
│   ├── sonarr/
│   ├── radarr/
│   └── prowlarr/
└── ai-services/
    ├── config.yaml
    └── models/
```

### Environment Variables
```bash
# Core settings
PUID=1000
PGID=1000  
TZ=UTC

# Feature flags
ENABLE_GPU=true
ENABLE_AI_SERVICES=true
ENABLE_PLEX=false
ENABLE_3D_UI=true

# Service URLs (auto-configured)
JELLYFIN_URL=http://localhost:8096
SONARR_URL=http://localhost:8989
RADARR_URL=http://localhost:7878
```

## 🛠️ Troubleshooting

### Common Issues

**Service won't start:**
```bash
# Check dependencies
s6-rc -l | grep <service-name>

# Check logs
cat /var/log/services/<service-name>.log

# Restart service
s6-rc -d change <service-name>
s6-rc -u change <service-name>
```

**Database connection issues:**
```bash
# Check PostgreSQL status
s6-rc -a list | grep postgres

# Test database connection
psql -h localhost -U postgres -l
```

**Permission problems:**
```bash
# Fix ownership
chown -R media-server:media-server /opt/media-server

# Fix permissions
chmod -R 755 /opt/media-server/config
chmod -R 755 /opt/media-server/data
```

This service configuration provides a robust, scalable foundation for running 30+ services in a single container with proper dependency management, health monitoring, and failure recovery.