#!/bin/bash
# Create proper s6-overlay service definitions

echo "Creating s6-overlay service definitions..."

# Create service directories
mkdir -p s6-services/{jellyfin,sonarr,radarr,prowlarr,lidarr,bazarr,qbittorrent,redis,traefik,dashboard,ai-assistant}

# Jellyfin service
cat > s6-services/jellyfin/run << 'EOF'
#!/command/with-contenv bash
export JELLYFIN_DATA_DIR=/config/jellyfin
mkdir -p "$JELLYFIN_DATA_DIR" /logs/jellyfin
chown -R mediaserver:mediaserver "$JELLYFIN_DATA_DIR"
exec s6-setuidgid mediaserver /usr/bin/jellyfin \
    --datadir "$JELLYFIN_DATA_DIR" \
    --configdir "$JELLYFIN_DATA_DIR/config" \
    --logdir /logs/jellyfin
EOF

# Sonarr service
cat > s6-services/sonarr/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /config/sonarr /logs/sonarr
chown -R mediaserver:mediaserver /config/sonarr
exec s6-setuidgid mediaserver /opt/sonarr/Sonarr \
    -nobrowser \
    -data=/config/sonarr
EOF

# Radarr service
cat > s6-services/radarr/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /config/radarr /logs/radarr
chown -R mediaserver:mediaserver /config/radarr
exec s6-setuidgid mediaserver /opt/radarr/Radarr \
    -nobrowser \
    -data=/config/radarr
EOF

# Prowlarr service
cat > s6-services/prowlarr/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /config/prowlarr /logs/prowlarr
chown -R mediaserver:mediaserver /config/prowlarr
exec s6-setuidgid mediaserver /opt/prowlarr/Prowlarr \
    -nobrowser \
    -data=/config/prowlarr
EOF

# Lidarr service
cat > s6-services/lidarr/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /config/lidarr /logs/lidarr
chown -R mediaserver:mediaserver /config/lidarr
exec s6-setuidgid mediaserver /opt/lidarr/Lidarr \
    -nobrowser \
    -data=/config/lidarr
EOF

# Bazarr service
cat > s6-services/bazarr/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /config/bazarr /logs/bazarr
chown -R mediaserver:mediaserver /config/bazarr
cd /opt/bazarr
exec s6-setuidgid mediaserver python3 bazarr.py \
    --config /config/bazarr \
    --no-update
EOF

# qBittorrent service
cat > s6-services/qbittorrent/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /config/qbittorrent /downloads/{complete,incomplete}
chown -R mediaserver:mediaserver /config/qbittorrent /downloads
exec s6-setuidgid mediaserver qbittorrent-nox \
    --webui-port=8080 \
    --profile=/config/qbittorrent
EOF

# Redis service
cat > s6-services/redis/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /data/redis
chown -R mediaserver:mediaserver /data/redis
exec s6-setuidgid mediaserver redis-server \
    --dir /data/redis \
    --bind 127.0.0.1 \
    --appendonly yes
EOF

# Traefik service
cat > s6-services/traefik/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /config/traefik /logs/traefik
exec traefik \
    --configfile=/config/traefik/traefik.yml \
    --log.filepath=/logs/traefik/traefik.log
EOF

# Dashboard service
cat > s6-services/dashboard/run << 'EOF'
#!/command/with-contenv bash
cd /app/dashboard
exec s6-setuidgid mediaserver npm start
EOF

# AI Assistant service
cat > s6-services/ai-assistant/run << 'EOF'
#!/command/with-contenv bash
mkdir -p /config/ollama /logs/ai
export OLLAMA_MODELS=/config/ollama
cd /app/ai-services
exec s6-setuidgid mediaserver python3 main.py
EOF

# Make all run scripts executable
chmod +x s6-services/*/run

# Create dependencies
mkdir -p s6-services/{sonarr,radarr,lidarr}/dependencies.d
touch s6-services/sonarr/dependencies.d/prowlarr
touch s6-services/radarr/dependencies.d/prowlarr
touch s6-services/lidarr/dependencies.d/prowlarr

echo "✅ S6 service definitions created!"