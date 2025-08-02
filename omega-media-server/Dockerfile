# Omega Media Server 2025 - Ultimate All-in-One Container
FROM alpine:3.19 AS base

# Metadata
LABEL maintainer="Omega Team <team@omega-server.com>"
LABEL description="The Ultimate All-in-One Media Server Solution"
LABEL version="2025.1.0"

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PUID=1000 \
    PGID=1000 \
    TZ=UTC \
    OMEGA_VERSION=2025.1.0 \
    OMEGA_HOME=/opt/omega \
    OMEGA_CONFIG=/config \
    OMEGA_MEDIA=/media \
    OMEGA_APPS=/apps \
    NODE_ENV=production \
    PYTHONUNBUFFERED=1

# Install base system dependencies
RUN apk add --no-cache \
    # System utilities
    bash curl wget git openssh-client \
    # Process management
    supervisor s6-overlay \
    # Networking
    nginx haproxy wireguard-tools \
    # Development tools
    build-base python3 python3-dev py3-pip nodejs npm \
    # Media tools
    ffmpeg mediainfo \
    # Hardware acceleration support
    intel-media-driver mesa-va-drivers \
    # Database
    postgresql15 postgresql15-contrib redis \
    # Security
    openssl ca-certificates gnupg \
    # Monitoring
    htop iotop nethogs \
    # Archive tools
    unzip unrar p7zip \
    # Additional libraries
    libxml2 libxslt libjpeg-turbo libpng libwebp \
    && rm -rf /var/cache/apk/*

# Install Docker-in-Docker
RUN apk add --no-cache docker docker-compose \
    && rc-update add docker boot

# Install K3s for container orchestration
RUN curl -sfL https://get.k3s.io | sh -s - --disable-agent --data-dir /var/lib/k3s

# Create directory structure
RUN mkdir -p \
    ${OMEGA_HOME} \
    ${OMEGA_CONFIG} \
    ${OMEGA_MEDIA} \
    ${OMEGA_APPS} \
    /var/log/omega \
    /var/run/omega \
    /var/cache/omega \
    /var/lib/omega

# Install Node.js dependencies for management UI
WORKDIR ${OMEGA_HOME}
COPY package.json package-lock.json ./
RUN npm ci --production

# Install Python dependencies for AI/ML features
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=${PUID}:${PGID} . ${OMEGA_HOME}/

# Build the web UI
WORKDIR ${OMEGA_HOME}/web
RUN npm ci && npm run build

# Install media server applications
WORKDIR ${OMEGA_APPS}

# Jellyfin
RUN JELLYFIN_VERSION=$(curl -s https://api.github.com/repos/jellyfin/jellyfin/releases/latest | grep tag_name | cut -d '"' -f 4) && \
    wget -O jellyfin.tar.gz "https://repo.jellyfin.org/releases/server/linux/stable/combined/jellyfin_${JELLYFIN_VERSION}_linux-amd64.tar.gz" && \
    tar xzf jellyfin.tar.gz && rm jellyfin.tar.gz

# Plex (optional, requires Plex Pass for some features)
RUN wget -O plexmediaserver.deb "https://downloads.plex.tv/plex-media-server-new/1.32.8.7639-fb6452ebf/debian/plexmediaserver_1.32.8.7639-fb6452ebf_amd64.deb" && \
    dpkg -x plexmediaserver.deb plex && \
    rm plexmediaserver.deb

# *arr Suite
RUN for app in radarr sonarr lidarr readarr prowlarr bazarr; do \
        mkdir -p /apps/${app} && \
        wget -O - "https://github.com/Servarr/${app^}/releases/latest/download/${app^}.linux-musl-x64.tar.gz" | \
        tar xzf - -C /apps/${app}; \
    done

# Additional media applications
RUN \
    # qBittorrent
    apk add --no-cache qbittorrent-nox && \
    # PhotoPrism
    wget -O photoprism.tar.gz "https://github.com/photoprism/photoprism/releases/latest/download/photoprism_linux_amd64.tar.gz" && \
    tar xzf photoprism.tar.gz -C /apps/ && rm photoprism.tar.gz && \
    # Navidrome
    wget -O navidrome.tar.gz "https://github.com/navidrome/navidrome/releases/latest/download/navidrome_Linux_x86_64.tar.gz" && \
    mkdir -p /apps/navidrome && tar xzf navidrome.tar.gz -C /apps/navidrome && rm navidrome.tar.gz

# Install AI/ML models
WORKDIR ${OMEGA_HOME}/models
RUN \
    # Download recommendation model
    wget -O recommendation_model.pb "https://omega-models.s3.amazonaws.com/recommendation_v2.pb" && \
    # Download subtitle model (Whisper)
    python3 -c "import whisper; whisper.load_model('base')" && \
    # Download image recognition model
    wget -O image_model.onnx "https://omega-models.s3.amazonaws.com/clip_vit_b32.onnx"

# Configure Nginx
COPY nginx.conf /etc/nginx/nginx.conf
COPY nginx-sites/* /etc/nginx/sites-available/

# Configure supervisord
COPY supervisord.conf /etc/supervisord.conf
COPY supervisor.d/* /etc/supervisor.d/

# Configure services
COPY services/* /etc/s6-overlay/s6-rc.d/

# Setup SSL certificates directory
RUN mkdir -p /etc/letsencrypt/live

# Create startup script
COPY scripts/startup.sh /usr/local/bin/omega-start
RUN chmod +x /usr/local/bin/omega-start

# Create health check script
COPY scripts/healthcheck.sh /usr/local/bin/omega-health
RUN chmod +x /usr/local/bin/omega-health

# Setup cron jobs for maintenance
COPY crontab /etc/crontabs/root

# Expose ports
EXPOSE 80 443 \
       32400 32410 32412 32413 32414 \
       8096 8920 \
       7878 8989 8686 6767 9696 \
       51413 51413/udp \
       1900/udp 3005 5353/udp 8324 8324/udp

# Setup volumes
VOLUME ["${OMEGA_CONFIG}", "${OMEGA_MEDIA}", "/downloads", "/transcode"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/omega-health || exit 1

# Set working directory
WORKDIR ${OMEGA_HOME}

# Switch to non-root user
RUN addgroup -g ${PGID} omega && \
    adduser -D -u ${PUID} -G omega omega && \
    chown -R omega:omega ${OMEGA_HOME} ${OMEGA_CONFIG} ${OMEGA_APPS}

# Entry point
ENTRYPOINT ["/init"]
CMD ["/usr/local/bin/omega-start"]