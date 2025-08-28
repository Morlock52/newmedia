# Ultimate All-in-One Media Server Container
# Contains all 30+ media applications with unified dashboard and MCP integration
FROM ubuntu:22.04

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/New_York

# Set up basic environment
ENV PUID=1000
ENV PGID=1000
ENV UMASK=002

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Core system tools
    curl wget gnupg2 software-properties-common apt-transport-https \
    ca-certificates lsb-release python3 python3-pip nodejs npm \
    # Media processing tools
    ffmpeg mediainfo imagemagick \
    # Network and web tools
    nginx supervisor \
    # Database tools
    sqlite3 \
    # Archive tools
    unrar p7zip-full \
    # Development tools
    git build-essential \
    # Process management
    htop procps \
    && rm -rf /var/lib/apt/lists/*

# Create application directories
RUN mkdir -p /app /config /data /downloads /media \
    /config/sonarr /config/radarr /config/lidarr /config/readarr /config/prowlarr \
    /config/bazarr /config/jellyfin /config/plex /config/emby /config/kodi \
    /config/qbittorrent /config/transmission /config/deluge /config/rtorrent \
    /config/sabnzbd /config/nzbget /config/tautulli /config/ombi /config/overseerr \
    /config/requestrr /config/notifiarr /config/unpackerr /config/recyclarr \
    /config/jackett /config/flaresolverr /config/prowlarr /config/autobrr \
    /config/cross-seed /config/maintainerr /config/plex-meta-manager \
    /config/komga /config/kavita /config/audiobookshelf /config/navidrome \
    /config/homepage /config/organizr /config/heimdall /config/homarr \
    /data/torrents /data/usenet /data/watch /downloads/complete /downloads/incomplete \
    /media/movies /media/tv /media/music /media/books /media/audiobooks /media/photos

# Install Node.js LTS
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs

# Install media applications via package managers where possible
RUN add-apt-repository universe \
    && apt-get update

# Copy application installation scripts
COPY scripts/ /tmp/scripts/
RUN chmod +x /tmp/scripts/*.sh

# Install core applications
RUN /tmp/scripts/install-sonarr.sh
RUN /tmp/scripts/install-radarr.sh
RUN /tmp/scripts/install-lidarr.sh
RUN /tmp/scripts/install-readarr.sh
RUN /tmp/scripts/install-prowlarr.sh
RUN /tmp/scripts/install-bazarr.sh
RUN /tmp/scripts/install-jellyfin.sh
RUN /tmp/scripts/install-qbittorrent.sh
RUN /tmp/scripts/install-transmission.sh
RUN /tmp/scripts/install-sabnzbd.sh
RUN /tmp/scripts/install-tautulli.sh
RUN /tmp/scripts/install-overseerr.sh
RUN /tmp/scripts/install-jackett.sh

# Install additional applications
RUN /tmp/scripts/install-additional-apps.sh

# Copy unified dashboard
COPY dashboard/ /app/dashboard/
WORKDIR /app/dashboard
RUN npm install && npm run build

# Copy unified MCP server
COPY unified-mcp/ /app/mcp/
WORKDIR /app/mcp
RUN npm install

# Copy nginx configuration for internal routing
COPY nginx/ /etc/nginx/
COPY supervisor/ /etc/supervisor/

# Copy startup and configuration scripts
COPY scripts/start-all-services.sh /usr/local/bin/
COPY scripts/configure-all-services.sh /usr/local/bin/
COPY scripts/health-check.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/*.sh

# Create service user
RUN groupadd -r mediaserver && useradd -r -g mediaserver -d /app -s /bin/bash mediaserver \
    && chown -R mediaserver:mediaserver /app /config /data /downloads /media

# Set up volume mounts
VOLUME ["/config", "/data", "/downloads", "/media"]

# Expose unified port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/health-check.sh

# Set working directory
WORKDIR /app

# Start all services via supervisor
CMD ["/usr/local/bin/start-all-services.sh"]