# Multi-stage Dockerfile for complete media server stack
FROM ubuntu:22.04 as base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PUID=1000
ENV PGID=1000
ENV TZ=America/New_York

# Install base dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    gnupg \
    software-properties-common \
    apt-transport-https \
    ca-certificates \
    lsb-release \
    supervisor \
    nginx \
    python3 \
    python3-pip \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Docker inside container (Docker-in-Docker)
RUN curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null \
    && apt-get update \
    && apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin \
    && rm -rf /var/lib/apt/lists/*

# Create media user
RUN groupadd -g $PGID mediauser && \
    useradd -u $PUID -g $PGID -m -s /bin/bash mediauser

# Create directory structure
RUN mkdir -p /media/{config,data} \
    && mkdir -p /media/config/{jellyfin,sonarr,radarr,qbittorrent,prowlarr,overseerr,tautulli,homarr} \
    && mkdir -p /media/data/{media,torrents} \
    && mkdir -p /media/data/media/{movies,tv,music} \
    && mkdir -p /media/data/torrents/{movies,tv,music} \
    && chown -R mediauser:mediauser /media

# Copy application files
COPY --chown=mediauser:mediauser docker-compose.yml /media/
COPY --chown=mediauser:mediauser scripts/ /media/scripts/
COPY --chown=mediauser:mediauser config/ /media/config/
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Set working directory
WORKDIR /media

# Expose all service ports
EXPOSE 8096 8989 7878 8080 9696 5055 8181 7575 80

# Start supervisor to manage all services
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]