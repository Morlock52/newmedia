# Single Container Media Server Deployment Guide

## 🚀 Quick Start - Single Container

This guide helps you run the all-in-one media server using a single Docker container with all services included.

### Prerequisites
- Docker installed
- At least 4GB RAM available
- 20GB+ free disk space

### Build the Container

```bash
# Build the multi-service container
docker build -t media-server-aio -f Dockerfile.multi-service .
```

### Run the Container

```bash
# Create directories
mkdir -p config data/media data/downloads

# Run with default ports
docker run -d \
  --name media-server \
  -p 80:80 \
  -p 8096:8096 \
  -p 8989:8989 \
  -p 7878:7878 \
  -p 9696:9696 \
  -p 8080:8080 \
  -p 3000:3000 \
  -v $(pwd)/config:/config \
  -v $(pwd)/data:/data \
  -e PUID=$(id -u) \
  -e PGID=$(id -g) \
  -e TZ=$(cat /etc/timezone 2>/dev/null || echo "America/New_York") \
  --restart unless-stopped \
  media-server-aio
```

### Access Services

After startup (may take 2-3 minutes), access services at:

- **Dashboard**: http://localhost/ or http://localhost:3000
- **Jellyfin**: http://localhost:8096
- **Sonarr**: http://localhost:8989
- **Radarr**: http://localhost:7878
- **Prowlarr**: http://localhost:9696
- **qBittorrent**: http://localhost:8080

### Default Credentials

- **qBittorrent**: Username: `admin`, Password: `adminadmin` (change immediately!)
- **Other services**: Set up on first access

### Container Management

```bash
# View logs
docker logs -f media-server

# Stop container
docker stop media-server

# Start container
docker start media-server

# Restart container
docker restart media-server

# Remove container (preserves data)
docker rm -f media-server
```

### Health Check

```bash
# Check if services are running
docker exec media-server /usr/local/bin/healthcheck
```

### Troubleshooting

1. **Services not starting**: Wait 3-5 minutes for initial setup
2. **Port conflicts**: Change ports in the run command
3. **Permission issues**: Ensure PUID/PGID match your user
4. **Out of memory**: Allocate more RAM to Docker

### Notes

- This is a monolithic container - not recommended for production
- First startup takes longer as services initialize
- All data is persisted in mounted volumes
- For production, use docker-compose with separate containers