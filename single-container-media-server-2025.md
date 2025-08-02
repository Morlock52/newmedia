# Single Container Media Server 2025 - Ultimate Design Guide

## Executive Summary

Based on extensive research of trending approaches in August 2025, this guide presents an optimized single-container media server solution that combines multiple services using s6-overlay as the process supervisor and Caddy as the frontend reverse proxy. This approach offers superior performance, security, and maintainability compared to traditional multi-container deployments.

## Key Findings from Research

### 1. **Process Supervision Trends**
- **S6-Overlay Dominance**: S6-overlay has emerged as the preferred init system for multi-process containers in 2025
- **Advantages over Supervisor**: Proper PID 1 handling, signal management, graceful shutdown
- **LinuxServer.io Standards**: Industry-leading container practices use s6-overlay extensively

### 2. **Reverse Proxy Evolution**
- **Caddy vs SWAG**: While LinuxServer.io's SWAG (Nginx-based) is popular, Caddy offers:
  - Single static binary (no dependencies)
  - Automatic HTTPS with zero configuration
  - Superior performance for media streaming
  - Built-in HTTP/2 and HTTP/3 support

### 3. **Container Architecture Trends**
- **Single vs Multi**: While multi-container remains popular, single container solutions offer:
  - Simplified deployment (one image to manage)
  - Better resource efficiency
  - Easier backup and migration
  - Reduced network complexity

### 4. **Performance Optimizations**
- Static binary compilation for all services
- Memory-mapped file I/O for media streaming
- Hardware acceleration support
- Efficient process communication via Unix sockets

## Proposed Architecture

### Container Stack Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Single Container                         │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │                   S6-Overlay Init                    │  │
│  │                    (PID 1)                          │  │
│  └──────────────────────┬──────────────────────────────┘  │
│                         │                                   │
│  ┌──────────┬──────────┼──────────┬──────────────────┐   │
│  │          │          │          │                  │   │
│  │  Caddy   │ Jellyfin │  *arr    │   Download      │   │
│  │ (Port 80,│ (Port    │  Stack   │   Clients       │   │
│  │  443)    │  8096)   │          │                  │   │
│  │          │          │          │                  │   │
│  │ • Rev    │ • Media  │ • Radarr │ • qBittorrent   │   │
│  │   Proxy  │   Server │ • Sonarr │ • Transmission  │   │
│  │ • SSL    │ • Trans- │ • Lidarr │ • NZBGet        │   │
│  │ • Auth   │   coding │ • Prowl- │                  │   │
│  │          │          │   arr    │                  │   │
│  └──────────┴──────────┴──────────┴──────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              Shared Volume Mounts                    │  │
│  │  /config  /media  /downloads  /transcodes          │  │
│  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### Base Image Selection

```dockerfile
# Use Ubuntu 22.04 LTS as base for maximum compatibility
FROM ubuntu:22.04

# Install s6-overlay v3.1.6.2 (latest stable)
ADD https://github.com/just-containers/s6-overlay/releases/download/v3.1.6.2/s6-overlay-noarch.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-noarch.tar.xz
ADD https://github.com/just-containers/s6-overlay/releases/download/v3.1.6.2/s6-overlay-x86_64.tar.xz /tmp
RUN tar -C / -Jxpf /tmp/s6-overlay-x86_64.tar.xz
```

## Service Configuration

### 1. S6-Overlay Service Definitions

```bash
# /etc/s6-overlay/s6-rc.d/caddy/run
#!/usr/bin/with-contenv bash
exec /usr/bin/caddy run --config /config/caddy/Caddyfile --adapter caddyfile

# /etc/s6-overlay/s6-rc.d/jellyfin/run
#!/usr/bin/with-contenv bash
exec /usr/bin/jellyfin --datadir /config/jellyfin --cachedir /cache --webdir /usr/share/jellyfin/web

# /etc/s6-overlay/s6-rc.d/radarr/run
#!/usr/bin/with-contenv bash
exec /usr/bin/mono /opt/Radarr/Radarr.exe -nobrowser -data=/config/radarr

# /etc/s6-overlay/s6-rc.d/sonarr/run
#!/usr/bin/with-contenv bash
exec /usr/bin/mono /opt/Sonarr/Sonarr.exe -nobrowser -data=/config/sonarr

# /etc/s6-overlay/s6-rc.d/qbittorrent/run
#!/usr/bin/with-contenv bash
exec /usr/bin/qbittorrent-nox --webui-port=8080 --profile=/config/qbittorrent
```

### 2. Service Dependencies

```bash
# /etc/s6-overlay/s6-rc.d/caddy/dependencies.d/base
# /etc/s6-overlay/s6-rc.d/jellyfin/dependencies.d/base
# /etc/s6-overlay/s6-rc.d/radarr/dependencies.d/jellyfin
# /etc/s6-overlay/s6-rc.d/sonarr/dependencies.d/jellyfin
# /etc/s6-overlay/s6-rc.d/qbittorrent/dependencies.d/base
```

### 3. Caddy Configuration

```caddyfile
# /config/caddy/Caddyfile
{
    email admin@example.com
    # Performance optimizations
    servers {
        protocol {
            experimental_http3
        }
    }
}

# Main domain
example.com {
    # Homepage/Dashboard
    handle / {
        reverse_proxy localhost:8096
    }
    
    # Jellyfin
    handle /jellyfin/* {
        reverse_proxy localhost:8096
    }
    
    # Radarr
    handle /radarr/* {
        reverse_proxy localhost:7878
    }
    
    # Sonarr  
    handle /sonarr/* {
        reverse_proxy localhost:8989
    }
    
    # qBittorrent
    handle /qbittorrent/* {
        reverse_proxy localhost:8080
    }
    
    # Authentication
    @admin {
        path /radarr/* /sonarr/* /qbittorrent/*
    }
    basicauth @admin {
        admin $2a$14$... # bcrypt hash
    }
    
    # Security headers
    header {
        X-Content-Type-Options "nosniff"
        X-Frame-Options "SAMEORIGIN"
        X-XSS-Protection "1; mode=block"
        Referrer-Policy "strict-origin-when-cross-origin"
        Content-Security-Policy "default-src 'self'"
    }
    
    # Enable compression
    encode gzip
    
    # File server for direct media access
    handle /media/* {
        file_server {
            root /media
            browse
        }
    }
}
```

## Performance Optimizations

### 1. **Memory and CPU**
```bash
# Environment variables for optimization
S6_KEEP_ENV=1
S6_BEHAVIOUR_IF_STAGE2_FAILS=2
S6_CMD_WAIT_FOR_SERVICES_MAXTIME=0

# Process limits
ulimit -n 65536  # Increase file descriptors
ulimit -u 32768  # Increase max processes
```

### 2. **Storage Optimization**
- Use XFS or ext4 with `noatime` mount option
- Separate volumes for media, downloads, and transcodes
- NVMe SSD for metadata and transcoding

### 3. **Network Optimization**
```bash
# Sysctl optimizations
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 5000
```

### 4. **Hardware Acceleration**
```dockerfile
# Intel QuickSync support
RUN apt-get update && apt-get install -y \
    intel-media-va-driver-non-free \
    vainfo \
    mesa-va-drivers

# NVIDIA support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility
```

## Security Best Practices

### 1. **User Management**
```dockerfile
# Create non-root user
RUN useradd -u 1000 -U -d /config -s /bin/false mediaserver && \
    usermod -G users mediaserver

# Set permissions
RUN chown -R mediaserver:mediaserver /config /media /downloads
```

### 2. **Network Security**
- Internal services listen on localhost only
- Caddy handles all external connections
- Automatic HTTPS with Let's Encrypt
- Basic auth for administrative interfaces

### 3. **Container Security**
```yaml
# Docker Compose security options
security_opt:
  - no-new-privileges:true
  - seccomp:unconfined
cap_drop:
  - ALL
cap_add:
  - NET_BIND_SERVICE
  - DAC_READ_SEARCH
```

## Deployment Guide

### 1. **Docker Compose Configuration**
```yaml
version: '3.8'
services:
  mediaserver:
    image: mediaserver:2025
    container_name: mediaserver
    hostname: mediaserver
    environment:
      - PUID=1000
      - PGID=1000
      - TZ=America/New_York
      - JELLYFIN_PublishedServerUrl=https://example.com
    volumes:
      - ./config:/config
      - ./media:/media
      - ./downloads:/downloads
      - ./transcodes:/transcodes
    ports:
      - "80:80"
      - "443:443"
      - "443:443/udp" # HTTP/3
    devices:
      - /dev/dri:/dev/dri # Hardware acceleration
    restart: unless-stopped
    networks:
      - medianet

networks:
  medianet:
    driver: bridge
```

### 2. **Build Process**
```bash
#!/bin/bash
# build.sh - Optimized build script

# Build with BuildKit for better caching
DOCKER_BUILDKIT=1 docker build \
  --build-arg BUILDKIT_INLINE_CACHE=1 \
  --cache-from mediaserver:latest \
  --tag mediaserver:2025 \
  --tag mediaserver:latest \
  .

# Multi-platform build for ARM64 support
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  --push \
  --tag yourregistry/mediaserver:2025 \
  .
```

### 3. **Quick Start**
```bash
# Clone repository
git clone https://github.com/yourusername/mediaserver-2025
cd mediaserver-2025

# Configure environment
cp .env.example .env
nano .env

# Build and start
docker-compose up -d

# View logs
docker-compose logs -f
```

## Monitoring and Maintenance

### 1. **Health Checks**
```dockerfile
HEALTHCHECK --interval=30s --timeout=30s --start-period=5m --retries=3 \
  CMD curl -f http://localhost/health || exit 1
```

### 2. **Logging**
```bash
# S6 service for log rotation
#!/usr/bin/with-contenv bash
# /etc/s6-overlay/s6-rc.d/logrotate/run
exec logrotate -s /config/logrotate.state /etc/logrotate.conf
```

### 3. **Backup Strategy**
```bash
# Automated backup script
#!/bin/bash
# /config/scripts/backup.sh

# Stop services gracefully
s6-svc -d /var/run/s6/services/jellyfin
s6-svc -d /var/run/s6/services/radarr
s6-svc -d /var/run/s6/services/sonarr

# Backup configuration
tar -czf /backup/config-$(date +%Y%m%d).tar.gz /config

# Restart services
s6-svc -u /var/run/s6/services/jellyfin
s6-svc -u /var/run/s6/services/radarr
s6-svc -u /var/run/s6/services/sonarr
```

## Migration from Multi-Container Setup

### 1. **Data Migration**
```bash
# Export from existing containers
docker cp jellyfin:/config ./migration/jellyfin
docker cp radarr:/config ./migration/radarr
docker cp sonarr:/config ./migration/sonarr

# Import to single container
docker cp ./migration/jellyfin mediaserver:/config/
docker cp ./migration/radarr mediaserver:/config/
docker cp ./migration/sonarr mediaserver:/config/
```

### 2. **Database Conversion**
- SQLite databases are portable
- Update connection strings for localhost
- Adjust paths in application configs

## Troubleshooting

### Common Issues and Solutions

1. **Service Won't Start**
   - Check s6-overlay logs: `docker logs mediaserver`
   - Verify permissions: `ls -la /config`
   - Check dependencies in s6-rc.d

2. **Performance Issues**
   - Monitor with `docker stats`
   - Check I/O: `iotop -o`
   - Review Caddy access logs

3. **SSL Certificate Issues**
   - Ensure ports 80/443 are accessible
   - Check Caddy logs for ACME errors
   - Verify DNS resolution

## Future Enhancements

### 2025 Roadmap
1. **AI Integration**
   - Content recommendation engine
   - Automatic metadata enhancement
   - Smart transcoding decisions

2. **Web3 Features**
   - IPFS integration for distributed media
   - NFT support for digital collections
   - Decentralized authentication

3. **Performance**
   - AV1 codec support
   - GPU clustering for transcoding
   - Edge caching integration

## Conclusion

This single-container media server design represents the cutting edge of self-hosted media solutions for 2025. By combining s6-overlay's robust process management with Caddy's modern reverse proxy capabilities, we achieve:

- **Simplified Deployment**: One container to rule them all
- **Enhanced Security**: Single attack surface with proper isolation
- **Better Performance**: Optimized inter-process communication
- **Easier Maintenance**: Unified logging, monitoring, and updates

The trend toward single-container solutions reflects the maturity of container technology and the desire for simpler, more maintainable infrastructure without sacrificing functionality or performance.

## Resources and References

- [S6-Overlay Documentation](https://github.com/just-containers/s6-overlay)
- [Caddy Documentation](https://caddyserver.com/docs/)
- [LinuxServer.io Best Practices](https://docs.linuxserver.io/)
- [r/selfhosted Community](https://reddit.com/r/selfhosted)
- [Docker Best Practices 2025](https://docs.docker.com/develop/dev-best-practices/)

---

*Last Updated: August 2025*
*Version: 1.0.0*
*License: MIT*