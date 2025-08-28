# Ultimate Media Server 2025 - Single Container Edition

## 🚀 ALL 30 Services in ONE Docker Container

**The most comprehensive single-container media server solution ever created!**

This implementation packages ALL 30 media server applications into a single Docker container with unified MCP (Model Context Protocol) integration, providing a complete media ecosystem that can be deployed and managed as one unit.

## 📊 Complete Service List (30 Total)

### 🎬 Media Servers (3 Services)
1. **Jellyfin** (Port 8096) - Open-source media server
2. **Plex** (Port 32400) - Premium media server platform  
3. **Emby** (Port 8097) - Alternative media server

### 📋 Content Management - *arr Suite (5 Services)
4. **Sonarr** (Port 8989) - TV series management
5. **Radarr** (Port 7878) - Movie management
6. **Lidarr** (Port 8686) - Music management
7. **Readarr** (Port 8787) - Book management
8. **Bazarr** (Port 6767) - Subtitle management

### 🔍 Indexers & Search (3 Services)
9. **Prowlarr** (Port 9696) - Modern indexer management
10. **Jackett** (Port 9117) - Legacy indexer proxy
11. **FlareSolverr** (Port 8191) - Cloudflare solver

### 📥 Download Clients (5 Services)
12. **qBittorrent** (Port 8080) - Primary torrent client
13. **Transmission** (Port 9091) - Alternative torrent client
14. **Deluge** (Port 8112) - Backup torrent client
15. **NZBGet** (Port 6789) - Usenet downloader
16. **SABnzbd** (Port 8085) - Alternative usenet client

### 🙋 Request Management (3 Services)
17. **Overseerr** (Port 5055) - Modern request system
18. **Requestrr** (Port 4545) - Discord/chat bot requests
19. **Ombi** (Port 3579) - Legacy request system

### 📊 Analytics & Monitoring (2 Services)
20. **Tautulli** (Port 8181) - Plex/Jellyfin analytics
21. **Netdata** (Port 19999) - System monitoring

### 🏠 Dashboards (4 Services)
22. **Homepage** (Port 3000) - Modern service dashboard
23. **Heimdall** (Port 7575) - Application dashboard
24. **Organizr** (Port 8081) - Classic dashboard solution
25. **Homarr** (Port 7576) - Modern dashboard alternative

### 🛡️ Infrastructure (5 Services)
26. **Nginx Proxy Manager** (Port 81) - Reverse proxy management
27. **Portainer** (Port 9000) - Container management
28. **Watchtower** (Internal) - Automated updates
29. **Gluetun VPN** (Internal) - VPN protection
30. **Unpackerr** (Internal) - Archive extraction

## 🌟 Revolutionary Features

### ✨ Single Container Architecture
- **ONE Docker container** running all 30 services
- **Supervisor-managed** process orchestration
- **Shared storage** and configuration
- **Internal networking** between services
- **Unified logging** and monitoring

### 🧠 Unified MCP Integration
- **Single MCP endpoint** managing all services
- **No SDK dependencies** - pure HTTP/JSON approach
- **Real-time service monitoring**
- **Cross-service search and management**
- **AI-powered recommendations**

### 🎨 Ultimate Dashboard
- **Glass morphism design** with modern UI
- **Real-time service status** indicators
- **Integrated MCP assistant** chat
- **Mobile-responsive** interface
- **Direct service access** links

### 🔧 Smart Service Management
- **Automatic service discovery**
- **Health monitoring** for all services
- **Centralized configuration**
- **Backup and restore** capabilities
- **Performance optimization**

## ⚡ Quick Start

### 1. Prerequisites
```bash
# Docker and Docker Compose
docker --version
docker-compose --version

# Minimum system requirements
# - 8GB RAM (16GB recommended)
# - 4 CPU cores (8 recommended)  
# - 100GB storage (500GB+ recommended for media)
```

### 2. Setup
```bash
# Clone the repository
git clone <repository-url>
cd mcp-architecture

# Copy environment file
cp .env.single-container .env

# Edit configuration (REQUIRED)
nano .env
# Set your OPENAI_API_KEY and other preferences

# Create media directories
mkdir -p media/{movies,tv,music,books,audiobooks,podcasts,documentaries,anime}
mkdir -p downloads/{complete,incomplete,watch}
mkdir -p backups
```

### 3. Deploy
```bash
# Build and start the container
docker-compose -f docker-compose.single-container.yml up -d

# Monitor startup (takes 2-5 minutes)
docker-compose -f docker-compose.single-container.yml logs -f

# Check container health
docker-compose -f docker-compose.single-container.yml ps
```

### 4. Access Services

**Main Dashboard:** http://localhost:8090

**Key Services:**
- **Jellyfin:** http://localhost:8096
- **Sonarr:** http://localhost:8989  
- **Radarr:** http://localhost:7878
- **qBittorrent:** http://localhost:8080
- **Overseerr:** http://localhost:5055
- **Tautulli:** http://localhost:8181

**All 30 services available on their respective ports!**

### 5. Initial Configuration

1. **Access each service** through the dashboard or direct URLs
2. **Complete initial setup** for each service
3. **Copy API keys** from service settings to `.env` file
4. **Restart container** to apply API key configuration:
   ```bash
   docker-compose -f docker-compose.single-container.yml restart
   ```

## 🔧 Configuration Guide

### Environment Variables (.env file)

**CRITICAL - Required:**
```bash
# OpenAI API Key (REQUIRED for MCP features)
OPENAI_API_KEY=sk-your-openai-api-key-here

# System settings
TZ=UTC
PUID=1000
PGID=1000
```

**Service API Keys (Configure after setup):**
```bash
JELLYFIN_API_KEY=your-jellyfin-api-key
SONARR_API_KEY=your-sonarr-api-key
RADARR_API_KEY=your-radarr-api-key
PROWLARR_API_KEY=your-prowlarr-api-key
# ... etc for all services
```

**Storage Paths:**
```bash
MEDIA_ROOT=./media
DOWNLOADS_ROOT=./downloads
BACKUP_ROOT=./backups
```

### Directory Structure
```
media/
├── movies/          # Movie files
├── tv/              # TV show files
├── music/           # Music files
├── books/           # Book files
├── audiobooks/      # Audiobook files
├── podcasts/        # Podcast files
├── documentaries/   # Documentary files
└── anime/           # Anime files

downloads/
├── complete/        # Completed downloads
├── incomplete/      # In-progress downloads
└── watch/           # Watch folder for auto-import

backups/             # Configuration backups
```

## 🌐 MCP Integration

### Claude Desktop Configuration

Add to your Claude Desktop config:
```json
{
  "mcpServers": {
    "ultimate-media-server": {
      "command": "node",
      "args": ["/path/to/project/src/simple-unified-mcp.js"],
      "env": {
        "MCP_PORT": "3001"
      }
    }
  }
}
```

### MCP Capabilities

**30 Unified Tools:**
- `get_all_services` - Overview of all services
- `check_service_health` - Health monitoring
- `search_across_services` - Cross-service search
- `get_download_status` - Download monitoring
- `manage_downloads` - Download control
- `get_library_stats` - Media statistics
- `get_requests_overview` - Content requests
- `get_system_overview` - System health
- And 22 more specialized tools!

**Resource Access:**
- `media://services` - Service configuration
- `media://health` - Real-time health data
- `media://stats` - System statistics
- `media://downloads` - Download status
- `media://requests` - Content requests

## 📊 Performance & Monitoring

### System Requirements

**Minimum:**
- **CPU:** 4 cores
- **RAM:** 8GB
- **Storage:** 100GB system + media storage
- **Network:** 100 Mbps

**Recommended:**
- **CPU:** 8+ cores
- **RAM:** 16-32GB
- **Storage:** 500GB+ NVMe SSD + large HDD array
- **Network:** Gigabit ethernet

### Health Monitoring

**Built-in Monitoring:**
- **Dashboard:** Real-time service status
- **Health endpoint:** http://localhost:8090/health
- **Netdata:** http://localhost:19999
- **Service status:** http://localhost:8090/services

**Health Check Script:**
```bash
# Manual health check
docker exec ultimate-media-server-2025 /opt/media-server/scripts/healthcheck.sh
```

## 🔒 Security Features

### Built-in Security
- **Non-root execution** for all services
- **Isolated networking** between services  
- **JWT authentication** for API access
- **Rate limiting** on endpoints
- **Security headers** in Nginx
- **VPN integration** with Gluetun

### Optional Enhancements
- **SSL/TLS termination** with Let's Encrypt
- **IP whitelisting** for admin access
- **Two-factor authentication** (service-dependent)
- **Reverse proxy** with authentication

## 🚀 Advanced Usage

### Custom Service Management
```bash
# Access container shell
docker exec -it ultimate-media-server-2025 bash

# Manage individual services
supervisorctl status                    # View all services
supervisorctl restart sonarr           # Restart specific service
supervisorctl stop jellyfin            # Stop service
supervisorctl start plex               # Start service

# View logs
tail -f /opt/media-server/logs/sonarr.log
tail -f /opt/media-server/logs/mcp-suite.log
```

### Backup & Restore
```bash
# Backup configuration
docker exec ultimate-media-server-2025 tar -czf /opt/media-server/backups/config-backup-$(date +%Y%m%d).tar.gz -C /opt/media-server config

# Restore configuration
docker exec ultimate-media-server-2025 tar -xzf /opt/media-server/backups/config-backup-20250803.tar.gz -C /opt/media-server
```

### VPN Configuration
```bash
# Enable VPN for download clients
# Edit .env file:
VPN_PROVIDER=surfshark
VPN_TYPE=wireguard
WIREGUARD_PRIVATE_KEY=your-private-key
WIREGUARD_ADDRESSES=10.64.0.1/32

# Restart container
docker-compose -f docker-compose.single-container.yml restart
```

## 🐛 Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker-compose -f docker-compose.single-container.yml logs

# Check system resources
docker stats

# Verify environment file
cat .env
```

**Service not accessible:**
```bash
# Check service status in container
docker exec ultimate-media-server-2025 supervisorctl status

# Check port binding
docker port ultimate-media-server-2025

# Check firewall
sudo ufw status
```

**MCP integration issues:**
```bash
# Test MCP server directly
curl http://localhost:3001/health

# Check MCP logs
docker exec ultimate-media-server-2025 tail -f /opt/media-server/logs/mcp-suite.log
```

### Performance Optimization

**For high-end systems:**
```bash
# Increase worker processes
WORKER_PROCESSES=8
NODE_MAX_OLD_SPACE_SIZE=4096

# Enable hardware acceleration
JELLYFIN_HARDWARE_ACCELERATION=true
PLEX_HARDWARE_TRANSCODING=true
```

**For low-end systems:**
```bash
# Reduce memory usage
NODE_MAX_OLD_SPACE_SIZE=1024
DISABLE_UNNECESSARY_SERVICES=emby,deluge,organizr
```

## 📈 Upgrade & Maintenance

### Updates
```bash
# Update to latest version
git pull origin main
docker-compose -f docker-compose.single-container.yml build --no-cache
docker-compose -f docker-compose.single-container.yml up -d
```

### Maintenance Tasks
```bash
# Clean up old logs
docker exec ultimate-media-server-2025 find /opt/media-server/logs -name "*.log" -mtime +7 -delete

# Optimize databases
docker exec ultimate-media-server-2025 supervisorctl restart sonarr radarr

# Clean Docker system
docker system prune -f
```

## 🤝 Support & Community

### Getting Help
- **Documentation:** Complete guides in `/docs`
- **Issues:** GitHub Issues for bug reports
- **Discussions:** GitHub Discussions for questions
- **Discord:** Community chat server

### Contributing
- **Bug Reports:** Use GitHub Issues
- **Feature Requests:** Use GitHub Discussions  
- **Pull Requests:** Always welcome!
- **Documentation:** Help improve guides

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ✨ Acknowledgments

- **All 30 service developers** for their amazing applications
- **Model Context Protocol** for the integration framework
- **Docker community** for containerization best practices
- **Open source community** for inspiration and support

---

**🎉 Welcome to the Ultimate Media Server 2025 - The most complete single-container media solution ever created!**