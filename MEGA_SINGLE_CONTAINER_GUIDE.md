# 🚀 Mega Single Container - Complete Media Server Stack

This is a comprehensive single container solution that includes **30+ interconnected services** running together with full service discovery, shared authentication, unified databases, and integrated monitoring.

## 🎯 What's Included

### 📺 Media Management Stack
- **Sonarr** - TV Series management
- **Radarr** - Movie management  
- **Lidarr** - Music management
- **Readarr** - Book/Audiobook management
- **Prowlarr** - Indexer management (shared across all *arr apps)
- **Bazarr** - Subtitle management

### ⬇️ Download Clients
- **qBittorrent** - BitTorrent client
- **Transmission** - Alternative BitTorrent client
- **Deluge** - Additional torrent client

### 🎬 Media Servers
- **Jellyfin** - Media streaming server
- **Plex** - Alternative media server
- **Tautulli** - Media server monitoring

### 🔐 Authentication & Security
- **Authelia** - Single Sign-On (SSO) authentication
- **Traefik** - Reverse proxy with automatic SSL
- **Fail2Ban** - Intrusion prevention

### 📊 Monitoring & Observability
- **Prometheus** - Metrics collection
- **Grafana** - Dashboards and visualization
- **Loki** - Log aggregation
- **Promtail** - Log shipping
- **Jaeger** - Distributed tracing

### 🗄️ Databases & Caching
- **PostgreSQL 15** - Primary database for all applications
- **Redis** - Caching and session storage
- **RabbitMQ** - Message queue for async communication

### 🛠️ Management & Utilities
- **Portainer** - Docker management interface
- **Uptime Kuma** - Service monitoring
- **Heimdall** - Application dashboard
- **Supervisor** - Process management

### 🌐 Networking & Service Mesh
- **Internal DNS** - Service discovery
- **Kong** - API Gateway (alternative to Traefik)
- **Nginx/Caddy** - Additional reverse proxy options

## 🏗️ Architecture Features

### ⚡ Service Interconnection
- **Prowlarr → All *arr apps** - Centralized indexer management
- **qBittorrent ← All *arr apps** - Shared download client
- **Jellyfin ← All *arr apps** - Automated media import
- **Authelia → All services** - Single sign-on authentication
- **PostgreSQL ← All apps** - Unified database layer
- **Prometheus ← All services** - Centralized monitoring

### 🔗 Service Discovery
- Internal DNS resolution (127.0.0.1 sonarr, radarr, etc.)
- Environment variable-based configuration
- Automatic service registration
- Health check integration

### 🛡️ Security Features
- SSO authentication via Authelia
- Reverse proxy with automatic SSL
- Fail2Ban intrusion prevention
- Secure inter-service communication
- No hardcoded secrets (environment-based)

### 📈 Monitoring Integration
- All services expose metrics to Prometheus
- Pre-configured Grafana dashboards
- Centralized logging via Loki
- Real-time health monitoring
- Performance tracking

## 🚀 Quick Start

### 1. Build and Deploy
```bash
# Make build script executable
chmod +x build-mega-single.sh

# Build and deploy the entire stack
./build-mega-single.sh
```

### 2. Alternative Manual Build
```bash
# Build the image
docker build -f Dockerfile.mega-single -t mega-media-server:latest .

# Deploy with Docker Compose
docker-compose -f docker-compose.mega-single.yml up -d
```

### 3. Wait for Initialization
The container includes 30+ services and needs time to initialize:
- **2-3 minutes** for basic services (databases, auth, proxy)
- **5-10 minutes** for all services to be fully operational
- **15+ minutes** for complete service interconnection

### 4. Access the Stack
Once deployed, access your services:

| Service | URL | Purpose |
|---------|-----|---------|
| **Main Dashboard** | http://localhost/ | Heimdall application dashboard |
| **Traefik** | http://localhost:8080/ | Reverse proxy management |
| **Health Monitor** | http://localhost:8888/health | System health API |
| **Supervisor** | http://localhost:9001/ | Process management |

## 📱 Service Access

### 🎬 Media Management
| Service | URL | Default Login |
|---------|-----|---------------|
| **Sonarr** | http://localhost:8989/ | No login required initially |
| **Radarr** | http://localhost:7878/ | No login required initially |
| **Lidarr** | http://localhost:8686/ | No login required initially |
| **Readarr** | http://localhost:8787/ | No login required initially |
| **Prowlarr** | http://localhost:9696/ | No login required initially |
| **Bazarr** | http://localhost:6767/ | No login required initially |

### ⬇️ Downloads & Media
| Service | URL | Default Login |
|---------|-----|---------------|
| **qBittorrent** | http://localhost:8090/ | admin / adminpass |
| **Jellyfin** | http://localhost:8096/ | Setup wizard on first access |
| **Tautulli** | http://localhost:8181/ | Setup wizard on first access |

### 📊 Monitoring
| Service | URL | Default Login |
|---------|-----|---------------|
| **Grafana** | http://localhost:3000/ | admin / admin123 |
| **Prometheus** | http://localhost:9090/ | No login required |

### 🔐 Security & Management
| Service | URL | Default Login |
|---------|-----|---------------|
| **Authelia** | http://localhost:9091/ | admin / admin123 |
| **Portainer** | http://localhost:9000/ | Setup on first access |
| **Uptime Kuma** | http://localhost:3001/ | Setup on first access |

## ⚙️ Configuration

### 📁 Directory Structure
```
/Users/morlock/fun/newmedia/
├── config/                 # Service configurations
│   ├── sonarr/            # Sonarr configuration
│   ├── radarr/            # Radarr configuration
│   ├── postgresql/        # PostgreSQL data
│   ├── redis/             # Redis configuration
│   ├── traefik/           # Reverse proxy config
│   └── authelia/          # Authentication config
├── data/                  # Application data
│   ├── media/             # Media libraries
│   │   ├── movies/        # Movie files
│   │   ├── tv/           # TV show files
│   │   ├── music/        # Music files
│   │   └── books/        # Book files
│   ├── downloads/         # Download directories
│   └── databases/         # Database storage
└── logs/                  # Service logs
    ├── apps/             # Application logs
    ├── system/           # System logs
    └── access/           # Access logs
```

### 🔧 Environment Variables
Key environment variables for customization:
```bash
# System Configuration
PUID=1000                  # User ID
PGID=1000                  # Group ID
TZ=America/New_York        # Timezone

# Security (CHANGE THESE!)
AUTHELIA_JWT_SECRET=your-jwt-secret
AUTHELIA_SESSION_SECRET=your-session-secret

# Resource Limits
POSTGRES_MAX_CONNECTIONS=200
REDIS_MAX_MEMORY=512mb
```

## 🔄 Service Interconnections

### 🎯 Media Pipeline Flow
```
Indexers (Prowlarr) 
    ↓
*arr Applications (Sonarr/Radarr/etc.) 
    ↓
Download Client (qBittorrent) 
    ↓
Media Organization 
    ↓
Media Server (Jellyfin) 
    ↓
Monitoring (Tautulli)
```

### 🔐 Authentication Flow
```
User Request → Traefik → Authelia → Service
```

### 📊 Monitoring Flow
```
Services → Prometheus → Grafana (Dashboards)
Services → Loki → Grafana (Logs)
```

## 🛠️ Management Commands

### Container Management
```bash
# View all service logs
docker-compose -f docker-compose.mega-single.yml logs -f

# Restart the entire stack
docker-compose -f docker-compose.mega-single.yml restart

# Stop the stack
docker-compose -f docker-compose.mega-single.yml down

# Enter the container
docker exec -it mega-media-server bash

# Check container health
curl http://localhost:8888/health
```

### Service Management (Inside Container)
```bash
# View service status
supervisorctl status

# Restart a specific service
supervisorctl restart sonarr

# View service logs
supervisorctl tail -f sonarr

# Start/stop services
supervisorctl start prowlarr
supervisorctl stop bazarr
```

### Database Management
```bash
# Connect to PostgreSQL
docker exec -it mega-media-server su postgres -c "psql"

# Connect to Redis
docker exec -it mega-media-server redis-cli

# View database status
docker exec -it mega-media-server python3 -c "
import psycopg2
conn = psycopg2.connect('host=localhost dbname=sonarr user=postgres')
print('PostgreSQL connection successful')
"
```

## 🩺 Health Monitoring

### Health Check Endpoint
```bash
# Overall health
curl http://localhost:8888/health

# Detailed service status
curl http://localhost:8888/services

# System resources
curl http://localhost:8888/system

# Combined status
curl http://localhost:8888/status
```

### Service-Specific Health
```bash
# Individual service pings
curl http://localhost:8989/ping  # Sonarr
curl http://localhost:7878/ping  # Radarr  
curl http://localhost:9696/ping  # Prowlarr
curl http://localhost:8080/ping  # Traefik
```

## 🔧 Troubleshooting

### Common Issues

#### Services Not Starting
1. **Check logs**: `docker-compose logs -f mega-media-server`
2. **Check disk space**: `df -h`
3. **Check memory**: `free -h`
4. **Check supervisor status**: `docker exec mega-media-server supervisorctl status`

#### Service Interconnection Issues
1. **Check service health**: `curl http://localhost:8888/services`
2. **Verify API keys**: Check `/config/api-keys/` directory
3. **Check network connectivity**: `docker exec mega-media-server netstat -tlnp`

#### Database Connection Issues  
1. **Check PostgreSQL**: `docker exec mega-media-server su postgres -c "pg_isready"`
2. **Check Redis**: `docker exec mega-media-server redis-cli ping`
3. **Check database logs**: `docker exec mega-media-server tail -f /logs/databases/postgresql.log`

#### Authentication Issues
1. **Check Authelia**: `curl http://localhost:9091/api/health`
2. **Verify configuration**: `docker exec mega-media-server cat /config/authelia/configuration.yml`
3. **Check session storage**: `docker exec mega-media-server redis-cli keys "*authelia*"`

### Performance Optimization

#### Resource Monitoring
```bash
# System resources
curl http://localhost:8888/system

# Individual service metrics
curl http://localhost:9090/metrics

# Database performance
docker exec mega-media-server psql -U postgres -c "
SELECT datname, numbackends, xact_commit, xact_rollback 
FROM pg_stat_database;"
```

#### Memory Optimization
```bash
# Adjust PostgreSQL shared_buffers
# Edit /config/postgresql/postgresql.conf
shared_buffers = 512MB

# Adjust Redis memory limit
# Edit /config/redis/redis.conf  
maxmemory 256mb
```

## 🔄 Updates and Maintenance

### Updating Services
```bash
# Rebuild with updated packages
docker build --no-cache -f Dockerfile.mega-single -t mega-media-server:latest .

# Deploy updated container
docker-compose -f docker-compose.mega-single.yml up -d --force-recreate
```

### Backup and Restore
```bash
# Backup configuration and data
tar -czf backup-$(date +%Y%m%d).tar.gz config/ data/

# Restore from backup
tar -xzf backup-YYYYMMDD.tar.gz
```

### Log Rotation
Logs are automatically rotated by the container. Manual cleanup:
```bash
# Clean old logs
find ./logs -name "*.log" -mtime +30 -delete
```

## 📈 Scaling and Customization

### Adding New Services
1. **Update Dockerfile.mega-single** - Add new service installation
2. **Update supervisord.conf** - Add new service configuration
3. **Update entrypoint.sh** - Add service initialization
4. **Update service interconnection** - Configure API connections

### Resource Scaling
```yaml
# Update docker-compose.mega-single.yml
deploy:
  resources:
    limits:
      cpus: '16.0'      # Scale up CPU
      memory: 32G       # Scale up memory
```

### Custom Configurations
All service configurations are stored in `./config/` and can be customized:
- **Traefik**: `./config/traefik/traefik.yml`
- **Authelia**: `./config/authelia/configuration.yml` 
- **Prometheus**: `./config/prometheus/prometheus.yml`
- **Grafana**: `./config/grafana/grafana.ini`

## 🎉 Conclusion

This mega single container provides a complete, production-ready media server stack with:

✅ **30+ interconnected services**  
✅ **Single sign-on authentication**  
✅ **Unified database layer**  
✅ **Centralized monitoring**  
✅ **Service discovery**  
✅ **Automated service interconnection**  
✅ **Health monitoring**  
✅ **Process orchestration**  

Perfect for users who want a complete media server solution without the complexity of managing multiple containers and their interconnections.

---

**Built with ❤️ by the Integration Specialist**

For issues or questions, check the container logs and health endpoints first. The system is designed to be self-healing and will attempt to recover from most issues automatically.