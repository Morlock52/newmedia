# Ultimate Media Server 2025 - Backend Architecture

## Overview

This directory contains the complete backend architecture for the Ultimate Media Server 2025, featuring a microservices-based design with comprehensive service orchestration, monitoring, and management capabilities.

## Architecture Components

### 1. Service Orchestrator (`/api/service-orchestrator`)
The core orchestration service that manages all Docker containers and services.

**Key Features:**
- Dynamic service installation/uninstallation
- Dependency resolution and management
- Health monitoring with auto-recovery
- Secure environment and secret management
- RESTful API with WebSocket support
- Comprehensive logging and metrics

**Technology Stack:**
- Node.js 20 with ES6 modules
- Express.js for API framework
- Docker SDK for container management
- PostgreSQL for persistent storage
- Redis for caching and pub/sub
- HashiCorp Vault for secret management

### 2. Service Management Scripts (`/scripts`)

#### `service-installer.sh`
Interactive CLI for installing and managing services:
```bash
./scripts/service-installer.sh
```

Features:
- Browse available services by category
- Install individual services or service groups
- Automatic dependency resolution
- Post-installation configuration guidance

#### `service-control.sh`
Control running services:
```bash
./scripts/service-control.sh
```

Options:
- Start/stop/restart services
- View service status
- Bulk operations on all services

### 3. Health Monitoring System

**Continuous Monitoring:**
- HTTP endpoint health checks
- Docker container status monitoring
- TCP port availability checks
- Custom health check commands
- Automatic recovery attempts

**Alert System:**
- Critical service failures
- Performance degradation warnings
- Disk space alerts
- Network connectivity issues

### 4. Environment Management

**Secure Configuration:**
- Encrypted secrets at rest
- Automatic secret rotation
- Environment-specific configurations
- Version-controlled settings

**Supported Formats:**
- Docker Compose environment files
- Kubernetes ConfigMaps and Secrets
- SystemD environment files
- Shell export scripts

### 5. API Endpoints

Base URL: `http://localhost:3000/api/v1`

**Service Management:**
- `GET /services` - List all services
- `POST /services` - Install new service
- `GET /services/{name}` - Get service details
- `PUT /services/{name}` - Update service
- `DELETE /services/{name}` - Uninstall service
- `POST /services/{name}/start` - Start service
- `POST /services/{name}/stop` - Stop service
- `POST /services/{name}/restart` - Restart service

**Health Monitoring:**
- `GET /health` - System health status
- `GET /health/services` - All services health
- `GET /health/dependencies` - Infrastructure health

**Configuration:**
- `GET /config/{service}` - Get configuration
- `PUT /config/{service}` - Update configuration
- `POST /config/validate` - Validate configuration
- `POST /config/{service}/reload` - Reload configuration

**Tasks:**
- `GET /tasks` - List all tasks
- `POST /tasks` - Create new task
- `GET /tasks/{id}` - Get task status
- `DELETE /tasks/{id}` - Cancel task

## Quick Start

### 1. Prerequisites
- Docker 20.10+
- Docker Compose 2.0+
- Node.js 20+ (for development)
- PostgreSQL 15+
- Redis 7+

### 2. Initial Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/ultimate-media-server-2025.git
cd ultimate-media-server-2025

# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env

# Start the orchestrator
cd api/service-orchestrator
docker-compose up -d

# Wait for services to be ready
docker-compose ps

# Run the installer
./scripts/service-installer.sh
```

### 3. Install Essential Services

```bash
# Option 1: Interactive installer
./scripts/service-installer.sh
# Select option 3: Install service group
# Select option 1: Essential

# Option 2: Direct API calls
curl -X POST http://localhost:3000/api/v1/services \
  -H "Content-Type: application/json" \
  -d @services/jellyfin/manifest.yml
```

### 4. Access Services

After installation, services are available at:
- Jellyfin: http://localhost:8096
- Sonarr: http://localhost:8989
- Radarr: http://localhost:7878
- Prowlarr: http://localhost:9696
- Overseerr: http://localhost:5055

## Security Features

### 1. Network Isolation
- Separate networks for different service groups
- Internal-only communication for sensitive services
- VPN integration for download clients

### 2. Authentication & Authorization
- JWT-based API authentication
- Role-based access control (RBAC)
- Service-to-service authentication
- API key management

### 3. Encryption
- AES-256-GCM for secrets at rest
- TLS for all external communications
- Encrypted backups
- Secure key rotation

### 4. Monitoring & Auditing
- Comprehensive audit logs
- Failed authentication alerts
- Suspicious activity detection
- Compliance reporting

## Performance Optimization

### 1. Caching Strategy
- Redis for API response caching
- Docker layer caching
- Static asset caching
- Database query caching

### 2. Resource Management
- CPU and memory limits per service
- Automatic scaling based on load
- Resource usage monitoring
- Performance bottleneck detection

### 3. Network Optimization
- HTTP/2 support
- Connection pooling
- Compression enabled
- CDN integration ready

## Backup & Recovery

### 1. Automated Backups
```bash
# Run backup script
./scripts/backup.sh

# Schedule daily backups
crontab -e
0 2 * * * /path/to/scripts/backup.sh
```

### 2. Backup Contents
- Service configurations
- Database dumps
- Environment settings
- SSL certificates
- Application data (optional)

### 3. Recovery Process
```bash
# Restore from backup
./scripts/restore.sh /path/to/backup.tar.gz
```

## Monitoring & Metrics

### 1. Prometheus Metrics
Available at: http://localhost:9090

Key metrics:
- Service health status
- Container resource usage
- API response times
- Error rates
- Task completion times

### 2. Grafana Dashboards
Available at: http://localhost:3001

Pre-configured dashboards:
- System Overview
- Service Health
- Performance Metrics
- Error Analysis
- Resource Usage

### 3. Logs
Centralized logging with:
- Service logs aggregation
- Error tracking
- Performance monitoring
- Security audit trails

## Development

### 1. Local Development
```bash
cd api/service-orchestrator
npm install
npm run dev
```

### 2. Testing
```bash
# Run unit tests
npm test

# Run integration tests
npm run test:integration

# Run end-to-end tests
npm run test:e2e
```

### 3. Code Style
```bash
# Lint code
npm run lint

# Format code
npm run format
```

## Troubleshooting

### Common Issues

1. **Service won't start**
   - Check logs: `docker logs <service-name>`
   - Verify dependencies are running
   - Check port conflicts
   - Ensure proper permissions

2. **Health check failures**
   - Verify service is accessible
   - Check network connectivity
   - Review health check configuration
   - Examine service logs

3. **Configuration errors**
   - Validate environment variables
   - Check file permissions
   - Verify paths exist
   - Review configuration syntax

### Debug Mode
```bash
# Enable debug logging
export DEBUG=orchestrator:*
npm run dev

# View detailed Docker events
docker events

# Monitor resource usage
docker stats
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Support

- Documentation: [Full Documentation](https://docs.ultimate-media-server.com)
- Issues: [GitHub Issues](https://github.com/yourusername/ultimate-media-server-2025/issues)
- Community: [Discord Server](https://discord.gg/mediaserver)