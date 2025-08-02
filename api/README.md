# Media Server Orchestration API

A comprehensive backend API for managing media server infrastructure with Docker Compose integration, health monitoring, configuration management, and seedbox automation.

## Features

- **🐳 Docker Integration**: Complete Docker Compose service management with profile support
- **📊 Health Monitoring**: Real-time system and service health monitoring with metrics
- **⚙️ Configuration Management**: Secure .env file parsing, validation, and management
- **🌱 Seedbox Integration**: Cross-seed automation and torrent management
- **📝 Logging**: Comprehensive log aggregation and streaming
- **🔌 WebSocket Support**: Real-time updates for service status and logs
- **🔒 Security**: Rate limiting, input validation, and secure error handling
- **📚 API Documentation**: Complete OpenAPI 3.0 specification
- **🧪 Testing**: Comprehensive integration test suite

## Quick Start

### Prerequisites

- Node.js 18.0.0 or higher
- Docker and Docker Compose v2
- Media server stack (docker-compose.yml) in project root

### Installation

1. **Clone and install dependencies:**
```bash
cd /path/to/your/media-server/api
npm install
```

2. **Set up environment variables:**
Copy the environment template and configure your settings:
```bash
cp ../env.template ../.env
# Edit .env with your configuration
```

3. **Start the API server:**
```bash
npm start
# Or for development:
npm run dev
```

The API will be available at `http://localhost:3002`

### Docker Integration

The API requires a `docker-compose.yml` file in the project root. It supports the following service profiles:

- **minimal**: Essential services only (jellyfin, homepage)
- **media**: Media management (jellyfin, sonarr, radarr, bazarr)
- **download**: Download clients (qbittorrent, vpn, sabnzbd)
- **monitoring**: Monitoring stack (prometheus, grafana, tautulli)
- **full**: All services

## API Endpoints

### Service Management
- `GET /api/services` - Get all services
- `GET /api/services/{service}/status` - Get service status
- `POST /api/services/start` - Start services
- `POST /api/services/stop` - Stop services
- `POST /api/services/restart` - Restart services
- `GET /api/services/{service}/logs` - Get service logs

### Configuration Management
- `GET /api/config` - Get configuration
- `PUT /api/config` - Update configuration
- `POST /api/config/validate` - Validate configuration
- `GET /api/config/env` - Get environment variables

### Health Monitoring
- `GET /api/health/overview` - Get health overview
- `GET /api/health/detailed` - Get detailed health check
- `GET /api/health/metrics` - Get system metrics

### Seedbox Management
- `GET /api/seedbox/status` - Get seedbox status
- `POST /api/seedbox/cross-seed/start` - Start cross-seed
- `GET /api/seedbox/torrents/stats` - Get torrent statistics

### Logging
- `GET /api/logs` - Get logs with filtering
- WebSocket at `/` for real-time log streaming

## Configuration

### Environment Variables

```bash
# API Configuration
API_PORT=3002
API_KEY=your-secret-api-key
NODE_ENV=production
LOG_LEVEL=info

# Docker Configuration
DOCKER_PROJECT_PATH=/path/to/your/media/server
DOCKER_COMPOSE_FILE=docker-compose.yml

# Cross-Seed Configuration
CROSS_SEED_ENABLED=true
CROSS_SEED_DELAY=30
CROSS_SEED_TIMEOUT=60

# qBittorrent Configuration
QBITTORRENT_USERNAME=admin
QBITTORRENT_PASSWORD=adminadmin

# CORS Configuration
CORS_ORIGIN=*
```

### Service Profiles

Configure service profiles in your Docker Compose file using labels:

```yaml
services:
  jellyfin:
    image: jellyfin/jellyfin:latest
    profiles: ["minimal", "media", "full"]
    # ... other configuration
```

## WebSocket API

Connect to `ws://localhost:3002` for real-time updates:

### Subscription Actions
```javascript
// Subscribe to health updates
ws.send(JSON.stringify({
    action: 'subscribe-health'
}));

// Subscribe to log streaming
ws.send(JSON.stringify({
    action: 'subscribe-logs',
    payload: {
        level: 'info',
        service: 'jellyfin'
    }
}));

// Ping/pong for connection testing
ws.send(JSON.stringify({
    action: 'ping'
}));
```

### Event Types
- `initial-status` - Initial service and health status
- `services-started/stopped/restarted` - Service actions
- `config-updated` - Configuration changes
- `health-update` - Health status updates
- `log-entry` - New log entries
- `cross-seed-started` - Cross-seed operations

## Security

### API Key Authentication

For write operations, include the API key in the request header:
```bash
curl -H "X-API-Key: your-secret-api-key" \
     -X POST \
     http://localhost:3002/api/services/start
```

### Rate Limiting

- **100 requests per 15 minutes** per IP address
- Applies to all `/api/*` endpoints
- Returns `429 Too Many Requests` when exceeded

### Input Validation

All endpoints include comprehensive input validation:
- Request size limits (10MB max)
- Content-Type validation
- Parameter sanitization
- Schema validation with Joi

## Health Monitoring

The API provides comprehensive health monitoring:

### System Health
- CPU usage and load average
- Memory utilization
- Disk space usage
- Network statistics
- Process information

### Service Health
- Container status and statistics
- Health check endpoints
- Response time monitoring
- Docker daemon status

### Health Scoring
- Overall health score (0-100)
- Status levels: healthy, warning, critical
- Trend analysis and historical data

## Logging

### Log Levels
- `error` - Error conditions
- `warn` - Warning conditions
- `info` - Informational messages (default)
- `debug` - Debug information
- `trace` - Detailed trace information

### Log Sources
- API server logs
- Service container logs
- System logs
- Docker logs

### Log Management
- Automatic log rotation (10MB files, 5 file retention)
- Real-time log streaming via WebSocket
- Log filtering and search
- Export capabilities (JSON, CSV, TXT)

## Development

### Running Tests
```bash
# Install dependencies
npm install

# Run integration tests
npm test

# Run with coverage
npm run test:coverage
```

### Development Mode
```bash
# Start with auto-reload
npm run dev

# Enable debug logging
LOG_LEVEL=debug npm run dev
```

### API Documentation

- Interactive docs: `http://localhost:3002/api/docs`
- OpenAPI spec: `/api/docs/openapi.yaml`
- Postman collection available

## Production Deployment

### Docker Deployment

1. **Create production environment file:**
```bash
cp env.template .env.production
# Configure production settings
```

2. **Add to your docker-compose.yml:**
```yaml
services:
  media-api:
    build: ./api
    container_name: media-api
    environment:
      - NODE_ENV=production
      - API_PORT=3002
    ports:
      - "3002:3002"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock:ro
      - .:/app/project:ro
    restart: unless-stopped
```

### Process Management

Use PM2 for production process management:
```bash
# Install PM2
npm install -g pm2

# Start with PM2
pm2 start start.js --name "media-api"

# Monitor
pm2 monit

# Auto-restart on system boot
pm2 startup
pm2 save
```

### Reverse Proxy

Configure Nginx or Traefik for HTTPS and load balancing:
```nginx
server {
    listen 443 ssl;
    server_name api.mediaserver.local;
    
    location / {
        proxy_pass http://localhost:3002;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

## Troubleshooting

### Common Issues

**Docker connection errors:**
- Ensure Docker daemon is running
- Check Docker socket permissions
- Verify docker-compose.yml path

**Port conflicts:**
- Change `API_PORT` environment variable
- Check for other services using the port

**Permission errors:**
- Run with appropriate user permissions
- Check file system permissions for config/logs

**Service health checks failing:**
- Verify service ports are accessible
- Check firewall and network settings
- Review service-specific health endpoints

### Debug Mode

Enable debug logging for troubleshooting:
```bash
LOG_LEVEL=debug npm start
```

### Health Check

Test API health:
```bash
curl http://localhost:3002/health
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add/update tests
5. Submit a pull request

### Code Style
- ESLint configuration provided
- Prettier for code formatting
- Comprehensive JSDoc comments
- Integration tests for all endpoints

## License

MIT License - see LICENSE file for details.

## Support

- Documentation: [API Docs](http://localhost:3002/api/docs)
- Issues: Create GitHub issues for bugs and features
- Security: Report security issues privately

---

Built with ❤️ for the media server community