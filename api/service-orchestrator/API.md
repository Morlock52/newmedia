# Service Orchestrator API Documentation

## Overview

The Service Orchestrator API provides comprehensive management of Docker-based services for the Ultimate Media Server 2025. It handles service installation, configuration, monitoring, and inter-service communication.

Base URL: `http://localhost:3000/api/v1`

## Authentication

All API endpoints (except health checks) require authentication using JWT tokens.

### Login
```http
POST /auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password"
}
```

Response:
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expiresIn": 3600,
  "user": {
    "id": "user-123",
    "username": "admin",
    "roles": ["admin"]
  }
}
```

Use the token in subsequent requests:
```http
Authorization: Bearer <token>
```

## Service Management

### List All Services
```http
GET /services
```

Query Parameters:
- `status` - Filter by status (running, stopped, installing)
- `category` - Filter by category (media-server, arr-suite, download, etc.)
- `sort` - Sort field (name, status, created)
- `order` - Sort order (asc, desc)

Response:
```json
{
  "services": [
    {
      "id": "jellyfin",
      "name": "jellyfin",
      "category": "media-server",
      "version": "latest",
      "status": "running",
      "health": "healthy",
      "ports": ["8096:8096"],
      "created": "2024-01-01T00:00:00Z",
      "lastUpdated": "2024-01-01T00:00:00Z"
    }
  ],
  "total": 15,
  "page": 1,
  "limit": 20
}
```

### Get Service Details
```http
GET /services/{serviceName}
```

Response:
```json
{
  "id": "jellyfin",
  "name": "jellyfin",
  "category": "media-server",
  "version": "latest",
  "status": "running",
  "health": {
    "status": "healthy",
    "lastCheck": "2024-01-01T00:00:00Z",
    "responseTime": 145,
    "details": {
      "checks": {
        "http": "passed",
        "docker": "passed"
      }
    }
  },
  "config": {
    "image": "jellyfin/jellyfin:latest",
    "ports": ["8096:8096", "8920:8920"],
    "volumes": [
      "./config/jellyfin:/config",
      "./media-data:/media:ro"
    ],
    "environment": {
      "PUID": "1000",
      "PGID": "1000",
      "TZ": "America/New_York"
    },
    "networks": ["media_network"]
  },
  "containers": [
    {
      "id": "abc123...",
      "name": "jellyfin",
      "state": "running",
      "uptime": "2024-01-01T00:00:00Z",
      "restartCount": 0
    }
  ],
  "dependencies": ["postgres", "redis"],
  "integrations": [
    {
      "service": "sonarr",
      "type": "api",
      "status": "connected"
    }
  ]
}
```

### Install Service
```http
POST /services
Content-Type: application/json

{
  "name": "sonarr",
  "category": "arr-suite",
  "version": "latest",
  "config": {
    "image": "lscr.io/linuxserver/sonarr:latest",
    "ports": ["8989:8989"],
    "volumes": [
      {
        "source": "./config/sonarr",
        "target": "/config"
      },
      {
        "source": "${MEDIA_PATH}",
        "target": "/media"
      }
    ],
    "environment": {
      "PUID": "1000",
      "PGID": "1000",
      "TZ": "America/New_York"
    },
    "networks": ["media_network"],
    "restart": "unless-stopped",
    "dependencies": ["prowlarr"],
    "healthcheck": {
      "test": ["CMD", "curl", "-f", "http://localhost:8989/ping"],
      "interval": "30s",
      "timeout": "10s",
      "retries": 3
    }
  }
}
```

Response:
```json
{
  "id": "sonarr",
  "name": "sonarr",
  "status": "installing",
  "message": "Service installation started",
  "taskId": "task-456"
}
```

### Update Service Configuration
```http
PUT /services/{serviceName}
Content-Type: application/json

{
  "config": {
    "environment": {
      "LOG_LEVEL": "debug",
      "CUSTOM_SETTING": "value"
    },
    "ports": ["8989:8989", "9999:9999"]
  }
}
```

### Uninstall Service
```http
DELETE /services/{serviceName}
```

Query Parameters:
- `removeData` - Remove configuration data (default: false)
- `force` - Force removal even with dependents (default: false)

### Service Control

#### Start Service
```http
POST /services/{serviceName}/start
```

#### Stop Service
```http
POST /services/{serviceName}/stop
```

Query Parameters:
- `graceful` - Graceful shutdown (default: true)
- `timeout` - Shutdown timeout in seconds (default: 30)

#### Restart Service
```http
POST /services/{serviceName}/restart
```

### Service Status
```http
GET /services/{serviceName}/status
```

Response:
```json
{
  "service": "jellyfin",
  "status": "running",
  "health": "healthy",
  "uptime": 3600,
  "containers": [
    {
      "id": "container-123",
      "status": "running",
      "cpu": 15.5,
      "memory": 512,
      "restarts": 0
    }
  ],
  "lastEvent": {
    "type": "started",
    "timestamp": "2024-01-01T00:00:00Z"
  }
}
```

### Service Logs
```http
GET /services/{serviceName}/logs
```

Query Parameters:
- `tail` - Number of lines to return (default: 100)
- `since` - Unix timestamp to start from
- `follow` - Stream logs (WebSocket upgrade)
- `timestamps` - Include timestamps (default: true)

Response:
```json
{
  "service": "jellyfin",
  "logs": [
    {
      "container": "jellyfin",
      "timestamp": "2024-01-01T00:00:00Z",
      "message": "[INFO] Server started successfully",
      "stream": "stdout"
    }
  ]
}
```

### Service Metrics
```http
GET /services/{serviceName}/metrics
```

Query Parameters:
- `period` - Time period (1h, 6h, 24h, 7d)
- `resolution` - Data resolution (1m, 5m, 1h)

Response:
```json
{
  "service": "jellyfin",
  "period": "1h",
  "metrics": {
    "cpu": {
      "current": 15.5,
      "average": 12.3,
      "max": 45.2,
      "data": [
        {
          "timestamp": "2024-01-01T00:00:00Z",
          "value": 15.5
        }
      ]
    },
    "memory": {
      "current": 512,
      "average": 480,
      "max": 650,
      "limit": 2048,
      "data": [...]
    },
    "network": {
      "rx_bytes": 1048576,
      "tx_bytes": 2097152,
      "data": [...]
    },
    "disk": {
      "read_bytes": 10485760,
      "write_bytes": 5242880,
      "data": [...]
    }
  }
}
```

## Health Monitoring

### System Health
```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "1.0.0",
  "uptime": 86400,
  "services": {
    "total": 15,
    "healthy": 14,
    "unhealthy": 1,
    "stopped": 0
  }
}
```

### Service Health
```http
GET /health/services
```

Response:
```json
{
  "status": "degraded",
  "services": [
    {
      "name": "jellyfin",
      "status": "healthy",
      "lastCheck": "2024-01-01T00:00:00Z",
      "responseTime": 145
    },
    {
      "name": "sonarr",
      "status": "unhealthy",
      "lastCheck": "2024-01-01T00:00:00Z",
      "error": "Connection timeout"
    }
  ]
}
```

### Dependency Health
```http
GET /health/dependencies
```

Response:
```json
{
  "docker": {
    "status": "healthy",
    "version": "24.0.0",
    "containers": 15,
    "images": 20
  },
  "network": {
    "status": "healthy",
    "latency": "low"
  },
  "storage": {
    "status": "healthy",
    "usage": "45%",
    "available": "550GB"
  },
  "database": {
    "status": "healthy",
    "connections": "active"
  }
}
```

## Configuration Management

### Get Configuration
```http
GET /config/{serviceName}
```

Query Parameters:
- `environment` - Environment name (default: production)
- `includeSecrets` - Include decrypted secrets (default: false)

Response:
```json
{
  "service": "jellyfin",
  "environment": "production",
  "variables": {
    "PUID": "1000",
    "PGID": "1000",
    "TZ": "America/New_York",
    "JELLYFIN_PublishedServerUrl": "https://media.example.com"
  },
  "secrets": {
    "JELLYFIN_API_KEY": "***REDACTED***"
  },
  "version": 3,
  "lastModified": "2024-01-01T00:00:00Z"
}
```

### Update Configuration
```http
PUT /config/{serviceName}
Content-Type: application/json

{
  "environment": "production",
  "variables": {
    "LOG_LEVEL": "debug",
    "CUSTOM_SETTING": "value"
  },
  "secrets": {
    "API_KEY": "new-secret-key"
  }
}
```

### Validate Configuration
```http
POST /config/validate
Content-Type: application/json

{
  "service": "jellyfin",
  "environment": "production",
  "variables": {
    "PUID": "1000",
    "TZ": "Invalid/Timezone"
  }
}
```

Response:
```json
{
  "valid": false,
  "errors": [
    "Invalid format for TZ: must match timezone pattern"
  ],
  "warnings": [
    "JELLYFIN_PublishedServerUrl is not set"
  ]
}
```

### Reload Configuration
```http
POST /config/{serviceName}/reload
```

This will reload configuration without restarting the service (if supported).

### Export Configuration
```http
GET /config/{serviceName}/export
```

Query Parameters:
- `format` - Export format (env, yaml, json)
- `includeSecrets` - Include secrets (default: false)

Response (format=env):
```text
# Generated by Ultimate Media Server
# Service: jellyfin
# Environment: production

PUID=1000
PGID=1000
TZ=America/New_York
JELLYFIN_PublishedServerUrl=https://media.example.com
JELLYFIN_API_KEY=***REDACTED***
```

### Import Configuration
```http
POST /config/{serviceName}/import
Content-Type: application/json

{
  "environment": "staging",
  "format": "env",
  "data": "PUID=1000\nPGID=1000\nTZ=America/New_York"
}
```

### Secret Rotation
```http
POST /config/{serviceName}/secrets/{secretKey}/rotate
```

Response:
```json
{
  "service": "jellyfin",
  "secret": "JELLYFIN_API_KEY",
  "rotated": true,
  "version": 2,
  "nextRotation": "2024-02-01T00:00:00Z"
}
```

### Schedule Secret Rotation
```http
POST /config/{serviceName}/secrets/{secretKey}/schedule
Content-Type: application/json

{
  "intervalDays": 30,
  "environment": "production"
}
```

## Task Management

### List Tasks
```http
GET /tasks
```

Query Parameters:
- `status` - Filter by status (pending, running, completed, failed)
- `service` - Filter by service name
- `type` - Filter by task type
- `since` - Tasks created after timestamp

Response:
```json
{
  "tasks": [
    {
      "id": "task-123",
      "type": "service_install",
      "service": "sonarr",
      "status": "running",
      "progress": 75,
      "message": "Pulling Docker images...",
      "created": "2024-01-01T00:00:00Z",
      "started": "2024-01-01T00:00:01Z"
    }
  ]
}
```

### Get Task Details
```http
GET /tasks/{taskId}
```

Response:
```json
{
  "id": "task-123",
  "type": "service_install",
  "service": "sonarr",
  "status": "completed",
  "progress": 100,
  "result": {
    "success": true,
    "message": "Service installed successfully"
  },
  "logs": [
    {
      "timestamp": "2024-01-01T00:00:01Z",
      "level": "info",
      "message": "Starting service installation"
    }
  ],
  "created": "2024-01-01T00:00:00Z",
  "started": "2024-01-01T00:00:01Z",
  "completed": "2024-01-01T00:02:00Z",
  "duration": 119
}
```

### Create Task
```http
POST /tasks
Content-Type: application/json

{
  "type": "backup",
  "service": "all",
  "config": {
    "includeMedia": false,
    "compression": true
  }
}
```

### Cancel Task
```http
DELETE /tasks/{taskId}
```

## WebSocket Events

Connect to WebSocket endpoint for real-time updates:
```
ws://localhost:3000/ws
```

### Event Types

#### Service Events
```json
{
  "type": "service:installed",
  "data": {
    "service": "jellyfin",
    "version": "latest",
    "status": "installed"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Other service events:
- `service:started`
- `service:stopped`
- `service:updated`
- `service:uninstalled`
- `service:health:changed`

#### Task Events
```json
{
  "type": "task:progress",
  "data": {
    "taskId": "task-123",
    "progress": 50,
    "message": "Configuring service..."
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

Other task events:
- `task:created`
- `task:started`
- `task:completed`
- `task:failed`

#### Health Events
```json
{
  "type": "health:alert",
  "data": {
    "service": "sonarr",
    "severity": "warning",
    "message": "Service unhealthy for 5 minutes"
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

#### System Events
```json
{
  "type": "system:backup:completed",
  "data": {
    "backup": "backup-2024-01-01.tar.gz",
    "size": 1073741824,
    "duration": 300
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Error Responses

All errors follow a consistent format:

```json
{
  "error": {
    "code": "SERVICE_NOT_FOUND",
    "message": "Service 'unknown' not found",
    "details": {
      "service": "unknown"
    }
  },
  "timestamp": "2024-01-01T00:00:00Z"
}
```

### Common Error Codes

- `UNAUTHORIZED` - Missing or invalid authentication
- `FORBIDDEN` - Insufficient permissions
- `SERVICE_NOT_FOUND` - Service does not exist
- `SERVICE_ALREADY_EXISTS` - Service already installed
- `INVALID_CONFIG` - Invalid service configuration
- `DEPENDENCY_ERROR` - Missing or conflicting dependencies
- `TASK_NOT_FOUND` - Task does not exist
- `OPERATION_FAILED` - Generic operation failure

## Rate Limiting

API requests are rate limited:
- Anonymous: 10 requests per minute
- Authenticated: 100 requests per minute
- Admin: 1000 requests per minute

Rate limit headers:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1704067200
```

## Metrics Endpoint

Prometheus-compatible metrics available at:
```http
GET /metrics
```

Example metrics:
```
# HELP api_requests_total Total API requests
# TYPE api_requests_total counter
api_requests_total{method="GET",endpoint="/services",status="200"} 1234

# HELP service_health_status Service health status
# TYPE service_health_status gauge
service_health_status{service="jellyfin"} 1
service_health_status{service="sonarr"} 0

# HELP api_response_duration_seconds API response duration
# TYPE api_response_duration_seconds histogram
api_response_duration_seconds_bucket{le="0.1"} 950
api_response_duration_seconds_bucket{le="0.5"} 990
api_response_duration_seconds_bucket{le="1"} 999
```

## OpenAPI Specification

Full OpenAPI 3.0 specification available at:
```http
GET /openapi.json
```

Interactive API documentation:
```http
GET /docs
```