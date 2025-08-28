# 🔌 Ultimate Media Server 2025 - API Documentation

<div align="center">
  <img src="https://img.shields.io/badge/API-v2.0-blue?style=for-the-badge" alt="API Version">
  <img src="https://img.shields.io/badge/REST-Compliant-green?style=for-the-badge" alt="REST">
  <img src="https://img.shields.io/badge/WebSocket-Support-orange?style=for-the-badge" alt="WebSocket">
</div>

---

## 📋 Table of Contents

1. [API Overview](#-api-overview)
2. [Authentication](#-authentication)
3. [Dashboard API](#-dashboard-api)
4. [Service Management](#-service-management)
5. [System Information](#-system-information)
6. [Download Management](#-download-management)
7. [Media Library](#-media-library)
8. [Monitoring & Metrics](#-monitoring--metrics)
9. [WebSocket Events](#-websocket-events)
10. [Service-Specific APIs](#-service-specific-apis)
11. [Error Handling](#-error-handling)
12. [Rate Limiting](#-rate-limiting)
13. [SDKs & Examples](#-sdks--examples)

---

## 🌐 API Overview

The Ultimate Media Server 2025 provides a comprehensive REST API for managing all aspects of your media server infrastructure. The API is designed to be:

- **RESTful**: Following REST principles with proper HTTP methods
- **Real-time**: WebSocket support for live updates
- **Comprehensive**: Complete coverage of all system functionality
- **Secure**: Token-based authentication with rate limiting
- **Documented**: OpenAPI 3.0 specification available

### Base URLs

| Environment | Base URL | Description |
|-------------|----------|-------------|
| **Local** | `http://localhost:3002/api` | Default local installation |
| **Custom** | `https://your-domain.com/api` | Custom domain setup |
| **Docker** | `http://container-name:3002/api` | Inter-container communication |

### API Versioning

The API uses URL path versioning:
- Current version: `v2`
- Deprecated: `v1` (supported until 2026)
- Beta features: `v3` (preview)

Example: `GET /api/v2/services`

### Content Types

- **Request**: `application/json`
- **Response**: `application/json`
- **File uploads**: `multipart/form-data`
- **Streaming**: `text/event-stream`

---

## 🔐 Authentication

### API Key Authentication

Most endpoints require API key authentication via header:

```http
Authorization: Bearer YOUR_API_KEY
```

### Getting an API Key

```bash
# Generate new API key
curl -X POST http://localhost:3002/api/v2/auth/token \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "your_password"
  }'
```

**Response:**
```json
{
  "success": true,
  "data": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "expires_in": 86400,
    "refresh_token": "refresh_token_here"
  }
}
```

### Token Refresh

```bash
curl -X POST http://localhost:3002/api/v2/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{
    "refresh_token": "your_refresh_token"
  }'
```

### Public Endpoints

Some endpoints don't require authentication:
- `GET /api/v2/health`
- `GET /api/v2/version`
- `GET /api/v2/status`

---

## 📊 Dashboard API

### Get Dashboard Overview

```http
GET /api/v2/dashboard
```

**Response:**
```json
{
  "success": true,
  "data": {
    "system": {
      "cpu_usage": 25.4,
      "memory_usage": 68.2,
      "disk_usage": 45.1,
      "uptime": 86400
    },
    "services": {
      "total": 15,
      "healthy": 13,
      "unhealthy": 1,
      "unknown": 1
    },
    "media": {
      "movies": 1245,
      "tv_shows": 89,
      "episodes": 3421,
      "music_tracks": 15678
    },
    "downloads": {
      "active": 3,
      "queued": 7,
      "completed_today": 12
    }
  }
}
```

### Get Widget Data

```http
GET /api/v2/dashboard/widgets/{widget_id}
```

**Parameters:**
- `widget_id`: Widget identifier (e.g., `system-stats`, `recent-downloads`)

**Example Response:**
```json
{
  "success": true,
  "data": {
    "widget_id": "system-stats",
    "title": "System Statistics",
    "data": {
      "cpu": { "value": 25.4, "unit": "%" },
      "memory": { "value": 68.2, "unit": "%" },
      "disk": { "value": 45.1, "unit": "%" }
    },
    "last_updated": "2025-01-15T10:30:00Z"
  }
}
```

### Update Widget Configuration

```http
PUT /api/v2/dashboard/widgets/{widget_id}
```

**Request Body:**
```json
{
  "title": "Custom Widget Title",
  "refresh_interval": 30,
  "settings": {
    "show_percentages": true,
    "theme": "dark"
  }
}
```

---

## ⚙️ Service Management

### List All Services

```http
GET /api/v2/services
```

**Query Parameters:**
- `status`: Filter by status (`healthy`, `unhealthy`, `unknown`)
- `type`: Filter by service type (`media-server`, `content-management`, etc.)
- `category`: Group by category

**Response:**
```json
{
  "success": true,
  "data": {
    "services": {
      "jellyfin": {
        "name": "Jellyfin",
        "type": "media-server",
        "url": "http://localhost:8096",
        "description": "Media streaming server",
        "health": {
          "status": "healthy",
          "last_check": "2025-01-15T10:30:00Z",
          "response_time": 150,
          "uptime": 99.9
        },
        "config": {
          "auto_start": true,
          "restart_policy": "unless-stopped"
        }
      }
    }
  },
  "meta": {
    "total": 15,
    "healthy": 13,
    "unhealthy": 1,
    "unknown": 1
  }
}
```

### Get Service Details

```http
GET /api/v2/services/{service_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "service_id": "jellyfin",
    "name": "Jellyfin",
    "status": "running",
    "health": {
      "status": "healthy",
      "checks": [
        {
          "name": "HTTP Response",
          "status": "passing",
          "last_check": "2025-01-15T10:30:00Z"
        },
        {
          "name": "Database Connection",
          "status": "passing",
          "last_check": "2025-01-15T10:29:45Z"
        }
      ]
    },
    "metrics": {
      "cpu_usage": 15.2,
      "memory_usage": 512.5,
      "network_in": 1024000,
      "network_out": 2048000
    },
    "logs": {
      "recent": [
        {
          "timestamp": "2025-01-15T10:30:00Z",
          "level": "info",
          "message": "User authenticated successfully"
        }
      ]
    }
  }
}
```

### Restart Service

```http
POST /api/v2/services/{service_id}/restart
```

**Response:**
```json
{
  "success": true,
  "message": "Service restart initiated",
  "data": {
    "service_id": "jellyfin",
    "action": "restart",
    "status": "in_progress",
    "estimated_duration": 30
  }
}
```

### Update Service Configuration

```http
PUT /api/v2/services/{service_id}/config
```

**Request Body:**
```json
{
  "auto_start": true,
  "restart_policy": "unless-stopped",
  "health_check": {
    "enabled": true,
    "interval": 30,
    "timeout": 10
  },
  "resource_limits": {
    "memory": "2G",
    "cpu": "1.0"
  }
}
```

### Service Actions

```http
POST /api/v2/services/{service_id}/actions
```

**Request Body:**
```json
{
  "action": "start|stop|restart|update|backup",
  "parameters": {
    "force": true,
    "timeout": 60
  }
}
```

**Available Actions:**
- `start`: Start the service
- `stop`: Stop the service  
- `restart`: Restart the service
- `update`: Update to latest image
- `backup`: Create configuration backup
- `health_check`: Force health check

---

## 💻 System Information

### Get System Status

```http
GET /api/v2/system
```

**Response:**
```json
{
  "success": true,
  "data": {
    "system": {
      "hostname": "mediaserver",
      "platform": "linux",
      "arch": "x64",
      "uptime": 86400,
      "load_average": [1.2, 1.5, 1.8],
      "cpu": {
        "model": "Intel Core i7-10700K",
        "cores": 8,
        "threads": 16,
        "usage": 25.4,
        "temperature": 45.2
      },
      "memory": {
        "total": 17179869184,
        "used": 11744051200,
        "free": 5435817984,
        "usage_percent": 68.2
      },
      "storage": [
        {
          "device": "/dev/sda1",
          "mountpoint": "/",
          "filesystem": "ext4",
          "total": 1000204886016,
          "used": 451289341952,
          "free": 548915544064,
          "usage_percent": 45.1
        }
      ],
      "network": {
        "interfaces": [
          {
            "name": "eth0",
            "address": "192.168.1.100",
            "rx_bytes": 1073741824,
            "tx_bytes": 536870912
          }
        ]
      }
    },
    "docker": {
      "version": "24.0.7",
      "containers": {
        "total": 15,
        "running": 14,
        "stopped": 1
      },
      "images": 25,
      "volumes": 18,
      "networks": 3
    }
  }
}
```

### Get System Health

```http
GET /api/v2/system/health
```

**Response:**
```json
{
  "success": true,
  "data": {
    "overall_status": "healthy",
    "checks": [
      {
        "name": "CPU Usage",
        "status": "healthy",
        "value": 25.4,
        "threshold": 80,
        "message": "CPU usage is normal"
      },
      {
        "name": "Memory Usage", 
        "status": "warning",
        "value": 85.2,
        "threshold": 90,
        "message": "Memory usage is high"
      },
      {
        "name": "Disk Space",
        "status": "healthy",
        "value": 45.1,
        "threshold": 85,
        "message": "Sufficient disk space available"
      }
    ],
    "services_health": {
      "healthy": 13,
      "unhealthy": 1,
      "unknown": 1
    }
  }
}
```

### Get System Logs

```http
GET /api/v2/system/logs
```

**Query Parameters:**
- `level`: Filter by log level (`error`, `warn`, `info`, `debug`)
- `service`: Filter by service name
- `since`: Show logs since timestamp (ISO 8601)
- `limit`: Number of logs to return (default: 100)

**Response:**
```json
{
  "success": true,
  "data": {
    "logs": [
      {
        "timestamp": "2025-01-15T10:30:00Z",
        "level": "info",
        "service": "dashboard",
        "message": "Health check completed successfully",
        "metadata": {
          "duration": 1.2,
          "services_checked": 15
        }
      }
    ],
    "meta": {
      "total": 1000,
      "returned": 100,
      "has_more": true
    }
  }
}
```

---

## 📥 Download Management

### Get Download Queue

```http
GET /api/v2/downloads
```

**Query Parameters:**
- `status`: Filter by status (`downloading`, `queued`, `completed`, `failed`)
- `category`: Filter by category (`movies`, `tv`, `music`)
- `client`: Filter by download client (`qbittorrent`, `transmission`)

**Response:**
```json
{
  "success": true,
  "data": {
    "downloads": [
      {
        "id": "download_123",
        "name": "Movie.Title.2024.1080p.BluRay.x264-GROUP",
        "status": "downloading",
        "progress": 65.2,
        "speed": 5242880,
        "eta": 1800,
        "size": 8589934592,
        "downloaded": 5603341312,
        "category": "movies",
        "client": "qbittorrent",
        "added": "2025-01-15T09:30:00Z",
        "completed": null
      }
    ],
    "stats": {
      "total_downloads": 10,
      "active": 3,
      "queued": 4,
      "completed": 2,
      "failed": 1,
      "total_speed": 15728640,
      "total_size": 85899345920
    }
  }
}
```

### Add Download

```http
POST /api/v2/downloads
```

**Request Body:**
```json
{
  "url": "magnet:?xt=urn:btih:...",
  "category": "movies",
  "priority": "high",
  "client": "qbittorrent",
  "options": {
    "skip_hash_check": false,
    "sequential_download": false,
    "first_last_piece_priority": true
  }
}
```

**Response:**
```json
{
  "success": true,
  "message": "Download added successfully",
  "data": {
    "download_id": "download_124",
    "status": "queued",
    "estimated_start": "2025-01-15T10:35:00Z"
  }
}
```

### Control Download

```http
POST /api/v2/downloads/{download_id}/actions
```

**Request Body:**
```json
{
  "action": "pause|resume|delete|force_start",
  "delete_files": false
}
```

**Available Actions:**
- `pause`: Pause the download
- `resume`: Resume paused download
- `delete`: Remove download (optionally delete files)
- `force_start`: Force start queued download
- `recheck`: Recheck download integrity

### Get Download Statistics

```http
GET /api/v2/downloads/stats
```

**Response:**
```json
{
  "success": true,
  "data": {
    "current_session": {
      "downloaded": 85899345920,
      "uploaded": 17179869184,
      "ratio": 0.2,
      "active_time": 7200
    },
    "all_time": {
      "downloaded": 1099511627776,
      "uploaded": 219902325555,
      "ratio": 0.2
    },
    "speed": {
      "download": 5242880,
      "upload": 1048576
    },
    "connections": {
      "max": 200,
      "current": 45
    }
  }
}
```

---

## 🎬 Media Library

### Get Library Overview

```http
GET /api/v2/media
```

**Response:**
```json
{
  "success": true,
  "data": {
    "libraries": [
      {
        "id": "movies",
        "name": "Movies",
        "type": "movie",
        "path": "/media/movies",
        "item_count": 1245,
        "total_size": 10995116277760,
        "last_scan": "2025-01-15T08:00:00Z"
      },
      {
        "id": "tv",
        "name": "TV Shows", 
        "type": "series",
        "path": "/media/tv",
        "item_count": 89,
        "episode_count": 3421,
        "total_size": 5497558138880,
        "last_scan": "2025-01-15T08:00:00Z"
      }
    ],
    "totals": {
      "movies": 1245,
      "tv_shows": 89,
      "episodes": 3421,
      "music_albums": 234,
      "music_tracks": 15678,
      "total_size": 21990232555520
    }
  }
}
```

### Get Library Items

```http
GET /api/v2/media/{library_id}/items
```

**Query Parameters:**
- `limit`: Number of items per page (default: 50)
- `offset`: Pagination offset (default: 0)
- `sort`: Sort field (`name`, `date_added`, `size`)
- `order`: Sort order (`asc`, `desc`)
- `search`: Search query

**Response:**
```json
{
  "success": true,
  "data": {
    "items": [
      {
        "id": "movie_123",
        "title": "Movie Title",
        "year": 2024,
        "path": "/media/movies/Movie Title (2024)",
        "size": 8589934592,
        "quality": "1080p",
        "codec": "x264",
        "added": "2025-01-15T10:00:00Z",
        "metadata": {
          "imdb_id": "tt1234567",
          "tmdb_id": 12345,
          "rating": 8.5,
          "genres": ["Action", "Thriller"],
          "poster": "/images/poster_123.jpg"
        }
      }
    ],
    "meta": {
      "total": 1245,
      "returned": 50,
      "page": 1,
      "pages": 25
    }
  }
}
```

### Scan Library

```http
POST /api/v2/media/{library_id}/scan
```

**Request Body:**
```json
{
  "force": false,
  "deep_scan": true,
  "update_metadata": true
}
```

### Get Recent Additions

```http
GET /api/v2/media/recent
```

**Query Parameters:**
- `days`: Number of days to look back (default: 7)
- `limit`: Number of items to return (default: 20)
- `type`: Media type filter (`movie`, `episode`, `album`)

**Response:**
```json
{
  "success": true,
  "data": {
    "recent_items": [
      {
        "id": "movie_125",
        "title": "New Movie",
        "type": "movie",
        "added": "2025-01-14T20:30:00Z",
        "library": "movies",
        "metadata": {
          "poster": "/images/poster_125.jpg",
          "year": 2024
        }
      }
    ]
  }
}
```

---

## 📊 Monitoring & Metrics

### Get System Metrics

```http
GET /api/v2/metrics
```

**Query Parameters:**
- `start`: Start time (ISO 8601)
- `end`: End time (ISO 8601)
- `interval`: Data interval (`1m`, `5m`, `1h`, `1d`)

**Response:**
```json
{
  "success": true,
  "data": {
    "metrics": {
      "cpu_usage": [
        {
          "timestamp": "2025-01-15T10:00:00Z",
          "value": 25.4
        }
      ],
      "memory_usage": [
        {
          "timestamp": "2025-01-15T10:00:00Z", 
          "value": 68.2
        }
      ],
      "disk_io": [
        {
          "timestamp": "2025-01-15T10:00:00Z",
          "read": 1048576,
          "write": 2097152
        }
      ],
      "network_traffic": [
        {
          "timestamp": "2025-01-15T10:00:00Z",
          "rx": 1073741824,
          "tx": 536870912
        }
      ]
    }
  }
}
```

### Get Service Metrics

```http
GET /api/v2/metrics/services/{service_id}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "service_id": "jellyfin",
    "metrics": {
      "response_time": [
        {
          "timestamp": "2025-01-15T10:00:00Z",
          "value": 150
        }
      ],
      "active_sessions": [
        {
          "timestamp": "2025-01-15T10:00:00Z",
          "value": 3
        }
      ],
      "transcoding_sessions": [
        {
          "timestamp": "2025-01-15T10:00:00Z",
          "value": 1
        }
      ]
    }
  }
}
```

### Export Metrics

```http
GET /api/v2/metrics/export
```

**Query Parameters:**
- `format`: Export format (`json`, `csv`, `prometheus`)
- `services`: Comma-separated service IDs
- `start`: Start time
- `end`: End time

**Response (Prometheus format):**
```
# HELP mediaserver_cpu_usage CPU usage percentage
# TYPE mediaserver_cpu_usage gauge
mediaserver_cpu_usage 25.4

# HELP mediaserver_memory_usage Memory usage percentage  
# TYPE mediaserver_memory_usage gauge
mediaserver_memory_usage 68.2
```

---

## 🔌 WebSocket Events

### Connection

Connect to WebSocket endpoint:
```javascript
const ws = new WebSocket('ws://localhost:3002/api/v2/ws');
```

### Authentication

```javascript
ws.send(JSON.stringify({
  type: 'auth',
  token: 'YOUR_API_KEY'
}));
```

### Event Types

#### Service Status Updates
```json
{
  "type": "service_status",
  "data": {
    "service_id": "jellyfin",
    "status": "healthy",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

#### System Metrics
```json
{
  "type": "system_metrics",
  "data": {
    "cpu_usage": 25.4,
    "memory_usage": 68.2,
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

#### Download Progress
```json
{
  "type": "download_progress",
  "data": {
    "download_id": "download_123",
    "progress": 75.2,
    "speed": 5242880,
    "eta": 900
  }
}
```

#### Media Library Updates
```json
{
  "type": "library_update",
  "data": {
    "library_id": "movies",
    "action": "item_added",
    "item": {
      "id": "movie_126",
      "title": "New Movie",
      "added": "2025-01-15T10:30:00Z"
    }
  }
}
```

### Subscribing to Events

```javascript
ws.send(JSON.stringify({
  type: 'subscribe',
  events: ['service_status', 'download_progress', 'system_metrics']
}));
```

---

## 🔧 Service-Specific APIs

### Jellyfin Integration

#### Get Jellyfin Stats
```http
GET /api/v2/jellyfin/stats
```

#### Get Active Sessions
```http
GET /api/v2/jellyfin/sessions
```

### Sonarr Integration

#### Get Series
```http
GET /api/v2/sonarr/series
```

#### Search for Series
```http
POST /api/v2/sonarr/series/search
```

### Radarr Integration

#### Get Movies
```http
GET /api/v2/radarr/movies
```

#### Search for Movies
```http
POST /api/v2/radarr/movies/search
```

### qBittorrent Integration

#### Get Torrents
```http
GET /api/v2/qbittorrent/torrents
```

#### Add Torrent
```http
POST /api/v2/qbittorrent/torrents
```

---

## ❌ Error Handling

### Error Response Format

```json
{
  "success": false,
  "error": {
    "code": "SERVICE_UNAVAILABLE",
    "message": "Service is currently unavailable",
    "details": "Jellyfin container is not responding",
    "timestamp": "2025-01-15T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

### HTTP Status Codes

| Code | Description | Usage |
|------|-------------|-------|
| `200` | OK | Successful request |
| `201` | Created | Resource created successfully |
| `400` | Bad Request | Invalid request parameters |
| `401` | Unauthorized | Missing or invalid authentication |
| `403` | Forbidden | Insufficient permissions |
| `404` | Not Found | Resource not found |
| `429` | Too Many Requests | Rate limit exceeded |
| `500` | Internal Server Error | Server error |
| `503` | Service Unavailable | Service temporarily unavailable |

### Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request validation failed |
| `AUTHENTICATION_FAILED` | Invalid credentials |
| `AUTHORIZATION_FAILED` | Insufficient permissions |
| `RESOURCE_NOT_FOUND` | Requested resource not found |
| `SERVICE_UNAVAILABLE` | External service unavailable |
| `RATE_LIMIT_EXCEEDED` | Too many requests |
| `INTERNAL_ERROR` | Unexpected server error |

---

## 🚦 Rate Limiting

### Default Limits

| Endpoint Category | Requests per Minute | Burst |
|-------------------|-------------------|-------|
| **Authentication** | 10 | 15 |
| **Service Management** | 30 | 50 |
| **System Information** | 60 | 100 |
| **Media Library** | 100 | 150 |
| **Download Management** | 50 | 75 |
| **Metrics** | 120 | 200 |

### Rate Limit Headers

```http
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1642276800
```

### Rate Limit Exceeded Response

```json
{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded",
    "details": "Maximum 60 requests per minute allowed",
    "retry_after": 30
  }
}
```

---

## 🛠️ SDKs & Examples

### JavaScript/Node.js SDK

```javascript
const MediaServerAPI = require('@ultimate-media-server/api');

const client = new MediaServerAPI({
  baseURL: 'http://localhost:3002/api/v2',
  apiKey: 'your_api_key'
});

// Get service status
const services = await client.services.list();
console.log(services);

// Restart service
await client.services.restart('jellyfin');

// Get system metrics
const metrics = await client.metrics.getSystem({
  start: '2025-01-15T00:00:00Z',
  end: '2025-01-15T23:59:59Z'
});
```

### Python SDK

```python
from ultimate_media_server import MediaServerAPI

client = MediaServerAPI(
    base_url='http://localhost:3002/api/v2',
    api_key='your_api_key'
)

# Get service status
services = client.services.list()
print(services)

# Add download
download = client.downloads.add(
    url='magnet:?xt=...',
    category='movies'
)

# Get recent media
recent = client.media.get_recent(days=7)
```

### cURL Examples

#### Health Check
```bash
curl -X GET http://localhost:3002/api/v2/health
```

#### Get Services with Authentication
```bash
curl -X GET http://localhost:3002/api/v2/services \
  -H "Authorization: Bearer YOUR_API_KEY"
```

#### Restart Service
```bash
curl -X POST http://localhost:3002/api/v2/services/jellyfin/restart \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json"
```

#### Add Download
```bash
curl -X POST http://localhost:3002/api/v2/downloads \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "url": "magnet:?xt=...",
    "category": "movies",
    "priority": "high"
  }'
```

### WebSocket Client Example

```javascript
class MediaServerWebSocket {
  constructor(apiKey) {
    this.apiKey = apiKey;
    this.ws = null;
  }
  
  connect() {
    this.ws = new WebSocket('ws://localhost:3002/api/v2/ws');
    
    this.ws.onopen = () => {
      // Authenticate
      this.ws.send(JSON.stringify({
        type: 'auth',
        token: this.apiKey
      }));
      
      // Subscribe to events
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        events: ['service_status', 'download_progress']
      }));
    };
    
    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      this.handleEvent(data);
    };
  }
  
  handleEvent(event) {
    switch (event.type) {
      case 'service_status':
        console.log('Service status:', event.data);
        break;
      case 'download_progress':
        console.log('Download progress:', event.data);
        break;
    }
  }
}

const wsClient = new MediaServerWebSocket('your_api_key');
wsClient.connect();
```

---

## 📝 OpenAPI Specification

The complete OpenAPI 3.0 specification is available at:
- **JSON**: `GET /api/v2/openapi.json`
- **YAML**: `GET /api/v2/openapi.yaml`
- **Interactive Docs**: http://localhost:3002/api/docs

### Generating SDKs

Use the OpenAPI specification to generate SDKs for your preferred language:

```bash
# Generate JavaScript SDK
openapi-generator generate -i openapi.yaml -g javascript -o ./js-sdk

# Generate Python SDK
openapi-generator generate -i openapi.yaml -g python -o ./python-sdk

# Generate Go SDK
openapi-generator generate -i openapi.yaml -g go -o ./go-sdk
```

---

## 🔄 Webhooks

### Configure Webhooks

```http
POST /api/v2/webhooks
```

**Request Body:**
```json
{
  "name": "Discord Notifications",
  "url": "https://discord.com/api/webhooks/...",
  "events": ["download_completed", "service_down"],
  "headers": {
    "Content-Type": "application/json"
  },
  "template": {
    "content": "Download completed: {{item.name}}"
  }
}
```

### Webhook Events

Available webhook events:
- `download_started`
- `download_completed`
- `download_failed`
- `service_started`
- `service_stopped`
- `service_down`
- `system_alert`
- `library_updated`

---

## 📚 Additional Resources

### Postman Collection

Import our Postman collection for easy API testing:
```bash
curl -o ultimate-media-server.postman.json \
  http://localhost:3002/api/v2/postman.json
```

### API Changelog

- **v2.1.0**: Added webhook support
- **v2.0.0**: WebSocket real-time updates
- **v1.9.0**: Enhanced metrics endpoints
- **v1.8.0**: Service-specific API integrations

### Support

- **Documentation**: Full API documentation
- **GitHub Issues**: Bug reports and feature requests
- **Discord**: Community support and discussions

---

<div align="center">
  <p><strong>Ultimate Media Server 2025 - API Documentation</strong></p>
  <p>Made with ❤️ by the self-hosting community</p>
</div>