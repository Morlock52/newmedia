# Ultimate Media Server API Documentation

## Overview

The Ultimate Media Server API provides comprehensive endpoints for managing media services, user authentication, real-time communications, and AI-powered features. This documentation covers all available endpoints, authentication methods, and integration patterns.

## Table of Contents

1. [Authentication](#authentication)
2. [Service Health Monitoring](#service-health-monitoring)
3. [Media Management](#media-management)
4. [Download Management](#download-management)
5. [Search & Discovery](#search--discovery)
6. [AI & Recommendations](#ai--recommendations)
7. [Real-time Features](#real-time-features)
8. [WebSocket Events](#websocket-events)
9. [Error Handling](#error-handling)
10. [Rate Limiting](#rate-limiting)

## Base URL

```
Development: http://localhost:3001
Production: https://your-domain.com
```

## Authentication

### JWT Token Authentication

The API uses JWT (JSON Web Tokens) for authentication. Include the token in the Authorization header:

```http
Authorization: Bearer <your-jwt-token>
```

### Registration

**POST** `/api/auth/register`

Register a new user account.

```json
{
  "email": "user@example.com",
  "username": "username",
  "password": "securepassword123",
  "confirmPassword": "securepassword123"
}
```

**Response:**
```json
{
  "user": {
    "id": "user123",
    "email": "user@example.com",
    "username": "username",
    "roles": ["user"],
    "permissions": ["media:read", "downloads:read"]
  },
  "tokens": {
    "accessToken": "eyJhbGciOiJIUzI1NiIs...",
    "refreshToken": "eyJhbGciOiJIUzI1NiIs...",
    "expiresIn": 900,
    "tokenType": "Bearer"
  }
}
```

### Login

**POST** `/api/auth/login`

Login with email and password.

```json
{
  "email": "user@example.com",
  "password": "securepassword123",
  "rememberMe": false
}
```

### OAuth2 Authentication

**POST** `/api/auth/google`

Authenticate using Google OAuth2.

```json
{
  "code": "google-auth-code"
}
```

**POST** `/api/auth/github`

Authenticate using GitHub OAuth2.

```json
{
  "code": "github-auth-code"
}
```

### Token Refresh

**POST** `/api/auth/refresh`

Refresh an expired access token.

```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIs..."
}
```

### Logout

**POST** `/api/auth/logout`

Logout and invalidate tokens.

```json
{
  "refreshToken": "eyJhbGciOiJIUzI1NiIs..."
}
```

## Service Health Monitoring

### Get All Services Health

**GET** `/api/health`

Returns health status for all configured services.

**Response:**
```json
{
  "timestamp": "2025-08-15T21:30:00.000Z",
  "services": [
    {
      "service": "jellyfin",
      "status": "healthy",
      "responseTime": 145,
      "lastCheck": "2025-08-15T21:30:00.000Z",
      "uptime": 99.5,
      "errorCount": 0,
      "circuitState": "closed"
    }
  ],
  "summary": {
    "total": 30,
    "healthy": 28,
    "unhealthy": 1,
    "degraded": 1
  }
}
```

### Get Specific Service Health

**GET** `/api/health/{service}`

Returns health status for a specific service.

**Parameters:**
- `service` (path): Service name (jellyfin, plex, sonarr, etc.)

### Circuit Breaker Management

**POST** `/api/health/{service}/circuit-breaker/reset`

Force reset circuit breaker for a service.

**GET** `/api/health/{service}/history`

Get health history for a service.

## Media Management

### Get Movies

**GET** `/api/media/movies`

Returns all movies from connected services.

**Response:**
```json
[
  {
    "service": "jellyfin",
    "movies": [
      {
        "id": "movie123",
        "title": "The Matrix",
        "type": "movie",
        "year": 1999,
        "overview": "A computer hacker learns...",
        "posterUrl": "https://...",
        "rating": 8.7,
        "genres": ["Action", "Sci-Fi"]
      }
    ]
  }
]
```

### Get TV Shows

**GET** `/api/media/tv`

Returns all TV shows from connected services.

### Get Media by Service

**GET** `/api/media/{service}/movies`
**GET** `/api/media/{service}/tv`

Get media from a specific service.

## Download Management

### Get Active Downloads

**GET** `/api/downloads`

Returns all active downloads across all services.

**Response:**
```json
[
  {
    "service": "qbittorrent",
    "downloads": [
      {
        "id": "download123",
        "title": "Movie Title",
        "progress": 65.5,
        "status": "downloading",
        "speed": "2.5 MB/s",
        "eta": "15m",
        "size": "4.2 GB"
      }
    ]
  }
]
```

### Download Controls

**POST** `/api/downloads/{id}/pause`

Pause a download.

**POST** `/api/downloads/{id}/resume`

Resume a paused download.

**DELETE** `/api/downloads/{id}`

Cancel and remove a download.

## Search & Discovery

### Universal Search

**POST** `/api/search`

Search for content across all connected services.

```json
{
  "query": "The Matrix",
  "type": "movie",
  "year": 1999
}
```

**Response:**
```json
[
  {
    "service": "prowlarr",
    "results": [
      {
        "id": "result123",
        "title": "The Matrix (1999)",
        "year": 1999,
        "quality": "1080p",
        "size": "8.5 GB",
        "seeders": 150,
        "leechers": 12,
        "indexer": "TorrentSite1"
      }
    ]
  }
]
```

### Advanced Search

**POST** `/api/search/advanced`

Perform advanced search with filters.

```json
{
  "query": "batman",
  "filters": {
    "type": ["movie", "tv"],
    "year": {"min": 2000, "max": 2023},
    "quality": ["1080p", "2160p"],
    "genre": ["action", "adventure"]
  },
  "sort": "seeders",
  "order": "desc"
}
```

## AI & Recommendations

### Get Recommendations

**GET** `/api/recommendations`

Get personalized content recommendations.

**Query Parameters:**
- `limit` (number): Number of recommendations (default: 20)
- `category` (string): Filter by category (trending, personal, similar, etc.)

**Response:**
```json
[
  {
    "contentId": "movie123",
    "score": 0.95,
    "reason": ["You like Action movies", "Highly rated", "Trending now"],
    "confidence": 0.87,
    "category": "personal",
    "metadata": {
      "title": "John Wick",
      "type": "movie",
      "genre": ["Action", "Thriller"],
      "rating": 7.4
    }
  }
]
```

### Viewing Analytics

**GET** `/api/analytics/viewing`

Get viewing pattern analysis for the user.

**POST** `/api/analytics/viewing`

Record a viewing pattern.

```json
{
  "contentId": "movie123",
  "contentType": "movie",
  "watchTime": 7200,
  "totalDuration": 7800,
  "completionRate": 0.92,
  "genre": ["Action", "Thriller"],
  "rating": 8,
  "device": "smart-tv",
  "timeOfDay": "evening"
}
```

### Smart Download Scheduling

**POST** `/api/downloads/schedule`

Generate optimal download schedule based on user patterns.

```json
{
  "contentIds": ["movie123", "movie456", "show789"]
}
```

**Response:**
```json
[
  {
    "contentId": "movie123",
    "priority": 95,
    "estimatedSize": 8500,
    "estimatedTime": 45,
    "optimalStartTime": "2025-08-15T22:00:00.000Z",
    "reason": "High user interest"
  }
]
```

### Cache Predictions

**GET** `/api/cache/predictions`

Get predictive caching recommendations.

## Real-time Features

### WebSocket Connection

Connect to WebSocket server at `/socket.io/`.

**Authentication:**
```javascript
socket.emit('authenticate', 'your-jwt-token')
```

### WebSocket Status

**GET** `/api/websocket/status`

Get WebSocket server statistics.

```json
{
  "connectedUsers": 15,
  "activeDownloads": 5,
  "totalDownloads": 23,
  "onlineServices": 28
}
```

## WebSocket Events

### Client → Server Events

#### Authentication
```javascript
socket.emit('authenticate', token)
```

#### Download Control
```javascript
socket.emit('download:pause', downloadId)
socket.emit('download:resume', downloadId)
socket.emit('download:cancel', downloadId)
```

#### Service Control
```javascript
socket.emit('service:restart', serviceName)
```

#### Chat/Messaging
```javascript
socket.emit('chat:join', channel)
socket.emit('chat:message', { channel, message, type })
socket.emit('chat:typing', { channel, isTyping })
```

#### Notifications
```javascript
socket.emit('notification:dismiss', notificationId)
socket.emit('notification:action', { notificationId, action, actionData })
```

### Server → Client Events

#### Authentication
```javascript
socket.on('authenticated', (data) => {
  // { user: {...} }
})
socket.on('auth:error', (data) => {
  // { message: 'Invalid token' }
})
```

#### Downloads
```javascript
socket.on('download:progress', (download) => {
  // Real-time progress updates
})
socket.on('download:completed', (download) => {
  // Download completion notification
})
socket.on('download:added', (download) => {
  // New download added
})
```

#### Services
```javascript
socket.on('service:status', (status) => {
  // Service status change
})
socket.on('services:initial', (statuses) => {
  // Initial service statuses on connect
})
```

#### Notifications
```javascript
socket.on('notification:new', (notification) => {
  // New notification
})
socket.on('notification:dismissed', (data) => {
  // Notification dismissed
})
```

#### Chat
```javascript
socket.on('chat:message', (message) => {
  // New chat message
})
socket.on('chat:history', (messages) => {
  // Chat history when joining channel
})
socket.on('chat:typing', (data) => {
  // User typing indicator
})
```

#### System
```javascript
socket.on('system:stats', (stats) => {
  // System statistics
})
```

## Service Discovery

### Get Service Topology

**GET** `/api/services/discovery`

Get current service discovery topology and load balancing statistics.

**Response:**
```json
{
  "topology": {
    "jellyfin": [
      {
        "id": "jellyfin-primary",
        "name": "jellyfin",
        "host": "localhost",
        "port": 8096,
        "protocol": "http",
        "health": "healthy",
        "weight": 1
      }
    ]
  },
  "loadBalancing": {
    "jellyfin": {
      "totalInstances": 1,
      "healthyInstances": 1,
      "healthPercentage": 100,
      "strategy": "health-aware"
    }
  }
}
```

## Error Handling

### Standard Error Response

All API errors follow this format:

```json
{
  "error": "Error type",
  "message": "Detailed error message",
  "code": "ERROR_CODE",
  "timestamp": "2025-08-15T21:30:00.000Z"
}
```

### HTTP Status Codes

- `200` - Success
- `201` - Created
- `400` - Bad Request
- `401` - Unauthorized
- `403` - Forbidden
- `404` - Not Found
- `429` - Too Many Requests
- `500` - Internal Server Error
- `502` - Bad Gateway (Service unavailable)
- `503` - Service Temporarily Unavailable

### Common Error Codes

- `INVALID_CREDENTIALS` - Login failed
- `TOKEN_EXPIRED` - Access token expired
- `SERVICE_UNAVAILABLE` - External service is down
- `CIRCUIT_BREAKER_OPEN` - Service circuit breaker is open
- `RATE_LIMIT_EXCEEDED` - Too many requests
- `VALIDATION_ERROR` - Request validation failed

## Rate Limiting

The API implements rate limiting to prevent abuse:

- **General API**: 100 requests per 15 minutes per IP
- **Authentication**: 10 requests per 5 minutes per IP
- **Search**: 50 requests per 10 minutes per authenticated user
- **WebSocket**: Connection-based limiting

Rate limit headers are included in responses:
```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1692135000
```

## SDK Examples

### JavaScript/Node.js

```javascript
const MediaServerAPI = require('@ultimate-media-server/api-client')

const api = new MediaServerAPI({
  baseURL: 'http://localhost:3001',
  timeout: 10000
})

// Login
const { tokens } = await api.auth.login({
  email: 'user@example.com',
  password: 'password'
})

// Set token for future requests
api.setToken(tokens.accessToken)

// Get recommendations
const recommendations = await api.recommendations.get({ limit: 10 })

// Search content
const results = await api.search.universal({ query: 'The Matrix' })

// WebSocket connection
const socket = api.websocket.connect()
socket.on('download:progress', (data) => {
  console.log('Download progress:', data)
})
```

### Python

```python
from ultimate_media_server import MediaServerAPI

api = MediaServerAPI(base_url='http://localhost:3001')

# Login
auth_result = api.auth.login(
    email='user@example.com',
    password='password'
)

# Set token
api.set_token(auth_result['tokens']['accessToken'])

# Get health status
health = api.health.get_all()

# Get recommendations
recommendations = api.recommendations.get(limit=10)
```

## Webhooks

The API supports webhooks for external integrations:

### Register Webhook

**POST** `/api/webhooks`

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["download.completed", "service.down", "user.registered"],
  "secret": "webhook-secret"
}
```

### Webhook Events

- `download.completed` - Download finished
- `download.failed` - Download failed
- `service.down` - Service went offline
- `service.up` - Service came online
- `user.registered` - New user registered
- `recommendation.generated` - New recommendations available

### Webhook Payload

```json
{
  "event": "download.completed",
  "timestamp": "2025-08-15T21:30:00.000Z",
  "data": {
    "downloadId": "download123",
    "title": "Movie Title",
    "service": "qbittorrent",
    "userId": "user123"
  },
  "signature": "sha256=..."
}
```

## Development

### Local Setup

1. Clone the repository
2. Install dependencies: `npm install`
3. Set environment variables
4. Start the server: `npm run dev:api`

### Environment Variables

```env
NODE_ENV=development
PORT=3001
JWT_SECRET=your-jwt-secret
REDIS_URL=redis://localhost:6379

# Service URLs
JELLYFIN_URL=http://localhost:8096
PLEX_URL=http://localhost:32400
SONARR_URL=http://localhost:8989
RADARR_URL=http://localhost:7878

# API Keys
SONARR_API_KEY=your-sonarr-api-key
RADARR_API_KEY=your-radarr-api-key

# OAuth2
GOOGLE_CLIENT_ID=your-google-client-id
GOOGLE_CLIENT_SECRET=your-google-client-secret
GITHUB_CLIENT_ID=your-github-client-id
GITHUB_CLIENT_SECRET=your-github-client-secret
```

### Testing

```bash
# Unit tests
npm test

# Integration tests
npm run test:integration

# E2E tests
npm run test:e2e

# Coverage
npm run test:coverage
```

## Support

For API support:
- Documentation: http://localhost:3001/api-docs
- GitHub Issues: https://github.com/your-repo/issues
- Email: support@ultimatemediaserver.com