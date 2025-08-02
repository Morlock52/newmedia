# Config Server API

A secure Node.js/Express server for managing .env files and Docker services with real-time WebSocket updates.

## Features

- **Secure .env File Management**
  - Read, write, and update environment variables
  - Preserve comments and file structure
  - File watching with real-time updates
  - Backup creation before modifications

- **Docker Service Control**
  - Start, stop, and restart services
  - View service status and logs
  - Pull latest images
  - Support for docker-compose

- **Security**
  - JWT-based authentication
  - Rate limiting
  - Input validation with Joi
  - Helmet.js security headers
  - Path traversal protection

- **Real-time Updates**
  - WebSocket support for live notifications
  - File change detection
  - Service status updates

## Installation

```bash
npm install
```

## Configuration

1. Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

2. Update the configuration values:
- `JWT_SECRET`: Change to a secure random string
- `ADMIN_PASSWORD_HASH`: Generate with: `node -e "console.log(require('bcryptjs').hashSync('your-password', 10))"`
- `DOCKER_COMPOSE_PATH`: Path to your docker-compose.yml file
- `ENV_FILES_PATH`: Directory containing .env files

## Running the Server

```bash
# Production
npm start

# Development (with auto-reload)
npm run dev
```

## API Endpoints

### Authentication

#### Login
```http
POST /api/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "your-password"
}
```

Response:
```json
{
  "token": "jwt-token",
  "user": {
    "username": "admin",
    "role": "admin"
  }
}
```

#### Verify Token
```http
POST /api/auth/verify
Authorization: Bearer <token>
```

### Environment File Management

All endpoints require authentication via `Authorization: Bearer <token>` header.

#### List .env Files
```http
GET /api/env/files
```

#### Read .env File
```http
GET /api/env/files/.env
```

#### Update Entire .env File
```http
PUT /api/env/files/.env
Content-Type: application/json

{
  "variables": {
    "API_KEY": "new-value",
    "DATABASE_URL": "postgres://..."
  }
}
```

Or with raw content:
```json
{
  "content": "API_KEY=value\nDATABASE_URL=postgres://..."
}
```

#### Update Single Variable
```http
PATCH /api/env/files/.env/variables/API_KEY
Content-Type: application/json

{
  "value": "new-api-key-value"
}
```

#### Delete Variable
```http
DELETE /api/env/files/.env/variables/OLD_KEY
```

#### Watch File for Changes
```http
POST /api/env/files/.env/watch
```

#### Stop Watching File
```http
DELETE /api/env/files/.env/watch
```

### Docker Service Management

#### List All Services Status
```http
GET /api/docker/services
```

#### Get Specific Service Status
```http
GET /api/docker/services/web
```

#### Start Services
```http
POST /api/docker/services/start
Content-Type: application/json

{
  "services": ["web", "db"]  // Optional, starts all if omitted
}
```

#### Stop Services
```http
POST /api/docker/services/stop
Content-Type: application/json

{
  "services": ["web"]  // Optional, stops all if omitted
}
```

#### Restart Services
```http
POST /api/docker/services/restart
Content-Type: application/json

{
  "services": ["web", "db"]  // Optional, restarts all if omitted
}
```

#### Get Service Logs
```http
GET /api/docker/services/web/logs?tail=100
```

For streaming logs:
```http
GET /api/docker/services/web/logs?follow=true&tail=50
```

#### Pull Latest Images
```http
POST /api/docker/services/pull
Content-Type: application/json

{
  "services": ["web"]  // Optional, pulls all if omitted
}
```

### Health Check
```http
GET /api/health
```

## WebSocket Connection

Connect to receive real-time updates:

```javascript
const ws = new WebSocket('ws://localhost:3000?token=<jwt-token>');

ws.on('message', (data) => {
  const event = JSON.parse(data);
  console.log('Event:', event);
});
```

Event types:
- `connection`: Initial connection success
- `env:file:updated`: Environment file was updated
- `env:file:changed`: File changed on disk (when watching)
- `env:file:deleted`: File was deleted
- `env:variable:updated`: Single variable updated
- `env:variable:deleted`: Variable deleted
- `docker:services:started`: Services started
- `docker:services:stopped`: Services stopped
- `docker:services:restarted`: Services restarted
- `docker:images:pulled`: Images pulled

## Security Considerations

1. **Always change default credentials** in production
2. **Use HTTPS** in production environments
3. **Restrict CORS origins** to trusted domains
4. **Keep JWT_SECRET** secure and rotate regularly
5. **Monitor rate limits** and adjust as needed
6. **Review file permissions** for .env files

## Error Handling

The API returns consistent error responses:

```json
{
  "error": "Human-readable error message",
  "details": "Additional error details (if available)"
}
```

HTTP Status Codes:
- `200`: Success
- `400`: Bad Request (validation error)
- `401`: Unauthorized (missing/invalid token)
- `403`: Forbidden (insufficient permissions)
- `404`: Not Found
- `500`: Internal Server Error

## Development

The server includes comprehensive error handling, input validation, and logging. In development mode (`NODE_ENV=development`), additional error details are included in responses.

## License

MIT