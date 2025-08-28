# API Integration Test Report

**Generated:** 2025-08-09T07:31:04.103Z

## Summary

- **Total Tests:** 45
- **Passed:** 17
- **Failed:** 28
- **Warnings:** 0
- **Success Rate:** 37.8%

## Test Results

### Jellyfin Media Server

✅ **Jellyfin Media Server - Health Check**
   - Status: ACCESSIBLE
   - Message: Service accessible (75ms)
   - Data: {"responseTime":75,"status":200}

✅ **Jellyfin Media Server - Response Time**
   - Status: GOOD_PERFORMANCE
   - Message: Average response time: 14.67ms

### Plex Media Server

❌ **Plex Media Server - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

### Sonarr TV Shows

✅ **Sonarr TV Shows - Health Check**
   - Status: ACCESSIBLE
   - Message: Service accessible (95ms)
   - Data: {"responseTime":95,"status":200}

✅ **Sonarr TV Shows - Response Time**
   - Status: GOOD_PERFORMANCE
   - Message: Average response time: 3.33ms

### Radarr Movies

✅ **Radarr Movies - Health Check**
   - Status: ACCESSIBLE
   - Message: Service accessible (87ms)
   - Data: {"responseTime":87,"status":200}

✅ **Radarr Movies - Response Time**
   - Status: GOOD_PERFORMANCE
   - Message: Average response time: 3.00ms

### Lidarr Music

❌ **Lidarr Music - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

### Bazarr Subtitles

❌ **Bazarr Subtitles - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

### Prowlarr Indexer

✅ **Prowlarr Indexer - Health Check**
   - Status: ACCESSIBLE
   - Message: Service accessible (102ms)
   - Data: {"responseTime":102,"status":200}

✅ **Prowlarr Indexer - Response Time**
   - Status: GOOD_PERFORMANCE
   - Message: Average response time: 4.33ms

### qBittorrent Client

✅ **qBittorrent Client - Health Check**
   - Status: ACCESSIBLE
   - Message: Service accessible (20ms)
   - Data: {"responseTime":20,"status":200}

### Transmission Client

❌ **Transmission Client - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

### SABnzbd Usenet

❌ **SABnzbd Usenet - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

### Media Server API

❌ **Media Server API - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

❌ **Media Server API - Response Time**
   - Status: SLOW_PERFORMANCE
   - Message: Average response time: 10000.00ms (slow)

### Media Dashboard

❌ **Media Dashboard - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

### Prometheus Monitoring

❌ **Prometheus Monitoring - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

### Grafana Dashboards

❌ **Grafana Dashboards - Health Check**
   - Status: NOT_ACCESSIBLE
   - Message: Service not running or port closed

### Uptime Kuma

✅ **Uptime Kuma - Health Check**
   - Status: ACCESSIBLE
   - Message: Service accessible (53ms)
   - Data: {"responseTime":53,"status":200}

### Portainer Docker UI

✅ **Portainer Docker UI - Health Check**
   - Status: ACCESSIBLE
   - Message: Service accessible (41ms)
   - Data: {"responseTime":41,"status":200}

### Jellyfin

✅ **Jellyfin - API Authentication**
   - Status: NO_AUTH_REQUIRED
   - Message: Jellyfin API accessible without authentication

### Sonarr

❌ **Sonarr - API Authentication**
   - Status: AUTH_FAILED
   - Message: Invalid API key

❌ **Sonarr - Download Clients**
   - Status: ERROR
   - Message: Request failed with status code 401

### Radarr

❌ **Radarr - API Authentication**
   - Status: AUTH_FAILED
   - Message: Invalid API key

❌ **Radarr - Download Clients**
   - Status: ERROR
   - Message: Request failed with status code 401

### Lidarr

❌ **Lidarr - API Authentication**
   - Status: ERROR
   - Message: 

❌ **Lidarr - Download Clients**
   - Status: ERROR
   - Message: 

### Bazarr

❌ **Bazarr - API Authentication**
   - Status: NO_API_KEY
   - Message: API key not found

### Prowlarr

❌ **Prowlarr - API Authentication**
   - Status: AUTH_FAILED
   - Message: Invalid API key

### qBittorrent

❌ **qBittorrent - API Authentication**
   - Status: ERROR
   - Message: Request failed with status code 403

### Transmission

❌ **Transmission - API Authentication**
   - Status: ERROR
   - Message: 

### SABnzbd

❌ **SABnzbd - API Authentication**
   - Status: ERROR
   - Message: 

### Prowlarr Integration Tests

❌ **Prowlarr Integration Tests**
   - Status: ERROR
   - Message: Request failed with status code 401

### Jellyfin API Endpoints

✅ **Jellyfin API Endpoints - System Info**
   - Status: SUCCESS
   - Message: Jellyfin version: 10.10.7

✅ **Jellyfin API Endpoints - Library**
   - Status: SUCCESS
   - Message: Library endpoint accessible

### Plex API Endpoints

❌ **Plex API Endpoints**
   - Status: ERROR
   - Message: 

### Custom Media API Server

❌ **Custom Media API Server**
   - Status: ERROR
   - Message: 

### PostgreSQL Database

✅ **PostgreSQL Database**
   - Status: RUNNING
   - Message: PostgreSQL appears to be running (non-HTTP response expected)

### Redis Database

✅ **Redis Database**
   - Status: RUNNING
   - Message: Redis appears to be running (non-HTTP response expected)

### MariaDB Database

❌ **MariaDB Database**
   - Status: SERVICE_DOWN
   - Message: MariaDB service not accessible

### Container Network Communication

✅ **Container Network Communication**
   - Status: DOCKER_NETWORK
   - Message: Services should communicate via Docker network (manual verification needed)

### Sonarr Webhooks

❌ **Sonarr Webhooks**
   - Status: ERROR
   - Message: Request failed with status code 401

### Radarr Webhooks

❌ **Radarr Webhooks**
   - Status: ERROR
   - Message: Request failed with status code 401

### Lidarr Webhooks

❌ **Lidarr Webhooks**
   - Status: ERROR
   - Message: 

## Recommendations

- Start missing services: docker-compose up -d
- Configure API keys for *ARR services in their respective web interfaces
- Consider optimizing system resources or increasing timeout values
- Verify database services are running and accessible
