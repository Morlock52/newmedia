# Media Server Web Interface Test Report
Generated: 2025-08-09T07:40:57.514Z
Total Services: 15
Accessible: 5/15 (33%)

## Summary

### ✅ Accessible Services (5)
- **Jellyfin** (http://localhost:8096) - Status: 302 - Response: 89ms
- **Prowlarr** (http://localhost:9696) - Status: 200 - Response: 177ms - Title: "Prowlarr"
- **qBittorrent** (http://localhost:8080) - Status: 200 - Response: 71ms - Title: "qBittorrent WebUI"
- **Portainer** (http://localhost:9000) - Status: 200 - Response: 12ms - Title: "Portainer"
- **Uptime Kuma** (http://localhost:3001) - Status: 302 - Response: 27ms

### ❌ Inaccessible Services (10)
- **Sonarr** (http://localhost:8989) - Error: read ECONNRESET
- **Radarr** (http://localhost:7878) - Error: read ECONNRESET
- **Bazarr** (http://localhost:6767) - Error: ERROR
- **Lidarr** (http://localhost:8686) - Error: ERROR
- **SABnzbd** (http://localhost:8085) - Error: ERROR
- **Transmission** (http://localhost:9091) - Error: ERROR
- **Jellyseerr** (http://localhost:5055) - Error: ERROR
- **Tautulli** (http://localhost:8181) - Error: ERROR
- **Nginx Proxy Manager** (http://localhost:81) - Error: ERROR
- **Dashboard** (http://localhost:3000) - Error: ERROR

## Detailed Results

### Jellyfin
- **URL**: http://localhost:8096
- **Status**: 302
- **Accessible**: ✅ Yes
- **Response Time**: 89ms
- **Content Length**: 0 bytes
- **Has HTML**: ❌ No
- **Page Title**: Not found
- **Has Login Form**: ❌ No
- **JavaScript Errors Detected**: ✅ No
- **Redirects To**: web/

### Sonarr
- **URL**: http://localhost:8989
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 49ms
- **Error**: read ECONNRESET

### Radarr
- **URL**: http://localhost:7878
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 50ms
- **Error**: read ECONNRESET

### Prowlarr
- **URL**: http://localhost:9696
- **Status**: 200
- **Accessible**: ✅ Yes
- **Response Time**: 177ms
- **Content Length**: 1972 bytes
- **Has HTML**: ✅ Yes
- **Page Title**: Prowlarr
- **Has Login Form**: ❌ No
- **JavaScript Errors Detected**: ✅ No

### qBittorrent
- **URL**: http://localhost:8080
- **Status**: 200
- **Accessible**: ✅ Yes
- **Response Time**: 71ms
- **Content Length**: 1808 bytes
- **Has HTML**: ✅ Yes
- **Page Title**: qBittorrent WebUI
- **Has Login Form**: ✅ Yes
- **JavaScript Errors Detected**: ✅ No

### Bazarr
- **URL**: http://localhost:6767
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 5ms
- **Error**: Unknown error

### Lidarr
- **URL**: http://localhost:8686
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 4ms
- **Error**: Unknown error

### SABnzbd
- **URL**: http://localhost:8085
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 4ms
- **Error**: Unknown error

### Transmission
- **URL**: http://localhost:9091
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 4ms
- **Error**: Unknown error

### Jellyseerr
- **URL**: http://localhost:5055
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 3ms
- **Error**: Unknown error

### Tautulli
- **URL**: http://localhost:8181
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 3ms
- **Error**: Unknown error

### Portainer
- **URL**: http://localhost:9000
- **Status**: 200
- **Accessible**: ✅ Yes
- **Response Time**: 12ms
- **Content Length**: 18729 bytes
- **Has HTML**: ✅ Yes
- **Page Title**: Portainer
- **Has Login Form**: ❌ No
- **JavaScript Errors Detected**: ✅ No

### Uptime Kuma
- **URL**: http://localhost:3001
- **Status**: 302
- **Accessible**: ✅ Yes
- **Response Time**: 27ms
- **Content Length**: 32 bytes
- **Has HTML**: ❌ No
- **Page Title**: Not found
- **Has Login Form**: ❌ No
- **JavaScript Errors Detected**: ✅ No
- **Redirects To**: /dashboard

### Nginx Proxy Manager
- **URL**: http://localhost:81
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 4ms
- **Error**: Unknown error

### Dashboard
- **URL**: http://localhost:3000
- **Status**: ERROR
- **Accessible**: ❌ No
- **Response Time**: 3ms
- **Error**: Unknown error

