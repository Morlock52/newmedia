# Ultimate Media Server 2025 - Project Structure

## Core Architecture
- **Frontend**: Next.js 15 + React 19 with App Router
- **3D Visualization**: Three.js + React Three Fiber
- **Animation**: Framer Motion + Lottie
- **Styling**: Tailwind CSS v4 + Shadcn/ui
- **State Management**: Zustand + React Query
- **Real-time**: WebSockets + Server-Sent Events

## Services Integration (30+ Services)
### Media Servers
- Jellyfin (Primary)
- Plex (Secondary)
- Emby (Optional)

### Media Management
- Sonarr (TV Shows)
- Radarr (Movies)
- Lidarr (Music)
- Readarr (Books)
- Bazarr (Subtitles)

### Download Clients
- qBittorrent
- Transmission
- SABnzbd
- NZBGet

### Request Management
- Overseerr
- Jellyseerr
- Ombi
- Requestrr

### Monitoring & Analytics
- Tautulli
- Varken
- Grafana
- Prometheus
- Uptime Kuma

### Infrastructure
- Nginx Proxy Manager
- Portainer
- Caddy
- Redis
- PostgreSQL

## Project Phases
1. Architecture Design & Planning
2. Backend API Development
3. Frontend Dashboard Creation
4. Service Integration
5. Testing & Optimization
6. Documentation & Deployment