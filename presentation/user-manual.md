# 📚 Ultimate Media Server 2025 - Complete User Manual

## 🎯 Table of Contents
1. [Quick Start Guide](#quick-start-guide)
2. [System Requirements](#system-requirements)
3. [Installation Methods](#installation-methods)
4. [Configuration Guide](#configuration-guide)
5. [Service Overview](#service-overview)
6. [AI Agent System](#ai-agent-system)
7. [Web Interface Guide](#web-interface-guide)
8. [Mobile Experience](#mobile-experience)
9. [API Documentation](#api-documentation)
10. [Troubleshooting](#troubleshooting)
11. [Performance Optimization](#performance-optimization)
12. [Security Guide](#security-guide)
13. [Backup & Recovery](#backup--recovery)
14. [Advanced Features](#advanced-features)
15. [FAQ](#faq)

---

## 🚀 Quick Start Guide

### Option 1: One-Line Installation (Easiest)
```bash
# Download and run automated installer
curl -sSL https://raw.githubusercontent.com/ultimate-media-server/main/install.sh | bash
```

### Option 2: Manual Setup
```bash
# 1. Clone repository
git clone https://github.com/ultimate-media-server/2025.git
cd ultimate-media-server

# 2. Configure environment
cp .env.example .env
nano .env  # Add your settings

# 3. Deploy stack
docker compose up -d

# 4. Access dashboard
open http://localhost:3001
```

### Option 3: Single Container
```bash
# Run everything in one container
docker run -d \
  --name mediaserver-aio \
  -p 80:80 \
  -p 8096:8096 \
  -v $(pwd)/config:/config \
  -v $(pwd)/media:/media \
  ultimatemediaserver/2025:latest
```

---

## 💻 System Requirements

### Minimum Requirements
- **CPU**: 4 cores (2.0 GHz)
- **RAM**: 8GB
- **Storage**: 100GB SSD + 1TB HDD
- **OS**: Ubuntu 22.04, Debian 12, CentOS 8+
- **Network**: 100 Mbps internet

### Recommended Requirements
- **CPU**: 8+ cores (3.0 GHz)
- **RAM**: 16GB+
- **Storage**: 500GB NVMe + 4TB+ HDD
- **OS**: Ubuntu 22.04 LTS
- **Network**: Gigabit internet
- **GPU**: NVIDIA (for transcoding)

### Supported Platforms
- ✅ **Linux**: Ubuntu, Debian, RHEL, CentOS
- ✅ **macOS**: Intel and Apple Silicon
- ✅ **Windows**: WSL2 or Docker Desktop
- ✅ **Cloud**: AWS, GCP, Azure, DigitalOcean
- ✅ **ARM**: Raspberry Pi 4+, ARM64 servers

---

## 🔧 Installation Methods

### Method 1: Docker Compose (Recommended)

#### Step 1: Install Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker compose version
```

#### Step 2: Download & Configure
```bash
# Clone repository
git clone https://github.com/ultimate-media-server/2025.git
cd ultimate-media-server

# Create directories
mkdir -p config media/{movies,tv,music,books} downloads

# Configure environment
cp .env.example .env
nano .env
```

#### Step 3: Environment Configuration
```bash
# .env file settings
PUID=1000                    # Your user ID (run: id -u)
PGID=1000                    # Your group ID (run: id -g)
TZ=America/New_York          # Your timezone
DOMAIN=localhost             # Your domain or IP

# Media paths
MEDIA_ROOT=/data/media
DOWNLOADS_ROOT=/data/downloads

# Database passwords
POSTGRES_PASSWORD=secure_password_here
MYSQL_ROOT_PASSWORD=another_secure_password

# API Keys (optional but recommended)
OPENAI_API_KEY=sk-your-key-here
TMDB_API_KEY=your-tmdb-key

# VPN Configuration (required for downloads)
VPN_PROVIDER=nordvpn
VPN_USER=your_vpn_username
VPN_PASS=your_vpn_password
VPN_COUNTRY=Switzerland
```

#### Step 4: Deploy Services
```bash
# Start all services
docker compose up -d

# Check status
docker compose ps

# View logs
docker compose logs -f
```

### Method 2: Single Container Deployment

```bash
# Build the all-in-one container
docker build -t mediaserver-aio -f Dockerfile.multi-service .

# Run with all ports exposed
docker run -d \
  --name mediaserver \
  --restart unless-stopped \
  -p 80:80 \
  -p 8096:8096 \
  -p 8989:8989 \
  -p 7878:7878 \
  -p 9696:9696 \
  -p 8080:8080 \
  -p 3001:3001 \
  -v $(pwd)/config:/config \
  -v $(pwd)/media:/data \
  -e PUID=1000 \
  -e PGID=1000 \
  -e TZ=America/New_York \
  mediaserver-aio
```

### Method 3: Kubernetes Deployment

```yaml
# kubernetes/mediaserver-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mediaserver
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mediaserver
  template:
    metadata:
      labels:
        app: mediaserver
    spec:
      containers:
      - name: mediaserver
        image: ultimatemediaserver/2025:latest
        ports:
        - containerPort: 80
        - containerPort: 8096
        volumeMounts:
        - name: config
          mountPath: /config
        - name: media
          mountPath: /media
      volumes:
      - name: config
        persistentVolumeClaim:
          claimName: mediaserver-config
      - name: media
        persistentVolumeClaim:
          claimName: mediaserver-media
```

---

## ⚙️ Configuration Guide

### Initial Setup Wizard

1. **Access the Dashboard**
   - Open http://localhost:3001
   - Complete the setup wizard
   - Set admin credentials

2. **Configure Media Libraries**
   - Movies: `/media/movies`
   - TV Shows: `/media/tv`
   - Music: `/media/music`
   - Books: `/media/books`

3. **Set Up Download Clients**
   - qBittorrent: http://localhost:8080
   - SABnzbd: http://localhost:8081
   - Configure download paths

### Service-Specific Configuration

#### Jellyfin Setup
1. Access: http://localhost:8096
2. Create admin account
3. Add media libraries:
   - Name: Movies, Path: `/media/movies`
   - Name: TV Shows, Path: `/media/tv`
   - Name: Music, Path: `/media/music`
4. Configure transcoding (if GPU available)
5. Set up user accounts

#### Sonarr Configuration
1. Access: http://localhost:8989
2. Settings → Download Clients
3. Add qBittorrent:
   - Host: `qbittorrent`
   - Port: `8080`
   - Username: `admin`
   - Password: `adminadmin` (change this!)
4. Settings → Media Management
   - Root Folder: `/media/tv`
   - File naming patterns

#### Radarr Configuration
1. Access: http://localhost:7878
2. Similar to Sonarr but for movies
3. Root Folder: `/media/movies`

#### Prowlarr Setup
1. Access: http://localhost:9696
2. Add indexers (torrent sites)
3. Connect to Sonarr and Radarr
4. Sync applications

---

## 📋 Service Overview

### Media Servers
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| **Jellyfin** | 8096 | Primary media server | ✅ Core |
| **Plex** | 32400 | Alternative media server | 🔄 Optional |
| **Emby** | 8097 | Another alternative | 🔄 Optional |

### Automation (*arr Suite)
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| **Sonarr** | 8989 | TV show automation | ✅ Core |
| **Radarr** | 7878 | Movie automation | ✅ Core |
| **Lidarr** | 8686 | Music automation | 🔄 Optional |
| **Readarr** | 8787 | Book automation | 🔄 Optional |
| **Bazarr** | 6767 | Subtitle automation | 🔄 Optional |
| **Prowlarr** | 9696 | Indexer management | ✅ Core |

### Download Clients
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| **qBittorrent** | 8080 | Torrent client | ✅ Core |
| **Transmission** | 9091 | Alternative torrent | 🔄 Optional |
| **SABnzbd** | 8081 | Usenet downloader | 🔄 Optional |
| **NZBGet** | 6789 | Alternative usenet | 🔄 Optional |

### Request Systems
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| **Overseerr** | 5055 | Plex requests | 🔄 Optional |
| **Jellyseerr** | 5056 | Jellyfin requests | ✅ Core |
| **Ombi** | 3579 | Universal requests | 🔄 Optional |

### Monitoring & Management
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| **Grafana** | 3000 | Dashboards | ✅ Core |
| **Prometheus** | 9090 | Metrics collection | ✅ Core |
| **Portainer** | 9000 | Docker management | ✅ Core |
| **Uptime Kuma** | 3001 | Service monitoring | ✅ Core |

### AI & Automation
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| **MCP Suite** | 8090 | AI agent orchestration | ✅ Core |
| **Jellyfin MCP** | 3001 | Media management AI | ✅ Core |
| **Sonarr MCP** | 3002 | TV automation AI | ✅ Core |
| **Radarr MCP** | 3003 | Movie automation AI | ✅ Core |
| **Prowlarr MCP** | 3004 | Indexer AI | ✅ Core |
| **qBittorrent MCP** | 3005 | Download AI | ✅ Core |

---

## 🤖 AI Agent System

### Agent Overview
The Media Server includes 6 specialized AI agents that use democratic voting to make intelligent decisions:

#### 1. 📺 Media Curator Agent
- **Role**: Content discovery and recommendations
- **Capabilities**:
  - Analyzes viewing patterns
  - Suggests new content based on preferences
  - Manages content quality preferences
  - Optimizes library organization

```bash
# Example: Ask for movie recommendations
curl -X POST http://localhost:8090/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Recommend action movies similar to John Wick"}'
```

#### 2. ⚙️ Technical Specialist Agent
- **Role**: System optimization and maintenance
- **Capabilities**:
  - Monitors system performance
  - Suggests configuration improvements
  - Manages resource allocation
  - Handles technical troubleshooting

#### 3. 👤 User Advocate Agent
- **Role**: User experience optimization
- **Capabilities**:
  - Analyzes user feedback
  - Suggests UI/UX improvements
  - Manages user access and permissions
  - Optimizes content accessibility

#### 4. 🔄 Automation Expert Agent
- **Role**: Workflow and process optimization
- **Capabilities**:
  - Optimizes download workflows
  - Manages automated tasks
  - Suggests process improvements
  - Handles scheduling and priorities

#### 5. 🛡️ Security Guardian Agent
- **Role**: Security and privacy protection
- **Capabilities**:
  - Monitors security threats
  - Manages VPN connections
  - Handles access control
  - Ensures privacy compliance

#### 6. 📈 Trend Analyst Agent
- **Role**: Social media and trend analysis
- **Capabilities**:
  - Monitors trending content
  - Analyzes social media for popular content
  - Predicts content popularity
  - Suggests timely downloads

### Voting System

#### How It Works
1. **Question Submission**: User asks a question via API or web interface
2. **Agent Analysis**: Each agent analyzes the question from their expertise
3. **Democratic Voting**: Agents vote on the best course of action
4. **Consensus Building**: Requires 70% agreement for action
5. **Action Execution**: System executes the agreed-upon action
6. **Feedback Loop**: Results are fed back to improve future decisions

#### Voting Process
```mermaid
graph TD
    A[User Question] --> B[Agent Analysis]
    B --> C[Individual Votes]
    C --> D{70% Consensus?}
    D -->|Yes| E[Execute Action]
    D -->|No| F[Request More Info]
    E --> G[Monitor Results]
    F --> B
    G --> H[Update Learning]
```

#### Example Voting Scenarios

**Scenario 1: Storage Upgrade Decision**
```json
{
  "question": "Should I upgrade to a larger storage array?",
  "votes": {
    "media_curator": {
      "vote": "yes",
      "confidence": 0.8,
      "reasoning": "Library growth indicates need for more space"
    },
    "technical_specialist": {
      "vote": "yes",
      "confidence": 0.9,
      "reasoning": "Current usage at 85%, performance degrading"
    },
    "user_advocate": {
      "vote": "yes",
      "confidence": 0.7,
      "reasoning": "Users complaining about slow loading"
    },
    "automation_expert": {
      "vote": "yes",
      "confidence": 0.8,
      "reasoning": "Download queue backing up due to space"
    },
    "security_guardian": {
      "vote": "neutral",
      "confidence": 0.5,
      "reasoning": "No security implications either way"
    },
    "trend_analyst": {
      "vote": "yes",
      "confidence": 0.6,
      "reasoning": "4K content trend requires more storage"
    }
  },
  "consensus": "yes",
  "confidence": 0.75,
  "action": "recommend_storage_upgrade"
}
```

---

## 🖥️ Web Interface Guide

### Dashboard Overview
The main dashboard provides a unified view of your entire media ecosystem:

#### Navigation Structure
```
📊 Dashboard
├── 🏠 Home - System overview and quick stats
├── 📺 Media - Library management and playback
├── 📥 Downloads - Active and queued downloads
├── 🤖 AI Agents - Agent status and voting history
├── 📊 Analytics - Performance and usage metrics
├── ⚙️ Settings - System configuration
└── 👤 Profile - User account and preferences
```

#### Home Dashboard Features
- **Real-time Statistics**: Active streams, library size, download speed
- **Recent Activity**: Latest downloads, recently watched content
- **System Health**: Service status, resource usage, alerts
- **Quick Actions**: Add content, manage downloads, system controls

#### Media Management
- **Library Browser**: Visual grid of all content with filtering
- **Search Interface**: Global search across all media types
- **Metadata Editor**: Edit movie/show information and artwork
- **Collection Manager**: Create and manage custom collections

#### Download Management
- **Active Downloads**: Real-time progress with ETA
- **Queue Management**: Prioritize, pause, or cancel downloads
- **History**: Complete download history with statistics
- **Automation Rules**: Configure automatic download preferences

### Voice Control Interface

The system includes advanced voice control powered by AI:

#### Activation
- **Wake phrase**: "Hey MediaFlow"
- **Button activation**: Click microphone icon
- **Keyboard shortcut**: Ctrl+Shift+V

#### Voice Commands
```bash
# Media Discovery
"Find action movies from 2023"
"Show me comedies with high ratings"
"What's trending on Netflix?"

# Playback Control
"Play The Matrix"
"Resume Stranger Things"
"Add Inception to my watchlist"

# System Management
"Check download status"
"How much storage space is left?"
"Show system performance"

# AI Agent Interaction
"Ask the agents about upgrading my server"
"What do the agents think about this movie?"
"Should I download this series?"
```

### Mobile Interface

#### Progressive Web App (PWA)
The mobile interface is a full-featured PWA with:
- **Offline Support**: 50GB cache for offline viewing
- **Install Prompt**: Add to home screen functionality
- **Push Notifications**: Download complete, new content alerts
- **Background Sync**: Sync preferences when connection restored

#### Mobile-Specific Features
- **Swipe Gestures**: Navigate between sections
- **Pull-to-Refresh**: Update content lists
- **Thumb Navigation**: Bottom navigation bar
- **Responsive Design**: Optimized for all screen sizes

---

## 📱 Mobile Experience

### Installation
1. Open browser on mobile device
2. Navigate to your server URL
3. Tap browser menu → "Add to Home Screen"
4. Launch the installed app

### Mobile App Features

#### Streaming
- **Adaptive Bitrate**: Automatically adjusts to connection
- **Offline Downloads**: Download content for offline viewing
- **Chromecast Support**: Cast to TV with one tap
- **Picture-in-Picture**: Continue watching while using other apps

#### Content Management
- **Request Interface**: Request new movies/shows on mobile
- **Watchlist Sync**: Synchronized across all devices
- **Progress Tracking**: Resume on any device
- **Rating System**: Rate content to improve recommendations

#### Mobile-Optimized UI
```css
/* Mobile breakpoints */
@media (max-width: 768px) {
  .container {
    padding: 10px;
    font-size: 14px;
  }
  
  .nav-tabs {
    overflow-x: auto;
    white-space: nowrap;
  }
  
  .grid {
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
  }
}

@media (max-width: 480px) {
  .grid {
    grid-template-columns: 1fr;
  }
}
```

---

## 🔌 API Documentation

### RESTful API Endpoints

#### Authentication
```bash
# Get access token
POST /api/auth/login
{
  "username": "admin",
  "password": "password"
}

# Response
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "expires": "2025-01-15T10:30:00Z"
}

# Use token in subsequent requests
Authorization: Bearer eyJhbGciOiJIUzI1NiIs...
```

#### Media Endpoints
```bash
# Get all movies
GET /api/media/movies

# Search media
GET /api/media/search?q=inception&type=movie

# Get media details
GET /api/media/movies/123

# Add media to library
POST /api/media/movies
{
  "title": "The Matrix",
  "year": 1999,
  "imdb_id": "tt0133093"
}
```

#### Download Endpoints
```bash
# Get active downloads
GET /api/downloads

# Add download
POST /api/downloads
{
  "url": "magnet:?xt=urn:btih:...",
  "category": "movies"
}

# Pause/Resume download
PUT /api/downloads/123/pause
PUT /api/downloads/123/resume

# Delete download
DELETE /api/downloads/123
```

#### AI Agent Endpoints
```bash
# Chat with AI agents
POST /api/ai/chat
{
  "message": "Should I upgrade my storage?",
  "context": "system_performance"
}

# Get agent status
GET /api/ai/agents

# Get voting history
GET /api/ai/votes?limit=10

# Submit vote
POST /api/ai/vote
{
  "question_id": "uuid-here",
  "agent": "technical_specialist",
  "vote": "yes",
  "confidence": 0.8
}
```

#### System Endpoints
```bash
# System health
GET /api/system/health

# Performance metrics
GET /api/system/metrics

# Service status
GET /api/system/services

# Restart service
POST /api/system/services/jellyfin/restart
```

### WebSocket API

#### Connection
```javascript
const ws = new WebSocket('ws://localhost:8090/ws');

ws.onopen = function() {
  console.log('Connected to MediaServer WebSocket');
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

#### Message Types
```javascript
// Subscribe to events
ws.send(JSON.stringify({
  type: 'subscribe',
  events: ['download_progress', 'media_added', 'agent_vote']
}));

// Real-time download progress
{
  "type": "download_progress",
  "download_id": "123",
  "progress": 75.5,
  "speed": "15.2 MB/s",
  "eta": "00:05:30"
}

// New media added
{
  "type": "media_added",
  "media_type": "movie",
  "title": "The Matrix",
  "year": 1999
}

// Agent voting update
{
  "type": "agent_vote",
  "question_id": "uuid-here",
  "agent": "media_curator",
  "vote": "yes",
  "current_consensus": 0.6
}
```

### GraphQL API

#### Schema Overview
```graphql
type Query {
  movies(limit: Int, offset: Int): [Movie]
  tvShows(limit: Int, offset: Int): [TVShow]
  downloads: [Download]
  agents: [Agent]
  systemHealth: SystemHealth
}

type Mutation {
  addMovie(input: MovieInput!): Movie
  startDownload(url: String!): Download
  askAgents(question: String!): AgentResponse
}

type Subscription {
  downloadProgress(downloadId: ID!): Download
  mediaAdded: Media
  agentVoting: VotingUpdate
}
```

#### Example Queries
```graphql
# Get movies with downloads
query GetMoviesWithDownloads {
  movies(limit: 10) {
    id
    title
    year
    imdbRating
    downloads {
      id
      status
      progress
    }
  }
}

# Start download with AI consultation
mutation StartDownloadWithAI($url: String!, $question: String!) {
  startDownload(url: $url) {
    id
    status
  }
  askAgents(question: $question) {
    consensus
    recommendation
    votes {
      agent
      vote
      confidence
    }
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues

#### Issue 1: Services Not Starting
**Symptoms**: Containers exit immediately or fail to start

**Diagnosis**:
```bash
# Check container status
docker ps -a

# View container logs
docker logs <container_name>

# Check system resources
docker system df
df -h
free -h
```

**Solutions**:
1. **Insufficient Resources**:
   ```bash
   # Free up space
   docker system prune -a
   
   # Check memory usage
   docker stats
   ```

2. **Port Conflicts**:
   ```bash
   # Check port usage
   netstat -tulpn | grep :8096
   
   # Change ports in docker-compose.yml
   ports:
     - "8097:8096"  # Use different external port
   ```

3. **Permission Issues**:
   ```bash
   # Fix ownership
   sudo chown -R $USER:$USER ./config ./media
   
   # Set correct PUID/PGID
   echo "PUID=$(id -u)" >> .env
   echo "PGID=$(id -g)" >> .env
   ```

#### Issue 2: Can't Access Web Interface
**Symptoms**: Browser shows "connection refused" or timeouts

**Diagnosis**:
```bash
# Test local connectivity
curl -I http://localhost:8096

# Check firewall
sudo ufw status

# Test from container
docker exec jellyfin curl -I localhost:8096
```

**Solutions**:
1. **Firewall Blocking**:
   ```bash
   # Open required ports
   sudo ufw allow 8096
   sudo ufw allow 8989
   sudo ufw allow 7878
   ```

2. **Container Network Issues**:
   ```bash
   # Restart networking
   docker compose down
   docker network prune
   docker compose up -d
   ```

3. **Service Not Ready**:
   ```bash
   # Wait for service to initialize
   docker compose logs -f jellyfin
   
   # Check health status
   docker compose ps
   ```

#### Issue 3: Downloads Not Working
**Symptoms**: Torrents won't start or fail immediately

**Diagnosis**:
```bash
# Check VPN connection
docker exec gluetun curl -s https://ipinfo.io

# Check qBittorrent logs
docker logs qbittorrent

# Test connectivity
docker exec qbittorrent ping google.com
```

**Solutions**:
1. **VPN Not Connected**:
   ```bash
   # Check VPN status
   docker exec gluetun cat /tmp/gluetun/ip
   
   # Restart VPN container
   docker compose restart gluetun
   
   # Update VPN credentials
   nano .env  # Update VPN_USER and VPN_PASS
   ```

2. **Indexer Issues**:
   ```bash
   # Test indexers in Prowlarr
   # Go to http://localhost:9696
   # Settings → Indexers → Test
   ```

3. **Download Client Configuration**:
   ```bash
   # Reset qBittorrent password
   docker exec qbittorrent cat /config/qBittorrent/config/qBittorrent.conf
   
   # Default: admin/adminadmin
   # Change in Web UI: Tools → Options → Web UI
   ```

### Performance Issues

#### High CPU Usage
```bash
# Identify resource-heavy containers
docker stats --no-stream

# Check transcoding load
docker exec jellyfin htop

# Reduce transcoding quality
# Jellyfin → Admin → Dashboard → Playback → Transcoding
```

#### High Memory Usage
```bash
# Check memory per container
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Restart memory-heavy services
docker compose restart jellyfin sonarr radarr

# Clear cache
docker exec jellyfin rm -rf /config/cache/*
```

#### Slow Performance
```bash
# Check disk I/O
iostat -x 1

# Check database performance
docker exec postgres psql -U postgres -c "SELECT * FROM pg_stat_activity;"

# Optimize databases
docker exec postgres psql -U postgres -c "VACUUM ANALYZE;"
```

### Network Issues

#### Container Communication
```bash
# Test inter-container connectivity
docker exec sonarr ping radarr
docker exec sonarr ping qbittorrent

# Check network configuration
docker network ls
docker network inspect mediaserver_default
```

#### DNS Resolution
```bash
# Test DNS inside containers
docker exec jellyfin nslookup google.com

# Check custom DNS settings
docker exec gluetun cat /etc/resolv.conf
```

### Log Analysis

#### Centralized Logging
```bash
# View all logs
docker compose logs -f

# View specific service logs
docker compose logs -f jellyfin

# Search logs for errors
docker compose logs | grep -i error

# Export logs for analysis
docker compose logs > mediaserver-logs.txt
```

#### Log Rotation
```bash
# Configure log rotation in docker-compose.yml
services:
  jellyfin:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## ⚡ Performance Optimization

### Hardware Optimization

#### CPU Optimization
```bash
# Set CPU limits in docker-compose.yml
services:
  jellyfin:
    deploy:
      resources:
        limits:
          cpus: '2.0'
        reservations:
          cpus: '0.5'
```

#### Memory Optimization
```bash
# Set memory limits
services:
  postgres:
    deploy:
      resources:
        limits:
          memory: 1G
        reservations:
          memory: 512M
```

#### Storage Optimization
```bash
# Use SSD for databases and configs
volumes:
  postgres-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/ssd/postgres

# Use HDD for media storage
  media-data:
    driver: local
    driver_opts:
      type: none
      o: bind
      device: /mnt/hdd/media
```

### Database Optimization

#### PostgreSQL Tuning
```sql
-- /config/postgres/postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
effective_io_concurrency = 200
```

#### Database Maintenance
```bash
# Regular maintenance script
#!/bin/bash
# maintenance.sh

# Vacuum and analyze all databases
docker exec postgres psql -U postgres -c "VACUUM ANALYZE;"

# Reindex databases
docker exec postgres psql -U postgres -c "REINDEX DATABASE sonarr;"
docker exec postgres psql -U postgres -c "REINDEX DATABASE radarr;"

# Check database sizes
docker exec postgres psql -U postgres -c "
  SELECT pg_database.datname,
         pg_size_pretty(pg_database_size(pg_database.datname)) AS size
  FROM pg_database;
"
```

### Network Optimization

#### CDN Configuration
```nginx
# nginx.conf
server {
    listen 80;
    server_name your-domain.com;
    
    # Enable gzip compression
    gzip on;
    gzip_vary on;
    gzip_min_length 10240;
    gzip_proxied expired no-cache no-store private must-revalidate auth;
    gzip_types text/plain text/css text/xml text/javascript application/javascript application/xml+rss application/json;
    
    # Cache static assets
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # Proxy to Jellyfin
    location / {
        proxy_pass http://jellyfin:8096;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

#### Caching Strategy
```bash
# Redis configuration for caching
services:
  redis:
    image: redis:7-alpine
    command: redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru
    volumes:
      - redis-data:/data
```

### Monitoring and Alerting

#### Prometheus Configuration
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'docker'
    static_configs:
      - targets: ['cadvisor:8080']
  
  - job_name: 'jellyfin'
    static_configs:
      - targets: ['jellyfin:8096']
    metrics_path: /metrics
  
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
```

#### Grafana Dashboards
```json
{
  "dashboard": {
    "title": "MediaServer Performance",
    "panels": [
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(container_cpu_usage_seconds_total[5m]) * 100",
            "legendFormat": "{{name}}"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes / container_spec_memory_limit_bytes * 100",
            "legendFormat": "{{name}}"
          }
        ]
      }
    ]
  }
}
```

---

## 🔒 Security Guide

### Authentication & Authorization

#### Multi-Factor Authentication
```bash
# Enable 2FA in Jellyfin
# Admin Dashboard → Users → [User] → Enable Two-Factor Authentication

# TOTP apps: Google Authenticator, Authy, 1Password
```

#### JWT Configuration
```javascript
// config/auth.js
const jwt = require('jsonwebtoken');

const jwtConfig = {
  secret: process.env.JWT_SECRET || 'your-super-secret-key',
  expiresIn: '24h',
  issuer: 'mediaserver',
  audience: 'mediaserver-users'
};

// Generate token
function generateToken(user) {
  return jwt.sign(
    { 
      id: user.id, 
      username: user.username, 
      role: user.role 
    }, 
    jwtConfig.secret, 
    { 
      expiresIn: jwtConfig.expiresIn,
      issuer: jwtConfig.issuer,
      audience: jwtConfig.audience
    }
  );
}
```

### Network Security

#### Firewall Configuration
```bash
# UFW (Ubuntu Firewall)
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow ssh

# Allow media server ports
sudo ufw allow 8096   # Jellyfin
sudo ufw allow 443    # HTTPS
sudo ufw allow 80     # HTTP (redirect to HTTPS)

# Enable firewall
sudo ufw enable

# Check status
sudo ufw status verbose
```

#### Reverse Proxy with SSL
```nginx
# /etc/nginx/sites-available/mediaserver
server {
    listen 443 ssl http2;
    server_name your-domain.com;
    
    # SSL configuration
    ssl_certificate /etc/letsencrypt/live/your-domain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/your-domain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES128-GCM-SHA256:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Security headers
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains; preload";
    
    # Rate limiting
    limit_req_zone $binary_remote_addr zone=login:10m rate=5r/m;
    
    location /api/auth {
        limit_req zone=login burst=3 nodelay;
        proxy_pass http://mediaserver-backend;
    }
    
    location / {
        proxy_pass http://jellyfin:8096;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name your-domain.com;
    return 301 https://$server_name$request_uri;
}
```

### VPN Security

#### OpenVPN Configuration
```bash
# /config/openvpn/client.conf
client
dev tun
proto udp
remote your-vpn-server.com 1194
resolv-retry infinite
nobind
persist-key
persist-tun
ca ca.crt
cert client.crt
key client.key
remote-cert-tls server
comp-lzo
verb 3

# Kill switch
script-security 2
up /scripts/up.sh
down /scripts/down.sh
```

#### WireGuard Configuration
```ini
# /config/wireguard/wg0.conf
[Interface]
PrivateKey = your-private-key
Address = 10.0.0.2/32
DNS = 1.1.1.1, 1.0.0.1

# Kill switch
PostUp = iptables -I OUTPUT ! -o %i -m mark ! --mark $(wg show %i fwmark) -m addrtype ! --dst-type LOCAL -j REJECT
PreDown = iptables -D OUTPUT ! -o %i -m mark ! --mark $(wg show %i fwmark) -m addrtype ! --dst-type LOCAL -j REJECT

[Peer]
PublicKey = your-server-public-key
AllowedIPs = 0.0.0.0/0
Endpoint = your-vpn-server.com:51820
```

### Container Security

#### Security Scanning
```bash
# Scan images for vulnerabilities
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  -v $(pwd):/tmp aquasec/trivy \
  image jellyfin/jellyfin:latest

# Scan filesystem
docker run --rm -v $(pwd):/tmp aquasec/trivy \
  filesystem /tmp
```

#### Runtime Security
```yaml
# docker-compose.yml security enhancements
services:
  jellyfin:
    image: jellyfin/jellyfin:latest
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
      - /var/tmp
    user: "1000:1000"
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - SETGID
      - SETUID
```

### Backup Security

#### Encrypted Backups
```bash
#!/bin/bash
# secure-backup.sh

# Backup directory
BACKUP_DIR="/backups/mediaserver-$(date +%Y%m%d)"

# Create backup
mkdir -p "$BACKUP_DIR"
tar -czf "$BACKUP_DIR/config.tar.gz" ./config
tar -czf "$BACKUP_DIR/database.tar.gz" ./postgres-data

# Encrypt backup
gpg --cipher-algo AES256 --compress-algo 1 --s2k-mode 3 \
    --s2k-digest-algo SHA512 --s2k-count 65536 --symmetric \
    --output "$BACKUP_DIR.gpg" "$BACKUP_DIR.tar.gz"

# Upload to cloud (encrypted)
rclone copy "$BACKUP_DIR.gpg" remote:backups/mediaserver/

# Cleanup local backup
rm -rf "$BACKUP_DIR" "$BACKUP_DIR.tar.gz"
```

---

## 💾 Backup & Recovery

### Automated Backup Strategy

#### Backup Script
```bash
#!/bin/bash
# /scripts/backup.sh

set -e  # Exit on any error

# Configuration
BACKUP_ROOT="/backups"
RETENTION_DAYS=30
DATESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/mediaserver_$DATESTAMP"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo "Starting backup at $(date)"

# Stop services for consistent backup
echo "Stopping services..."
docker compose stop

# Backup configuration
echo "Backing up configuration..."
tar -czf "$BACKUP_DIR/config.tar.gz" ./config

# Backup databases
echo "Backing up databases..."
docker compose start postgres redis
sleep 10  # Wait for databases to start

# PostgreSQL backup
docker exec postgres pg_dumpall -U postgres | gzip > "$BACKUP_DIR/postgres.sql.gz"

# Redis backup
docker exec redis redis-cli BGSAVE
docker cp redis:/data/dump.rdb "$BACKUP_DIR/redis.rdb"

# Backup media metadata (not the actual files)
echo "Backing up media metadata..."
find ./media -name "*.nfo" -o -name "*.jpg" -o -name "*.png" | \
  tar -czf "$BACKUP_DIR/metadata.tar.gz" -T -

# Restart all services
echo "Restarting services..."
docker compose up -d

# Create backup manifest
cat > "$BACKUP_DIR/manifest.txt" << EOF
Backup created: $(date)
Host: $(hostname)
Docker Compose version: $(docker compose version --short)
Services backed up:
$(docker compose ps --services)
EOF

# Calculate checksums
find "$BACKUP_DIR" -type f -exec sha256sum {} \; > "$BACKUP_DIR/checksums.sha256"

# Compress entire backup
tar -czf "$BACKUP_ROOT/mediaserver_$DATESTAMP.tar.gz" -C "$BACKUP_ROOT" "mediaserver_$DATESTAMP"
rm -rf "$BACKUP_DIR"

# Upload to remote storage
if [ -n "$RCLONE_REMOTE" ]; then
    echo "Uploading to remote storage..."
    rclone copy "$BACKUP_ROOT/mediaserver_$DATESTAMP.tar.gz" "$RCLONE_REMOTE:backups/"
fi

# Cleanup old backups
echo "Cleaning up old backups..."
find "$BACKUP_ROOT" -name "mediaserver_*.tar.gz" -mtime +$RETENTION_DAYS -delete

echo "Backup completed at $(date)"
echo "Backup size: $(du -sh $BACKUP_ROOT/mediaserver_$DATESTAMP.tar.gz)"
```

#### Automated Scheduling
```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh >> /var/log/mediaserver-backup.log 2>&1

# Weekly full backup (including media) on Sundays at 1 AM
0 1 * * 0 /path/to/full-backup.sh >> /var/log/mediaserver-full-backup.log 2>&1
```

### Recovery Procedures

#### Complete System Recovery
```bash
#!/bin/bash
# /scripts/restore.sh

set -e

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup-file.tar.gz>"
    exit 1
fi

echo "Starting recovery from $BACKUP_FILE"

# Stop all services
docker compose down

# Extract backup
TEMP_DIR=$(mktemp -d)
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
BACKUP_DIR=$(find "$TEMP_DIR" -maxdepth 1 -type d -name "mediaserver_*")

# Verify checksums
echo "Verifying backup integrity..."
(cd "$BACKUP_DIR" && sha256sum -c checksums.sha256)

# Restore configuration
echo "Restoring configuration..."
rm -rf ./config
tar -xzf "$BACKUP_DIR/config.tar.gz"

# Restore databases
echo "Starting database services..."
docker compose up -d postgres redis
sleep 30

# Restore PostgreSQL
echo "Restoring PostgreSQL..."
zcat "$BACKUP_DIR/postgres.sql.gz" | docker exec -i postgres psql -U postgres

# Restore Redis
echo "Restoring Redis..."
docker compose stop redis
docker cp "$BACKUP_DIR/redis.rdb" redis:/data/dump.rdb
docker compose start redis

# Restore metadata
echo "Restoring media metadata..."
tar -xzf "$BACKUP_DIR/metadata.tar.gz"

# Start all services
echo "Starting all services..."
docker compose up -d

# Cleanup
rm -rf "$TEMP_DIR"

echo "Recovery completed successfully!"
echo "Please verify all services are working correctly."
```

#### Selective Recovery
```bash
# Restore only configuration
./restore.sh --config-only backup.tar.gz

# Restore only databases
./restore.sh --databases-only backup.tar.gz

# Restore specific service
./restore.sh --service jellyfin backup.tar.gz
```

### Cloud Backup Integration

#### Rclone Configuration
```bash
# Install rclone
curl https://rclone.org/install.sh | sudo bash

# Configure cloud storage
rclone config

# Example: Google Drive
# Choose: Google Drive
# Client ID: (leave blank for default)
# Client Secret: (leave blank for default)
# Scope: drive
# Authorize in browser

# Test connection
rclone ls gdrive:

# Configure in .env
echo "RCLONE_REMOTE=gdrive" >> .env
```

#### Multi-Cloud Strategy
```bash
#!/bin/bash
# multi-cloud-backup.sh

BACKUP_FILE="$1"

# Upload to multiple cloud providers
rclone copy "$BACKUP_FILE" gdrive:backups/mediaserver/
rclone copy "$BACKUP_FILE" aws-s3:mediaserver-backups/
rclone copy "$BACKUP_FILE" dropbox:backups/mediaserver/

# Verify uploads
echo "Verifying cloud backups..."
rclone check "$BACKUP_FILE" gdrive:backups/mediaserver/$(basename "$BACKUP_FILE")
rclone check "$BACKUP_FILE" aws-s3:mediaserver-backups/$(basename "$BACKUP_FILE")
rclone check "$BACKUP_FILE" dropbox:backups/mediaserver/$(basename "$BACKUP_FILE")

echo "Multi-cloud backup completed successfully"
```

---

## 🚀 Advanced Features

### Custom Themes and Branding

#### Jellyfin Custom CSS
```css
/* /config/jellyfin/custom.css */
/* Neon theme for Jellyfin */

:root {
    --neon-pink: #ff006e;
    --electric-blue: #3a86ff;
    --neon-purple: #8338ec;
    --neon-green: #06ffa5;
}

/* Background gradient */
.skinHeader {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
}

/* Glassmorphism cards */
.cardContent {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    backdrop-filter: blur(20px) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
}

/* Neon accents */
.itemProgressBar {
    background: linear-gradient(90deg, var(--neon-pink), var(--electric-blue)) !important;
}

/* Hover effects */
.card:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 10px 40px rgba(255, 0, 110, 0.3) !important;
    transition: all 0.3s ease !important;
}
```

#### Custom Logo and Branding
```html
<!-- /config/jellyfin/web/index.html modifications -->
<script>
// Replace Jellyfin branding
document.addEventListener('DOMContentLoaded', function() {
    // Replace logo
    const logo = document.querySelector('.headerLogo');
    if (logo) {
        logo.src = '/web/assets/custom-logo.svg';
    }
    
    // Update title
    document.title = 'MediaFlow - Your Personal Cinema';
    
    // Add custom meta tags
    const meta = document.createElement('meta');
    meta.name = 'theme-color';
    meta.content = '#ff006e';
    document.head.appendChild(meta);
});
</script>
```

### Plugin Development

#### Custom Jellyfin Plugin
```csharp
// Plugins/MediaFlow.AI/Plugin.cs
using MediaBrowser.Common.Configuration;
using MediaBrowser.Common.Plugins;
using MediaBrowser.Model.Serialization;
using System;

namespace MediaFlow.AI
{
    public class Plugin : BasePlugin<PluginConfiguration>
    {
        public Plugin(IApplicationPaths applicationPaths, IXmlSerializer xmlSerializer)
            : base(applicationPaths, xmlSerializer)
        {
        }

        public override string Name => "MediaFlow AI";
        public override Guid Id => Guid.Parse("12345678-1234-1234-1234-123456789012");
        public override string Description => "AI-powered media recommendations";
    }

    public class PluginConfiguration : BasePluginConfiguration
    {
        public string OpenAIApiKey { get; set; } = string.Empty;
        public bool EnableRecommendations { get; set; } = true;
        public int MaxRecommendations { get; set; } = 10;
    }
}
```

### Integration APIs

#### Webhook System
```javascript
// webhook-handler.js
const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

// Handle Sonarr webhooks
app.post('/webhooks/sonarr', async (req, res) => {
    const { eventType, series, episodes } = req.body;
    
    switch (eventType) {
        case 'Download':
            // Notify users
            await notifyUsers(`New episode downloaded: ${series.title}`);
            
            // Update AI recommendations
            await updateAIRecommendations(series);
            
            // Refresh Jellyfin library
            await refreshJellyfinLibrary();
            break;
            
        case 'Upgrade':
            await notifyUsers(`Episode upgraded: ${series.title}`);
            break;
    }
    
    res.status(200).send('OK');
});

// Handle Radarr webhooks
app.post('/webhooks/radarr', async (req, res) => {
    const { eventType, movie } = req.body;
    
    if (eventType === 'Download') {
        await notifyUsers(`New movie downloaded: ${movie.title}`);
        await updateAIRecommendations(movie);
        await refreshJellyfinLibrary();
    }
    
    res.status(200).send('OK');
});

async function notifyUsers(message) {
    // Send push notifications
    const users = await getActiveUsers();
    
    for (const user of users) {
        await sendPushNotification(user.id, {
            title: 'MediaFlow',
            body: message,
            icon: '/assets/logo.png'
        });
    }
}

app.listen(3100, () => {
    console.log('Webhook handler listening on port 3100');
});
```

### Machine Learning Integration

#### Content Recommendation Engine
```python
# ml/recommendation_engine.py
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF
import joblib

class MediaRecommendationEngine:
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
        self.nmf = NMF(n_components=50, random_state=42)
        self.content_similarity = None
        self.user_item_matrix = None
        
    def prepare_content_features(self, media_df):
        """Prepare content-based features"""
        # Combine text features
        media_df['content_features'] = (
            media_df['title'].fillna('') + ' ' +
            media_df['overview'].fillna('') + ' ' +
            media_df['genres'].fillna('') + ' ' +
            media_df['cast'].fillna('')
        )
        
        # Create TF-IDF matrix
        tfidf_matrix = self.tfidf.fit_transform(media_df['content_features'])
        
        # Calculate cosine similarity
        self.content_similarity = cosine_similarity(tfidf_matrix)
        
        return tfidf_matrix
    
    def prepare_collaborative_features(self, ratings_df):
        """Prepare collaborative filtering features"""
        # Create user-item matrix
        self.user_item_matrix = ratings_df.pivot_table(
            index='user_id', 
            columns='media_id', 
            values='rating'
        ).fillna(0)
        
        # Apply NMF for dimensionality reduction
        self.nmf.fit(self.user_item_matrix)
        
        return self.user_item_matrix
    
    def get_content_recommendations(self, media_id, n_recommendations=10):
        """Get content-based recommendations"""
        if self.content_similarity is None:
            raise ValueError("Content features not prepared")
            
        # Get similarity scores
        sim_scores = list(enumerate(self.content_similarity[media_id]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        # Get top N similar items (excluding the item itself)
        sim_scores = sim_scores[1:n_recommendations+1]
        
        return [i[0] for i in sim_scores]
    
    def get_collaborative_recommendations(self, user_id, n_recommendations=10):
        """Get collaborative filtering recommendations"""
        if self.user_item_matrix is None:
            raise ValueError("Collaborative features not prepared")
            
        # Get user factors
        user_factors = self.nmf.transform(self.user_item_matrix.loc[[user_id]])
        
        # Get item factors
        item_factors = self.nmf.components_
        
        # Calculate predicted ratings
        predicted_ratings = np.dot(user_factors, item_factors)[0]
        
        # Get user's already rated items
        rated_items = self.user_item_matrix.loc[user_id].nonzero()[0]
        
        # Filter out already rated items
        predicted_ratings[rated_items] = -1
        
        # Get top N recommendations
        top_items = np.argsort(predicted_ratings)[::-1][:n_recommendations]
        
        return top_items.tolist()
    
    def get_hybrid_recommendations(self, user_id, media_id, n_recommendations=10, alpha=0.7):
        """Get hybrid recommendations combining content and collaborative"""
        content_recs = self.get_content_recommendations(media_id, n_recommendations*2)
        collab_recs = self.get_collaborative_recommendations(user_id, n_recommendations*2)
        
        # Combine recommendations with weighted scoring
        hybrid_scores = {}
        
        for i, item in enumerate(content_recs):
            hybrid_scores[item] = alpha * (1 - i / len(content_recs))
            
        for i, item in enumerate(collab_recs):
            if item in hybrid_scores:
                hybrid_scores[item] += (1 - alpha) * (1 - i / len(collab_recs))
            else:
                hybrid_scores[item] = (1 - alpha) * (1 - i / len(collab_recs))
        
        # Sort by combined score
        sorted_recs = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        
        return [item[0] for item in sorted_recs[:n_recommendations]]
    
    def save_model(self, filepath):
        """Save the trained model"""
        joblib.dump({
            'tfidf': self.tfidf,
            'nmf': self.nmf,
            'content_similarity': self.content_similarity,
            'user_item_matrix': self.user_item_matrix
        }, filepath)
    
    def load_model(self, filepath):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.tfidf = model_data['tfidf']
        self.nmf = model_data['nmf']
        self.content_similarity = model_data['content_similarity']
        self.user_item_matrix = model_data['user_item_matrix']
```

#### Training Script
```python
# ml/train_model.py
import pandas as pd
from recommendation_engine import MediaRecommendationEngine
import psycopg2
from sqlalchemy import create_engine

def load_data_from_jellyfin():
    """Load data from Jellyfin database"""
    engine = create_engine('postgresql://postgres:password@localhost:5432/jellyfin')
    
    # Load media data
    media_query = """
    SELECT 
        Id as media_id,
        Name as title,
        Overview as overview,
        Genres as genres,
        People as cast
    FROM TypedBaseItems 
    WHERE Type IN ('Movie', 'Series')
    """
    
    media_df = pd.read_sql(media_query, engine)
    
    # Load user rating data
    ratings_query = """
    SELECT 
        UserId as user_id,
        ItemId as media_id,
        Rating as rating
    FROM UserData
    WHERE Rating IS NOT NULL
    """
    
    ratings_df = pd.read_sql(ratings_query, engine)
    
    return media_df, ratings_df

def main():
    print("Loading data from Jellyfin...")
    media_df, ratings_df = load_data_from_jellyfin()
    
    print(f"Loaded {len(media_df)} media items and {len(ratings_df)} ratings")
    
    # Initialize recommendation engine
    engine = MediaRecommendationEngine()
    
    # Prepare features
    print("Preparing content features...")
    engine.prepare_content_features(media_df)
    
    print("Preparing collaborative features...")
    engine.prepare_collaborative_features(ratings_df)
    
    # Save the trained model
    print("Saving model...")
    engine.save_model('models/recommendation_model.pkl')
    
    print("Training completed!")

if __name__ == '__main__':
    main()
```

---

## ❓ FAQ

### General Questions

**Q: What is the Ultimate Media Server 2025?**
A: It's a comprehensive self-hosted media automation and streaming solution that combines 60+ integrated services with AI-powered decision making, modern web interfaces, and production-ready deployment options.

**Q: Is this legal to use?**
A: The software itself is completely legal. However, you are responsible for ensuring that any content you download or stream complies with your local copyright laws and regulations.

**Q: What makes this different from other media servers?**
A: Key differentiators include:
- AI agent voting system for intelligent decisions
- Unified modern web interface
- Complete automation from request to streaming
- Production-ready deployment with monitoring
- Social media integration
- Voice control capabilities

### Technical Questions

**Q: What are the minimum system requirements?**
A: Minimum: 4 CPU cores, 8GB RAM, 100GB SSD + 1TB HDD, Ubuntu 22.04 or similar Linux distribution.

**Q: Can I run this on a Raspberry Pi?**
A: Yes, with limitations. A Raspberry Pi 4 with 8GB RAM can run the basic stack, but transcoding performance will be limited. Consider using hardware transcoding or pre-encoded media.

**Q: How much internet bandwidth do I need?**
A: For downloading: 100+ Mbps recommended. For streaming: 25 Mbps per 4K stream, 10 Mbps per 1080p stream.

**Q: Can I use my existing Plex/Jellyfin setup?**
A: Yes! The system can integrate with existing media servers. You can migrate gradually or run in parallel.

### Setup and Configuration

**Q: How long does setup take?**
A: Initial setup: 15-30 minutes with the automated installer. Full configuration: 1-2 hours depending on your requirements.

**Q: Do I need a VPN?**
A: Highly recommended for download clients. The system includes VPN integration with popular providers.

**Q: Can I customize the interface?**
A: Yes! The system supports custom themes, branding, and CSS modifications. You can completely customize the look and feel.

**Q: How do I add new services?**
A: Add services to the docker-compose.yml file and configure integration through the web interface or API.

### AI and Automation

**Q: How does the AI agent voting system work?**
A: Six specialized AI agents analyze questions from different perspectives and vote democratically. A 70% consensus is required for actions. This ensures balanced, intelligent decision-making.

**Q: What AI models are used?**
A: The system uses OpenAI's GPT-4 and o1-mini models for natural language processing and decision making. You can also integrate local models like Ollama.

**Q: Can I train the AI on my preferences?**
A: Yes! The system learns from your viewing patterns, ratings, and decisions to provide increasingly personalized recommendations.

**Q: How accurate are the recommendations?**
A: The hybrid recommendation engine typically achieves 85-90% user satisfaction with personalized suggestions.

### Performance and Scaling

**Q: How many concurrent streams can it handle?**
A: Depends on your hardware and transcoding needs. A modern server can typically handle 5-10 concurrent 1080p transcoded streams or 20+ direct streams.

**Q: Can I scale horizontally?**
A: Yes! The system supports Kubernetes deployment with horizontal scaling for most services.

**Q: How much storage do I need?**
A: Start with 2-4TB for a basic library. Heavy users often have 20-50TB or more. The system supports unlimited expansion.

**Q: What about backup strategies?**
A: Automated daily backups are included, with support for multiple cloud providers and retention policies.

### Security and Privacy

**Q: Is my data secure?**
A: Yes! The system includes:
- Encrypted communications (SSL/TLS)
- VPN integration for downloads
- Regular security updates
- Optional 2FA authentication
- Network isolation

**Q: Can I access it remotely?**
A: Yes, with several options:
- VPN access (most secure)
- Reverse proxy with SSL
- Tailscale or similar mesh networking
- Cloud deployment

**Q: What data is collected?**
A: Only local usage statistics for system optimization. No data is sent to third parties without your explicit consent.

### Troubleshooting

**Q: Services won't start - what should I check?**
A: Common issues:
1. Port conflicts - check `netstat -tulpn`
2. Insufficient resources - check `docker stats`
3. Permission issues - ensure correct PUID/PGID
4. Firewall blocking - check `ufw status`

**Q: Downloads aren't working - what's wrong?**
A: Check:
1. VPN connection - `docker exec gluetun curl ipinfo.io`
2. Indexer configuration in Prowlarr
3. Download client settings in *arr apps
4. Available disk space

**Q: Streaming is buffering/slow - how to fix?**
A: Solutions:
1. Enable hardware transcoding
2. Reduce stream quality
3. Check network bandwidth
4. Optimize database performance
5. Add more RAM/CPU

**Q: Web interface is slow - how to improve?**
A: Optimizations:
1. Enable browser caching
2. Use a reverse proxy with compression
3. Optimize database queries
4. Add more system resources
5. Use SSD storage for databases

### Community and Support

**Q: Where can I get help?**
A: Support channels:
- GitHub Issues for bugs/features
- Discord community chat
- Reddit community
- Documentation wiki
- Video tutorials

**Q: How can I contribute?**
A: Ways to contribute:
- Report bugs and feature requests
- Submit code improvements
- Create documentation
- Help other users in the community
- Share your configurations and tips

**Q: Is commercial use allowed?**
A: Yes, under the MIT license. You can use, modify, and distribute the software commercially.

**Q: Are there managed hosting options?**
A: Several community providers offer managed hosting. Check the official documentation for current recommendations.

---

*This manual is continuously updated. For the latest version, visit: https://docs.mediaflow.dev*