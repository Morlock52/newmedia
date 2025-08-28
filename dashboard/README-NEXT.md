# Ultimate Media Dashboard 2025

A modern, responsive media server dashboard built with **Next.js 15**, **Shadcn UI**, **Tailwind CSS v4**, and **real-time WebSocket monitoring**. Designed to manage and monitor 30+ media services including Plex, Jellyfin, Sonarr, Radarr, and more.

## ✨ Features

### 🎨 Modern UI/UX
- **Next.js 15 App Router** with React 19 support
- **Shadcn UI components** with Tailwind CSS v4
- **Responsive design** optimized for mobile and desktop
- **Dark/Light theme** support with system preference detection
- **Smooth animations** with Framer Motion
- **Glassmorphism effects** and modern card designs

### 📊 Service Management
- **30+ supported services** (Media servers, *ARR stack, downloaders, monitoring)
- **Real-time status monitoring** with WebSocket connections
- **Service categorization** and filtering
- **Quick actions** (start, stop, restart, configure)
- **Performance metrics** with CPU, memory, and disk usage
- **Health checks** and uptime tracking

### 🔧 Dashboard Features
- **Multiple view modes**: Grid, List, Categorized, Metrics
- **Advanced search and filtering**
- **System resource monitoring**
- **Real-time notifications**
- **Service dependency tracking**
- **Mobile-optimized navigation**

### 🚀 Technical Features
- **TypeScript** for type safety
- **Zustand** for state management
- **React Query** for server state
- **WebSocket** real-time updates
- **Progressive Web App** (PWA) support
- **Performance optimized** with code splitting

## 🛠️ Installation & Setup

### Prerequisites
- **Node.js 18+** 
- **npm or yarn or pnpm**
- **Docker** (for running media services)

### 1. Clone and Install

```bash
cd /path/to/newmedia/dashboard
npm install
```

### 2. Environment Configuration

```bash
cp .env.local.example .env.local
```

Edit `.env.local` with your service URLs:

```env
# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:3000
NEXT_PUBLIC_WS_URL=ws://localhost:3010/ws

# Service URLs
NEXT_PUBLIC_JELLYFIN_URL=http://localhost:8096
NEXT_PUBLIC_PLEX_URL=http://localhost:32400
NEXT_PUBLIC_SONARR_URL=http://localhost:8989
NEXT_PUBLIC_RADARR_URL=http://localhost:7878
# ... more services
```

### 3. Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) to view the dashboard.

### 4. Production Build

```bash
npm run build
npm start
```

## 🐳 Docker Integration

The dashboard is designed to work with the docker-compose.yml in the parent directory:

```bash
# From the newmedia directory
docker-compose up -d

# Dashboard will monitor all services defined in docker-compose.yml
```

## 📁 Project Structure

```
dashboard/
├── src/
│   ├── app/                 # Next.js App Router
│   │   ├── api/            # API routes
│   │   ├── globals.css     # Global styles
│   │   ├── layout.tsx      # Root layout
│   │   └── page.tsx        # Homepage
│   ├── components/         # React components
│   │   ├── ui/            # Shadcn UI components
│   │   ├── ServiceCard.tsx # Service cards
│   │   ├── ServiceGrid.tsx # Service grid
│   │   └── SystemMetrics.tsx # Metrics dashboard
│   ├── hooks/             # Custom hooks
│   │   └── useWebSocket.ts # WebSocket management
│   ├── lib/               # Utilities
│   │   ├── utils.ts       # Helper functions
│   │   └── services-config.ts # Service definitions
│   ├── store/             # Zustand stores
│   │   └── services-store.ts # Service state
│   └── types/             # TypeScript types
│       └── services.ts    # Type definitions
├── package.json
├── tailwind.config.js     # Tailwind CSS v4 config
├── tsconfig.json          # TypeScript config
└── next.config.js         # Next.js config
```

## 🎯 Supported Services

### Media Servers
- **Jellyfin** - Free media server
- **Plex** - Premium media server  
- **Emby** - Alternative media server

### Media Management (*ARR Stack)
- **Sonarr** - TV show management
- **Radarr** - Movie management
- **Lidarr** - Music management
- **Readarr** - Book management
- **Bazarr** - Subtitle management
- **Prowlarr** - Indexer management

### Download Clients
- **qBittorrent** - Torrent client
- **Transmission** - Alternative torrent client
- **SABnzbd** - Usenet downloader
- **NZBGet** - Alternative usenet client

### Request Managers
- **Jellyseerr** - Jellyfin requests
- **Overseerr** - Plex requests
- **Ombi** - Media request management

### Monitoring & Management
- **Prometheus** - Metrics collection
- **Grafana** - Metrics visualization
- **Uptime Kuma** - Service monitoring
- **Portainer** - Docker management
- **Nginx Proxy Manager** - Reverse proxy

### Content Libraries
- **Calibre-Web** - E-book library
- **AudioBookshelf** - Audiobook server
- **Navidrome** - Music server
- **PhotoPrism** - Photo management
- **Immich** - Photo backup
- **Paperless-ngx** - Document management

### Utilities
- **Nextcloud** - Cloud storage
- **Vaultwarden** - Password manager
- **Pi-hole** - DNS ad blocker
- **And 10+ more services...**

## 🚀 What's New in 2.0

- **Next.js 15** with App Router
- **React 19** with new features
- **Tailwind CSS v4** with modern features
- **Enhanced animations** and transitions
- **Improved mobile experience**
- **Better TypeScript** integration
- **30+ service support**
- **Real-time monitoring**
- **Advanced filtering** and search

---

**Ultimate Media Dashboard 2025** - The future of media server management! 🎬✨