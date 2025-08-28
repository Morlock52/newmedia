# 🚀 Minimal Single Container Media Server

## What You Actually Need

For a single Docker container deployment, you only need these files:

### Essential Files:
```
.
├── Dockerfile.multi-service      # Builds the single container
├── .env                         # Your configuration
├── modern-landing.html          # Modern UI
├── dashboard-enhanced.html      # Dashboard
├── mobile-ui.css               # Styles
└── social-share.js             # Social features
```

### That's it! Everything else is bloat.

## 🧹 Clean Up Commands

```bash
# Remove all the bloat
./cleanup-bloat.sh

# Or manually remove the worst offenders:
rm -f docker-compose*.yml        # 20+ compose files you don't need
rm -f *dashboard*.html          # 15+ duplicate dashboards  
rm -f *DASHBOARD*.html
rm -rf TEST_REPORTS TEST_RESULTS
rm -rf holographic-* quantum-* blockchain-* web3-*
```

## 🎯 Single Container Deployment

```bash
# 1. Build the container (all services in one)
docker build -t mediaserver -f Dockerfile.multi-service .

# 2. Run it
docker run -d \
  --name mediaserver \
  --restart unless-stopped \
  -p 80:80 \
  -p 8096:8096 \
  -p 8989:8989 \
  -p 7878:7878 \
  -p 9696:9696 \
  -p 8080:8080 \
  -v $(pwd)/config:/config \
  -v $(pwd)/media:/media \
  -e PUID=$(id -u) \
  -e PGID=$(id -g) \
  mediaserver
```

## 📁 Volume Structure

Create these directories:
```bash
mkdir -p config media/movies media/tv downloads
```

## 🌐 Access Services

- **Caddy Dashboard**: http://localhost
- **Jellyfin**: http://localhost:8096
- **Sonarr**: http://localhost:8989
- **Radarr**: http://localhost:7878
- **Prowlarr**: http://localhost:9696
- **qBittorrent**: http://localhost:8080

## ⚡ Quick .env File

```bash
cat > .env << 'EOF'
# Basic settings
PUID=1000
PGID=1000
TZ=America/New_York

# Paths
CONFIG_PATH=./config
MEDIA_PATH=./media
DOWNLOADS_PATH=./downloads
EOF
```

## 🎯 The Actual Bloat

You have:
- **20 docker-compose files** (you need 0 for single container)
- **23 HTML files** (you need 2)
- **50+ markdown docs** (you need 1)
- **Multiple test suites** (not needed)
- **Experimental features** (quantum, blockchain, AR/VR - not needed)

Run `./cleanup-bloat.sh` to remove ~200MB of unnecessary files!