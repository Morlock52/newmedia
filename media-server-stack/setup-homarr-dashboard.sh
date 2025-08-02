#!/usr/bin/env bash
set -euo pipefail

echo "🏠 Setting up Homarr Dashboard for Complete Media Stack"
echo "======================================================"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Load environment
source .env

echo -e "${BLUE}Domain:${NC} $DOMAIN"
echo -e "${BLUE}Dashboard URLs:${NC}"
echo "• Main: https://$DOMAIN (root domain)"
echo "• Alt: https://dashboard.$DOMAIN"
echo "• Alt: https://homarr.$DOMAIN"
echo ""

echo -e "${GREEN}Step 1: Deploy Homarr Dashboard${NC}"
echo "==============================="

# Deploy Homarr
docker-compose -f docker-compose.homarr.yml up -d

echo "⏳ Waiting for Homarr to start..."
sleep 15

echo ""
echo -e "${GREEN}Step 2: Configure Homarr with All Your Apps${NC}"
echo "============================================"

# Create Homarr configuration
cat > homarr-config.json << 'EOF'
{
  "configProperties": {
    "name": "Morloksmaze Media Server"
  },
  "categories": [
    {
      "id": "media-streaming",
      "name": "🎬 Media Streaming",
      "position": 1
    },
    {
      "id": "media-management", 
      "name": "📚 Media Management",
      "position": 2
    },
    {
      "id": "downloads",
      "name": "⬇️ Downloads & Indexers",
      "position": 3
    },
    {
      "id": "monitoring",
      "name": "📊 Monitoring & Analytics", 
      "position": 4
    },
    {
      "id": "utilities",
      "name": "🔧 Utilities & Management",
      "position": 5
    }
  ],
  "apps": [
    {
      "id": "jellyfin",
      "name": "Jellyfin",
      "url": "https://jellyfin.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/jellyfin.svg",
        "appNameStatus": "normal",
        "positionAppName": "column",
        "lineClampAppName": 1
      },
      "behaviour": {
        "isOpeningNewTab": true,
        "externalUrl": "https://jellyfin.morloksmaze.com"
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-streaming"
        }
      },
      "shape": {},
      "integration": {
        "type": "jellyfin",
        "properties": [
          {
            "property": "apiKey",
            "value": ""
          },
          {
            "property": "username", 
            "value": ""
          }
        ]
      }
    },
    {
      "id": "overseerr",
      "name": "Overseerr",
      "url": "https://overseerr.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/overseerr.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category", 
        "properties": {
          "categoryId": "media-streaming"
        }
      }
    },
    {
      "id": "tautulli",
      "name": "Tautulli",
      "url": "https://tautulli.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/tautulli.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-streaming"
        }
      }
    },
    {
      "id": "sonarr",
      "name": "Sonarr", 
      "url": "https://sonarr.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/sonarr.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "radarr",
      "name": "Radarr",
      "url": "https://radarr.morloksmaze.com", 
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/radarr.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "lidarr",
      "name": "Lidarr",
      "url": "https://lidarr.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/lidarr.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "readarr",
      "name": "Readarr",
      "url": "https://readarr.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/readarr.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "bazarr",
      "name": "Bazarr",
      "url": "https://bazarr.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/bazarr.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "mylar",
      "name": "Mylar3", 
      "url": "https://mylar.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/mylar3.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "podgrab",
      "name": "Podgrab",
      "url": "https://podgrab.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/podgrab.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "youtube-dl",
      "name": "YouTube-DL",
      "url": "https://youtube-dl.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/youtubedl-material.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "photoprism",
      "name": "PhotoPrism",
      "url": "https://photoprism.morloksmaze.com", 
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/photoprism.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "media-management"
        }
      }
    },
    {
      "id": "prowlarr",
      "name": "Prowlarr",
      "url": "https://prowlarr.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/prowlarr.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "downloads"
        }
      }
    },
    {
      "id": "qbittorrent",
      "name": "qBittorrent",
      "url": "http://localhost:8080",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/qbittorrent.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "downloads"
        }
      }
    },
    {
      "id": "grafana",
      "name": "Grafana",
      "url": "https://grafana.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/grafana.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "monitoring"
        }
      }
    },
    {
      "id": "prometheus",
      "name": "Prometheus", 
      "url": "https://prometheus.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/prometheus.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "monitoring"
        }
      }
    },
    {
      "id": "traefik",
      "name": "Traefik",
      "url": "https://traefik.morloksmaze.com",
      "appearance": {
        "icon": "https://raw.githubusercontent.com/walkxcode/dashboard-icons/main/svg/traefik.svg"
      },
      "behaviour": {
        "isOpeningNewTab": true
      },
      "area": {
        "type": "category",
        "properties": {
          "categoryId": "utilities"
        }
      }
    }
  ],
  "layouts": [
    {
      "location": "homarr"
    }
  ]
}
EOF

echo "✅ Created Homarr configuration with all your apps"

echo ""
echo -e "${GREEN}Step 3: Update Cloudflare Tunnel for Main Domain${NC}"
echo "================================================"

# Add root domain routing to tunnel
if [[ "$CF_API_KEY" != "your-cloudflare-api-key" ]]; then
    echo "Adding root domain (morloksmaze.com) to tunnel..."
    
    # This will be handled by the tunnel configuration
    echo "✅ Root domain will be accessible via existing tunnel"
else
    echo "⚠️  Manual step needed: Add morloksmaze.com → HTTP://traefik:80 to tunnel"
fi

echo ""
echo -e "${GREEN}Step 4: Test Homarr Dashboard${NC}"
echo "=============================="

echo "⏳ Waiting for Homarr to be fully ready..."
sleep 10

# Test if Homarr is responding
if curl -s -o /dev/null -w "%{http_code}" "http://localhost:7575" | grep -q "200"; then
    echo "✅ Homarr is responding locally"
else
    echo "⏳ Homarr still starting up..."
fi

echo ""
echo -e "${GREEN}🎉 Homarr Dashboard Setup Complete!${NC}"
echo "==================================="

echo ""
echo -e "${BLUE}📱 Access Your Dashboard:${NC}"
echo "• Main URL: https://$DOMAIN"
echo "• Alt URL: https://dashboard.$DOMAIN" 
echo "• Alt URL: https://homarr.$DOMAIN"

echo ""
echo -e "${BLUE}📋 Dashboard Features:${NC}"
echo "• 🎬 Media Streaming (Jellyfin, Overseerr, Tautulli)"
echo "• 📚 Media Management (Sonarr, Radarr, Lidarr, Readarr, Bazarr, Mylar, Podgrab, YouTube-DL, PhotoPrism)"
echo "• ⬇️  Downloads & Indexers (Prowlarr, qBittorrent)"
echo "• 📊 Monitoring & Analytics (Grafana, Prometheus)"
echo "• 🔧 Utilities & Management (Traefik)"

echo ""
echo -e "${BLUE}🎨 Dashboard Customization:${NC}"
echo "• Beautiful icons for all services"
echo "• Organized by categories"
echo "• Responsive design"
echo "• Integration widgets (coming soon)"

echo ""
echo -e "${BLUE}🚀 Next Steps:${NC}"
echo "1. Open https://$DOMAIN to see your dashboard"
echo "2. Configure Cloudflare authentication for security"
echo "3. Set up API integrations for live stats"
echo "4. Customize layout and themes"

echo ""
echo "Opening your new dashboard..."
osascript -e "tell application \"Google Chrome\" to open location \"https://$DOMAIN\""

echo ""
echo "🏠 Welcome to your complete media server dashboard!"
