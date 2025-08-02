#!/bin/bash

# Automated Homarr Configuration Script
echo "🚀 Automated Homarr Dashboard Configuration"
echo "==========================================="

HOMARR_URL="http://localhost:8080"
HOST_HEADER="Host: home.morloksmaze.com"

# Test if Homarr is accessible
echo "🔍 Testing Homarr accessibility..."
if ! curl -s -H "$HOST_HEADER" "$HOMARR_URL" > /dev/null 2>&1; then
    echo "❌ Homarr is not accessible. Please ensure services are running."
    exit 1
fi

echo "✅ Homarr is accessible!"

# Copy pre-made configuration
echo "📋 Applying dashboard configuration..."

# Use Docker exec to copy configuration directly to container
docker exec homarr sh -c '
mkdir -p /appdata/configs
cat > /appdata/configs/default.json << "EOF"
{
  "schemaVersion": 2,
  "configProperties": {
    "name": "Media Server Dashboard"
  },
  "categories": [
    {
      "id": "media-services",
      "name": "Media Services",
      "position": 0
    },
    {
      "id": "management",
      "name": "Management Tools",
      "position": 1
    },
    {
      "id": "system",
      "name": "System",
      "position": 2
    }
  ],
  "wrappers": [
    {
      "id": "default",
      "position": 0
    }
  ],
  "apps": [
    {
      "id": "jellyfin-app",
      "name": "Jellyfin",
      "url": "https://jellyfin.morloksmaze.com",
      "behaviour": {
        "onClickUrl": "https://jellyfin.morloksmaze.com",
        "externalUrl": "https://jellyfin.morloksmaze.com",
        "isOpeningNewTab": true
      },
      "network": {
        "enabledStatusChecker": true,
        "statusCodes": ["200", "301", "302", "307", "401"]
      },
      "appearance": {
        "iconUrl": "https://jellyfin.org/images/logo.svg",
        "appNameStatus": "normal",
        "positionAppName": "column",
        "lineClampAppName": 1
      },
      "area": {
        "type": "wrapper",
        "properties": {
          "gridstack": {
            "w": 1,
            "h": 1,
            "x": 0,
            "y": 0
          }
        }
      },
      "shape": {
        "lg": {
          "location": {"x": 0, "y": 0},
          "size": {"width": 1, "height": 1}
        }
      }
    },
    {
      "id": "overseerr-app",
      "name": "Overseerr",
      "url": "https://overseerr.morloksmaze.com",
      "behaviour": {
        "onClickUrl": "https://overseerr.morloksmaze.com",
        "externalUrl": "https://overseerr.morloksmaze.com",
        "isOpeningNewTab": true
      },
      "network": {
        "enabledStatusChecker": true,
        "statusCodes": ["200", "301", "302", "307", "401"]
      },
      "appearance": {
        "iconUrl": "https://docs.overseerr.dev/os_logo_filled.svg",
        "appNameStatus": "normal",
        "positionAppName": "column",
        "lineClampAppName": 1
      },
      "area": {
        "type": "wrapper",
        "properties": {
          "gridstack": {
            "w": 1,
            "h": 1,
            "x": 1,
            "y": 0
          }
        }
      },
      "shape": {
        "lg": {
          "location": {"x": 1, "y": 0},
          "size": {"width": 1, "height": 1}
        }
      }
    },
    {
      "id": "sonarr-app",
      "name": "Sonarr",
      "url": "https://sonarr.morloksmaze.com",
      "behaviour": {
        "onClickUrl": "https://sonarr.morloksmaze.com",
        "externalUrl": "https://sonarr.morloksmaze.com",
        "isOpeningNewTab": true
      },
      "network": {
        "enabledStatusChecker": true,
        "statusCodes": ["200", "301", "302", "307", "401"]
      },
      "appearance": {
        "iconUrl": "https://raw.githubusercontent.com/Sonarr/Sonarr/develop/Logo/256.png",
        "appNameStatus": "normal",
        "positionAppName": "column",
        "lineClampAppName": 1
      },
      "area": {
        "type": "wrapper",
        "properties": {
          "gridstack": {
            "w": 1,
            "h": 1,
            "x": 0,
            "y": 1
          }
        }
      },
      "shape": {
        "lg": {
          "location": {"x": 0, "y": 1},
          "size": {"width": 1, "height": 1}
        }
      }
    },
    {
      "id": "radarr-app",
      "name": "Radarr",
      "url": "https://radarr.morloksmaze.com",
      "behaviour": {
        "onClickUrl": "https://radarr.morloksmaze.com",
        "externalUrl": "https://radarr.morloksmaze.com",
        "isOpeningNewTab": true
      },
      "network": {
        "enabledStatusChecker": true,
        "statusCodes": ["200", "301", "302", "307", "401"]
      },
      "appearance": {
        "iconUrl": "https://raw.githubusercontent.com/Radarr/Radarr/develop/Logo/256.png",
        "appNameStatus": "normal",
        "positionAppName": "column",
        "lineClampAppName": 1
      },
      "area": {
        "type": "wrapper",
        "properties": {
          "gridstack": {
            "w": 1,
            "h": 1,
            "x": 1,
            "y": 1
          }
        }
      },
      "shape": {
        "lg": {
          "location": {"x": 1, "y": 1},
          "size": {"width": 1, "height": 1}
        }
      }
    }
  ],
  "widgets": [],
  "layouts": []
}
EOF
'

# Restart Homarr to apply configuration
echo "🔄 Restarting Homarr to apply configuration..."
docker-compose restart homarr

# Wait for restart
sleep 5

echo ""
echo "✅ Configuration Applied Successfully!"
echo ""
echo "🌐 Dashboard Access:"
echo "   External: https://home.morloksmaze.com"
echo "   Local: http://localhost:8080 (with Host: home.morloksmaze.com)"
echo ""
echo "🔑 Login Credentials:"
echo "   Username: morlock"
echo "   Password: changeme123"
echo ""
echo "📊 Your dashboard now includes:"
echo "   • Jellyfin (Media Server)"
echo "   • Overseerr (Request Management)"
echo "   • Sonarr (TV Shows)"
echo "   • Radarr (Movies)"
echo ""
echo "🎯 Next Steps:"
echo "   1. Open https://home.morloksmaze.com"
echo "   2. Login with Authelia"
echo "   3. Your dashboard should be automatically configured!"
echo ""
echo "✨ All status indicators should show green when services are healthy"