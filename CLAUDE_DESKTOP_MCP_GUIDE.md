# 🚀 Claude Desktop MCP Integration Guide

## Complete Setup for Media Server Control via Claude Desktop

### ✨ What This Enables

Once configured, you can control all your media services directly from Claude Desktop:
- **Natural Language Control**: "Download the latest episode of Show X"
- **Unified Search**: "Find all 4K movies in my library"
- **Smart Automation**: "Set up monitoring for new releases this week"
- **AI-Powered Insights**: "What are the most watched shows this month?"

---

## 📋 Prerequisites

1. **Claude Desktop** installed (https://claude.ai/download)
2. **Media services running** (via Docker Compose)
3. **Node.js 18+** installed
4. **API keys** from your media services

---

## 🛠️ Quick Setup (Automated)

Run the automated setup script:

```bash
# Make sure you're in the project directory
cd /Users/morlock/fun/newmedia

# Run the setup script
./setup-claude-desktop-mcp.sh
```

This will:
- ✅ Extract API keys from your services
- ✅ Configure Claude Desktop
- ✅ Install dependencies
- ✅ Create startup scripts

---

## 🔧 Manual Setup

### Step 1: Get Your API Keys

#### Jellyfin
1. Open Jellyfin: http://localhost:8096
2. Go to Dashboard → API Keys
3. Create new API key for "Claude Desktop"

#### Sonarr
```bash
# Extract from config file
grep -oP '(?<=<ApiKey>)[^<]+' sonarr-config/config.xml
```
Or visit: http://localhost:8989/settings/general

#### Radarr
```bash
# Extract from config file
grep -oP '(?<=<ApiKey>)[^<]+' radarr-config/config.xml
```
Or visit: http://localhost:7878/settings/general

#### Prowlarr
```bash
# Extract from config file
grep -oP '(?<=<ApiKey>)[^<]+' prowlarr-config/config.xml
```
Or visit: http://localhost:9696/settings/general

### Step 2: Configure Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "media-server-suite": {
      "command": "node",
      "args": [
        "/Users/morlock/fun/newmedia/mcp-architecture/src/index.js"
      ],
      "env": {
        "JELLYFIN_URL": "http://localhost:8096",
        "JELLYFIN_API_KEY": "YOUR_JELLYFIN_KEY",
        "SONARR_URL": "http://localhost:8989",
        "SONARR_API_KEY": "YOUR_SONARR_KEY",
        "RADARR_URL": "http://localhost:7878",
        "RADARR_API_KEY": "YOUR_RADARR_KEY",
        "PROWLARR_URL": "http://localhost:9696",
        "PROWLARR_API_KEY": "YOUR_PROWLARR_KEY",
        "QBITTORRENT_URL": "http://localhost:8080",
        "QBITTORRENT_USERNAME": "admin",
        "QBITTORRENT_PASSWORD": "adminadmin"
      }
    }
  }
}
```

### Step 3: Install Dependencies

```bash
cd mcp-architecture
npm install
```

### Step 4: Test MCP Server

```bash
# Test the MCP server
cd mcp-architecture
npm start

# Should see:
# ✅ jellyfin MCP server started on port 3001
# ✅ sonarr MCP server started on port 3002
# ✅ radarr MCP server started on port 3003
# ✅ prowlarr MCP server started on port 3004
# ✅ qbittorrent MCP server started on port 3005
```

### Step 5: Restart Claude Desktop

1. Quit Claude Desktop completely
2. Reopen Claude Desktop
3. Look for "MCP" indicator in the interface

---

## 💬 Example Commands

Once configured, try these in Claude Desktop:

### Media Search & Discovery
- "Search for sci-fi movies in my Jellyfin library"
- "Show me recently added TV shows"
- "Find all 4K content available"
- "What episodes am I missing from The Mandalorian?"

### Download Management
- "What's currently downloading in qBittorrent?"
- "Pause all downloads"
- "Search for Ubuntu 24.04 ISO on indexers"
- "Add The Last of Us to my TV show monitoring"

### Library Management
- "Scan my movie library for new content"
- "Show me library statistics"
- "Find duplicate movies"
- "What's the total size of my media library?"

### Automation
- "Set up automatic downloading for all Marvel movies"
- "Monitor for new episodes of my favorite shows"
- "Configure quality profiles for 4K content"
- "Show me failed download history"

### Smart Insights
- "What are the most watched movies this month?"
- "Show me trending content I don't have"
- "Analyze my viewing patterns"
- "Recommend similar shows to what I watch"

---

## 🔍 Troubleshooting

### MCP Not Showing in Claude Desktop

1. Check config file location:
```bash
cat ~/Library/Application\ Support/Claude/claude_desktop_config.json
```

2. Verify MCP server is running:
```bash
curl http://localhost:8090/health
```

3. Check logs:
```bash
tail -f mcp-architecture/logs/combined.log
```

### API Key Issues

Test API keys directly:
```bash
# Test Sonarr
curl -H "X-Api-Key: YOUR_KEY" http://localhost:8989/api/v3/system/status

# Test Radarr  
curl -H "X-Api-Key: YOUR_KEY" http://localhost:7878/api/v3/system/status

# Test Jellyfin
curl -H "X-Emby-Token: YOUR_KEY" http://localhost:8096/System/Info
```

### Connection Errors

1. Ensure all services are running:
```bash
docker ps | grep -E "(sonarr|radarr|jellyfin|prowlarr|qbittorrent)"
```

2. Check network connectivity:
```bash
nc -zv localhost 8096  # Jellyfin
nc -zv localhost 8989  # Sonarr
nc -zv localhost 7878  # Radarr
```

---

## 🎯 Advanced Features

### Custom Tools

The MCP implementation provides these tools:

| Tool | Description | Example |
|------|-------------|---------|
| `search_media` | Search across all libraries | "Find action movies from 2024" |
| `get_library_stats` | Get library statistics | "Show me storage usage" |
| `control_playback` | Control active sessions | "Pause playback on living room TV" |
| `manage_library` | Trigger scans and updates | "Refresh movie metadata" |
| `monitor_downloads` | Check download status | "Show active downloads" |
| `manage_series` | Add/remove TV shows | "Add Breaking Bad to monitoring" |
| `search_indexers` | Search torrent indexers | "Search for Ubuntu ISO" |

### AI Agent Voting

The system includes AI agents that vote on actions:
- **Content Curator**: Recommends what to download
- **Quality Manager**: Ensures optimal quality settings
- **Storage Optimizer**: Manages disk space efficiently
- **Trend Analyzer**: Identifies popular content

### WebSocket Real-time Updates

Connect to WebSocket for live updates:
```javascript
const socket = io('http://localhost:8090');
socket.on('mcp-activity', (data) => {
  console.log('MCP Activity:', data);
});
```

---

## 📚 Additional Resources

- **MCP Documentation**: https://modelcontextprotocol.io/
- **Claude Desktop**: https://claude.ai/download
- **Project Repository**: https://github.com/yourusername/media-server

---

## 🤝 Support

If you encounter issues:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Review logs in `mcp-architecture/logs/`
3. Verify all services are running with `docker ps`
4. Ensure API keys are correctly configured

---

## 🎉 Success Indicators

You'll know everything is working when:

✅ "MCP" badge appears in Claude Desktop  
✅ Media commands get specific responses about YOUR content  
✅ Real-time updates appear when downloads start/complete  
✅ Claude can search and control your actual media library  

---

**Enjoy seamless media management with Claude Desktop! 🎬🤖**