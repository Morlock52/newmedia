# 🔑 How to Add Your Sonarr API Key

## Quick Steps:

### 1. Get Your Sonarr API Key

**If Sonarr is running:**
1. Open http://localhost:8989 in your browser
2. Go to **Settings** → **General** 
3. Scroll to **Security** section
4. Click **Show** next to API Key
5. Copy the 32-character key

**If Sonarr is in Docker:**
```bash
# Find Sonarr container
docker ps | grep sonarr

# Get API key from config
docker exec sonarr cat /config/config.xml | grep -i apikey
```

### 2. Add API Key to Claude Desktop

Open the config file:
```bash
nano ~/.claude/claude_desktop_config.json
```

Find this section:
```json
"sonarr": {
  "command": "node", 
  "args": ["/Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js"],
  "env": {
    "DEBUG": "true",
    "SONARR_URL": "http://localhost:8989",
    "SONARR_API_KEY": ""  ← ADD YOUR KEY HERE
  }
}
```

Replace the empty quotes with your API key:
```json
"SONARR_API_KEY": "your-32-character-api-key-here"
```

Save the file (Ctrl+X, then Y, then Enter).

### 3. Restart Claude Desktop

1. Quit Claude Desktop completely (Cmd+Q)
2. Open Claude Desktop again

### 4. Test the Connection

Ask Claude:
- "Show me my Sonarr system status"
- "What TV series do I have?"
- "Search for Breaking Bad"

## 🎯 Troubleshooting

### Can't find API key?
Check Sonarr config file directly:
```bash
# Local installation
cat ~/.config/Sonarr/config.xml | grep ApiKey

# Docker installation  
docker exec sonarr cat /config/config.xml | grep ApiKey
```

### Sonarr not running?
```bash
# Start with Docker
docker start sonarr

# Or with Docker Compose
cd /path/to/your/media-server
docker-compose up -d sonarr

# Check if running
curl http://localhost:8989
```

### Different Sonarr URL?
If Sonarr runs on a different port or host, update the config:
```json
"SONARR_URL": "http://your-server:your-port"
```

## 📺 Demo Mode

**No Sonarr instance? No problem!** The MCP works in demo mode without configuration. You'll get sample data for all tools.

## ✅ Success Indicators

You'll know it's working when:
- Claude can show your real TV series list
- Episode counts match your Sonarr library
- System status shows your Sonarr version
- Search returns actual results from your indexers

**Current Status**: Running in Demo Mode (add API key for real data)