# MCP Claude Desktop Troubleshooting Guide

## Common Issues and Solutions

### Issue: "Server disconnected" in Claude Desktop

#### Possible Causes:
1. **Node.js path issues** - Claude can't find node
2. **Shell environment** - PATH not set correctly
3. **File permissions** - Scripts not executable
4. **JSON-RPC protocol** - Server not responding correctly

#### Solutions Tried:

### 1. Direct Node Path
```json
{
  "command": "/Users/morlock/.nvm/versions/node/v22.16.0/bin/node",
  "args": ["/path/to/script.js"]
}
```
**Result**: May fail if Claude doesn't inherit environment

### 2. Simple Node Command
```json
{
  "command": "node",
  "args": ["/path/to/script.js"]
}
```
**Result**: Works if node is in PATH

### 3. Shell Wrapper
```json
{
  "command": "/bin/zsh",
  "args": ["-c", "source ~/.zshrc && node /path/to/script.js"]
}
```
**Result**: Most reliable - ensures environment is loaded

### 4. Custom Launcher Script
```bash
#!/bin/zsh
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
nvm use 22.16.0 >/dev/null 2>&1
exec node "$@"
```

## Debugging Steps

1. **Test MCP Server Manually**
```bash
echo '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}' | node /path/to/mcp.js
```
Should return: `{"jsonrpc":"2.0","id":1,"result":{...}}`

2. **Check File Permissions**
```bash
ls -la /path/to/mcp.js
chmod +x /path/to/mcp.js
```

3. **Verify Node Path**
```bash
which node
node --version
```

4. **Claude Desktop Logs**
- Open Developer Console: View → Developer Tools
- Check Console tab for errors
- Look for "MCP server connected" messages

## Working Configurations

### Option 1: Shell Command (Most Reliable)
```json
{
  "mcpServers": {
    "media-server": {
      "command": "/bin/zsh",
      "args": [
        "-c",
        "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js"
      ]
    }
  }
}
```

### Option 2: Direct Node (If PATH is correct)
```json
{
  "mcpServers": {
    "media-server": {
      "command": "node",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js"]
    }
  }
}
```

### Option 3: Custom Launcher
```json
{
  "mcpServers": {
    "media-server": {
      "command": "/Users/morlock/fun/newmedia/mcp-architecture/mcp-launcher.sh",
      "args": ["/Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js"]
    }
  }
}
```

## Final Working Configuration for All 5 Servers

Save this to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "media-server": {
      "command": "/bin/zsh",
      "args": [
        "-c",
        "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/standalone-mcp.js"
      ]
    },
    "sonarr": {
      "command": "/bin/zsh",
      "args": [
        "-c",
        "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/sonarr-mcp-standalone.js"
      ]
    },
    "jellyfin": {
      "command": "/bin/zsh",
      "args": [
        "-c",
        "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/jellyfin-mcp-standalone.js"
      ]
    },
    "radarr": {
      "command": "/bin/zsh",
      "args": [
        "-c",
        "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/radarr-mcp-standalone.js"
      ]
    },
    "prowlarr": {
      "command": "/bin/zsh",
      "args": [
        "-c",
        "source ~/.zshrc && node /Users/morlock/fun/newmedia/mcp-architecture/prowlarr-mcp-standalone.js"
      ]
    }
  }
}
```

## Testing Steps

1. Save configuration
2. Completely quit Claude Desktop (Cmd+Q)
3. Wait 5 seconds
4. Open Claude Desktop
5. Ask: "What MCP tools are available?"

## If Still Not Working

1. **Check zsh configuration**
```bash
cat ~/.zshrc | grep -i nvm
```

2. **Test with absolute node path**
```bash
/Users/morlock/.nvm/versions/node/v22.16.0/bin/node --version
```

3. **Create test script**
```bash
#!/bin/zsh
source ~/.zshrc
which node
node --version
```

4. **Contact Claude Support**
- Include your configuration
- Include any error messages from Developer Console
- Mention you're using nvm with Node v22.16.0