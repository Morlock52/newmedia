# Quick Setup Guide

## 1. Dependencies are already installed ✅

## 2. Test it works locally
```bash
cd /Users/morlock/fun/newmedia/mcp-architecture/simple-mcp
npx simple-mcp
```
You should see: "Simple MCP server running on stdio"
Press Ctrl+C to stop.

## 3. Add to Claude Desktop

Edit your Claude Desktop config file:
- **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
- **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`

Add this configuration:
```json
{
  "mcpServers": {
    "simple-mcp": {
      "command": "npx",
      "args": [
        "-p",
        "/Users/morlock/fun/newmedia/mcp-architecture/simple-mcp",
        "simple-mcp"
      ]
    }
  }
}
```

## 4. Restart Claude Desktop completely

## 5. Test in Claude Desktop
Ask: "Use the hello tool to say hello to me"

## That's it! 🎉

This is the simplest possible MCP server:
- Only one tool: `hello`
- Zero dependencies except MCP SDK
- Uses npx (no global installs)
- Cannot fail if setup correctly

If this doesn't work, check:
1. Path is correct in config
2. npm install completed
3. Claude Desktop fully restarted