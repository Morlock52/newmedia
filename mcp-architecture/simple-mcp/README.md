# Simple MCP Server

The absolute simplest MCP server that will definitely work with Claude Desktop.

## What it does

Provides one tool: `hello` that greets you with a message.

## Setup Instructions

### 1. Install dependencies
```bash
cd mcp-architecture/simple-mcp
npm install
```

### 2. Test the server works
```bash
npx simple-mcp
# Should show: Simple MCP server running on stdio
# Press Ctrl+C to exit
```

### 3. Add to Claude Desktop

Copy the configuration from `claude-desktop-config.json` to your Claude Desktop config file:

**On macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**On Windows:** `%APPDATA%\Claude\claude_desktop_config.json`

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

**Important:** Change the path `/Users/morlock/fun/newmedia/mcp-architecture/simple-mcp` to your actual path!

### 4. Restart Claude Desktop

Close and reopen Claude Desktop completely.

### 5. Test it works

In Claude Desktop, you should now have access to a tool called `hello`. Try:
"Use the hello tool to greet me"

## Troubleshooting

If it doesn't work:

1. Check the path in your config is correct
2. Make sure npm install completed successfully
3. Test `npx simple-mcp` works in terminal
4. Check Claude Desktop logs for errors

## Files

- `package.json` - Minimal package configuration
- `index.js` - The MCP server implementation
- `claude-desktop-config.json` - Configuration for Claude Desktop
- `README.md` - This file

## Features

- ✅ Uses official MCP SDK
- ✅ Zero external dependencies except MCP SDK
- ✅ Works with npx
- ✅ Proper error handling
- ✅ Simple tool that just works
- ✅ Follows MCP protocol correctly

This is the simplest possible MCP server. If this doesn't work, there's a deeper issue with your Claude Desktop setup.