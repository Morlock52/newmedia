# MCP Server Fix Guide

## Problem Analysis

The main issues with the original MCP server implementations were:

1. **Stdout Pollution**: Using `console.log()` for responses, which can add extra formatting
2. **Debug Output to Stdout**: Debug messages going to stdout instead of stderr
3. **Improper JSON Formatting**: Not ensuring each message ends with a newline
4. **Process Lifecycle**: Not properly handling the readline interface to keep the process alive
5. **Error Handling**: Not returning proper JSON-RPC error responses

## Key Fixes Applied

### 1. Stdout is Sacred

```javascript
// ❌ WRONG - console.log can add extra formatting
console.log(JSON.stringify(response));

// ✅ CORRECT - use process.stdout.write with explicit newline
process.stdout.write(JSON.stringify(response) + '\n');
```

### 2. All Logging to Stderr

```javascript
// ❌ WRONG - debug output pollutes stdout
console.log(`Debug: ${message}`);

// ✅ CORRECT - use stderr for all debug/log output
process.stderr.write(`[DEBUG] ${message}\n`);
```

### 3. Proper JSON-RPC Protocol

```javascript
// Every response must follow JSON-RPC 2.0 spec
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": { /* result data */ }
}

// Error responses must be properly formatted
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32603,
    "message": "Internal error",
    "data": "Optional error details"
  }
}
```

### 4. Process Lifecycle Management

```javascript
const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
  terminal: false  // Important: disable terminal mode
});

// Handle graceful shutdown
rl.on('close', () => {
  process.exit(0);
});

// Handle signals properly
process.on('SIGINT', () => {
  rl.close();
});
```

## Testing Your Fixed Server

### 1. Manual Test with Fixed Server

```bash
# Make the test script executable
chmod +x test-fixed-server.js

# Run the test
./test-fixed-server.js
```

### 2. Test with Claude Desktop

Use this configuration in Claude Desktop settings:

```json
{
  "mcpServers": {
    "fixed-test-server": {
      "command": "node",
      "args": ["/full/path/to/fixed-mcp-server.js"],
      "env": {
        "MCP_DEBUG": "false"
      }
    }
  }
}
```

### 3. Debug Mode

Enable debug output by setting the environment variable:

```json
{
  "mcpServers": {
    "fixed-test-server": {
      "command": "node",
      "args": ["/full/path/to/fixed-mcp-server.js"],
      "env": {
        "MCP_DEBUG": "true"
      }
    }
  }
}
```

Debug output will appear in Claude Desktop's logs, not in the chat.

## Common Pitfalls to Avoid

### 1. Don't Use Console Methods

```javascript
// ❌ AVOID ALL OF THESE for responses:
console.log()
console.error()
console.warn()
console.info()

// ✅ USE THESE INSTEAD:
process.stdout.write()  // For JSON-RPC responses only
process.stderr.write()  // For debug/log output
```

### 2. Always Validate Requests

```javascript
// Check JSON-RPC version
if (!request.jsonrpc || request.jsonrpc !== '2.0') {
  sendError(request.id || null, -32600, 'Invalid JSON-RPC version');
  return;
}

// Check for required fields
if (!request.method) {
  sendError(request.id, -32600, 'Missing method');
  return;
}
```

### 3. Handle All Error Cases

```javascript
try {
  const result = await handleRequest(request);
  sendResponse(request.id, result);
} catch (error) {
  sendError(request.id, -32603, error.message);
}
```

## Converting Existing Servers

To convert an existing MCP server:

1. Replace all `console.log()` for responses with `process.stdout.write()`
2. Move all debug output to `process.stderr.write()`
3. Ensure each JSON response ends with `\n`
4. Add proper error handling with JSON-RPC error responses
5. Test with the provided test script

## Example Fixed Server Structure

```javascript
class FixedMCPServer {
  log(message) {
    if (process.env.MCP_DEBUG === 'true') {
      process.stderr.write(`[DEBUG] ${message}\n`);
    }
  }

  sendResponse(id, result) {
    const response = {
      jsonrpc: '2.0',
      id: id,
      result: result
    };
    process.stdout.write(JSON.stringify(response) + '\n');
  }

  sendError(id, code, message) {
    const response = {
      jsonrpc: '2.0',
      id: id,
      error: { code, message }
    };
    process.stdout.write(JSON.stringify(response) + '\n');
  }

  start() {
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      terminal: false
    });

    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        const result = await this.handleRequest(request);
        this.sendResponse(request.id, result);
      } catch (error) {
        // Handle errors properly
      }
    });
  }
}
```

## Jellyfin Server Configuration

For the fixed Jellyfin server, use this configuration:

```json
{
  "mcpServers": {
    "jellyfin-fixed": {
      "command": "node",
      "args": ["/full/path/to/fixed-jellyfin-mcp.js"],
      "env": {
        "JELLYFIN_URL": "http://localhost:8096",
        "JELLYFIN_API_KEY": "your-api-key-here",
        "MCP_DEBUG": "false"
      }
    }
  }
}
```

## Verification Checklist

- [ ] Server only outputs JSON-RPC to stdout
- [ ] All debug/log messages go to stderr
- [ ] Each JSON message ends with newline
- [ ] Server stays running after initialization
- [ ] Errors return proper JSON-RPC error responses
- [ ] No console.log() calls for responses
- [ ] Process handles SIGINT/SIGTERM gracefully
- [ ] Test script passes all tests