#!/usr/bin/env node

const readline = require('readline');

class WorkingMinimalMCP {
  constructor() {
    this.protocolVersion = '2025-06-18';
    this.serverInfo = { name: 'sonarr-mcp', version: '1.0.0' };
  }

  log(msg) {
    if (process.env.MCP_DEBUG === 'true') {
      console.error(`[sonarr] ${new Date().toISOString()} ${msg}`);
    }
  }

  async handleRequest(req) {
    this.log(`Handling: ${req.method}`);
    
    switch (req.method) {
      case 'initialize':
        return {
          protocolVersion: this.protocolVersion,
          capabilities: { tools: {}, resources: {}, prompts: { listChanged: true } },
          serverInfo: this.serverInfo
        };
      case 'tools/list':
        return { tools: [{ name: 'test', description: 'Test tool', inputSchema: { type: 'object', properties: {} } }] };
      case 'tools/call':
        return { content: [{ type: 'text', text: '✅ MCP Server Working!' }] };
      case 'resources/list':
        return { resources: [] };
      case 'prompts/list':
        return { prompts: [{ name: 'test_prompt', description: 'Test prompt', arguments: [] }] };
      case 'prompts/get':
        return { messages: [{ role: 'user', content: { type: 'text', text: 'Test prompt message' } }] };
      default:
        throw new Error(`Unknown method: ${req.method}`);
    }
  }

  start() {
    console.error('[sonarr] Starting...');
    const rl = readline.createInterface({ input: process.stdin, output: process.stdout, terminal: false });
    const keepAlive = setInterval(() => {}, 60000);

    rl.on('line', async (line) => {
      try {
        const request = JSON.parse(line);
        const result = await this.handleRequest(request);
        const response = { jsonrpc: '2.0', id: request.id, result };
        process.stdout.write(JSON.stringify(response) + '\n');
      } catch (error) {
        const req = JSON.parse(line);
        const errResp = { jsonrpc: '2.0', id: req.id, error: { code: -32603, message: error.message } };
        process.stdout.write(JSON.stringify(errResp) + '\n');
      }
    });

    rl.on('close', () => { clearInterval(keepAlive); process.exit(0); });
    process.on('SIGINT', () => { clearInterval(keepAlive); process.exit(0); });
    console.error('[sonarr] Ready');
  }
}

new WorkingMinimalMCP().start();