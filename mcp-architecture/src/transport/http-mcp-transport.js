/**
 * HTTP/SSE Transport for MCP Servers
 * Converts MCP servers to be accessible via HTTP with Server-Sent Events
 */

const express = require('express');
const cors = require('cors');
const { EventEmitter } = require('events');

class HttpMcpTransport extends EventEmitter {
  constructor(mcpServer, options = {}) {
    super();
    this.mcpServer = mcpServer;
    this.port = options.port || 3000;
    this.name = options.name || 'mcp-server';
    this.app = express();
    this.clients = new Set();
    
    this.setupMiddleware();
    this.setupRoutes();
  }

  setupMiddleware() {
    // CORS configuration
    this.app.use(cors({
      origin: '*',
      methods: ['GET', 'POST', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'Accept'],
      credentials: true
    }));

    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true }));

    // Add request logging
    this.app.use((req, res, next) => {
      console.log(`[${this.name}] ${req.method} ${req.path}`);
      next();
    });
  }

  setupRoutes() {
    // Health check endpoint
    this.app.get('/health', (req, res) => {
      res.json({
        status: 'healthy',
        server: this.name,
        timestamp: new Date().toISOString(),
        clients: this.clients.size
      });
    });

    // Server info endpoint
    this.app.get('/info', (req, res) => {
      res.json({
        name: this.mcpServer.serverInfo?.name || this.name,
        version: this.mcpServer.serverInfo?.version || '1.0.0',
        capabilities: this.mcpServer.serverInfo?.capabilities || {},
        transport: 'http-sse',
        endpoints: {
          health: '/health',
          info: '/info',
          events: '/events',
          resources: '/resources',
          tools: '/tools',
          call: '/call'
        }
      });
    });

    // Server-Sent Events endpoint for real-time updates
    this.app.get('/events', (req, res) => {
      // Set SSE headers
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control'
      });

      // Add client to active connections
      const clientId = Date.now() + Math.random();
      const client = { id: clientId, res };
      this.clients.add(client);

      // Send initial connection event
      this.sendEvent(client, 'connected', {
        server: this.name,
        timestamp: new Date().toISOString(),
        clientId
      });

      // Handle client disconnect
      req.on('close', () => {
        this.clients.delete(client);
        console.log(`[${this.name}] Client ${clientId} disconnected. Active clients: ${this.clients.size}`);
      });

      req.on('error', (err) => {
        console.error(`[${this.name}] Client ${clientId} error:`, err);
        this.clients.delete(client);
      });

      console.log(`[${this.name}] Client ${clientId} connected. Active clients: ${this.clients.size}`);
    });

    // List available resources
    this.app.get('/resources', async (req, res) => {
      try {
        const result = await this.handleMcpRequest('resources/list', {});
        res.json({
          success: true,
          data: result,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        res.status(500).json({
          success: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Read a specific resource
    this.app.get('/resources/*', async (req, res) => {
      try {
        const uri = req.path.replace('/resources/', '');
        const result = await this.handleMcpRequest('resources/read', { uri });
        res.json({
          success: true,
          data: result,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        res.status(500).json({
          success: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    });

    // List available tools
    this.app.get('/tools', async (req, res) => {
      try {
        const result = await this.handleMcpRequest('tools/list', {});
        res.json({
          success: true,
          data: result,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        res.status(500).json({
          success: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Call a tool
    this.app.post('/call/:toolName', async (req, res) => {
      try {
        const { toolName } = req.params;
        const { arguments: args = {} } = req.body;

        // Broadcast tool call start to SSE clients
        this.broadcastEvent('tool_call_start', {
          tool: toolName,
          arguments: args,
          timestamp: new Date().toISOString()
        });

        const result = await this.handleMcpRequest('tools/call', {
          name: toolName,
          arguments: args
        });

        // Broadcast tool call completion to SSE clients
        this.broadcastEvent('tool_call_complete', {
          tool: toolName,
          success: true,
          timestamp: new Date().toISOString()
        });

        res.json({
          success: true,
          data: result,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        // Broadcast tool call error to SSE clients
        this.broadcastEvent('tool_call_error', {
          tool: req.params.toolName,
          error: error.message,
          timestamp: new Date().toISOString()
        });

        res.status(500).json({
          success: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    });

    // Generic MCP request endpoint
    this.app.post('/mcp', async (req, res) => {
      try {
        const { method, params = {} } = req.body;

        if (!method) {
          return res.status(400).json({
            success: false,
            error: 'Method is required',
            timestamp: new Date().toISOString()
          });
        }

        const result = await this.handleMcpRequest(method, params);
        res.json({
          success: true,
          data: result,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        res.status(500).json({
          success: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    });

    // CORS preflight handler
    this.app.options('*', (req, res) => {
      res.header('Access-Control-Allow-Origin', '*');
      res.header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
      res.header('Access-Control-Allow-Headers', 'Content-Type, Authorization, Accept');
      res.status(200).send();
    });
  }

  async handleMcpRequest(method, params) {
    try {
      // Handle different request methods
      if (method === 'resources/list') {
        // Get resources from MCP server
        if (this.mcpServer.getResources) {
          const resources = this.mcpServer.getResources();
          return { resources };
        }
        return { resources: [] };
        
      } else if (method === 'tools/list') {
        // Get tools from MCP server
        if (this.mcpServer.getTools) {
          const tools = this.mcpServer.getTools();
          return { tools };
        }
        return { tools: [] };
        
      } else if (method === 'tools/call') {
        // Handle tool calling
        const toolName = params.name;
        const toolArgs = params.arguments || {};
        
        if (this.mcpServer.callTool) {
          return await this.mcpServer.callTool(toolName, toolArgs);
        }
        
        throw new Error(`Tool calling not supported`);
        
      } else {
        throw new Error(`Unknown method: ${method}`);
      }
    } catch (error) {
      throw error;
    }
  }

  sendEvent(client, event, data) {
    try {
      const eventData = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`;
      client.res.write(eventData);
    } catch (error) {
      console.error(`[${this.name}] Error sending event to client:`, error);
      this.clients.delete(client);
    }
  }

  broadcastEvent(event, data) {
    console.log(`[${this.name}] Broadcasting event: ${event} to ${this.clients.size} clients`);
    
    const deadClients = new Set();
    
    for (const client of this.clients) {
      try {
        this.sendEvent(client, event, data);
      } catch (error) {
        console.error(`[${this.name}] Failed to send to client ${client.id}:`, error);
        deadClients.add(client);
      }
    }

    // Remove dead clients
    for (const client of deadClients) {
      this.clients.delete(client);
    }
  }

  async start() {
    return new Promise((resolve, reject) => {
      const server = this.app.listen(this.port, (err) => {
        if (err) {
          reject(err);
        } else {
          console.log(`[${this.name}] HTTP MCP Transport running on port ${this.port}`);
          console.log(`[${this.name}] Health check: http://localhost:${this.port}/health`);
          console.log(`[${this.name}] Server info: http://localhost:${this.port}/info`);
          console.log(`[${this.name}] Events stream: http://localhost:${this.port}/events`);
          resolve(server);
        }
      });

      server.on('error', (error) => {
        console.error(`[${this.name}] Server error:`, error);
        reject(error);
      });
    });
  }

  // Method to simulate MCP server methods for external access
  simulateMcpMethods(mcpServerInstance) {
    // Store reference to actual MCP server methods
    this.mcpServer._tools = mcpServerInstance._tools || new Map();
    this.mcpServer._resources = mcpServerInstance._resources || new Map();
    this.mcpServer._resourceHandlers = mcpServerInstance._resourceHandlers || new Map();
    
    // Hook into tool calls to broadcast events
    if (mcpServerInstance._tools) {
      for (const [toolName, tool] of mcpServerInstance._tools) {
        const originalHandler = tool.handler;
        tool.handler = async (args) => {
          this.broadcastEvent('tool_start', { tool: toolName, args });
          try {
            const result = await originalHandler(args);
            this.broadcastEvent('tool_complete', { tool: toolName, success: true });
            return result;
          } catch (error) {
            this.broadcastEvent('tool_error', { tool: toolName, error: error.message });
            throw error;
          }
        };
      }
    }
  }
}

module.exports = HttpMcpTransport;