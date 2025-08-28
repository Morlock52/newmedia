#!/usr/bin/env node

/**
 * MediaServer MCP Suite - Main Entry Point
 * Orchestrates all MCP servers and AI agent voting system
 * 
 * Features:
 * - MCP servers for Jellyfin, Sonarr, Radarr, Prowlarr, qBittorrent
 * - AI agent voting system with OpenAI o1-mini
 * - Social media research integration
 * - Real-time agent coordination
 */

require('dotenv').config();
const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const cors = require('cors');
const helmet = require('helmet');
const winston = require('winston');
const path = require('path');

// MCP Server imports
const JellyfinMCP = require('./servers/jellyfin-mcp');
const SonarrMCP = require('./servers/sonarr-mcp');
const RadarrMCP = require('./servers/radarr-mcp');
const ProwlarrMCP = require('./servers/prowlarr-mcp');
const QBittorrentMCP = require('./servers/qbittorrent-mcp');

// HTTP Transport for streamable MCP servers
const HttpMcpTransport = require('./transport/http-mcp-transport');

// AI Agent imports
const AgentOrchestrator = require('./agents/orchestrator');
const VotingSystem = require('./agents/voting-system');
const SocialResearcher = require('./agents/social-researcher');

// Chatbot interface
const ChatbotInterface = require('./chatbot/interface');

// Configure logger
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'mediaserver-mcp-suite' },
  transports: [
    new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: 'logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

class MediaServerMCPSuite {
  constructor() {
    this.app = express();
    this.server = http.createServer(this.app);
    this.io = socketIo(this.server, {
      cors: {
        origin: process.env.ALLOWED_ORIGINS?.split(',') || ["http://localhost:3000", "http://localhost:8090"],
        methods: ["GET", "POST"]
      }
    });
    
    this.mcpServers = {};
    this.httpTransports = {}; // HTTP/SSE transports for MCP servers
    this.agents = {};
    this.votingSystem = null;
    this.chatbot = null;
    this.port = process.env.PORT || 8090;
    
    this.setupMiddleware();
    this.setupRoutes();
    this.setupSocketHandlers();
  }

  setupMiddleware() {
    // Security and CORS
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'", "'unsafe-inline'", "https://cdn.tailwindcss.com", "https://cdn.jsdelivr.net"],
          styleSrc: ["'self'", "'unsafe-inline'", "https://cdn.tailwindcss.com"],
          connectSrc: ["'self'", "ws://localhost:8090", "https://api.openai.com"],
          imgSrc: ["'self'", "data:", "https:"],
        },
      },
    }));
    
    this.app.use(cors());
    this.app.use(express.json({ limit: '50mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '50mb' }));
    
    // Static files
    this.app.use(express.static(path.join(__dirname, '../public')));
    
    // Request logging
    this.app.use((req, res, next) => {
      logger.info(`${req.method} ${req.path}`, { 
        ip: req.ip, 
        userAgent: req.get('User-Agent') 
      });
      next();
    });
  }

  setupRoutes() {
    // Health check
    this.app.get('/health', (req, res) => {
      const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        uptime: process.uptime(),
        mcpServers: Object.keys(this.mcpServers).reduce((acc, key) => {
          acc[key] = this.mcpServers[key].isRunning ? 'running' : 'stopped';
          return acc;
        }, {}),
        agents: Object.keys(this.agents).reduce((acc, key) => {
          acc[key] = this.agents[key].isActive ? 'active' : 'inactive';
          return acc;
        }, {})
      };
      res.json(health);
    });

    // MCP Server status and controls
    this.app.get('/api/mcp/status', (req, res) => {
      const status = {};
      Object.keys(this.mcpServers).forEach(key => {
        const server = this.mcpServers[key];
        status[key] = {
          running: server.isRunning,
          port: server.port,
          lastActivity: server.lastActivity,
          requestCount: server.requestCount || 0,
          errorCount: server.errorCount || 0
        };
      });
      res.json(status);
    });

    // Agent voting system
    this.app.get('/api/agents/status', (req, res) => {
      if (!this.votingSystem) {
        return res.status(503).json({ error: 'Voting system not initialized' });
      }
      
      res.json({
        activeAgents: this.votingSystem.getActiveAgents(),
        recentVotes: this.votingSystem.getRecentVotes(),
        systemStats: this.votingSystem.getSystemStats()
      });
    });

    // Chat interface
    this.app.post('/api/chat', async (req, res) => {
      try {
        const { message, sessionId } = req.body;
        
        if (!this.chatbot) {
          return res.status(503).json({ error: 'Chatbot not initialized' });
        }

        const response = await this.chatbot.processMessage(message, sessionId);
        res.json(response);
      } catch (error) {
        logger.error('Chat processing error:', error);
        res.status(500).json({ error: 'Failed to process message' });
      }
    });

    // Social research endpoint
    this.app.post('/api/research', async (req, res) => {
      try {
        const { query, platforms } = req.body;
        
        if (!this.agents.socialResearcher) {
          return res.status(503).json({ error: 'Social researcher not available' });
        }

        const results = await this.agents.socialResearcher.research(query, platforms);
        res.json(results);
      } catch (error) {
        logger.error('Research error:', error);
        res.status(500).json({ error: 'Failed to conduct research' });
      }
    });

    // Serve main interface
    this.app.get('/', (req, res) => {
      res.sendFile(path.join(__dirname, '../public/index.html'));
    });
  }

  setupSocketHandlers() {
    this.io.on('connection', (socket) => {
      logger.info('Client connected:', socket.id);

      // Agent voting updates
      socket.on('subscribe-voting', () => {
        socket.join('voting-updates');
      });

      // MCP server logs
      socket.on('subscribe-logs', (serverId) => {
        socket.join(`logs-${serverId}`);
      });

      // Chat messages
      socket.on('chat-message', async (data) => {
        try {
          const response = await this.chatbot.processMessage(data.message, data.sessionId);
          socket.emit('chat-response', response);
          
          // Broadcast to voting room if agents were involved
          if (response.agentVotes) {
            this.io.to('voting-updates').emit('agent-vote', response.agentVotes);
          }
        } catch (error) {
          logger.error('Socket chat error:', error);
          socket.emit('chat-error', { error: 'Failed to process message' });
        }
      });

      socket.on('disconnect', () => {
        logger.info('Client disconnected:', socket.id);
      });
    });
  }

  async initializeMCPServers() {
    logger.info('Initializing MCP servers with HTTP/SSE transport...');

    try {
      // Initialize each MCP server with HTTP transport
      const mcpConfigs = {
        jellyfin: {
          mcpClass: JellyfinMCP,
          port: 3001,
          config: {
            jellyfinUrl: process.env.JELLYFIN_URL || 'http://localhost:8096',
            apiKey: process.env.JELLYFIN_API_KEY,
            io: this.io
          }
        },
        sonarr: {
          mcpClass: SonarrMCP,
          port: 3002,
          config: {
            sonarrUrl: process.env.SONARR_URL || 'http://localhost:8989',
            apiKey: process.env.SONARR_API_KEY,
            io: this.io
          }
        },
        radarr: {
          mcpClass: RadarrMCP,
          port: 3003,
          config: {
            radarrUrl: process.env.RADARR_URL || 'http://localhost:7878',
            apiKey: process.env.RADARR_API_KEY,
            io: this.io
          }
        },
        prowlarr: {
          mcpClass: ProwlarrMCP,
          port: 3004,
          config: {
            prowlarrUrl: process.env.PROWLARR_URL || 'http://localhost:9696',
            apiKey: process.env.PROWLARR_API_KEY,
            io: this.io
          }
        },
        qbittorrent: {
          mcpClass: QBittorrentMCP,
          port: 3005,
          config: {
            qbittorrentUrl: process.env.QBITTORRENT_URL || 'http://localhost:8080',
            username: process.env.QBITTORRENT_USERNAME || 'admin',
            password: process.env.QBITTORRENT_PASSWORD || 'adminadmin',
            io: this.io
          }
        }
      };

      // Create HTTP transports for each MCP server
      this.httpTransports = {};
      
      for (const [name, config] of Object.entries(mcpConfigs)) {
        try {
          // Create the MCP server instance
          const mcpServer = new config.mcpClass({
            port: config.port,
            ...config.config
          });
          
          // Create HTTP transport wrapper
          const httpTransport = new HttpMcpTransport(mcpServer, {
            port: config.port,
            name: `${name}-mcp`
          });
          
          // Connect transport to MCP server methods
          httpTransport.simulateMcpMethods(mcpServer);
          
          // Store references
          this.mcpServers[name] = mcpServer;
          this.httpTransports[name] = httpTransport;
          
          // Start the HTTP transport
          await httpTransport.start();
          
          logger.info(`✅ ${name} MCP server started with HTTP/SSE transport on port ${config.port}`);
          
        } catch (error) {
          logger.error(`❌ Failed to start ${name} MCP server:`, error);
        }
      }
      
      logger.info('🚀 All MCP servers initialized with HTTP/SSE streaming support');

    } catch (error) {
      logger.error('Failed to initialize MCP servers:', error);
      throw error;
    }
  }

  async initializeAgents() {
    logger.info('Initializing AI agents...');

    try {
      // Initialize voting system
      this.votingSystem = new VotingSystem({
        openaiApiKey: process.env.OPENAI_API_KEY,
        model: process.env.OPENAI_MODEL || 'o1-mini',
        io: this.io
      });

      // Initialize agent orchestrator
      this.agents.orchestrator = new AgentOrchestrator({
        mcpServers: this.mcpServers,
        votingSystem: this.votingSystem,
        openaiApiKey: process.env.OPENAI_API_KEY,
        io: this.io
      });

      // Initialize social researcher
      this.agents.socialResearcher = new SocialResearcher({
        openaiApiKey: process.env.OPENAI_API_KEY,
        twitterApiKey: process.env.TWITTER_API_KEY,
        redditClientId: process.env.REDDIT_CLIENT_ID,
        io: this.io
      });

      // Initialize chatbot interface
      this.chatbot = new ChatbotInterface({
        agents: this.agents,
        votingSystem: this.votingSystem,
        mcpServers: this.mcpServers,
        openaiApiKey: process.env.OPENAI_API_KEY,
        io: this.io
      });

      logger.info('AI agents initialization complete');

    } catch (error) {
      logger.error('Failed to initialize agents:', error);
      throw error;
    }
  }

  async start() {
    try {
      logger.info('Starting MediaServer MCP Suite...');

      // Create logs directory
      const fs = require('fs');
      if (!fs.existsSync('logs')) {
        fs.mkdirSync('logs');
      }

      // Initialize systems
      await this.initializeMCPServers();
      await this.initializeAgents();

      // Start main server
      this.server.listen(this.port, () => {
        logger.info(`MediaServer MCP Suite running on port ${this.port}`);
        logger.info(`Dashboard: http://localhost:${this.port}`);
        logger.info(`Health check: http://localhost:${this.port}/health`);
        
        // Log MCP server status
        Object.keys(this.mcpServers).forEach(key => {
          const server = this.mcpServers[key];
          logger.info(`${key.toUpperCase()} MCP Server: http://localhost:${server.port}`);
        });
      });

      // Graceful shutdown
      process.on('SIGTERM', this.shutdown.bind(this));
      process.on('SIGINT', this.shutdown.bind(this));

    } catch (error) {
      logger.error('Failed to start MediaServer MCP Suite:', error);
      process.exit(1);
    }
  }

  async shutdown() {
    logger.info('Shutting down MediaServer MCP Suite...');

    try {
      // Stop MCP servers
      const stopPromises = Object.values(this.mcpServers).map(server => 
        server.stop().catch(error => {
          logger.error('Error stopping MCP server:', error);
        })
      );
      
      await Promise.allSettled(stopPromises);

      // Close main server
      this.server.close(() => {
        logger.info('MediaServer MCP Suite shut down gracefully');
        process.exit(0);
      });

    } catch (error) {
      logger.error('Error during shutdown:', error);
      process.exit(1);
    }
  }
}

// Start the suite if run directly
if (require.main === module) {
  const suite = new MediaServerMCPSuite();
  suite.start();
}

module.exports = MediaServerMCPSuite;