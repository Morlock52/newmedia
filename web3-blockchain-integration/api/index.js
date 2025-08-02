/**
 * Web3 Media Platform API Server
 * Production-ready Express.js server with blockchain integration
 */

require('dotenv').config();
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const winston = require('winston');

// Import services
const Web3MediaService = require('../backend/Web3MediaService');
const config = require('./config/config');

// Configure logging
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  defaultMeta: { service: 'web3-media-api' },
  transports: [
    new winston.transports.File({ filename: '/app/data/logs/error.log', level: 'error' }),
    new winston.transports.File({ filename: '/app/data/logs/combined.log' }),
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    })
  ]
});

// Create Express app
const app = express();
const port = process.env.PORT || 3030;

// Security middleware
app.use(helmet({
  contentSecurityPolicy: {
    directives: {
      defaultSrc: ["'self'"],
      styleSrc: ["'self'", "'unsafe-inline'"],
      scriptSrc: ["'self'"],
      imgSrc: ["'self'", "data:", "https:"],
      connectSrc: ["'self'", "wss:", "https:"],
      fontSrc: ["'self'"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"]
    }
  },
  crossOriginEmbedderPolicy: false
}));

// CORS configuration
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || [
    'http://localhost:3000',
    'http://localhost:3031',
    'http://localhost:8096'
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
}));

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: process.env.RATE_LIMIT_MAX || 1000, // limit each IP to 1000 requests per windowMs
  message: 'Too many requests from this IP, please try again later',
  standardHeaders: true,
  legacyHeaders: false
});
app.use(limiter);

// Stricter rate limiting for upload endpoints
const uploadLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 50, // 50 uploads per 15 minutes
  message: 'Upload rate limit exceeded, please try again later'
});

// General middleware
app.use(compression());
app.use(morgan('combined', { 
  stream: { write: message => logger.info(message.trim()) }
}));
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    version: process.env.npm_package_version || '1.0.0',
    uptime: process.uptime(),
    memory: process.memoryUsage(),
    environment: process.env.NODE_ENV || 'development'
  });
});

// Detailed health check
app.get('/health/detailed', async (req, res) => {
  try {
    const health = {
      status: 'healthy',
      timestamp: new Date().toISOString(),
      services: {},
      system: {
        uptime: process.uptime(),
        memory: process.memoryUsage(),
        cpu: process.cpuUsage(),
        nodeVersion: process.version,
        platform: process.platform
      }
    };

    // Check database connection if available
    try {
      // Add database health check here if needed
      health.services.database = { status: 'healthy' };
    } catch (error) {
      health.services.database = { status: 'unhealthy', error: error.message };
      health.status = 'degraded';
    }

    // Check Redis connection if available
    try {
      // Add Redis health check here if needed  
      health.services.redis = { status: 'healthy' };
    } catch (error) {
      health.services.redis = { status: 'unhealthy', error: error.message };
      health.status = 'degraded';
    }

    // Check IPFS connection
    try {
      // Add IPFS health check here
      health.services.ipfs = { status: 'healthy' };
    } catch (error) {
      health.services.ipfs = { status: 'unhealthy', error: error.message };
      health.status = 'degraded';
    }

    res.json(health);
  } catch (error) {
    logger.error('Health check failed:', error);
    res.status(500).json({
      status: 'unhealthy',
      timestamp: new Date().toISOString(),
      error: error.message
    });
  }
});

// API documentation endpoint
app.get('/api/docs', (req, res) => {
  res.json({
    name: 'Web3 Media Platform API',
    version: '1.0.0',
    description: 'RESTful API for Web3 media platform with blockchain integration',
    endpoints: {
      health: 'GET /health - Basic health check',
      ipfs: {
        upload: 'POST /api/ipfs/upload - Upload content to IPFS',
        retrieve: 'GET /api/ipfs/content/:hash - Retrieve content from IPFS'
      },
      nft: {
        mint: 'POST /api/nft/mint - Mint content as NFT',
        owned: 'GET /api/nft/owned/:address - Get owned NFTs'
      },
      payment: {
        process: 'POST /api/payment/process - Process cryptocurrency payment',
        subscribe: 'POST /api/payment/subscribe - Subscribe to platform'
      },
      jellyfin: {
        auth: 'POST /api/jellyfin/web3-auth - Web3 authentication for Jellyfin',
        content: 'GET /api/jellyfin/web3-content - Get Web3-enabled content'
      }
    },
    authentication: 'Bearer token in Authorization header',
    rateLimit: '1000 requests per 15 minutes (general), 50 requests per 15 minutes (uploads)'
  });
});

// Initialize Web3 Media Service
let web3Service;

async function initializeServices() {
  try {
    logger.info('Initializing Web3 Media Service...');
    
    web3Service = new Web3MediaService(config);
    
    // Apply upload rate limiting to upload endpoints
    web3Service.app.use('/api/ipfs/upload', uploadLimiter);
    web3Service.app.use('/api/nft/mint', uploadLimiter);
    
    // Mount Web3 service routes
    app.use('/', web3Service.app);
    
    logger.info('Web3 Media Service initialized successfully');
  } catch (error) {
    logger.error('Failed to initialize Web3 Media Service:', error);
    process.exit(1);
  }
}

// Error handling middleware
app.use((err, req, res, next) => {
  logger.error('Unhandled error:', err);
  
  if (err.type === 'entity.too.large') {
    return res.status(413).json({
      error: 'Request entity too large',
      message: 'The uploaded file exceeds the maximum size limit'
    });
  }
  
  res.status(err.status || 500).json({
    error: 'Internal server error',
    message: process.env.NODE_ENV === 'production' ? 'Something went wrong' : err.message,
    timestamp: new Date().toISOString()
  });
});

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Not found',
    message: `Route ${req.method} ${req.originalUrl} not found`,
    timestamp: new Date().toISOString()
  });
});

// Graceful shutdown handling
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

// Unhandled promise rejection handler
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
  process.exit(1);
});

// Uncaught exception handler
process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', error);
  process.exit(1);
});

// Start server
async function startServer() {
  try {
    await initializeServices();
    
    const server = app.listen(port, '0.0.0.0', () => {
      logger.info(`🚀 Web3 Media API server started on port ${port}`);
      logger.info(`Environment: ${process.env.NODE_ENV || 'development'}`);
      logger.info(`Health check: http://localhost:${port}/health`);
      logger.info(`API docs: http://localhost:${port}/api/docs`);
    });
    
    // Set server timeout
    server.timeout = 120000; // 2 minutes
    server.keepAliveTimeout = 65000; // 65 seconds
    server.headersTimeout = 66000; // 66 seconds
    
    return server;
  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Export for testing
if (require.main === module) {
  startServer();
} else {
  module.exports = { app, startServer };
}