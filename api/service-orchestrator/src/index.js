import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import dotenv from 'dotenv';
import { createServer } from 'http';
import { WebSocketServer } from 'ws';
import { logger } from './utils/logger.js';
import { errorHandler } from './middleware/errorHandler.js';
import { authenticateToken } from './middleware/auth.js';
import { rateLimiter } from './middleware/rateLimiter.js';
import { metricsMiddleware } from './middleware/metrics.js';
import servicesRouter from './routes/services.js';
import healthRouter from './routes/health.js';
import configRouter from './routes/config.js';
import tasksRouter from './routes/tasks.js';
import authRouter from './routes/auth.js';
import { ServiceOrchestrator } from './services/orchestrator.js';
import { HealthMonitor } from './services/healthMonitor.js';
import { ConfigManager } from './services/configManager.js';
import { EventBus } from './services/eventBus.js';
import { connectDatabase } from './database/connection.js';
import { connectRedis } from './cache/redis.js';
import { connectMessageQueue } from './queue/rabbitmq.js';

// Load environment variables
dotenv.config();

// Create Express app
const app = express();
const server = createServer(app);

// Initialize WebSocket server
const wss = new WebSocketServer({ 
  server,
  path: '/ws',
  verifyClient: (info, cb) => {
    // Verify WebSocket connections
    const token = info.req.headers.authorization?.split(' ')[1];
    if (token) {
      // Verify JWT token
      cb(true);
    } else {
      cb(false, 401, 'Unauthorized');
    }
  }
});

// Middleware
app.use(helmet());
app.use(cors({
  origin: process.env.ALLOWED_ORIGINS?.split(',') || '*',
  credentials: true
}));
app.use(compression());
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(morgan('combined', { stream: logger.stream }));
app.use(metricsMiddleware);

// Rate limiting
app.use('/api/', rateLimiter);

// Health check endpoint (no auth required)
app.use('/api/v1/health', healthRouter);

// Auth endpoints
app.use('/api/v1/auth', authRouter);

// Protected routes
app.use('/api/v1/services', authenticateToken, servicesRouter);
app.use('/api/v1/config', authenticateToken, configRouter);
app.use('/api/v1/tasks', authenticateToken, tasksRouter);

// Error handling
app.use(errorHandler);

// Initialize services
let orchestrator;
let healthMonitor;
let configManager;
let eventBus;

async function initializeServices() {
  try {
    // Connect to databases
    await connectDatabase();
    await connectRedis();
    await connectMessageQueue();

    // Initialize core services
    eventBus = new EventBus();
    configManager = new ConfigManager(eventBus);
    orchestrator = new ServiceOrchestrator(eventBus, configManager);
    healthMonitor = new HealthMonitor(eventBus);

    // Start health monitoring
    await healthMonitor.startMonitoring();

    // WebSocket connection handling
    wss.on('connection', (ws, req) => {
      logger.info('New WebSocket connection established');

      // Subscribe to events
      const unsubscribe = eventBus.subscribe('*', (event) => {
        ws.send(JSON.stringify(event));
      });

      ws.on('close', () => {
        logger.info('WebSocket connection closed');
        unsubscribe();
      });

      ws.on('error', (error) => {
        logger.error('WebSocket error:', error);
      });
    });

    // Graceful shutdown
    process.on('SIGTERM', async () => {
      logger.info('SIGTERM received, shutting down gracefully');
      
      // Stop accepting new connections
      server.close(() => {
        logger.info('HTTP server closed');
      });

      // Close WebSocket connections
      wss.clients.forEach((client) => {
        client.close();
      });

      // Stop services
      await healthMonitor.stopMonitoring();
      await eventBus.close();
      
      process.exit(0);
    });

    // Start server
    const PORT = process.env.PORT || 3000;
    server.listen(PORT, () => {
      logger.info(`Service Orchestrator API running on port ${PORT}`);
      logger.info(`WebSocket server available at ws://localhost:${PORT}/ws`);
    });

  } catch (error) {
    logger.error('Failed to initialize services:', error);
    process.exit(1);
  }
}

// Export for testing
export { app, orchestrator, healthMonitor, configManager, eventBus };

// Start the application
if (process.env.NODE_ENV !== 'test') {
  initializeServices();
}