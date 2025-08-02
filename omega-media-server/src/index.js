const fastify = require('fastify');
const path = require('path');
const { setupDatabase } = require('./lib/database');
const { setupRedis } = require('./lib/redis');
const { setupWebSocket } = require('./lib/websocket');
const { setupGraphQL } = require('./lib/graphql');
const { setupAuth } = require('./lib/auth');
const { setupAI } = require('./lib/ai');
const { setupMediaServices } = require('./lib/media-services');
const { setupMonitoring } = require('./lib/monitoring');
const { logger } = require('./lib/logger');

// Import routes
const systemRoutes = require('./routes/system');
const authRoutes = require('./routes/auth');
const mediaRoutes = require('./routes/media');
const appsRoutes = require('./routes/apps');
const aiRoutes = require('./routes/ai');
const statsRoutes = require('./routes/stats');

// Configuration
const config = require('./config');

async function start() {
  // Create Fastify instance
  const app = fastify({
    logger: logger,
    trustProxy: true,
  });

  try {
    // Register plugins
    await app.register(require('@fastify/cors'), {
      origin: config.cors.origin,
      credentials: true,
    });

    await app.register(require('@fastify/helmet'), {
      contentSecurityPolicy: false,
    });

    await app.register(require('@fastify/multipart'), {
      limits: {
        fileSize: 100 * 1024 * 1024 * 1024, // 100GB for large media files
      },
    });

    await app.register(require('@fastify/static'), {
      root: path.join(__dirname, '../web/dist'),
      prefix: '/',
    });

    // Setup core services
    await setupDatabase(app);
    await setupRedis(app);
    await setupAuth(app);
    await setupWebSocket(app);
    await setupGraphQL(app);
    
    // Setup feature services
    if (config.features.ai) {
      await setupAI(app);
    }
    
    await setupMediaServices(app);
    await setupMonitoring(app);

    // Register API routes
    await app.register(systemRoutes, { prefix: '/api/system' });
    await app.register(authRoutes, { prefix: '/api/auth' });
    await app.register(mediaRoutes, { prefix: '/api/media' });
    await app.register(appsRoutes, { prefix: '/api/apps' });
    await app.register(aiRoutes, { prefix: '/api/ai' });
    await app.register(statsRoutes, { prefix: '/api/stats' });

    // Catch-all route for SPA
    app.get('*', async (request, reply) => {
      return reply.sendFile('index.html');
    });

    // Start server
    const address = await app.listen({
      port: config.server.port,
      host: '0.0.0.0',
    });

    logger.info(`Omega Media Server started on ${address}`);
    logger.info('Features enabled:', {
      ai: config.features.ai,
      '8k': config.features['8k'],
      vpn: config.features.vpn,
      plex: config.features.plex,
    });

    // Setup graceful shutdown
    process.on('SIGTERM', async () => {
      logger.info('SIGTERM received, shutting down gracefully...');
      await app.close();
      process.exit(0);
    });

  } catch (error) {
    logger.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start the server
start();