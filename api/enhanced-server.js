#!/usr/bin/env node

// Compatibility wrapper for legacy enhanced-server.js
// The original implementation contained literal \n tokens and broke parsing.
// This wrapper delegates to the maintained server implementation.

try {
  const MediaServerAPI = require('./server');

  const port = process.env.API_PORT || '3004';
  process.env.API_PORT = port;

  const api = new MediaServerAPI();

  process.on('uncaughtException', (err) => {
  const logger = require('../middleware/logger');
  logger.error('Uncaught Exception in enhanced-server wrapper:', err);
    process.exit(1);
  });
  process.on('unhandledRejection', (reason) => {
  logger.error('Unhandled Rejection in enhanced-server wrapper:', reason);
    process.exit(1);
  });

  process.on('SIGTERM', () => api.shutdown && api.shutdown());
  process.on('SIGINT', () => api.shutdown && api.shutdown());

  api.start().then(() => {
  logger.info(`✅ Enhanced server wrapper running on http://localhost:${port}`);
  }).catch((err) => {
  logger.error('Failed to start enhanced server wrapper:', err);
    process.exit(1);
  });
} catch (err) {
  logger.error('Failed to load server implementation from enhanced-server.js wrapper:', err);
  process.exit(1);
}

