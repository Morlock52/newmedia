#!/usr/bin/env node
require('../scripts/console-shim');

/**
 * Media Server API Startup Script
 * Production-ready startup with proper error handling and logging
 */

const path = require('path');
const fs = require('fs');

const logger = require('../middleware/logger');
// Load environment variables
const dotenv = require('dotenv');
const envPath = path.join(__dirname, '../.env');

if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
    logger.info('✓ Environment variables loaded from .env');
} else {
    logger.warn('⚠ No .env file found, using default environment');
}

// Validate required environment variables
const requiredEnvVars = [
    'DOCKER_PROJECT_PATH'
];

const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);
if (missingEnvVars.length > 0) {
    logger.error('❌ Missing required environment variables: %s', missingEnvVars.join(', '));
    logger.error('Please check your .env file or set these variables manually');
    process.exit(1);
}

// Set default values
process.env.DOCKER_PROJECT_PATH = process.env.DOCKER_PROJECT_PATH || path.join(__dirname, '../');
process.env.API_PORT = process.env.API_PORT || '3002';
process.env.NODE_ENV = process.env.NODE_ENV || 'production';
process.env.LOG_LEVEL = process.env.LOG_LEVEL || 'info';

// Validate Docker project path
if (!fs.existsSync(process.env.DOCKER_PROJECT_PATH)) {
    logger.error('❌ Docker project path does not exist: %s', process.env.DOCKER_PROJECT_PATH);
    process.exit(1);
}

// Check for docker-compose.yml
const composePath = path.join(process.env.DOCKER_PROJECT_PATH, 'docker-compose.yml');
if (!fs.existsSync(composePath)) {
    logger.error('❌ docker-compose.yml not found at: %s', composePath);
    process.exit(1);
}

logger.info('🚀 Starting Media Server API...');
logger.info('📁 Project Path: %s', process.env.DOCKER_PROJECT_PATH);
logger.info('🌐 Port: %s', process.env.API_PORT);
logger.info('🏃 Environment: %s', process.env.NODE_ENV);
logger.info('📝 Log Level: %s', process.env.LOG_LEVEL);

// Import and start the API server
const MediaServerAPI = require('./server');

async function startServer() {
    try {
        const api = new MediaServerAPI();
        await api.start();
        
        logger.info('✅ Media Server API started successfully!');
        logger.info(`📚 API Documentation: http://localhost:${process.env.API_PORT}/api/docs`);
        logger.info(`🔌 WebSocket: ws://localhost:${process.env.API_PORT}`);
        logger.info(`❤️ Health Check: http://localhost:${process.env.API_PORT}/health`);
        
    } catch (error) {
        logger.error('❌ Failed to start Media Server API: %s', error.message);
        
        if (error.message.includes('Docker')) {
            logger.error('💡 Make sure Docker and Docker Compose are installed and running');
        }
        
        if (error.message.includes('EADDRINUSE')) {
            logger.error(`💡 Port ${process.env.API_PORT} is already in use. Try setting a different API_PORT`);
        }
        
        process.exit(1);
    }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    logger.error('💥 Uncaught Exception: %o', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    logger.error('💥 Unhandled Rejection at: %o reason: %o', promise, reason);
    process.exit(1);
});

// Handle process signals
process.on('SIGINT', () => {
    logger.info('\n🛑 Received SIGINT, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    logger.info('\n🛑 Received SIGTERM, shutting down gracefully...');
    process.exit(0);
});

// Start the server
startServer();