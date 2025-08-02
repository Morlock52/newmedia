#!/usr/bin/env node

/**
 * Media Server API Startup Script
 * Production-ready startup with proper error handling and logging
 */

const path = require('path');
const fs = require('fs');

// Load environment variables
const dotenv = require('dotenv');
const envPath = path.join(__dirname, '../.env');

if (fs.existsSync(envPath)) {
    dotenv.config({ path: envPath });
    console.log('✓ Environment variables loaded from .env');
} else {
    console.log('⚠ No .env file found, using default environment');
}

// Validate required environment variables
const requiredEnvVars = [
    'DOCKER_PROJECT_PATH'
];

const missingEnvVars = requiredEnvVars.filter(envVar => !process.env[envVar]);
if (missingEnvVars.length > 0) {
    console.error('❌ Missing required environment variables:', missingEnvVars.join(', '));
    console.error('Please check your .env file or set these variables manually');
    process.exit(1);
}

// Set default values
process.env.DOCKER_PROJECT_PATH = process.env.DOCKER_PROJECT_PATH || path.join(__dirname, '../');
process.env.API_PORT = process.env.API_PORT || '3002';
process.env.NODE_ENV = process.env.NODE_ENV || 'production';
process.env.LOG_LEVEL = process.env.LOG_LEVEL || 'info';

// Validate Docker project path
if (!fs.existsSync(process.env.DOCKER_PROJECT_PATH)) {
    console.error('❌ Docker project path does not exist:', process.env.DOCKER_PROJECT_PATH);
    process.exit(1);
}

// Check for docker-compose.yml
const composePath = path.join(process.env.DOCKER_PROJECT_PATH, 'docker-compose.yml');
if (!fs.existsSync(composePath)) {
    console.error('❌ docker-compose.yml not found at:', composePath);
    process.exit(1);
}

console.log('🚀 Starting Media Server API...');
console.log('📁 Project Path:', process.env.DOCKER_PROJECT_PATH);
console.log('🌐 Port:', process.env.API_PORT);
console.log('🏃 Environment:', process.env.NODE_ENV);
console.log('📝 Log Level:', process.env.LOG_LEVEL);

// Import and start the API server
const MediaServerAPI = require('./server');

async function startServer() {
    try {
        const api = new MediaServerAPI();
        await api.start();
        
        console.log('✅ Media Server API started successfully!');
        console.log(`📚 API Documentation: http://localhost:${process.env.API_PORT}/api/docs`);
        console.log(`🔌 WebSocket: ws://localhost:${process.env.API_PORT}`);
        console.log(`❤️ Health Check: http://localhost:${process.env.API_PORT}/health`);
        
    } catch (error) {
        console.error('❌ Failed to start Media Server API:', error.message);
        
        if (error.message.includes('Docker')) {
            console.error('💡 Make sure Docker and Docker Compose are installed and running');
        }
        
        if (error.message.includes('EADDRINUSE')) {
            console.error(`💡 Port ${process.env.API_PORT} is already in use. Try setting a different API_PORT`);
        }
        
        process.exit(1);
    }
}

// Handle uncaught exceptions
process.on('uncaughtException', (error) => {
    console.error('💥 Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('💥 Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Handle process signals
process.on('SIGINT', () => {
    console.log('\n🛑 Received SIGINT, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
    process.exit(0);
});

// Start the server
startServer();