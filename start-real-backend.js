#!/usr/bin/env node

/**
 * Startup Script for Real Media Server Backend
 * Launches the complete, functional backend server
 */

const RealMediaServerAPI = require('./real-backend-server');

console.log(`
🚀 STARTING REAL MEDIA SERVER BACKEND
====================================

This is the COMPLETE, WORKING backend server that replaces the mock-based implementation.

Features:
✅ Docker Service Management (/api/services/*)
✅ Configuration Management (/api/config)
✅ Health Monitoring (/api/health, /api/metrics)
✅ Authentication (/api/auth/login, /api/auth/profile)
✅ WebSocket Support (ws://localhost:3333)
✅ Media Operations (/api/media/*)
✅ Download Management (/api/downloads/*)
✅ Service Integrations (/api/integrations/*)
✅ User Management (/api/users/*)
✅ Notifications (/api/notifications/*)

Default Credentials:
- Username: admin / Password: admin123
- Username: user / Password: user123

Environment Variables (optional):
- API_PORT=3333
- JWT_SECRET=your-secret-key
- ADMIN_PASSWORD=admin123
- DOCKER_PROJECT_PATH=${__dirname}
- NODE_ENV=development

`);

// Set default environment variables if not set
if (!process.env.API_PORT) process.env.API_PORT = '3333';
if (!process.env.JWT_SECRET) process.env.JWT_SECRET = 'media-server-secret-key-change-in-production';
if (!process.env.ADMIN_PASSWORD) process.env.ADMIN_PASSWORD = 'admin123';
if (!process.env.NODE_ENV) process.env.NODE_ENV = 'development';

// Create and start the server
const api = new RealMediaServerAPI();

// Enhanced error handling
process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
    process.exit(1);
});

// Graceful shutdown
process.on('SIGTERM', async () => {
    console.log('SIGTERM received, shutting down gracefully...');
    await api.shutdown();
});

process.on('SIGINT', async () => {
    console.log('SIGINT received, shutting down gracefully...');
    await api.shutdown();
});

// Start the server
api.start().catch((error) => {
    console.error('Failed to start server:', error);
    process.exit(1);
});