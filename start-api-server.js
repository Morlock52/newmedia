#!/usr/bin/env node
require('./scripts/console-shim');

/**
 * Start API Server with Dependencies Check
 * Ensures all required services are initialized before starting
 */

const path = require('path');

// Set environment variables
process.env.NODE_ENV = process.env.NODE_ENV || 'development';
process.env.API_PORT = process.env.API_PORT || '3002';
process.env.JWT_SECRET = process.env.JWT_SECRET || 'development-secret-key-change-in-production';
process.env.ADMIN_PASSWORD = process.env.ADMIN_PASSWORD || 'admin123';

console.log('🚀 Starting Media Server API...');
console.log('================================');
console.log(`Environment: ${process.env.NODE_ENV}`);
console.log(`Port: ${process.env.API_PORT}`);
console.log(`Node.js: ${process.version}`);
console.log('');

// Check required dependencies
console.log('📦 Checking dependencies...');

const requiredModules = [
    'express',
    'cors', 
    'helmet',
    'express-rate-limit',
    'ws',
    'joi',
    'axios',
    'js-yaml',
    'winston'
];

const missingModules = [];

requiredModules.forEach(mod => {
    try {
        require.resolve(mod);
        console.log(`✅ ${mod}`);
    } catch (error) {
        console.log(`❌ ${mod} - Missing`);
        missingModules.push(mod);
    }
});

// Check optional modules
const optionalModules = [
    'socket.io',
    'bcryptjs', 
    'jsonwebtoken'
];

console.log('\n📦 Checking optional dependencies...');
optionalModules.forEach(mod => {
    try {
        require.resolve(mod);
        console.log(`✅ ${mod}`);
    } catch (error) {
        console.log(`⚠️  ${mod} - Optional (some features may be limited)`);
    }
});

if (missingModules.length > 0) {
    console.log('\n❌ Missing required dependencies. Run:');
    console.log(`npm install ${missingModules.join(' ')}`);
    process.exit(1);
}

console.log('\n🔧 Initializing services...');

// Create logs directory
const fs = require('fs');
const logsDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logsDir)) {
    fs.mkdirSync(logsDir, { recursive: true });
    console.log('✅ Created logs directory');
}

// Load the API server
try {
    const MediaServerAPI = require('./api/server.js');
    const api = new MediaServerAPI();
    
    console.log('✅ MediaServerAPI loaded');
    console.log('\n🌟 Starting server...\n');
    
    api.start().then(() => {
        console.log('\n🎉 Server started successfully!');
        console.log('📋 Available endpoints:');
        console.log(`   Health: http://localhost:${process.env.API_PORT}/health`);
        console.log(`   System: http://localhost:${process.env.API_PORT}/api/system`);
        console.log(`   Docs:   http://localhost:${process.env.API_PORT}/api/docs`);
        console.log(`   Login:  http://localhost:${process.env.API_PORT}/api/auth/login`);
        console.log('\n🧪 To test the fixes, run:');
        console.log('   node test-backend-fixes.js');
        console.log('\n💡 Default login: admin / admin123');
    }).catch(error => {
        console.error('❌ Failed to start server:', error.message);
        process.exit(1);
    });
    
} catch (error) {
    console.error('❌ Failed to load API server:', error.message);
    console.error('Stack:', error.stack);
    process.exit(1);
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\n🛑 Received SIGINT, shutting down gracefully...');
    process.exit(0);
});

process.on('SIGTERM', () => {
    console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
    process.exit(0);
});