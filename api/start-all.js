#!/usr/bin/env node
const logger = require('../middleware/logger.js');
require('../scripts/console-shim');

/**
 * Start All API Services
 * Starts both the main API server and Socket.IO server
 */

const { spawn } = require('child_process');
const path = require('path');

class APIServiceManager {
    constructor() {
        this.processes = new Map();
        this.baseDir = __dirname;
    }

    startService(name, scriptPath, port) {
        logger.info(`🚀 Starting ${name} on port ${port}...`);
        
        const process = spawn('node', [scriptPath], {
            cwd: path.dirname(scriptPath),
            stdio: 'inherit',
            env: {
                ...process.env,
                API_PORT: port,
                NODE_ENV: process.env.NODE_ENV || 'development'
            }
        });

        process.on('error', (error) => {
            logger.error(`❌ ${name} failed to start:`, error.message);
        });

        process.on('exit', (code, signal) => {
            logger.info(`📱 ${name} exited with code ${code} and signal ${signal}`);
            this.processes.delete(name);
        });

        this.processes.set(name, {
            process,
            port,
            startTime: new Date()
        });

        return process;
    }

    async start() {
        logger.info('🎬 Starting Media Server API Services...\n');

        // Start Main API Server
        this.startService(
            'Main API Server',
            path.join(this.baseDir, 'server.js'),
            3004
        );

        // Wait a bit before starting Socket server
        await new Promise(resolve => setTimeout(resolve, 2000));

        // Start Socket.IO Server
        this.startService(
            'Socket.IO Server',
            path.join(this.baseDir, 'socket-server.js'),
            3003
        );

        // Display service status
        setTimeout(() => {
            this.displayStatus();
        }, 5000);

        // Handle graceful shutdown
        process.on('SIGINT', () => this.shutdown());
        process.on('SIGTERM', () => this.shutdown());
    }

    displayStatus() {
        logger.info('\n📊 API Services Status:');
        logger.info('─'.repeat(50));
        
        for (const [name, service] of this.processes) {
            const uptime = Math.floor((Date.now() - service.startTime.getTime()) / 1000);
            logger.info(`✅ ${name.padEnd(25)} | Port: ${service.port} | Uptime: ${uptime}s`);
        }
        
        logger.info('─'.repeat(50));
        logger.info('🌐 Available Endpoints:');
        logger.info('  📚 API Docs:     http://localhost:3004/api/docs');
        logger.info('  🏥 Health:       http://localhost:3004/health');
        logger.info('  🔌 Socket.IO:    ws://localhost:3003');
        logger.info('  📡 Socket API:   http://localhost:3003/api/services/status');
        logger.info('\n💡 Use Ctrl+C to stop all services');
    }

    shutdown() {
        logger.info('\n🛑 Shutting down all API services...');
        
        for (const [name, service] of this.processes) {
            logger.info(`📱 Stopping ${name}...`);
            service.process.kill('SIGTERM');
        }

        setTimeout(() => {
            logger.info('✅ All API services stopped');
            process.exit(0);
        }, 3000);
    }

    async checkHealth() {
        const axios = require('axios');
        const healthChecks = [
            { name: 'Enhanced API', url: 'http://localhost:3004/health' },
            { name: 'Socket.IO API', url: 'http://localhost:3003/health' }
        ];

        logger.info('\n🏥 Health Check Results:');
        logger.info('─'.repeat(40));

        for (const check of healthChecks) {
            try {
                const response = await axios.get(check.url, { timeout: 5000 });
                const status = response.data.status === 'healthy' ? '✅' : '⚠️';
                logger.info(`${status} ${check.name.padEnd(20)} | ${response.data.status}`);
            } catch (error) {
                logger.info(`❌ ${check.name.padEnd(20)} | Unreachable`);
            }
        }
        logger.info('─'.repeat(40));
    }
}

// CLI commands
const command = process.argv[2];
const manager = new APIServiceManager();

switch (command) {
    case 'start':
        manager.start();
        break;
    case 'health':
        manager.checkHealth();
        break;
    case 'status':
        manager.displayStatus();
        break;
    default:
        logger.info('📖 Available commands:');
        logger.info('  node start-all.js start   - Start all API services');
        logger.info('  node start-all.js health  - Check service health');
        logger.info('  node start-all.js status  - Show service status');
        break;
}

module.exports = APIServiceManager;
