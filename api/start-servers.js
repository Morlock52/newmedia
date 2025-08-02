#!/usr/bin/env node

/**
 * Development Server Startup Script
 * Starts all API servers with proper error handling and health checks
 */

const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs').promises;

class ServerManager {
    constructor() {
        this.servers = new Map();
        this.healthCheckInterval = null;
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const colors = {
            info: '\x1b[36m',    // Cyan
            success: '\x1b[32m', // Green
            error: '\x1b[31m',   // Red
            warning: '\x1b[33m', // Yellow
            reset: '\x1b[0m'     // Reset
        };
        
        console.log(`${colors[type]}[${timestamp}] ${message}${colors.reset}`);
    }

    async startServer(name, scriptPath, port, env = {}) {
        this.log(`Starting ${name} on port ${port}...`, 'info');
        
        const server = spawn('node', [scriptPath], {
            env: { 
                ...process.env, 
                PORT: port.toString(),
                NODE_ENV: 'development',
                ...env 
            },
            stdio: 'pipe'
        });

        // Store server reference
        this.servers.set(name, {
            process: server,
            port,
            status: 'starting',
            startTime: Date.now()
        });

        // Handle server output
        server.stdout.on('data', (data) => {
            const output = data.toString().trim();
            if (output) {
                this.log(`[${name}] ${output}`, 'info');
                
                // Check for startup success
                if (output.includes(`running on port ${port}`) || 
                    output.includes(`listening on port ${port}`) ||
                    output.includes(`server running`)) {
                    
                    const serverInfo = this.servers.get(name);
                    if (serverInfo) {
                        serverInfo.status = 'running';
                        this.log(`✅ ${name} started successfully`, 'success');
                    }
                }
            }
        });

        server.stderr.on('data', (data) => {
            const error = data.toString().trim();
            if (error && !error.includes('ExperimentalWarning')) {
                this.log(`[${name}] ERROR: ${error}`, 'error');
                
                // Mark as failed if critical error
                if (error.includes('EADDRINUSE') || 
                    error.includes('Cannot find module') ||
                    error.includes('SyntaxError')) {
                    
                    const serverInfo = this.servers.get(name);
                    if (serverInfo) {
                        serverInfo.status = 'failed';
                    }
                }
            }
        });

        server.on('close', (code) => {
            const serverInfo = this.servers.get(name);
            if (serverInfo) {
                serverInfo.status = 'stopped';
                this.log(`${name} exited with code ${code}`, code === 0 ? 'info' : 'error');
            }
        });

        server.on('error', (error) => {
            this.log(`Failed to start ${name}: ${error.message}`, 'error');
            const serverInfo = this.servers.get(name);
            if (serverInfo) {
                serverInfo.status = 'failed';
            }
        });

        return server;
    }

    async checkHealth(port) {
        try {
            const response = await fetch(`http://localhost:${port}/api/health`, {
                signal: AbortSignal.timeout(5000)
            });
            return response.ok;
        } catch (error) {
            return false;
        }
    }

    async startHealthChecks() {
        this.log('Starting health check monitoring...', 'info');
        
        this.healthCheckInterval = setInterval(async () => {
            for (const [name, serverInfo] of this.servers) {
                if (serverInfo.status === 'running') {
                    const isHealthy = await this.checkHealth(serverInfo.port);
                    if (!isHealthy && serverInfo.status === 'running') {
                        this.log(`⚠️  ${name} health check failed`, 'warning');
                        serverInfo.status = 'unhealthy';
                    } else if (isHealthy && serverInfo.status === 'unhealthy') {
                        this.log(`✅ ${name} health restored`, 'success');
                        serverInfo.status = 'running';
                    }
                }
            }
        }, 30000); // Check every 30 seconds
    }

    async waitForServers(timeout = 30000) {
        this.log('Waiting for servers to start...', 'info');
        
        return new Promise((resolve, reject) => {
            const startTime = Date.now();
            
            const checkInterval = setInterval(() => {
                const allServers = Array.from(this.servers.values());
                const runningServers = allServers.filter(s => s.status === 'running');
                const failedServers = allServers.filter(s => s.status === 'failed');
                
                // Check if all servers are running
                if (runningServers.length === allServers.length) {
                    clearInterval(checkInterval);
                    this.log(`🎉 All ${allServers.length} servers started successfully!`, 'success');
                    resolve();
                    return;
                }
                
                // Check for failures
                if (failedServers.length > 0) {
                    clearInterval(checkInterval);
                    const failedNames = failedServers.map(s => 
                        Array.from(this.servers.entries())
                            .find(([, info]) => info === s)?.[0]
                    );
                    reject(new Error(`Failed to start servers: ${failedNames.join(', ')}`));
                    return;
                }
                
                // Check timeout
                if (Date.now() - startTime > timeout) {
                    clearInterval(checkInterval);
                    const stillStarting = allServers.filter(s => s.status === 'starting');
                    const startingNames = stillStarting.map(s => 
                        Array.from(this.servers.entries())
                            .find(([, info]) => info === s)?.[0]
                    );
                    reject(new Error(`Timeout waiting for servers: ${startingNames.join(', ')}`));
                    return;
                }
            }, 1000);
        });
    }

    async shutdown() {
        this.log('Shutting down servers...', 'info');
        
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }
        
        // Gracefully stop all servers
        for (const [name, serverInfo] of this.servers) {
            if (serverInfo.process && !serverInfo.process.killed) {
                this.log(`Stopping ${name}...`, 'info');
                serverInfo.process.kill('SIGTERM');
                
                // Force kill after 5 seconds
                setTimeout(() => {
                    if (!serverInfo.process.killed) {
                        serverInfo.process.kill('SIGKILL');
                    }
                }, 5000);
            }
        }
        
        this.log('All servers shut down', 'info');
    }

    getStatus() {
        const status = {};
        for (const [name, serverInfo] of this.servers) {
            status[name] = {
                status: serverInfo.status,
                port: serverInfo.port,
                uptime: serverInfo.status === 'running' ? Date.now() - serverInfo.startTime : 0
            };
        }
        return status;
    }

    printStatus() {
        this.log('📊 Server Status:', 'info');
        console.log('='.repeat(50));
        
        for (const [name, info] of Object.entries(this.getStatus())) {
            const statusEmoji = {
                'running': '🟢',
                'starting': '🟡',
                'failed': '🔴',
                'stopped': '⚫',
                'unhealthy': '🟠'
            }[info.status] || '❓';
            
            const uptime = info.uptime > 0 ? `(${Math.round(info.uptime / 1000)}s)` : '';
            console.log(`  ${statusEmoji} ${name.padEnd(20)} Port: ${info.port.toString().padEnd(5)} Status: ${info.status} ${uptime}`);
        }
        console.log('='.repeat(50));
    }
}

// Polyfill fetch for Node.js
if (typeof fetch === 'undefined') {
    global.fetch = require('node-fetch');
}

async function main() {
    const manager = new ServerManager();
    
    // Handle graceful shutdown
    process.on('SIGINT', async () => {
        console.log('\n🛑 Received SIGINT, shutting down gracefully...');
        await manager.shutdown();
        process.exit(0);
    });
    
    process.on('SIGTERM', async () => {
        console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
        await manager.shutdown();
        process.exit(0);
    });
    
    try {
        // Check if required files exist
        const chatbotPath = path.join(__dirname, 'chatbot-server.js');
        const configServerPath = path.join(__dirname, '..', 'config-server', 'server.js');
        
        try {
            await fs.access(chatbotPath);
        } catch (error) {
            throw new Error(`Chatbot server not found: ${chatbotPath}`);
        }
        
        try {
            await fs.access(configServerPath);
        } catch (error) {
            manager.log(`Config server not found: ${configServerPath} (optional)`, 'warning');
        }
        
        // Start servers
        manager.log('🚀 Starting development servers...', 'info');
        
        // Start chatbot API server
        await manager.startServer(
            'Chatbot API',
            chatbotPath,
            3001,
            { 
                OPENAI_API_KEY: process.env.OPENAI_API_KEY || 'demo-key-for-testing'
            }
        );
        
        // Start config server (if available)
        try {
            await fs.access(configServerPath);
            await manager.startServer(
                'Config Server',
                configServerPath,
                3000
            );
        } catch (error) {
            manager.log('Config server not available, skipping...', 'warning');
        }
        
        // Wait for servers to start
        try {
            await manager.waitForServers(15000); // 15 second timeout
            await manager.startHealthChecks();
            
            manager.printStatus();
            
            manager.log('', 'info');
            manager.log('📱 Access your services:', 'info');
            manager.log('  • Chatbot API Health: http://localhost:3001/api/health', 'info');
            manager.log('  • Config Server Health: http://localhost:3000/api/health', 'info');
            manager.log('  • Media Assistant: Open holographic-dashboard/media-assistant.html', 'info');
            manager.log('  • Service Dashboard: Open holographic-dashboard/service-dashboard.html', 'info');
            manager.log('', 'info');
            manager.log('Press Ctrl+C to stop all servers', 'info');
            
            // Keep the process alive
            setInterval(() => {
                // Print status every 5 minutes
                if (Date.now() % 300000 < 1000) {
                    manager.printStatus();
                }
            }, 1000);
            
        } catch (error) {
            manager.log(`Failed to start servers: ${error.message}`, 'error');
            await manager.shutdown();
            process.exit(1);
        }
        
    } catch (error) {
        manager.log(`Startup error: ${error.message}`, 'error');
        process.exit(1);
    }
}

// Run if this file is executed directly
if (require.main === module) {
    main().catch(error => {
        console.error('Server manager error:', error);
        process.exit(1);
    });
}

module.exports = ServerManager;