/**
 * Socket.IO Server for Real-time Media Server Communication
 * Provides WebSocket connectivity for real-time updates and control
 */

const express = require('express');
const { createServer } = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const axios = require('axios');
const Docker = require('dockerode');

class MediaSocketServer {
    constructor(port = 3003) {
        this.app = express();
        this.server = createServer(this.app);
        this.io = new Server(this.server, {
            cors: {
                origin: "*",
                methods: ["GET", "POST"]
            },
            transports: ['websocket', 'polling']
        });
        
        this.port = port;
        this.docker = new Docker();
        this.clients = new Map();
        this.serviceStatus = new Map();
        this.monitoringInterval = null;
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupSocketHandlers();
        this.startMonitoring();
    }

    setupMiddleware() {
        this.app.use(cors());
        this.app.use(express.json());
        this.app.use(express.static('public'));

        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                server: 'socket-server',
                clients: this.clients.size,
                timestamp: new Date().toISOString()
            });
        });
    }

    setupRoutes() {
        // API endpoint to get current service status
        this.app.get('/api/services/status', async (req, res) => {
            try {
                const status = await this.getAllServicesStatus();
                res.json({
                    success: true,
                    data: status,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                res.status(500).json({
                    success: false,
                    error: error.message,
                    timestamp: new Date().toISOString()
                });
            }
        });

        // API endpoint to control services
        this.app.post('/api/services/:service/:action', async (req, res) => {
            try {
                const { service, action } = req.params;
                const result = await this.controlService(service, action);
                
                // Broadcast the action to all connected clients
                this.io.emit('service-action', {
                    service,
                    action,
                    result,
                    timestamp: new Date().toISOString()
                });
                
                res.json({
                    success: true,
                    data: result,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                res.status(500).json({
                    success: false,
                    error: error.message,
                    timestamp: new Date().toISOString()
                });
            }
        });

        // API endpoint to get service logs
        this.app.get('/api/services/:service/logs', async (req, res) => {
            try {
                const { service } = req.params;
                const { lines = 100 } = req.query;
                
                const logs = await this.getServiceLogs(service, parseInt(lines));
                res.json({
                    success: true,
                    data: { logs },
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                res.status(500).json({
                    success: false,
                    error: error.message,
                    timestamp: new Date().toISOString()
                });
            }
        });
    }

    setupSocketHandlers() {
        this.io.on('connection', (socket) => {
            console.log(`📡 Client connected: ${socket.id}`);
            
            // Store client information
            this.clients.set(socket.id, {
                socket,
                connectedAt: new Date().toISOString(),
                subscriptions: new Set()
            });

            // Send initial service status
            this.sendServiceStatusUpdate(socket);

            // Handle client requests
            socket.on('subscribe-service-status', () => {
                const client = this.clients.get(socket.id);
                if (client) {
                    client.subscriptions.add('service-status');
                    socket.emit('subscription-confirmed', {
                        type: 'service-status',
                        message: 'Subscribed to service status updates'
                    });
                }
            });

            socket.on('subscribe-logs', (data) => {
                const client = this.clients.get(socket.id);
                if (client) {
                    const service = data.service || 'all';
                    client.subscriptions.add(`logs-${service}`);
                    socket.emit('subscription-confirmed', {
                        type: 'logs',
                        service,
                        message: `Subscribed to logs for ${service}`
                    });
                }
            });

            socket.on('get-service-status', async (data) => {
                try {
                    const { service } = data;
                    let status;
                    
                    if (service) {
                        status = await this.getServiceStatus(service);
                    } else {
                        status = await this.getAllServicesStatus();
                    }
                    
                    socket.emit('service-status-response', {
                        service,
                        status,
                        timestamp: new Date().toISOString()
                    });
                } catch (error) {
                    socket.emit('error', {
                        message: error.message,
                        timestamp: new Date().toISOString()
                    });
                }
            });

            socket.on('control-service', async (data) => {
                try {
                    const { service, action } = data;
                    const result = await this.controlService(service, action);
                    
                    socket.emit('service-control-response', {
                        service,
                        action,
                        result,
                        timestamp: new Date().toISOString()
                    });
                    
                    // Broadcast to all clients
                    this.io.emit('service-action', {
                        service,
                        action,
                        result,
                        timestamp: new Date().toISOString()
                    });
                } catch (error) {
                    socket.emit('error', {
                        message: error.message,
                        service: data.service,
                        action: data.action,
                        timestamp: new Date().toISOString()
                    });
                }
            });

            socket.on('get-logs', async (data) => {
                try {
                    const { service, lines = 100 } = data;
                    const logs = await this.getServiceLogs(service, lines);
                    
                    socket.emit('logs-response', {
                        service,
                        logs,
                        timestamp: new Date().toISOString()
                    });
                } catch (error) {
                    socket.emit('error', {
                        message: error.message,
                        timestamp: new Date().toISOString()
                    });
                }
            });

            socket.on('ping', () => {
                socket.emit('pong', {
                    timestamp: new Date().toISOString()
                });
            });

            socket.on('disconnect', () => {
                console.log(`📱 Client disconnected: ${socket.id}`);
                this.clients.delete(socket.id);
            });
        });
    }

    async getAllServicesStatus() {
        const services = [
            'jellyfin', 'plex', 'emby', 'sonarr', 'radarr', 'lidarr',
            'bazarr', 'prowlarr', 'qbittorrent', 'transmission', 'sabnzbd',
            'overseerr', 'jellyseerr', 'ombi', 'portainer', 'nginx-proxy-manager'
        ];

        const statusPromises = services.map(async (service) => {
            try {
                const status = await this.getServiceStatus(service);
                return { service, ...status };
            } catch (error) {
                return {
                    service,
                    status: 'error',
                    error: error.message,
                    timestamp: new Date().toISOString()
                };
            }
        });

        const results = await Promise.allSettled(statusPromises);
        const serviceStatus = {};

        results.forEach((result) => {
            if (result.status === 'fulfilled') {
                const service = result.value.service;
                serviceStatus[service] = result.value;
            }
        });

        return serviceStatus;
    }

    async getServiceStatus(serviceName) {
        try {
            // First check Docker container status
            const containers = await this.docker.listContainers({ all: true });
            const container = containers.find(c => 
                c.Names.some(name => name.toLowerCase().includes(serviceName.toLowerCase()))
            );

            if (!container) {
                return {
                    status: 'not-found',
                    message: 'Container not found',
                    timestamp: new Date().toISOString()
                };
            }

            const containerStatus = {
                containerId: container.Id.substring(0, 12),
                containerName: container.Names[0]?.replace('/', ''),
                dockerStatus: container.State,
                image: container.Image,
                created: container.Created,
                ports: container.Ports
            };

            // Then check HTTP health if container is running
            if (container.State === 'running') {
                const healthStatus = await this.checkServiceHealth(serviceName);
                return {
                    status: healthStatus.healthy ? 'healthy' : 'unhealthy',
                    ...containerStatus,
                    ...healthStatus,
                    timestamp: new Date().toISOString()
                };
            } else {
                return {
                    status: 'stopped',
                    ...containerStatus,
                    timestamp: new Date().toISOString()
                };
            }
        } catch (error) {
            return {
                status: 'error',
                error: error.message,
                timestamp: new Date().toISOString()
            };
        }
    }

    async checkServiceHealth(serviceName) {
        const serviceEndpoints = {
            jellyfin: { port: 8096, path: '/health' },
            plex: { port: 32400, path: '/identity' },
            emby: { port: 8097, path: '/health' },
            sonarr: { port: 8989, path: '/api/v3/system/status' },
            radarr: { port: 7878, path: '/api/v3/system/status' },
            lidarr: { port: 8686, path: '/api/v1/system/status' },
            bazarr: { port: 6767, path: '/api/system/status' },
            prowlarr: { port: 9696, path: '/api/v1/system/status' },
            qbittorrent: { port: 8080, path: '/api/v2/app/version' },
            transmission: { port: 9091, path: '/transmission/rpc' },
            sabnzbd: { port: 8081, path: '/sabnzbd/api?mode=version' },
            overseerr: { port: 5055, path: '/api/v1/status' },
            jellyseerr: { port: 5055, path: '/api/v1/status' },
            ombi: { port: 3579, path: '/api/v1/status' },
            portainer: { port: 9000, path: '/api/status' },
            'nginx-proxy-manager': { port: 81, path: '/api/nginx/proxy-hosts' }
        };

        const endpoint = serviceEndpoints[serviceName];
        if (!endpoint) {
            return { healthy: false, error: 'Unknown service' };
        }

        try {
            const startTime = Date.now();
            const response = await axios.get(
                `http://localhost:${endpoint.port}${endpoint.path}`,
                { 
                    timeout: 5000,
                    validateStatus: (status) => status < 500
                }
            );
            const responseTime = Date.now() - startTime;

            return {
                healthy: true,
                httpStatus: response.status,
                responseTime,
                lastCheck: new Date().toISOString()
            };
        } catch (error) {
            return {
                healthy: false,
                error: error.code || error.message,
                lastCheck: new Date().toISOString()
            };
        }
    }

    async controlService(serviceName, action) {
        const validActions = ['start', 'stop', 'restart'];
        if (!validActions.includes(action)) {
            throw new Error(`Invalid action: ${action}`);
        }

        try {
            const containers = await this.docker.listContainers({ all: true });
            const container = containers.find(c => 
                c.Names.some(name => name.toLowerCase().includes(serviceName.toLowerCase()))
            );

            if (!container) {
                throw new Error(`Container for service ${serviceName} not found`);
            }

            const dockerContainer = this.docker.getContainer(container.Id);

            switch (action) {
                case 'start':
                    await dockerContainer.start();
                    break;
                case 'stop':
                    await dockerContainer.stop({ t: 10 });
                    break;
                case 'restart':
                    await dockerContainer.restart({ t: 10 });
                    break;
            }

            return {
                success: true,
                message: `Service ${serviceName} ${action} completed`,
                containerId: container.Id.substring(0, 12),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            if (error.statusCode === 304) {
                return {
                    success: true,
                    message: `Service ${serviceName} already in desired state`,
                    timestamp: new Date().toISOString()
                };
            }
            throw error;
        }
    }

    async getServiceLogs(serviceName, lines = 100) {
        try {
            const containers = await this.docker.listContainers({ all: true });
            const container = containers.find(c => 
                c.Names.some(name => name.toLowerCase().includes(serviceName.toLowerCase()))
            );

            if (!container) {
                throw new Error(`Container for service ${serviceName} not found`);
            }

            const dockerContainer = this.docker.getContainer(container.Id);
            const logs = await dockerContainer.logs({
                stdout: true,
                stderr: true,
                tail: lines,
                timestamps: true
            });

            return logs.toString('utf8');
        } catch (error) {
            throw new Error(`Failed to get logs for ${serviceName}: ${error.message}`);
        }
    }

    startMonitoring() {
        // Monitor service status every 30 seconds
        this.monitoringInterval = setInterval(async () => {
            try {
                const currentStatus = await this.getAllServicesStatus();
                
                // Check for status changes
                for (const [service, status] of Object.entries(currentStatus)) {
                    const previousStatus = this.serviceStatus.get(service);
                    
                    if (!previousStatus || previousStatus.status !== status.status) {
                        // Status changed, notify subscribers
                        this.serviceStatus.set(service, status);
                        this.broadcastServiceStatusUpdate(service, status);
                    }
                }
            } catch (error) {
                console.error('Monitoring error:', error.message);
            }
        }, 30000);

        console.log('🔄 Service monitoring started');
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
            console.log('⏹️ Service monitoring stopped');
        }
    }

    sendServiceStatusUpdate(socket) {
        this.getAllServicesStatus().then(status => {
            socket.emit('service-status-update', {
                services: status,
                timestamp: new Date().toISOString()
            });
        }).catch(error => {
            socket.emit('error', {
                message: error.message,
                timestamp: new Date().toISOString()
            });
        });
    }

    broadcastServiceStatusUpdate(service, status) {
        // Send to clients subscribed to service status updates
        for (const [clientId, client] of this.clients) {
            if (client.subscriptions.has('service-status')) {
                client.socket.emit('service-status-change', {
                    service,
                    status,
                    timestamp: new Date().toISOString()
                });
            }
        }
    }

    start() {
        this.server.listen(this.port, () => {
            console.log(`🚀 Socket.IO Media Server running on port ${this.port}`);
            console.log(`🔌 Socket.IO endpoint: ws://localhost:${this.port}`);
            console.log(`📡 REST API: http://localhost:${this.port}/api`);
        });

        // Graceful shutdown
        process.on('SIGTERM', () => this.shutdown());
        process.on('SIGINT', () => this.shutdown());
    }

    shutdown() {
        console.log('🛑 Shutting down Socket.IO server...');
        
        this.stopMonitoring();
        
        // Close all client connections
        this.io.close(() => {
            console.log('✅ Socket.IO server shut down complete');
            process.exit(0);
        });
    }
}

// Start the server if this file is run directly
if (require.main === module) {
    const server = new MediaSocketServer(process.env.SOCKET_PORT || 3003);
    server.start();
}

module.exports = MediaSocketServer;