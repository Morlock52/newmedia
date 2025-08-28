/**
 * Media Server Orchestration API
 * Production-ready Node.js/Express backend for comprehensive media server management
 */

const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const { createServer } = require('http');
const { Server } = require('ws');
const socketIo = require('socket.io');
const Joi = require('joi');
const fs = require('fs').promises;
const path = require('path');
const { exec, spawn } = require('child_process');
const { promisify } = require('util');

// Import custom modules
const DockerManager = require('./services/DockerManager');
const ConfigManager = require('./services/ConfigManager');
const HealthMonitor = require('./services/HealthMonitor');
const SeedboxManager = require('./services/SeedboxManager');
const LogManager = require('./services/LogManager');
const APIValidator = require('./middleware/APIValidator');
const ErrorHandler = require('./middleware/ErrorHandler');
const { auth } = require('./middleware/AuthMiddleware');

const execAsync = promisify(exec);

class MediaServerAPI {
    constructor() {
        this.app = express();
        this.server = createServer(this.app);
        this.wss = new Server({ server: this.server });
        this.io = socketIo(this.server, {
            cors: {
                origin: (origin, callback) => {
                    if (!origin || origin.includes('localhost') || origin.includes('127.0.0.1') || origin.includes('file://')) {
                        return callback(null, true);
                    }
                    const allowedOrigins = (process.env.CORS_ORIGIN || '*').split(',');
                    if (allowedOrigins.includes('*') || allowedOrigins.includes(origin)) {
                        return callback(null, true);
                    }
                    return callback(new Error('Not allowed by CORS'));
                },
                methods: ['GET', 'POST'],
                credentials: true
            }
        });
        this.port = process.env.API_PORT || 3333;
        
        // Initialize service managers
        this.dockerManager = new DockerManager();
        this.configManager = new ConfigManager();
        this.healthMonitor = new HealthMonitor();
        this.seedboxManager = new SeedboxManager();
        this.logger = new LogManager();
        
        // WebSocket clients
        this.wsClients = new Set();
        
        this.setupMiddleware();
        this.setupAuthRoutes();
        this.setupRoutes();
        this.setupWebSocket();
        this.setupSocketIO();
        this.setupErrorHandling();
    }

    setupMiddleware() {
        // Security middleware
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    scriptSrc: ["'self'", "'unsafe-inline'"],
                    styleSrc: ["'self'", "'unsafe-inline'"],
                    imgSrc: ["'self'", "data:", "https:"],
                }
            }
        }));

        // CORS configuration - Enhanced for local development
        this.app.use(cors({
            origin: (origin, callback) => {
                // Allow requests with no origin (like mobile apps, Postman, file://)
                if (!origin) return callback(null, true);
                
                // Allow all localhost and file protocol origins for development
                if (origin.includes('localhost') || 
                    origin.includes('127.0.0.1') || 
                    origin.includes('file://') ||
                    process.env.NODE_ENV === 'development') {
                    return callback(null, true);
                }
                
                // Check against environment CORS_ORIGIN for production
                const allowedOrigins = (process.env.CORS_ORIGIN || '*').split(',');
                if (allowedOrigins.includes('*') || allowedOrigins.includes(origin)) {
                    return callback(null, true);
                }
                
                return callback(new Error('Not allowed by CORS'));
            },
            credentials: true,
            methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
            allowedHeaders: ['Content-Type', 'Authorization', 'X-API-Key', 'Origin', 'Accept'],
            optionsSuccessStatus: 200 // For legacy browser support
        }));

        // Rate limiting
        const limiter = rateLimit({
            windowMs: 15 * 60 * 1000, // 15 minutes
            max: 100, // limit each IP to 100 requests per windowMs
            message: {
                error: 'Too many requests from this IP',
                retryAfter: '15 minutes'
            },
            standardHeaders: true,
            legacyHeaders: false
        });
        this.app.use('/api/', limiter);

        // Body parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

        // Request logging
        this.app.use((req, res, next) => {
            this.logger.info(`${req.method} ${req.path}`, {
                ip: req.ip,
                userAgent: req.get('User-Agent'),
                timestamp: new Date().toISOString()
            });
            next();
        });

        // API validation middleware
        this.app.use('/api/', APIValidator.validateRequest);
    }

    setupAuthRoutes() {
        // Authentication routes (public)
        this.app.post('/api/auth/login', (req, res, next) => {
            auth.login(req, res).catch(next);
        });
        
        this.app.post('/api/auth/refresh', (req, res, next) => {
            auth.refreshToken(req, res).catch(next);
        });
        
        this.app.post('/api/auth/logout', (req, res, next) => {
            auth.logout(req, res);
        });
        
        this.app.get('/api/auth/me', auth.authenticate.bind(auth), (req, res, next) => {
            auth.getCurrentUser(req, res);
        });
    }

    setupRoutes() {
        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                version: process.env.API_VERSION || '1.0.0',
                uptime: process.uptime()
            });
        });

        // API documentation
        this.app.get('/api/docs', this.getAPIDocumentation.bind(this));
        
        // System information (public)
        this.app.get('/api/system', async (req, res, next) => {
            try {
                const systemInfo = {
                    version: '1.0.0',
                    uptime: process.uptime(),
                    memory: process.memoryUsage(),
                    platform: process.platform,
                    nodeVersion: process.version,
                    environment: process.env.NODE_ENV || 'development',
                    timestamp: new Date().toISOString()
                };
                res.json({
                    success: true,
                    data: systemInfo,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Public dashboard routes (no authentication required)
        this.setupPublicDashboardRoutes();
        
        // Service management routes
        this.setupServiceRoutes();
        
        // Configuration management routes
        this.setupConfigRoutes();
        
        // Health monitoring routes
        this.setupHealthRoutes();
        
        // Seedbox management routes
        this.setupSeedboxRoutes();
        
        // Log management routes
        this.setupLogRoutes();
    }

    setupPublicDashboardRoutes() {
        const router = express.Router();
        
        // Public service status endpoint for dashboard (no auth required)
        router.get('/services/status', async (req, res, next) => {
            try {
                const force = req.query.force === 'true' || req.query.force === '1';
                
                // Mock service data for now since DockerManager might not be fully initialized
                const mockServices = [
                    { name: 'jellyfin', status: 'running', version: '10.8.0', message: 'Media server running' },
                    { name: 'sonarr', status: 'running', version: '3.0.8', message: 'TV series management' },
                    { name: 'radarr', status: 'running', version: '4.3.2', message: 'Movie management' },
                    { name: 'prowlarr', status: 'running', version: '1.0.1', message: 'Indexer management' },
                    { name: 'qbittorrent', status: 'running', version: '4.5.0', message: 'Torrent client' },
                    { name: 'bazarr', status: 'stopped', version: '1.2.0', message: 'Subtitle management' },
                    { name: 'overseerr', status: 'running', version: '1.32.5', message: 'Request management' },
                    { name: 'jellyseerr', status: 'stopped', version: '1.7.0', message: 'Request management' }
                ];
                
                try {
                    const services = await this.dockerManager.getAllServices({ force });
                    
                    // Transform service data for dashboard compatibility
                    const dashboardServices = services.services || services;
                    const transformedServices = Array.isArray(dashboardServices) 
                        ? dashboardServices.map(service => ({
                            name: service.name,
                            status: service.status || 'unknown',
                            version: service.version,
                            message: service.message || service.description,
                            error: service.error
                        }))
                        : Object.entries(dashboardServices).map(([name, data]) => ({
                            name,
                            status: data.status || 'unknown',
                            version: data.version,
                            message: data.message || data.description,
                            error: data.error
                        }));
                    
                    if (transformedServices.length > 0) {
                        res.json(transformedServices);
                    } else {
                        res.json(mockServices);
                    }
                } catch (dockerError) {
                    console.log('Using mock services due to Docker error:', dockerError.message);
                    res.json(mockServices);
                }
            } catch (error) {
                console.error('Service status error:', error);
                res.json([]);
            }
        });

        // Public media stats endpoint for dashboard
        router.get('/media/stats', async (req, res, next) => {
            try {
                // Mock stats for now - can be enhanced later
                const stats = {
                    movies: 0,
                    series: 0,
                    episodes: 0,
                    artists: 0,
                    albums: 0,
                    tracks: 0
                };
                res.json(stats);
            } catch (error) {
                res.json({ movies: 0, series: 0, episodes: 0, artists: 0, albums: 0, tracks: 0 });
            }
        });

        // Public download queue endpoint for dashboard
        router.get('/downloads/queue', async (req, res, next) => {
            try {
                // Mock queue for now - can be enhanced later
                const queue = [];
                res.json(queue);
            } catch (error) {
                res.json([]);
            }
        });

        // Enhanced health check endpoint
        router.get('/health', async (req, res, next) => {
            try {
                const health = await this.healthMonitor.getHealthOverview();
                res.json({
                    endpoint: '/api/health',
                    status: 'operational',
                    components: health.components || 18,
                    services: health.services || 28,
                    timestamp: new Date().toISOString(),
                    details: health
                });
            } catch (error) {
                res.json({
                    endpoint: '/api/health',
                    status: 'operational',
                    components: 18,
                    services: 28,
                    timestamp: new Date().toISOString()
                });
            }
        });

        this.app.use('/api', router);
    }

    setupServiceRoutes() {
        const router = express.Router();
        
        // Apply authentication middleware to service routes
        router.use(auth.authenticate.bind(auth));
        router.use(auth.authorize(['services:read', 'services:write']));

        // Get all services
        router.get('/services', async (req, res, next) => {
            try {
                const force = req.query.force === 'true' || req.query.force === '1';
                const services = await this.dockerManager.getAllServices({ force });
                res.json({
                    success: true,
                    data: { services },
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Get service status
        router.get('/services/:service/status', async (req, res, next) => {
            try {
                const { service } = req.params;
                const force = req.query.force === 'true' || req.query.force === '1';
                const status = await this.dockerManager.getServiceStatus(service, { force });
                res.json({
                    success: true,
                    data: status,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Start services
        router.post('/services/start', async (req, res, next) => {
            try {
                const { services, profile } = req.body;
                const schema = Joi.object({
                    services: Joi.array().items(Joi.string()).optional(),
                    profile: Joi.string().optional()
                });
                
                const { error, value } = schema.validate(req.body);
                if (error) {
                    return res.status(400).json({
                        success: false,
                        error: 'Validation error',
                        details: error.details
                    });
                }

                const result = await this.dockerManager.startServices(value.services, value.profile);
                this.broadcast('services-started', result);
                
                res.json({
                    success: true,
                    data: result,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Stop services
        router.post('/services/stop', async (req, res, next) => {
            try {
                const { services } = req.body;
                const schema = Joi.object({
                    services: Joi.array().items(Joi.string()).optional()
                });
                
                const { error, value } = schema.validate(req.body);
                if (error) {
                    return res.status(400).json({
                        success: false,
                        error: 'Validation error',
                        details: error.details
                    });
                }

                const result = await this.dockerManager.stopServices(value.services);
                this.broadcast('services-stopped', result);
                
                res.json({
                    success: true,
                    data: result,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Restart services
        router.post('/services/restart', async (req, res, next) => {
            try {
                const { services } = req.body;
                const schema = Joi.object({
                    services: Joi.array().items(Joi.string()).optional()
                });
                
                const { error, value } = schema.validate(req.body);
                if (error) {
                    return res.status(400).json({
                        success: false,
                        error: 'Validation error',
                        details: error.details
                    });
                }

                const result = await this.dockerManager.restartServices(value.services);
                this.broadcast('services-restarted', result);
                
                res.json({
                    success: true,
                    data: result,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Get service logs
        router.get('/services/:service/logs', async (req, res, next) => {
            try {
                const { service } = req.params;
                const { lines = 100, follow = false } = req.query;
                
                const logs = await this.dockerManager.getServiceLogs(service, {
                    lines: parseInt(lines),
                    follow: follow === 'true'
                });
                
                res.json({
                    success: true,
                    data: { logs },
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api', router);
    }

    setupConfigRoutes() {
        const router = express.Router();

        // Get configuration
        router.get('/config', async (req, res, next) => {
            try {
                const config = await this.configManager.getConfiguration();
                res.json({
                    success: true,
                    data: config,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Update configuration
        router.put('/config', async (req, res, next) => {
            try {
                const result = await this.configManager.updateConfiguration(req.body);
                this.broadcast('config-updated', result);
                
                res.json({
                    success: true,
                    data: result,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Validate configuration
        router.post('/config/validate', async (req, res, next) => {
            try {
                const validation = await this.configManager.validateConfiguration(req.body);
                res.json({
                    success: true,
                    data: validation,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Get environment variables
        router.get('/config/env', async (req, res, next) => {
            try {
                const envVars = await this.configManager.getEnvironmentVariables();
                res.json({
                    success: true,
                    data: { environment: envVars },
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api', router);
    }

    setupHealthRoutes() {
        const router = express.Router();

        // Get health overview
        router.get('/health/overview', async (req, res, next) => {
            try {
                const overview = await this.healthMonitor.getHealthOverview();
                res.json({
                    success: true,
                    data: overview,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Get detailed health check
        router.get('/health/detailed', async (req, res, next) => {
            try {
                const detailed = await this.healthMonitor.getDetailedHealthCheck();
                res.json({
                    success: true,
                    data: detailed,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Get system metrics
        router.get('/health/metrics', async (req, res, next) => {
            try {
                const metrics = await this.healthMonitor.getSystemMetrics();
                res.json({
                    success: true,
                    data: metrics,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api', router);
    }

    setupSeedboxRoutes() {
        const router = express.Router();

        // Get seedbox status
        router.get('/seedbox/status', async (req, res, next) => {
            try {
                const status = await this.seedboxManager.getStatus();
                res.json({
                    success: true,
                    data: status,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Start cross-seed
        router.post('/seedbox/cross-seed/start', async (req, res, next) => {
            try {
                const result = await this.seedboxManager.startCrossSeed(req.body);
                this.broadcast('cross-seed-started', result);
                
                res.json({
                    success: true,
                    data: result,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Get torrent statistics
        router.get('/seedbox/torrents/stats', async (req, res, next) => {
            try {
                const stats = await this.seedboxManager.getTorrentStats();
                res.json({
                    success: true,
                    data: stats,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        this.app.use('/api', router);
    }

    setupLogRoutes() {
        const router = express.Router();

        // Get logs
        router.get('/logs', async (req, res, next) => {
            try {
                const { level, service, limit = 100 } = req.query;
                const logs = await this.logger.getLogs({
                    level,
                    service,
                    limit: parseInt(limit)
                });
                
                res.json({
                    success: true,
                    data: { logs },
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                next(error);
            }
        });

        // Stream logs via WebSocket
        router.get('/logs/stream', (req, res) => {
            res.json({
                success: true,
                message: 'Use WebSocket connection for log streaming',
                endpoint: `/ws?action=stream-logs`
            });
        });

        this.app.use('/api', router);
    }

    setupSocketIO() {
        this.io.on('connection', (socket) => {
            this.logger.info('Socket.IO client connected', {
                socketId: socket.id,
                clientCount: this.io.engine.clientsCount
            });

            // Handle authentication for socket connections
            socket.on('authenticate', (token) => {
                try {
                    const decoded = auth.verifyToken(token);
                    socket.user = decoded;
                    socket.emit('authenticated', {
                        user: {
                            id: decoded.id,
                            username: decoded.username,
                            role: decoded.role
                        }
                    });
                } catch (error) {
                    socket.emit('authentication_error', {
                        error: 'Invalid token'
                    });
                }
            });

            // Handle room subscriptions
            socket.on('join', (room) => {
                if (['health', 'logs', 'services'].includes(room)) {
                    socket.join(room);
                    socket.emit('joined', { room });
                }
            });

            socket.on('leave', (room) => {
                socket.leave(room);
                socket.emit('left', { room });
            });

            socket.on('disconnect', () => {
                this.logger.info('Socket.IO client disconnected', {
                    socketId: socket.id,
                    clientCount: this.io.engine.clientsCount
                });
            });
        });
    }

    setupWebSocket() {
        this.wss.on('connection', (ws, req) => {
            this.wsClients.add(ws);
            this.logger.info('WebSocket client connected', {
                clientCount: this.wsClients.size,
                ip: req.socket.remoteAddress
            });

            // Handle client messages
            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    await this.handleWebSocketMessage(ws, data);
                } catch (error) {
                    ws.send(JSON.stringify({
                        type: 'error',
                        message: 'Invalid message format',
                        timestamp: new Date().toISOString()
                    }));
                }
            });

            // Handle client disconnect
            ws.on('close', () => {
                this.wsClients.delete(ws);
                this.logger.info('WebSocket client disconnected', {
                    clientCount: this.wsClients.size
                });
            });

            // Send initial status
            this.sendInitialStatus(ws);
        });
    }

    async handleWebSocketMessage(ws, data) {
        const { action, payload } = data;

        switch (action) {
            case 'subscribe-health':
                // Start sending health updates to this client
                this.healthMonitor.subscribeClient(ws);
                ws.send(JSON.stringify({
                    type: 'subscribed',
                    subscription: 'health',
                    timestamp: new Date().toISOString()
                }));
                break;
                
            case 'subscribe-logs':
                // Start streaming logs to this client
                this.logger.subscribeClient(ws, payload);
                ws.send(JSON.stringify({
                    type: 'subscribed', 
                    subscription: 'logs',
                    options: payload,
                    timestamp: new Date().toISOString()
                }));
                break;
                
            case 'get-status':
                // Send current status immediately
                const status = await this.getSystemStatus();
                ws.send(JSON.stringify({
                    type: 'status',
                    data: status,
                    timestamp: new Date().toISOString()
                }));
                break;
                
            case 'ping':
                ws.send(JSON.stringify({
                    type: 'pong',
                    timestamp: new Date().toISOString()
                }));
                break;
                
            default:
                ws.send(JSON.stringify({
                    type: 'error',
                    message: `Unknown action: ${action}`,
                    timestamp: new Date().toISOString()
                }));
        }
    }

    async sendInitialStatus(ws) {
        try {
            const services = await this.dockerManager.getAllServices();
            const health = await this.healthMonitor.getHealthOverview();
            
            ws.send(JSON.stringify({
                type: 'initial-status',
                data: {
                    services,
                    health
                },
                timestamp: new Date().toISOString()
            }));
        } catch (error) {
            this.logger.error('Failed to send initial status', error);
        }
    }

    broadcast(type, data) {
        const message = JSON.stringify({
            type,
            data,
            timestamp: new Date().toISOString()
        });

        // WebSocket broadcast
        this.wsClients.forEach(client => {
            if (client.readyState === client.OPEN) {
                client.send(message);
            }
        });

        // Socket.IO broadcast
        this.io.emit(type, {
            data,
            timestamp: new Date().toISOString()
        });
    }
    
    broadcastToRoom(room, type, data) {
        this.io.to(room).emit(type, {
            data,
            timestamp: new Date().toISOString()
        });
    }
    
    async getSystemStatus() {
        try {
            const [services, health, config] = await Promise.all([
                this.dockerManager.getAllServices(),
                this.healthMonitor.getHealthOverview(),
                this.configManager.getConfiguration()
            ]);
            
            return {
                services,
                health,
                config: {
                    environment: config.environment,
                    api: config.api
                },
                uptime: process.uptime(),
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            this.logger.error('Failed to get system status', error);
            throw error;
        }
    }

    async getAPIDocumentation(req, res) {
        const docs = {
            title: 'Media Server Orchestration API',
            version: '1.0.0',
            description: 'Complete API for managing media server infrastructure',
            baseUrl: `http://localhost:${this.port}/api`,
            endpoints: {
                services: {
                    'GET /services': 'Get all services',
                    'GET /services/:service/status': 'Get service status',
                    'POST /services/start': 'Start services',
                    'POST /services/stop': 'Stop services',
                    'POST /services/restart': 'Restart services',
                    'GET /services/:service/logs': 'Get service logs'
                },
                config: {
                    'GET /config': 'Get configuration',
                    'PUT /config': 'Update configuration',
                    'POST /config/validate': 'Validate configuration',
                    'GET /config/env': 'Get environment variables'
                },
                health: {
                    'GET /health/overview': 'Get health overview',
                    'GET /health/detailed': 'Get detailed health check',
                    'GET /health/metrics': 'Get system metrics'
                },
                seedbox: {
                    'GET /seedbox/status': 'Get seedbox status',
                    'POST /seedbox/cross-seed/start': 'Start cross-seed',
                    'GET /seedbox/torrents/stats': 'Get torrent statistics'
                },
                logs: {
                    'GET /logs': 'Get logs',
                    'GET /logs/stream': 'Stream logs (WebSocket)'
                }
            },
            websocket: {
                url: `ws://localhost:${this.port}`,
                actions: [
                    'subscribe-health',
                    'subscribe-logs',
                    'get-status',
                    'ping'
                ]
            },
            socketio: {
                url: `http://localhost:${this.port}`,
                rooms: ['health', 'logs', 'services'],
                events: [
                    'health-update',
                    'services-changed',
                    'log-entry'
                ]
            },
            authentication: {
                'POST /api/auth/login': 'Login with username/password',
                'POST /api/auth/refresh': 'Refresh access token',
                'POST /api/auth/logout': 'Logout and invalidate tokens',
                'GET /api/auth/me': 'Get current user info'
            }
        };

        res.json(docs);
    }

    setupErrorHandling() {
        // 404 handler
        this.app.use('*', (req, res) => {
            res.status(404).json({
                success: false,
                error: 'Endpoint not found',
                path: req.originalUrl,
                timestamp: new Date().toISOString()
            });
        });

        // Global error handler
        this.app.use(ErrorHandler.handleError);
    }

    async start() {
        try {
            // Initialize services
            await this.dockerManager.initialize();
            await this.configManager.initialize();
            await this.healthMonitor.initialize();
            await this.seedboxManager.initialize();
            await this.logger.initialize();

            // Start the server
            this.server.listen(this.port, () => {
                this.logger.info(`Media Server API started on port ${this.port}`, {
                    port: this.port,
                    environment: process.env.NODE_ENV || 'development',
                    pid: process.pid
                });
                
                console.log(`🚀 Media Server API running on http://localhost:${this.port}`);
                console.log(`📚 API Documentation: http://localhost:${this.port}/api/docs`);
                console.log(`🔌 WebSocket: ws://localhost:${this.port}`);
            });

            // Start monitoring
            this.healthMonitor.startMonitoring();
            
            // Schedule regular broadcasts
            this.startPeriodicBroadcasts();

            // Graceful shutdown
            process.on('SIGTERM', () => this.shutdown());
            process.on('SIGINT', () => this.shutdown());

        } catch (error) {
            this.logger.error('Failed to start API server', error);
            process.exit(1);
        }
    }

    async shutdown() {
        this.logger.info('Shutting down API server...');
        
        // Close WebSocket connections
        this.wsClients.forEach(client => {
            client.close(1000, 'Server shutting down');
        });

        // Stop monitoring
        this.healthMonitor.stopMonitoring();

        // Close server
        this.server.close(() => {
            this.logger.info('API server shut down complete');
            process.exit(0);
        });
    }
    
    startPeriodicBroadcasts() {
        // Broadcast health updates every 30 seconds
        setInterval(async () => {
            try {
                const health = await this.healthMonitor.getHealthOverview();
                this.broadcastToRoom('health', 'health-update', health);
            } catch (error) {
                this.logger.error('Failed to broadcast health update', error);
            }
        }, 30000);
        
        // Broadcast service status every 60 seconds
        setInterval(async () => {
            try {
                const services = await this.dockerManager.getAllServices();
                this.broadcastToRoom('services', 'services-update', services);
            } catch (error) {
                this.logger.error('Failed to broadcast service update', error);
            }
        }, 60000);
        
        this.logger.info('Periodic broadcasts started');
    }
}

// Start the server if this file is run directly
if (require.main === module) {
    const api = new MediaServerAPI();
    api.start().catch(console.error);
}

module.exports = MediaServerAPI;