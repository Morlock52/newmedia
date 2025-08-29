// Performance-Optimized Express API Server
// Implements advanced caching, compression, and optimization strategies

const express = require('express');
const compression = require('compression');
const helmet = require('helmet');
const rateLimit = require('express-rate-limit');
const cors = require('cors');
const cluster = require('cluster');
const os = require('os');
const { promisify } = require('util');
const fs = require('fs');
const path = require('path');

const winston = require('winston');

// Performance optimization modules
const LRU = require('lru-cache');
const NodeCache = require('node-cache');
const redis = require('redis');

const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.simple()
  ),
  transports: [new winston.transports.Console()]
});

class PerformanceOptimizedAPI {
    constructor(options = {}) {
        this.options = {
            port: process.env.PORT || 3004,
            enableClustering: process.env.NODE_ENV === 'production',
            cacheEnabled: true,
            redisUrl: process.env.REDIS_URL,
            compressionLevel: 6,
            rateLimitWindow: 15 * 60 * 1000, // 15 minutes
            rateLimitMax: 1000, // requests per window
            ...options
        };

        this.app = express();
        this.cache = new LRU({
            max: 10000,
            ttl: 5 * 60 * 1000 // 5 minutes
        });
        
        this.memoryCache = new NodeCache({
            stdTTL: 300, // 5 minutes
            checkperiod: 60, // check for expired keys every minute
            useClones: false
        });

        this.redisClient = null;
        this.performanceMetrics = new Map();
        this.requestStats = {
            total: 0,
            errors: 0,
            averageResponseTime: 0,
            lastReset: Date.now()
        };

        this.init();
    }

    async init() {
        logger.info('🚀 Initializing Performance-Optimized API Server...');

        // Initialize Redis if available
        await this.initRedis();

        // Setup middleware
        this.setupMiddleware();

        // Setup routes
        this.setupRoutes();

        // Setup performance monitoring
        this.setupPerformanceMonitoring();

        // Setup error handling
        this.setupErrorHandling();

        logger.info('✅ API Server initialized successfully');
    }

    async initRedis() {
        if (this.options.redisUrl) {
            try {
                this.redisClient = redis.createClient({
                    url: this.options.redisUrl
                });

                await this.redisClient.connect();
                logger.info('✅ Redis connected for caching');
            } catch (error) {
                logger.warn('⚠️  Redis connection failed, using memory cache:', error.message);
            }
        }
    }

    setupMiddleware() {
        const self = this;
        // Security headers
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    styleSrc: ["'self'", "'unsafe-inline'"],
                    scriptSrc: ["'self'", "'unsafe-eval'"],
                    connectSrc: ["'self'", "http://localhost:*"],
                    imgSrc: ["'self'", "data:", "blob:"],
                    fontSrc: ["'self'", "data:"]
                }
            },
            crossOriginEmbedderPolicy: false
        }));

        // CORS with optimized settings
        this.app.use(cors({
            origin: (origin, callback) => {
                // Allow requests from localhost on any port
                if (!origin || origin.match(/^http:\/\/localhost:\d+$/)) {
                    callback(null, true);
                } else {
                    callback(new Error('Not allowed by CORS'));
                }
            },
            credentials: true,
            optionsSuccessStatus: 200,
            maxAge: 86400 // Cache preflight for 24 hours
        }));

        // Advanced compression
        this.app.use(compression({
            level: this.options.compressionLevel,
            threshold: 1024, // Only compress if > 1KB
            filter: (req, res) => {
                // Don't compress if response is already compressed
                if (res.getHeader('content-encoding')) {
                    return false;
                }
                // Compress text-based responses
                return compression.filter(req, res);
            }
        }));

        // Request parsing with limits
        this.app.use(express.json({
            limit: '10mb',
            type: ['application/json', 'application/vnd.api+json']
        }));

        this.app.use(express.urlencoded({
            extended: true,
            limit: '10mb',
            parameterLimit: 1000
        }));

        // Rate limiting with Redis store if available
        const rateLimitStore = this.redisClient ? 
            new (require('rate-limit-redis'))({
                client: this.redisClient,
                prefix: 'rl:'
            }) : undefined;

        this.app.use(rateLimit({
            windowMs: this.options.rateLimitWindow,
            max: this.options.rateLimitMax,
            store: rateLimitStore,
            message: {
                error: 'Too many requests',
                retryAfter: Math.ceil(this.options.rateLimitWindow / 1000)
            },
            standardHeaders: true,
            legacyHeaders: false,
            handler: (req, res) => {
                this.requestStats.errors++;
                res.status(429).json({
                    error: 'Rate limit exceeded',
                    retryAfter: Math.ceil(this.options.rateLimitWindow / 1000)
                });
            }
        }));

        this.app.use((req, res, next) => {
            req.startTime = process.hrtime.bigint();

            const originalJson = res.json;
            res.json = function(data) {
                const duration = Number(process.hrtime.bigint() - req.startTime) / 1000000;
                res.set('X-Response-Time', `${duration.toFixed(2)}ms`);

                try {
                    self.updateRequestStats(duration);
                } catch (e) {
                    logger.debug('Failed to update request stats', e && e.message);
                }

                return originalJson.call(this, data);
            };

            next();
        });

        // Caching middleware
        this.app.use(this.createCacheMiddleware());

        // Static file serving with optimizations
        this.app.use(express.static(path.join(__dirname, 'public'), {
            maxAge: '1d',
            etag: true,
            lastModified: true,
            setHeaders: (res, path) => {
                // Set cache control based on file type
                if (path.endsWith('.html')) {
                    res.setHeader('Cache-Control', 'no-cache');
                } else if (path.match(/\.(css|js|png|jpg|jpeg|gif|svg|woff|woff2)$/)) {
                    res.setHeader('Cache-Control', 'public, max-age=31536000'); // 1 year
                }
            }
        }));
    }

    createCacheMiddleware() {
        return async (req, res, next) => {
            // Only cache GET requests
            if (req.method !== 'GET') {
                return next();
            }

            // Skip caching for certain paths
            if (req.path.startsWith('/api/realtime') || 
                req.path.startsWith('/api/stream') ||
                req.path.includes('no-cache')) {
                return next();
            }

            const cacheKey = `${req.method}:${req.path}:${JSON.stringify(req.query)}`;
            
            try {
                // Try Redis first, then memory cache
                let cachedData = null;
                
                if (this.redisClient) {
                    cachedData = await this.redisClient.get(cacheKey);
                } else {
                    cachedData = this.memoryCache.get(cacheKey);
                }

                if (cachedData) {
                    const data = typeof cachedData === 'string' ? 
                        JSON.parse(cachedData) : cachedData;
                    
                    res.set('X-Cache', 'HIT');
                    res.set('X-Cache-TTL', data.ttl || 'unknown');
                    return res.json(data.body);
                }

                // Cache miss - override res.json to cache the response
                const originalJson = res.json;
                res.json = async (data) => {
                    // Don't cache errors
                    if (res.statusCode >= 400) {
                        return originalJson.call(res, data);
                    }

                    const cacheData = {
                        body: data,
                        ttl: Date.now() + (5 * 60 * 1000), // 5 minutes
                        timestamp: Date.now()
                    };

                    // Store in cache
                    try {
                        if (this.redisClient) {
                            await this.redisClient.setEx(cacheKey, 300, JSON.stringify(cacheData));
                        } else {
                            this.memoryCache.set(cacheKey, cacheData, 300);
                        }
                    } catch (error) {
                    logger.warn('Cache storage failed:', error.message);
                    }

                    res.set('X-Cache', 'MISS');
                    return originalJson.call(res, data);
                };

                next();
                } catch (error) {
                    logger.warn('Cache middleware error:', error.message);
                    next();
                }
        };
    }

    setupRoutes() {
        // Health check endpoint
        this.app.get('/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                uptime: process.uptime(),
                memory: process.memoryUsage(),
                cpu: process.cpuUsage(),
                version: require('./package.json').version
            });
        });

        // Performance metrics endpoint
        this.app.get('/api/performance', (req, res) => {
            const metrics = {
                requestStats: this.requestStats,
                memoryUsage: process.memoryUsage(),
                cpuUsage: process.cpuUsage(),
                cacheStats: {
                    memory: {
                        keys: this.memoryCache.keys().length,
                        hits: this.memoryCache.getStats().hits,
                        misses: this.memoryCache.getStats().misses
                    },
                    lru: {
                        size: this.cache.size,
                        max: this.cache.max
                    }
                },
                uptime: process.uptime(),
                timestamp: Date.now()
            };

            res.json(metrics);
        });

        // Service status with connection pooling and timeouts
        this.app.get('/api/services/status', async (req, res) => {
            const services = [
                { name: 'jellyfin', port: 8096, timeout: 2000 },
                { name: 'sonarr', port: 8989, timeout: 2000 },
                { name: 'radarr', port: 7878, timeout: 2000 },
                { name: 'prowlarr', port: 9696, timeout: 2000 },
                { name: 'qbittorrent', port: 8080, timeout: 2000 }
            ];

            const results = await Promise.allSettled(
                services.map(service => this.checkServiceStatus(service))
            );

            const serviceStatus = services.map((service, index) => ({
                name: service.name,
                port: service.port,
                status: results[index].status === 'fulfilled' ? 
                    results[index].value : { available: false, error: results[index].reason.message },
                timestamp: Date.now()
            }));

            res.json({
                services: serviceStatus,
                summary: {
                    total: services.length,
                    online: serviceStatus.filter(s => s.status.available).length,
                    offline: serviceStatus.filter(s => !s.status.available).length
                },
                timestamp: Date.now()
            });
        });

        // Bulk service operations
        this.app.post('/api/services/bulk', async (req, res) => {
            const { action, services } = req.body;
            
            if (!action || !Array.isArray(services)) {
                return res.status(400).json({
                    error: 'Invalid request format',
                    expected: { action: 'string', services: 'array' }
                });
            }

            const results = await Promise.allSettled(
                services.map(service => this.performServiceAction(service, action))
            );

            res.json({
                action,
                results: results.map((result, index) => ({
                    service: services[index],
                    success: result.status === 'fulfilled',
                    data: result.status === 'fulfilled' ? result.value : null,
                    error: result.status === 'rejected' ? result.reason.message : null
                })),
                timestamp: Date.now()
            });
        });

        // Resource optimization endpoint
        this.app.get('/api/optimize/resources', (req, res) => {
            const recommendations = this.generateOptimizationRecommendations();
            res.json(recommendations);
        });

        // Cache management endpoints
        this.app.post('/api/cache/clear', async (req, res) => {
            try {
                if (this.redisClient) {
                    await this.redisClient.flushDb();
                }
                this.memoryCache.flushAll();
                this.cache.clear();
                
                res.json({
                    message: 'Cache cleared successfully',
                    timestamp: Date.now()
                });
            } catch (error) {
                res.status(500).json({
                    error: 'Failed to clear cache',
                    message: error.message
                });
            }
        });

        this.app.get('/api/cache/stats', (req, res) => {
            const stats = {
                memory: this.memoryCache.getStats(),
                lru: {
                    size: this.cache.size,
                    max: this.cache.max,
                    calculatedSize: this.cache.calculatedSize
                },
                redis: this.redisClient ? 'connected' : 'not available',
                timestamp: Date.now()
            };

            res.json(stats);
        });

        // Performance testing endpoint
        this.app.get('/api/test/performance', async (req, res) => {
            const startTime = process.hrtime.bigint();
            
            // Simulate various operations
            const operations = [
                () => this.simulateCPUIntensiveTask(),
                () => this.simulateMemoryOperation(),
                () => this.simulateIOOperation(),
                () => this.simulateNetworkOperation()
            ];

            const results = await Promise.all(
                operations.map(async (op, index) => {
                    const opStart = process.hrtime.bigint();
                    try {
                        await op();
                        return {
                            operation: index,
                            duration: Number(process.hrtime.bigint() - opStart) / 1000000,
                            success: true
                        };
                    } catch (error) {
                        return {
                            operation: index,
                            duration: Number(process.hrtime.bigint() - opStart) / 1000000,
                            success: false,
                            error: error.message
                        };
                    }
                })
            );

            const totalDuration = Number(process.hrtime.bigint() - startTime) / 1000000;

            res.json({
                totalDuration,
                operations: results,
                performanceScore: this.calculatePerformanceScore(results),
                timestamp: Date.now()
            });
        });
    }

    async checkServiceStatus(service) {
        const axios = require('axios');

        try {
            const start = Date.now();
            const response = await axios.get(`http://localhost:${service.port}`, {
                timeout: service.timeout,
                headers: { 'User-Agent': 'MediaServer-HealthCheck/1.0' }
            });

            const responseTime = Date.now() - start;

            return {
                available: true,
                responseTime,
                statusCode: response.status,
                timestamp: Date.now()
            };
        } catch (error) {
            return {
                available: false,
                error: error.code || error.message,
                responseTime: service.timeout,
                timestamp: Date.now()
            };
        }
    }

    async performServiceAction(service, action) {
        // Implement service-specific actions
        switch (action) {
            case 'restart':
                return await this.restartService(service);
            case 'health-check':
                return await this.checkServiceStatus({ name: service, port: this.getServicePort(service), timeout: 5000 });
            case 'get-info':
                return await this.getServiceInfo(service);
            default:
                throw new Error(`Unknown action: ${action}`);
        }
    }

    getServicePort(serviceName) {
        const ports = {
            jellyfin: 8096,
            sonarr: 8989,
            radarr: 7878,
            prowlarr: 9696,
            qbittorrent: 8080
        };
        return ports[serviceName] || null;
    }

    generateOptimizationRecommendations() {
        const memUsage = process.memoryUsage();
        const cpuUsage = process.cpuUsage();
        const cacheStats = this.memoryCache.getStats();
        
        const recommendations = [];

        // Memory optimization
        if (memUsage.heapUsed > memUsage.heapTotal * 0.8) {
            recommendations.push({
                type: 'memory',
                priority: 'high',
                message: 'High memory usage detected',
                suggestion: 'Consider reducing cache size or increasing heap limit',
                impact: 'performance'
            });
        }

        // Cache optimization
        const totalCacheOps = (cacheStats.hits || 0) + (cacheStats.misses || 0);
        if (totalCacheOps > 0 && (cacheStats.hits / totalCacheOps) < 0.7) {
            recommendations.push({
                type: 'cache',
                priority: 'medium',
                message: 'Low cache hit ratio',
                suggestion: 'Review cache TTL settings and cache key strategies',
                impact: 'response-time'
            });
        }

        // Request rate optimization
        if (this.requestStats.averageResponseTime > 1000) {
            recommendations.push({
                type: 'response-time',
                priority: 'high',
                message: 'High average response time',
                suggestion: 'Optimize database queries and enable more aggressive caching',
                impact: 'user-experience'
            });
        }

        return {
            recommendations,
            summary: {
                total: recommendations.length,
                high: recommendations.filter(r => r.priority === 'high').length,
                medium: recommendations.filter(r => r.priority === 'medium').length,
                low: recommendations.filter(r => r.priority === 'low').length
            },
            timestamp: Date.now()
        };
    }

    setupPerformanceMonitoring() {
        // GC monitoring
        if (global.gc) {
            const originalGC = global.gc;

            global.gc = function() {
                const before = process.memoryUsage();
                const result = originalGC();
                const after = process.memoryUsage();

                logger.info(`GC: Freed ${Math.round((before.heapUsed - after.heapUsed) / 1024 / 1024)}MB`);
                return result;
            };
        }

        // Memory leak detection
        let memoryBaseline = process.memoryUsage().heapUsed;
        let lastBaselineReset = Date.now();
        setInterval(() => {
            const current = process.memoryUsage().heapUsed;
            const growth = current - memoryBaseline;

            if (growth > 50 * 1024 * 1024) {
                logger.warn(`⚠️  Memory leak detected: ${Math.round(growth / 1024 / 1024)}MB growth`);
            }

            const now = Date.now();
            if (now - lastBaselineReset >= 10 * 60 * 1000) {
                memoryBaseline = current;
                lastBaselineReset = now;
            }
        }, 30000);

        // Request stats reset
        setInterval(() => {
            const now = Date.now();
            const duration = now - this.requestStats.lastReset;

            logger.info(`📊 Request Stats (${Math.round(duration / 1000)}s): ${this.requestStats.total} total, ${this.requestStats.errors} errors, ${this.requestStats.averageResponseTime.toFixed(2)}ms avg`);

            this.requestStats = {
                total: 0,
                errors: 0,
                averageResponseTime: 0,
                lastReset: now
            };
        }, 5 * 60 * 1000);
    }

    updateRequestStats(duration) {
        this.requestStats.total++;
        this.requestStats.averageResponseTime = 
            (this.requestStats.averageResponseTime * (this.requestStats.total - 1) + duration) / 
            this.requestStats.total;
    }

    setupErrorHandling() {
        // Global error handler
        this.app.use((error, req, res, next) => {
            logger.error('API Error:', error);
            
            this.requestStats.errors++;
            
            // Don't leak error details in production
            const isDevelopment = process.env.NODE_ENV === 'development';
            
            res.status(error.status || 500).json({
                error: isDevelopment ? error.message : 'Internal server error',
                stack: isDevelopment ? error.stack : undefined,
                timestamp: Date.now(),
                requestId: req.id
            });
        });

        // 404 handler
        this.app.use((req, res) => {
            res.status(404).json({
                error: 'Not found',
                path: req.path,
                method: req.method,
                timestamp: Date.now()
            });
        });

        // Graceful shutdown
        process.on('SIGTERM', () => this.gracefulShutdown());
        process.on('SIGINT', () => this.gracefulShutdown());
        
        process.on('uncaughtException', (error) => {
            logger.error('Uncaught Exception:', error);
            this.gracefulShutdown();
        });

        process.on('unhandledRejection', (reason, promise) => {
            logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
        });
    }

    // Performance testing utilities
    async simulateCPUIntensiveTask() {
        return new Promise(resolve => {
            const start = Date.now();
            let iterations = 0;
            
            while (Date.now() - start < 100) { // 100ms of CPU work
                Math.random() * Math.random();
                iterations++;
            }
            
            resolve({ iterations });
        });
    }

    async simulateMemoryOperation() {
        const array = new Array(10000).fill(0).map(() => ({
            id: Math.random(),
            data: new Array(100).fill(Math.random())
        }));
        
        // Sort to simulate processing
        array.sort((a, b) => a.id - b.id);
        
        return { processed: array.length };
    }

    async simulateIOOperation() {
        const fs = require('fs').promises;
        const path = require('path');
        
        try {
            const stats = await fs.stat(__filename);
            return { size: stats.size };
        } catch (error) {
            throw new Error(`IO operation failed: ${error.message}`);
        }
    }

    async simulateNetworkOperation() {
        // Simulate network delay
        await new Promise(resolve => setTimeout(resolve, Math.random() * 100));
        return { latency: Math.random() * 100 };
    }

    calculatePerformanceScore(results) {
        const totalDuration = results.reduce((sum, r) => sum + r.duration, 0);
        const successCount = results.filter(r => r.success).length;
        
        // Score based on speed and success rate
        const speedScore = Math.max(0, 100 - (totalDuration / 10)); // Penalty for slow operations
        const reliabilityScore = (successCount / results.length) * 100;
        
        return Math.round((speedScore + reliabilityScore) / 2);
    }

    async gracefulShutdown() {
        logger.info('🛑 Graceful shutdown initiated...');
        
        // Close Redis connection
        if (this.redisClient) {
            await this.redisClient.quit();
        }
        
        // Clear caches
        this.memoryCache.close();
        this.cache.clear();
        
        logger.info('✅ Shutdown complete');
        process.exit(0);
    }

    start() {
        if (this.options.enableClustering && cluster.isMaster) {
            const numCPUs = os.cpus().length;
            logger.info(`🚀 Starting ${numCPUs} workers...`);
            
            for (let i = 0; i < numCPUs; i++) {
                cluster.fork();
            }
            
            cluster.on('exit', (worker, code, signal) => {
                logger.warn(`Worker ${worker.process.pid} died. Restarting...`);
                cluster.fork();
            });
        } else {
            this.server = this.app.listen(this.options.port, () => {
                logger.info(`🚀 Performance-Optimized API Server running on port ${this.options.port}`);
                logger.info(`📊 Process ID: ${process.pid}`);
                logger.info(`💾 Memory usage: ${Math.round(process.memoryUsage().heapUsed / 1024 / 1024)}MB`);
            });

            // Set up server optimizations
            this.server.keepAliveTimeout = 65000;
            this.server.headersTimeout = 66000;
        }
    }
}

// Export for use
module.exports = PerformanceOptimizedAPI;

// Start server if called directly
if (require.main === module) {
    const api = new PerformanceOptimizedAPI();
    api.start();
}