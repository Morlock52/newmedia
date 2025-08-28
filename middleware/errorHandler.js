/**
 * Comprehensive Error Handling Middleware for Media Server APIs
 * Provides consistent error responses, logging, and recovery mechanisms
 */

const fs = require('fs').promises;
const path = require('path');

class APIErrorHandler {
    constructor(options = {}) {
        this.logLevel = options.logLevel || 'error';
        this.logFile = options.logFile || '/var/log/api-errors.log';
        this.enableStackTrace = options.enableStackTrace !== false;
        this.enableErrorReporting = options.enableErrorReporting !== false;
        this.rateLimitWindow = options.rateLimitWindow || 60000; // 1 minute
        this.rateLimitThreshold = options.rateLimitThreshold || 100;
        
        this.errorCounts = new Map();
        this.circuitBreakers = new Map();
        
        // Initialize logging
        this.initializeLogging();
    }

    async initializeLogging() {
        try {
            const logDir = path.dirname(this.logFile);
            await fs.mkdir(logDir, { recursive: true });
        } catch (error) {
            console.error('Failed to initialize error logging:', error);
        }
    }

    // Main error handling middleware
    handleError() {
        return async (error, req, res, next) => {
            const errorId = this.generateErrorId();
            const timestamp = new Date().toISOString();
            
            // Categorize the error
            const errorInfo = this.categorizeError(error);
            
            // Log the error
            await this.logError(error, req, errorId, timestamp);
            
            // Update error metrics
            this.updateErrorMetrics(errorInfo.type, req.path);
            
            // Check if circuit breaker should be triggered
            await this.checkCircuitBreaker(req.path, errorInfo);
            
            // Send error response
            const response = this.formatErrorResponse(error, errorId, errorInfo);
            
            res.status(errorInfo.statusCode).json(response);
        };
    }

    // Async error wrapper for route handlers
    asyncHandler(fn) {
        return (req, res, next) => {
            Promise.resolve(fn(req, res, next)).catch(next);
        };
    }

    // API rate limiting middleware
    rateLimiter() {
        const requests = new Map();
        
        return (req, res, next) => {
            const clientId = this.getClientId(req);
            const now = Date.now();
            
            // Clean old entries
            this.cleanOldRequests(requests, now);
            
            // Check current rate
            const clientRequests = requests.get(clientId) || [];
            const recentRequests = clientRequests.filter(
                time => now - time < this.rateLimitWindow
            );
            
            if (recentRequests.length >= this.rateLimitThreshold) {
                return res.status(429).json({
                    error: 'Too Many Requests',
                    message: 'Rate limit exceeded. Please try again later.',
                    retryAfter: Math.ceil(this.rateLimitWindow / 1000)
                });
            }
            
            // Add current request
            recentRequests.push(now);
            requests.set(clientId, recentRequests);
            
            next();
        };
    }

    // Request timeout middleware
    timeoutHandler(timeout = 30000) {
        return (req, res, next) => {
            const timeoutId = setTimeout(() => {
                if (!res.headersSent) {
                    res.status(408).json({
                        error: 'Request Timeout',
                        message: 'The request took too long to complete.',
                        timeout: timeout
                    });
                }
            }, timeout);

            res.on('finish', () => {
                clearTimeout(timeoutId);
            });

            res.on('close', () => {
                clearTimeout(timeoutId);
            });

            next();
        };
    }

    // Validation error handler
    validationErrorHandler() {
        return (error, req, res, next) => {
            if (error.name === 'ValidationError') {
                const validationErrors = Object.values(error.errors).map(err => ({
                    field: err.path,
                    message: err.message,
                    value: err.value
                }));

                return res.status(400).json({
                    error: 'Validation Error',
                    message: 'Request validation failed',
                    details: validationErrors,
                    timestamp: new Date().toISOString()
                });
            }
            next(error);
        };
    }

    // Service unavailable handler
    serviceUnavailableHandler() {
        return (req, res, next) => {
            const serviceName = req.headers['x-service-name'];
            if (serviceName && this.isServiceDown(serviceName)) {
                return res.status(503).json({
                    error: 'Service Unavailable',
                    message: `${serviceName} service is currently unavailable`,
                    service: serviceName,
                    retryAfter: 60,
                    fallbacks: this.getFallbackOptions(serviceName)
                });
            }
            next();
        };
    }

    categorizeError(error) {
        // Network/Connection errors
        if (error.code === 'ECONNREFUSED' || error.code === 'ETIMEDOUT') {
            return {
                type: 'CONNECTION_ERROR',
                statusCode: 503,
                userMessage: 'Service temporarily unavailable',
                recoverable: true
            };
        }

        // Authentication errors
        if (error.name === 'UnauthorizedError' || error.status === 401) {
            return {
                type: 'AUTHENTICATION_ERROR',
                statusCode: 401,
                userMessage: 'Authentication required',
                recoverable: false
            };
        }

        // Permission errors
        if (error.status === 403) {
            return {
                type: 'PERMISSION_ERROR',
                statusCode: 403,
                userMessage: 'Access forbidden',
                recoverable: false
            };
        }

        // Validation errors
        if (error.name === 'ValidationError' || error.status === 400) {
            return {
                type: 'VALIDATION_ERROR',
                statusCode: 400,
                userMessage: 'Invalid request data',
                recoverable: false
            };
        }

        // Database errors
        if (error.name === 'SequelizeError' || error.name === 'MongoError') {
            return {
                type: 'DATABASE_ERROR',
                statusCode: 500,
                userMessage: 'Database operation failed',
                recoverable: true
            };
        }

        // File system errors
        if (error.code === 'ENOENT' || error.code === 'EACCES') {
            return {
                type: 'FILE_SYSTEM_ERROR',
                statusCode: 500,
                userMessage: 'File operation failed',
                recoverable: true
            };
        }

        // Default to internal server error
        return {
            type: 'INTERNAL_ERROR',
            statusCode: 500,
            userMessage: 'An unexpected error occurred',
            recoverable: true
        };
    }

    formatErrorResponse(error, errorId, errorInfo) {
        const response = {
            error: errorInfo.type,
            message: errorInfo.userMessage,
            errorId: errorId,
            timestamp: new Date().toISOString(),
            path: error.path || null
        };

        // Add stack trace in development
        if (this.enableStackTrace && process.env.NODE_ENV !== 'production') {
            response.stack = error.stack;
            response.details = error.message;
        }

        // Add recovery suggestions for recoverable errors
        if (errorInfo.recoverable) {
            response.recovery = this.getRecoveryOptions(errorInfo.type);
        }

        return response;
    }

    getRecoveryOptions(errorType) {
        const recoveryMap = {
            'CONNECTION_ERROR': [
                'Check service status',
                'Retry request in a few moments',
                'Use cached data if available'
            ],
            'DATABASE_ERROR': [
                'Retry the operation',
                'Check database connectivity',
                'Use fallback data source'
            ],
            'FILE_SYSTEM_ERROR': [
                'Check file permissions',
                'Verify file path exists',
                'Retry with alternative path'
            ],
            'INTERNAL_ERROR': [
                'Retry the request',
                'Contact system administrator',
                'Check system logs'
            ]
        };

        return recoveryMap[errorType] || ['Retry the request'];
    }

    async logError(error, req, errorId, timestamp) {
        const logEntry = {
            errorId,
            timestamp,
            method: req.method,
            url: req.url,
            path: req.path,
            userAgent: req.get('User-Agent'),
            ip: req.ip,
            error: {
                name: error.name,
                message: error.message,
                stack: error.stack,
                code: error.code
            },
            headers: req.headers,
            body: req.body,
            params: req.params,
            query: req.query
        };

        try {
            const logLine = JSON.stringify(logEntry) + '\n';
            await fs.appendFile(this.logFile, logLine);
        } catch (logError) {
            console.error('Failed to log error:', logError);
        }

        // Also log to console
        console.error(`[${timestamp}] ${error.name}: ${error.message}`, {
            errorId,
            path: req.path,
            method: req.method
        });
    }

    updateErrorMetrics(errorType, path) {
        const key = `${errorType}:${path}`;
        const count = this.errorCounts.get(key) || 0;
        this.errorCounts.set(key, count + 1);

        // Clean old metrics periodically
        if (Math.random() < 0.01) { // 1% chance
            this.cleanOldMetrics();
        }
    }

    async checkCircuitBreaker(path, errorInfo) {
        const key = path;
        let breaker = this.circuitBreakers.get(key);

        if (!breaker) {
            breaker = {
                state: 'CLOSED',
                failureCount: 0,
                lastFailureTime: null,
                successCount: 0
            };
            this.circuitBreakers.set(key, breaker);
        }

        if (errorInfo.recoverable) {
            breaker.failureCount++;
            breaker.lastFailureTime = Date.now();
            breaker.successCount = 0;

            // Open circuit if failure threshold reached
            if (breaker.failureCount >= 5) {
                breaker.state = 'OPEN';
                console.warn(`Circuit breaker OPENED for ${path}`);
            }
        } else {
            // Success - reset circuit breaker
            breaker.successCount++;
            if (breaker.successCount >= 3) {
                breaker.state = 'CLOSED';
                breaker.failureCount = 0;
                breaker.lastFailureTime = null;
            }
        }
    }

    isServiceDown(serviceName) {
        // Check if service has too many recent errors
        const errorKey = `CONNECTION_ERROR:/api/${serviceName}`;
        const errorCount = this.errorCounts.get(errorKey) || 0;
        return errorCount > 10;
    }

    getFallbackOptions(serviceName) {
        const fallbackMap = {
            'jellyfin': ['Use Plex', 'Direct file access', 'Check service status'],
            'sonarr': ['Manual search', 'Check download queue', 'Use Prowlarr directly'],
            'radarr': ['Manual search', 'Check download queue', 'Use Prowlarr directly'],
            'qbittorrent': ['Check downloads folder', 'Use alternative client', 'Manual download'],
            'plex': ['Use Jellyfin', 'Direct file access', 'Check service status']
        };

        return fallbackMap[serviceName] || ['Check service status', 'Try again later'];
    }

    generateErrorId() {
        return `err_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    }

    getClientId(req) {
        return req.ip + ':' + (req.get('User-Agent') || 'unknown');
    }

    cleanOldRequests(requests, now) {
        for (const [clientId, times] of requests.entries()) {
            const recent = times.filter(time => now - time < this.rateLimitWindow);
            if (recent.length === 0) {
                requests.delete(clientId);
            } else {
                requests.set(clientId, recent);
            }
        }
    }

    cleanOldMetrics() {
        // Keep metrics for last hour only
        const oneHourAgo = Date.now() - (60 * 60 * 1000);
        
        for (const [key, count] of this.errorCounts.entries()) {
            // This is a simplified cleanup - in production, store timestamps with counts
            if (Math.random() < 0.1) { // Randomly clean 10% of old entries
                this.errorCounts.delete(key);
            }
        }
    }

    // Get error statistics
    getErrorStats() {
        const stats = {
            totalErrors: 0,
            errorsByType: {},
            circuitBreakers: {},
            timestamp: new Date().toISOString()
        };

        for (const [key, count] of this.errorCounts.entries()) {
            const [type] = key.split(':');
            stats.totalErrors += count;
            stats.errorsByType[type] = (stats.errorsByType[type] || 0) + count;
        }

        for (const [path, breaker] of this.circuitBreakers.entries()) {
            stats.circuitBreakers[path] = {
                state: breaker.state,
                failureCount: breaker.failureCount,
                lastFailureTime: breaker.lastFailureTime
            };
        }

        return stats;
    }

    // Express middleware for 404 errors
    notFoundHandler() {
        return (req, res) => {
            res.status(404).json({
                error: 'NOT_FOUND',
                message: 'The requested resource was not found',
                path: req.path,
                method: req.method,
                timestamp: new Date().toISOString(),
                suggestions: [
                    'Check the URL for typos',
                    'Verify the API endpoint exists',
                    'Check API documentation'
                ]
            });
        };
    }

    // Health check endpoint
    healthCheck() {
        return (req, res) => {
            const stats = this.getErrorStats();
            const health = {
                status: 'healthy',
                uptime: process.uptime(),
                timestamp: new Date().toISOString(),
                errorHandler: {
                    totalErrors: stats.totalErrors,
                    circuitBreakersOpen: Object.values(stats.circuitBreakers)
                        .filter(cb => cb.state === 'OPEN').length
                }
            };

            // Mark as unhealthy if too many circuit breakers are open
            if (health.errorHandler.circuitBreakersOpen > 2) {
                health.status = 'degraded';
            }

            res.json(health);
        };
    }
}

// Usage example and export
const createErrorHandler = (options = {}) => {
    return new APIErrorHandler(options);
};

module.exports = {
    APIErrorHandler,
    createErrorHandler
};

// Example usage:
// const { createErrorHandler } = require('./errorHandler');
// const errorHandler = createErrorHandler({
//     logLevel: 'error',
//     logFile: '/var/log/api-errors.log',
//     enableStackTrace: true
// });
//
// app.use(errorHandler.rateLimiter());
// app.use(errorHandler.timeoutHandler(30000));
// app.use(errorHandler.serviceUnavailableHandler());
// app.use(errorHandler.validationErrorHandler());
// app.use(errorHandler.handleError());
// app.use(errorHandler.notFoundHandler());