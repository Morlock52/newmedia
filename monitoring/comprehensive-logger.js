// Comprehensive Logging and Monitoring System
// Advanced logger with structured logging, performance tracking, and alerting

const winston = require('winston');
const DailyRotateFile = require('winston-daily-rotate-file');
const { ElasticsearchTransport } = require('winston-elasticsearch');
const Sentry = require('@sentry/node');
const { StatsD } = require('node-statsd');
const crypto = require('crypto');
const os = require('os');
const cluster = require('cluster');

class ComprehensiveLogger {
    constructor(config = {}) {
        this.config = {
            service: config.service || 'media-server',
            environment: config.environment || process.env.NODE_ENV || 'development',
            logLevel: config.logLevel || 'info',
            enableConsole: config.enableConsole !== false,
            enableFile: config.enableFile !== false,
            enableElastic: config.enableElastic || false,
            enableSentry: config.enableSentry || false,
            enableMetrics: config.enableMetrics !== false,
            logDirectory: config.logDirectory || './logs',
            maxFileSize: config.maxFileSize || '100m',
            maxFiles: config.maxFiles || '30d',
            elasticNode: config.elasticNode || process.env.ELASTIC_NODE || 'http://localhost:9200',
            sentryDsn: config.sentryDsn || process.env.SENTRY_DSN,
            statsdHost: config.statsdHost || 'localhost',
            statsdPort: config.statsdPort || 8125,
            ...config
        };

        this.sessionId = this.generateSessionId();
        this.processInfo = this.getProcessInfo();
        this.logger = this.createLogger();
        this.metrics = this.initializeMetrics();
        this.performanceMarks = new Map();
        this.requestContexts = new Map();
        
        // Initialize error tracking
        if (this.config.enableSentry && this.config.sentryDsn) {
            this.initializeSentry();
        }

        // Set up structured logging context
        this.contextData = {
            service: this.config.service,
            environment: this.config.environment,
            sessionId: this.sessionId,
            ...this.processInfo
        };

        // Performance monitoring
        this.startPerformanceMonitoring();
    }

    createLogger() {
        const transports = [];

        // Console transport with colorization
        if (this.config.enableConsole) {
            transports.push(new winston.transports.Console({
                format: winston.format.combine(
                    winston.format.timestamp(),
                    winston.format.colorize(),
                    winston.format.printf(this.consoleFormat.bind(this))
                )
            }));
        }

        // File transports with rotation
        if (this.config.enableFile) {
            // General log file
            transports.push(new DailyRotateFile({
                filename: `${this.config.logDirectory}/%DATE%-app.log`,
                datePattern: 'YYYY-MM-DD',
                maxSize: this.config.maxFileSize,
                maxFiles: this.config.maxFiles,
                format: winston.format.combine(
                    winston.format.timestamp(),
                    winston.format.json()
                )
            }));

            // Error log file
            transports.push(new DailyRotateFile({
                filename: `${this.config.logDirectory}/%DATE%-error.log`,
                datePattern: 'YYYY-MM-DD',
                level: 'error',
                maxSize: this.config.maxFileSize,
                maxFiles: this.config.maxFiles,
                format: winston.format.combine(
                    winston.format.timestamp(),
                    winston.format.json()
                )
            }));

            // Performance log file
            transports.push(new DailyRotateFile({
                filename: `${this.config.logDirectory}/%DATE%-performance.log`,
                datePattern: 'YYYY-MM-DD',
                maxSize: this.config.maxFileSize,
                maxFiles: this.config.maxFiles,
                format: winston.format.combine(
                    winston.format.timestamp(),
                    winston.format.json()
                )
            }));

            // Security log file
            transports.push(new DailyRotateFile({
                filename: `${this.config.logDirectory}/%DATE%-security.log`,
                datePattern: 'YYYY-MM-DD',
                maxSize: this.config.maxFileSize,
                maxFiles: this.config.maxFiles,
                format: winston.format.combine(
                    winston.format.timestamp(),
                    winston.format.json()
                )
            }));
        }

        // Elasticsearch transport for centralized logging
        if (this.config.enableElastic) {
            transports.push(new ElasticsearchTransport({
                level: 'info',
                clientOpts: { node: this.config.elasticNode },
                index: `${this.config.service}-logs`,
                dataStream: true,
                transformer: this.elasticTransformer.bind(this)
            }));
        }

        return winston.createLogger({
            level: this.config.logLevel,
            format: winston.format.combine(
                winston.format.timestamp(),
                winston.format.errors({ stack: true }),
                winston.format.json()
            ),
            defaultMeta: this.contextData,
            transports,
            exitOnError: false
        });
    }

    initializeMetrics() {
        if (!this.config.enableMetrics) return null;

        const client = new StatsD({
            host: this.config.statsdHost,
            port: this.config.statsdPort,
            prefix: `${this.config.service}.${this.config.environment}.`,
            errorHandler: (error) => {
                this.logger.error('StatsD error', { error: error.message });
            }
        });

        // Set up common metrics
        this.setupCommonMetrics(client);

        return client;
    }

    initializeSentry() {
        Sentry.init({
            dsn: this.config.sentryDsn,
            environment: this.config.environment,
            serverName: os.hostname(),
            release: process.env.APP_VERSION || 'unknown',
            integrations: [
                new Sentry.Integrations.Http({ tracing: true }),
                new Sentry.Integrations.Express({ app: true }),
            ],
            tracesSampleRate: this.config.environment === 'production' ? 0.1 : 1.0,
            beforeSend: (event, hint) => {
                // Filter sensitive data
                if (event.request) {
                    delete event.request.cookies;
                    delete event.request.headers?.authorization;
                }
                return event;
            }
        });

        this.logger.info('Sentry initialized', { dsn: this.config.sentryDsn });
    }

    // Logging Methods

    log(level, message, meta = {}) {
        const enrichedMeta = {
            ...meta,
            timestamp: new Date().toISOString(),
            requestId: this.getCurrentRequestId(),
            correlationId: this.getCorrelationId(meta),
            ...this.getPerformanceContext()
        };

        this.logger.log(level, message, enrichedMeta);

        // Send metrics
        if (this.metrics) {
            this.metrics.increment(`logs.${level}`);
        }

        // Send to Sentry for errors
        if (level === 'error' && this.config.enableSentry) {
            Sentry.captureException(new Error(message), {
                level: 'error',
                extra: enrichedMeta
            });
        }
    }

    info(message, meta = {}) {
        this.log('info', message, meta);
    }

    warn(message, meta = {}) {
        this.log('warn', message, meta);
    }

    error(message, error, meta = {}) {
        const errorMeta = {
            ...meta,
            error: {
                message: error?.message || error,
                stack: error?.stack,
                code: error?.code,
                type: error?.constructor?.name
            }
        };

        this.log('error', message, errorMeta);
    }

    debug(message, meta = {}) {
        this.log('debug', message, meta);
    }

    // Performance Logging

    startTimer(label) {
        const id = `${label}-${Date.now()}`;
        this.performanceMarks.set(id, {
            label,
            start: process.hrtime.bigint(),
            startTime: Date.now()
        });
        return id;
    }

    endTimer(id, meta = {}) {
        const mark = this.performanceMarks.get(id);
        if (!mark) return;

        const duration = Number(process.hrtime.bigint() - mark.start) / 1000000; // Convert to ms
        this.performanceMarks.delete(id);

        const perfLog = {
            label: mark.label,
            duration,
            durationUnit: 'ms',
            startTime: mark.startTime,
            endTime: Date.now(),
            ...meta
        };

        this.logger.info('Performance measurement', {
            type: 'performance',
            performance: perfLog
        });

        // Send metrics
        if (this.metrics) {
            this.metrics.timing(`performance.${mark.label}`, duration);
        }

        return duration;
    }

    // Request Context

    createRequestContext(req) {
        const requestId = req.headers['x-request-id'] || this.generateRequestId();
        const context = {
            requestId,
            method: req.method,
            path: req.path,
            ip: req.ip || req.connection.remoteAddress,
            userAgent: req.headers['user-agent'],
            startTime: Date.now(),
            correlationId: req.headers['x-correlation-id'] || requestId
        };

        this.requestContexts.set(requestId, context);
        req.requestId = requestId;

        return context;
    }

    logRequest(req, res, responseTime) {
        const context = this.requestContexts.get(req.requestId);
        if (!context) return;

        const logData = {
            type: 'http',
            request: {
                ...context,
                query: req.query,
                params: req.params,
                body: this.sanitizeBody(req.body)
            },
            response: {
                statusCode: res.statusCode,
                responseTime,
                headers: res.getHeaders()
            }
        };

        const level = res.statusCode >= 500 ? 'error' : 
                     res.statusCode >= 400 ? 'warn' : 'info';

        this.log(level, `${req.method} ${req.path} ${res.statusCode}`, logData);

        // Send metrics
        if (this.metrics) {
            this.metrics.increment(`http.requests.${req.method.toLowerCase()}`);
            this.metrics.increment(`http.responses.${res.statusCode}`);
            this.metrics.timing('http.response_time', responseTime);
        }

        this.requestContexts.delete(req.requestId);
    }

    // Security Logging

    logSecurityEvent(event, severity = 'medium', meta = {}) {
        const securityLog = {
            type: 'security',
            event,
            severity,
            timestamp: new Date().toISOString(),
            ...meta
        };

        // Always log security events to dedicated file
        this.logger.warn('Security event', securityLog);

        // Send high severity events to Sentry
        if (severity === 'high' && this.config.enableSentry) {
            Sentry.captureMessage(`Security Event: ${event}`, {
                level: 'warning',
                extra: securityLog
            });
        }

        // Send metrics
        if (this.metrics) {
            this.metrics.increment(`security.events.${event}`);
            this.metrics.increment(`security.severity.${severity}`);
        }
    }

    // Audit Logging

    logAudit(action, userId, resourceType, resourceId, meta = {}) {
        const auditLog = {
            type: 'audit',
            action,
            userId,
            resourceType,
            resourceId,
            timestamp: new Date().toISOString(),
            ...meta
        };

        this.logger.info('Audit event', auditLog);

        // Send metrics
        if (this.metrics) {
            this.metrics.increment(`audit.actions.${action}`);
            this.metrics.increment(`audit.resources.${resourceType}`);
        }
    }

    // Business Metrics

    logBusinessMetric(metric, value, meta = {}) {
        const businessLog = {
            type: 'business',
            metric,
            value,
            timestamp: new Date().toISOString(),
            ...meta
        };

        this.logger.info('Business metric', businessLog);

        // Send to metrics system
        if (this.metrics) {
            if (typeof value === 'number') {
                this.metrics.gauge(`business.${metric}`, value);
            } else {
                this.metrics.increment(`business.${metric}`);
            }
        }
    }

    // Health Check Logging

    logHealthCheck(service, status, details = {}) {
        const healthLog = {
            type: 'health',
            service,
            status,
            details,
            timestamp: new Date().toISOString()
        };

        const level = status === 'healthy' ? 'info' : 
                     status === 'degraded' ? 'warn' : 'error';

        this.log(level, `Health check: ${service} is ${status}`, healthLog);

        // Send metrics
        if (this.metrics) {
            this.metrics.gauge(`health.${service}`, status === 'healthy' ? 1 : 0);
        }
    }

    // Utility Methods

    generateSessionId() {
        return crypto.randomBytes(16).toString('hex');
    }

    generateRequestId() {
        return crypto.randomBytes(8).toString('hex');
    }

    getCurrentRequestId() {
        // In a real implementation, this would get from async context
        return null;
    }

    getCorrelationId(meta) {
        return meta.correlationId || this.getCurrentRequestId() || this.sessionId;
    }

    getProcessInfo() {
        return {
            hostname: os.hostname(),
            pid: process.pid,
            workerId: cluster.isWorker ? cluster.worker.id : 'master',
            platform: os.platform(),
            nodeVersion: process.version
        };
    }

    getPerformanceContext() {
        return {
            memory: process.memoryUsage(),
            cpu: process.cpuUsage(),
            uptime: process.uptime()
        };
    }

    sanitizeBody(body) {
        if (!body) return body;
        
        const sensitive = ['password', 'token', 'secret', 'key', 'authorization'];
        const sanitized = { ...body };

        for (const field of sensitive) {
            if (sanitized[field]) {
                sanitized[field] = '[REDACTED]';
            }
        }

        return sanitized;
    }

    consoleFormat(info) {
        const level = info.level.toUpperCase().padEnd(7);
        const timestamp = info.timestamp;
        const message = info.message;
        
        let format = `${timestamp} ${level} ${message}`;
        
        if (info.error) {
            format += `\n${info.error.stack || info.error.message}`;
        }
        
        return format;
    }

    elasticTransformer(logData) {
        return {
            '@timestamp': logData.timestamp || new Date().toISOString(),
            level: logData.level,
            message: logData.message,
            service: this.config.service,
            environment: this.config.environment,
            ...logData.meta
        };
    }

    setupCommonMetrics(client) {
        // System metrics collection every 10 seconds
        setInterval(() => {
            const memUsage = process.memoryUsage();
            const cpuUsage = process.cpuUsage();
            
            client.gauge('system.memory.rss', memUsage.rss);
            client.gauge('system.memory.heap_used', memUsage.heapUsed);
            client.gauge('system.memory.heap_total', memUsage.heapTotal);
            client.gauge('system.cpu.user', cpuUsage.user);
            client.gauge('system.cpu.system', cpuUsage.system);
            client.gauge('system.uptime', process.uptime());
        }, 10000);
    }

    startPerformanceMonitoring() {
        // Monitor event loop lag
        let lastCheck = Date.now();
        setInterval(() => {
            const now = Date.now();
            const lag = now - lastCheck - 1000;
            
            if (lag > 50) {
                this.warn('Event loop lag detected', {
                    lag,
                    threshold: 50
                });
            }
            
            if (this.metrics) {
                this.metrics.gauge('system.event_loop_lag', lag);
            }
            
            lastCheck = now;
        }, 1000);
    }

    // Express Middleware

    expressMiddleware() {
        return (req, res, next) => {
            const context = this.createRequestContext(req);
            const startTime = Date.now();

            // Log request start
            this.debug(`Incoming ${req.method} ${req.path}`, {
                type: 'http_start',
                request: context
            });

            // Capture response
            const originalSend = res.send;
            res.send = function(data) {
                res.send = originalSend;
                const responseTime = Date.now() - startTime;
                
                // Log request completion
                setImmediate(() => {
                    this.logRequest(req, res, responseTime);
                });

                return res.send(data);
            }.bind(this);

            next();
        };
    }

    // Graceful Shutdown

    async shutdown() {
        this.info('Logger shutting down');
        
        // Flush any pending logs
        await new Promise((resolve) => {
            this.logger.end(() => resolve());
        });

        // Close metrics connection
        if (this.metrics) {
            this.metrics.close();
        }

        // Flush Sentry
        if (this.config.enableSentry) {
            await Sentry.close(2000);
        }
    }
}

// Export singleton instance
let loggerInstance;

function createLogger(config) {
    if (!loggerInstance) {
        loggerInstance = new ComprehensiveLogger(config);
    }
    return loggerInstance;
}

module.exports = {
    ComprehensiveLogger,
    createLogger,
    logger: createLogger()
};