/**
 * Log Manager Service
 * Comprehensive logging system with multiple transports and filtering
 */

const fs = require('fs').promises;
const path = require('path');
const { createWriteStream } = require('fs');

class LogManager {
    constructor() {
        this.logs = [];
        this.maxLogs = process.env.MAX_LOGS || 10000;
        this.logLevels = {
            error: 0,
            warn: 1,
            info: 2,
            debug: 3
        };
        this.currentLevel = this.logLevels[process.env.LOG_LEVEL || 'info'];
        this.logDir = path.join(__dirname, '../../logs');
        this.logFile = path.join(this.logDir, 'api.log');
        this.subscribers = new Set();
        this.fileStream = null;
        this.initialized = false;
    }

    async initialize() {
        try {
            // Ensure log directory exists
            await fs.mkdir(this.logDir, { recursive: true });
            
            // Create file stream for persistent logging
            this.fileStream = createWriteStream(this.logFile, { flags: 'a' });
            
            this.initialized = true;
            this.info('LogManager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize LogManager:', error);
            throw error;
        }
    }

    // Core logging methods
    error(message, metadata = {}) {
        this.log('error', message, metadata);
    }

    warn(message, metadata = {}) {
        this.log('warn', message, metadata);
    }

    info(message, metadata = {}) {
        this.log('info', message, metadata);
    }

    debug(message, metadata = {}) {
        this.log('debug', message, metadata);
    }

    log(level, message, metadata = {}) {
        // Check if log level is enabled
        if (this.logLevels[level] > this.currentLevel) {
            return;
        }

        const logEntry = {
            id: this.generateLogId(),
            timestamp: new Date().toISOString(),
            level,
            message,
            metadata,
            service: metadata.service || 'api',
            ip: metadata.ip || null,
            userId: metadata.userId || null,
            sessionId: metadata.sessionId || null,
            userAgent: metadata.userAgent || null
        };

        // Add to in-memory store
        this.logs.unshift(logEntry);
        
        // Trim logs if exceeded max
        if (this.logs.length > this.maxLogs) {
            this.logs = this.logs.slice(0, this.maxLogs);
        }

        // Console output with colors
        this.outputToConsole(logEntry);

        // File output
        if (this.fileStream && this.initialized) {
            this.outputToFile(logEntry);
        }

        // Notify subscribers
        this.notifySubscribers(logEntry);
    }

    outputToConsole(logEntry) {
        const colors = {
            error: '\x1b[31m', // Red
            warn: '\x1b[33m',  // Yellow
            info: '\x1b[36m',  // Cyan
            debug: '\x1b[90m'  // Gray
        };
        
        const reset = '\x1b[0m';
        const color = colors[logEntry.level] || '';
        
        const timestamp = new Date(logEntry.timestamp).toLocaleString();
        const service = logEntry.service ? `[${logEntry.service}]` : '';
        
        console.log(
            `${color}[${timestamp}] [${logEntry.level.toUpperCase()}]${service} ${logEntry.message}${reset}`
        );
        
        // Log metadata if present and debug level
        if (Object.keys(logEntry.metadata).length > 0 && this.currentLevel >= this.logLevels.debug) {
            console.log(`${color}Metadata:${reset}`, JSON.stringify(logEntry.metadata, null, 2));
        }
    }

    outputToFile(logEntry) {
        const logLine = JSON.stringify(logEntry) + '\n';
        this.fileStream.write(logLine);
    }

    // Retrieve logs with filtering
    getLogs(options = {}) {
        let filteredLogs = [...this.logs];
        
        // Filter by level
        if (options.level) {
            filteredLogs = filteredLogs.filter(log => log.level === options.level);
        }
        
        // Filter by service
        if (options.service) {
            filteredLogs = filteredLogs.filter(log => log.service === options.service);
        }
        
        // Filter by date range
        if (options.since) {
            const sinceDate = new Date(options.since);
            filteredLogs = filteredLogs.filter(log => new Date(log.timestamp) >= sinceDate);
        }
        
        if (options.until) {
            const untilDate = new Date(options.until);
            filteredLogs = filteredLogs.filter(log => new Date(log.timestamp) <= untilDate);
        }
        
        // Filter by search term
        if (options.search) {
            const searchTerm = options.search.toLowerCase();
            filteredLogs = filteredLogs.filter(log => 
                log.message.toLowerCase().includes(searchTerm) ||
                JSON.stringify(log.metadata).toLowerCase().includes(searchTerm)
            );
        }
        
        // Limit results
        const limit = options.limit || 100;
        if (filteredLogs.length > limit) {
            filteredLogs = filteredLogs.slice(0, limit);
        }
        
        return filteredLogs;
    }

    // Get log statistics
    getLogStats(hours = 24) {
        const since = new Date(Date.now() - hours * 60 * 60 * 1000);
        const recentLogs = this.logs.filter(log => new Date(log.timestamp) >= since);
        
        const stats = {
            total: recentLogs.length,
            levels: {
                error: 0,
                warn: 0,
                info: 0,
                debug: 0
            },
            services: {},
            period: `${hours} hours`,
            timestamp: new Date().toISOString()
        };
        
        recentLogs.forEach(log => {
            // Count by level
            stats.levels[log.level] = (stats.levels[log.level] || 0) + 1;
            
            // Count by service
            stats.services[log.service] = (stats.services[log.service] || 0) + 1;
        });
        
        return stats;
    }

    // Get recent errors
    getRecentErrors(limit = 10) {
        return this.logs
            .filter(log => log.level === 'error')
            .slice(0, limit);
    }

    // Real-time log streaming
    subscribeClient(client, options = {}) {
        const subscription = {
            client,
            options,
            id: this.generateLogId()
        };
        
        this.subscribers.add(subscription);
        
        // Clean up on disconnect
        client.on('close', () => {
            this.subscribers.delete(subscription);
        });
        
        // Send recent logs immediately
        const recentLogs = this.getLogs({ ...options, limit: 50 });
        client.send(JSON.stringify({
            type: 'log-history',
            data: recentLogs,
            subscriptionId: subscription.id
        }));
        
        return subscription.id;
    }

    notifySubscribers(logEntry) {
        for (const subscription of this.subscribers) {
            try {
                const { client, options } = subscription;
                
                // Check if client is still connected
                if (client.readyState !== client.OPEN) {
                    this.subscribers.delete(subscription);
                    continue;
                }
                
                // Apply filters
                let shouldSend = true;
                
                if (options.level && logEntry.level !== options.level) {
                    shouldSend = false;
                }
                
                if (options.service && logEntry.service !== options.service) {
                    shouldSend = false;
                }
                
                if (options.minLevel) {
                    const minLevel = this.logLevels[options.minLevel];
                    const entryLevel = this.logLevels[logEntry.level];
                    if (entryLevel > minLevel) {
                        shouldSend = false;
                    }
                }
                
                if (shouldSend) {
                    client.send(JSON.stringify({
                        type: 'log-entry',
                        data: logEntry,
                        subscriptionId: subscription.id
                    }));
                }
            } catch (error) {
                console.error('Failed to notify log subscriber:', error);
                this.subscribers.delete(subscription);
            }
        }
    }

    // Log rotation and cleanup
    async rotateLogs() {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const archiveFile = path.join(this.logDir, `api-${timestamp}.log`);
            
            // Close current stream
            if (this.fileStream) {
                this.fileStream.end();
            }
            
            // Move current log to archive
            await fs.rename(this.logFile, archiveFile);
            
            // Create new stream
            this.fileStream = createWriteStream(this.logFile, { flags: 'a' });
            
            this.info('Log rotation completed', { archiveFile });
            
            // Clean up old archives (keep last 10)
            await this.cleanupOldLogs();
            
        } catch (error) {
            this.error('Log rotation failed', { error: error.message });
        }
    }

    async cleanupOldLogs() {
        try {
            const files = await fs.readdir(this.logDir);
            const logFiles = files
                .filter(file => file.startsWith('api-') && file.endsWith('.log'))
                .sort()
                .reverse(); // Newest first
            
            // Keep only the 10 most recent archived logs
            const filesToDelete = logFiles.slice(10);
            
            for (const file of filesToDelete) {
                await fs.unlink(path.join(this.logDir, file));
                this.debug('Deleted old log file', { file });
            }
            
        } catch (error) {
            this.error('Failed to cleanup old logs', { error: error.message });
        }
    }

    // Performance logging
    startTimer(label) {
        const startTime = Date.now();
        return {
            end: (metadata = {}) => {
                const duration = Date.now() - startTime;
                this.info(`Timer: ${label}`, {
                    duration: `${duration}ms`,
                    ...metadata
                });
                return duration;
            }
        };
    }

    // HTTP request logging
    logRequest(req, res, responseTime) {
        const logData = {
            method: req.method,
            url: req.url,
            status: res.statusCode,
            responseTime: `${responseTime}ms`,
            ip: req.ip || req.connection.remoteAddress,
            userAgent: req.get('User-Agent'),
            contentLength: res.get('Content-Length')
        };
        
        const level = res.statusCode >= 400 ? 'warn' : 'info';
        this.log(level, `${req.method} ${req.url} ${res.statusCode}`, logData);
    }

    // Export logs to file
    async exportLogs(options = {}) {
        const logs = this.getLogs(options);
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const filename = `logs-export-${timestamp}.json`;
        const filepath = path.join(this.logDir, filename);
        
        await fs.writeFile(filepath, JSON.stringify(logs, null, 2));
        
        this.info('Logs exported', { filename, count: logs.length });
        return { filename, filepath, count: logs.length };
    }

    // Clear logs
    clearLogs() {
        const count = this.logs.length;
        this.logs = [];
        this.info('Logs cleared', { clearedCount: count });
        return { clearedCount: count };
    }

    // Utility methods
    generateLogId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }

    setLogLevel(level) {
        if (this.logLevels.hasOwnProperty(level)) {
            this.currentLevel = this.logLevels[level];
            this.info(`Log level set to ${level}`);
        } else {
            this.error(`Invalid log level: ${level}`);
        }
    }

    getLogLevel() {
        return Object.keys(this.logLevels)[this.currentLevel];
    }

    // Shutdown cleanup
    async shutdown() {
        this.info('LogManager shutting down...');
        
        if (this.fileStream) {
            this.fileStream.end();
        }
        
        // Close all subscriber connections
        for (const subscription of this.subscribers) {
            try {
                subscription.client.close();
            } catch (error) {
                // Ignore close errors during shutdown
            }
        }
        
        this.subscribers.clear();
    }
}

module.exports = LogManager;
