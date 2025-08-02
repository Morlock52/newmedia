/**
 * Log Manager Service
 * Comprehensive logging system with aggregation, streaming, and analysis
 */

const fs = require('fs').promises;
const path = require('path');
const { createWriteStream } = require('fs');
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

class LogManager {
    constructor() {
        this.logsPath = path.join(__dirname, '../../logs');
        this.logFiles = {
            api: path.join(this.logsPath, 'api.log'),
            error: path.join(this.logsPath, 'error.log'),
            access: path.join(this.logsPath, 'access.log'),
            docker: path.join(this.logsPath, 'docker.log'),
            system: path.join(this.logsPath, 'system.log')
        };
        
        this.logStreams = {};
        this.subscribers = new Map();
        this.logBuffer = [];
        this.maxBufferSize = 1000;
        
        // Log levels
        this.levels = {
            error: 0,
            warn: 1,
            info: 2,
            debug: 3,
            trace: 4
        };
        
        this.currentLevel = this.levels[process.env.LOG_LEVEL || 'info'];
        
        // Log rotation settings
        this.rotationSettings = {
            maxSize: 10 * 1024 * 1024, // 10MB
            maxFiles: 5,
            interval: '1d' // daily
        };

        // Service log paths (Docker container logs)
        this.serviceLogPaths = {
            jellyfin: '/var/log/jellyfin',
            sonarr: './config/sonarr/logs',
            radarr: './config/radarr/logs',
            lidarr: './config/lidarr/logs',
            prowlarr: './config/prowlarr/logs',
            bazarr: './config/bazarr/log',
            qbittorrent: './config/qbittorrent/logs',
            overseerr: './config/overseerr/logs',
            tautulli: './config/tautulli/logs'
        };
    }

    async initialize() {
        try {
            // Ensure logs directory exists
            await this.ensureLogsDirectory();
            
            // Initialize log streams
            await this.initializeLogStreams();
            
            // Set up log rotation
            this.setupLogRotation();
            
            console.log('LogManager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize LogManager:', error);
            throw error;
        }
    }

    async ensureLogsDirectory() {
        try {
            await fs.access(this.logsPath);
        } catch (error) {
            await fs.mkdir(this.logsPath, { recursive: true });
            console.log('Created logs directory');
        }
    }

    async initializeLogStreams() {
        for (const [type, logFile] of Object.entries(this.logFiles)) {
            this.logStreams[type] = createWriteStream(logFile, { flags: 'a' });
            
            this.logStreams[type].on('error', (error) => {
                console.error(`Log stream error for ${type}:`, error);
            });
        }
    }

    setupLogRotation() {
        // Set up daily log rotation
        setInterval(async () => {
            await this.rotateLogs();
        }, 24 * 60 * 60 * 1000); // Daily
    }

    async rotateLogs() {
        try {
            for (const [type, logFile] of Object.entries(this.logFiles)) {
                const stats = await fs.stat(logFile);
                
                if (stats.size > this.rotationSettings.maxSize) {
                    await this.rotateLogFile(type, logFile);
                }
            }
        } catch (error) {
            console.error('Log rotation error:', error);
        }
    }

    async rotateLogFile(type, logFile) {
        try {
            // Close current stream
            this.logStreams[type].end();
            
            // Move current log to rotated name
            const timestamp = new Date().toISOString().split('T')[0];
            const rotatedFile = `${logFile}.${timestamp}`;
            await fs.rename(logFile, rotatedFile);
            
            // Create new stream
            this.logStreams[type] = createWriteStream(logFile, { flags: 'a' });
            
            // Clean up old log files
            await this.cleanupOldLogs(path.dirname(logFile), type);
            
            console.log(`Rotated log file: ${type}`);
        } catch (error) {
            console.error(`Failed to rotate log file ${type}:`, error);
        }
    }

    async cleanupOldLogs(logDir, type) {
        try {
            const files = await fs.readdir(logDir);
            const logFiles = files
                .filter(file => file.startsWith(`${type}.log.`))
                .map(file => ({
                    name: file,
                    path: path.join(logDir, file),
                    mtime: null
                }));

            // Get modification times
            for (const file of logFiles) {
                const stats = await fs.stat(file.path);
                file.mtime = stats.mtime;
            }

            // Sort by modification time (newest first)
            logFiles.sort((a, b) => b.mtime - a.mtime);

            // Remove old files beyond maxFiles limit
            if (logFiles.length > this.rotationSettings.maxFiles) {
                const filesToRemove = logFiles.slice(this.rotationSettings.maxFiles);
                
                for (const file of filesToRemove) {
                    await fs.unlink(file.path);
                    console.log(`Removed old log file: ${file.name}`);
                }
            }
        } catch (error) {
            console.error('Failed to cleanup old logs:', error);
        }
    }

    log(level, message, meta = {}) {
        if (this.levels[level] > this.currentLevel) {
            return;
        }

        const logEntry = {
            timestamp: new Date().toISOString(),
            level: level.toUpperCase(),
            message,
            meta,
            pid: process.pid
        };

        // Add to buffer
        this.logBuffer.push(logEntry);
        if (this.logBuffer.length > this.maxBufferSize) {
            this.logBuffer.shift();
        }

        // Format log entry
        const formattedEntry = this.formatLogEntry(logEntry);

        // Write to appropriate streams
        if (level === 'error') {
            this.writeToStream('error', formattedEntry);
        }
        
        this.writeToStream('api', formattedEntry);

        // Broadcast to subscribers
        this.broadcastToSubscribers('log-entry', logEntry);

        // Console output in development
        if (process.env.NODE_ENV !== 'production') {
            console.log(formattedEntry);
        }
    }

    formatLogEntry(entry) {
        const metaStr = Object.keys(entry.meta).length > 0 ? JSON.stringify(entry.meta) : '';
        return `${entry.timestamp} [${entry.level}] ${entry.message} ${metaStr}\n`;
    }

    writeToStream(streamType, content) {
        if (this.logStreams[streamType]) {
            this.logStreams[streamType].write(content);
        }
    }

    // Convenience methods
    error(message, meta = {}) {
        this.log('error', message, meta);
    }

    warn(message, meta = {}) {
        this.log('warn', message, meta);
    }

    info(message, meta = {}) {
        this.log('info', message, meta);
    }

    debug(message, meta = {}) {
        this.log('debug', message, meta);
    }

    trace(message, meta = {}) {
        this.log('trace', message, meta);
    }

    async getLogs(options = {}) {
        try {
            const {
                level = null,
                service = null,
                limit = 100,
                since = null,
                until = null,
                search = null
            } = options;

            let logs = [];

            if (service && this.serviceLogPaths[service]) {
                // Get service-specific logs
                logs = await this.getServiceLogs(service, options);
            } else {
                // Get API logs
                logs = await this.getAPILogs(options);
            }

            // Apply filters
            if (level) {
                logs = logs.filter(log => log.level === level.toUpperCase());
            }

            if (since) {
                const sinceDate = new Date(since);
                logs = logs.filter(log => new Date(log.timestamp) >= sinceDate);
            }

            if (until) {
                const untilDate = new Date(until);
                logs = logs.filter(log => new Date(log.timestamp) <= untilDate);
            }

            if (search) {
                const searchLower = search.toLowerCase();
                logs = logs.filter(log => 
                    log.message.toLowerCase().includes(searchLower) ||
                    JSON.stringify(log.meta).toLowerCase().includes(searchLower)
                );
            }

            // Apply limit
            logs = logs.slice(-limit);

            return logs;
        } catch (error) {
            throw new Error('Failed to get logs: ' + error.message);
        }
    }

    async getAPILogs(options = {}) {
        const { limit = 100 } = options;
        
        // First, return from buffer for recent logs
        let logs = [...this.logBuffer];

        // If we need more logs, read from file
        if (logs.length < limit) {
            try {
                const fileContent = await fs.readFile(this.logFiles.api, 'utf8');
                const fileLines = fileContent.trim().split('\n').filter(line => line);
                
                const fileLogs = fileLines.map(line => this.parseLogLine(line)).filter(Boolean);
                
                // Merge and deduplicate
                const allLogs = [...fileLogs, ...logs];
                const uniqueLogs = this.deduplicateLogs(allLogs);
                
                logs = uniqueLogs.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            } catch (error) {
                console.error('Failed to read log file:', error);
            }
        }

        return logs.slice(-limit);
    }

    async getServiceLogs(service, options = {}) {
        try {
            const { limit = 100, lines = 100 } = options;
            
            // Try to get Docker container logs first
            try {
                const { stdout } = await execAsync(`docker logs ${service} --tail ${lines} 2>&1`);
                const logs = stdout.split('\n')
                    .filter(line => line.trim())
                    .map(line => this.parseDockerLogLine(service, line))
                    .filter(Boolean);
                
                return logs.slice(-limit);
            } catch (dockerError) {
                console.log(`Docker logs not available for ${service}, trying file logs`);
            }

            // Fallback to service log files
            const servicePath = this.serviceLogPaths[service];
            if (servicePath) {
                const logs = await this.readServiceLogFiles(service, servicePath, options);
                return logs.slice(-limit);
            }

            return [];
        } catch (error) {
            throw new Error(`Failed to get logs for service ${service}: ` + error.message);
        }
    }

    async readServiceLogFiles(service, logPath, options = {}) {
        try {
            const { limit = 100 } = options;
            const logs = [];

            // Check if path exists
            try {
                await fs.access(logPath);
            } catch (error) {
                return []; // Path doesn't exist
            }

            // Read directory or file
            const stats = await fs.stat(logPath);
            
            if (stats.isDirectory()) {
                // Read all log files in directory
                const files = await fs.readdir(logPath);
                const logFiles = files.filter(file => 
                    file.endsWith('.log') || file.endsWith('.txt')
                ).sort();

                for (const file of logFiles.slice(-3)) { // Read last 3 files
                    const filePath = path.join(logPath, file);
                    const content = await fs.readFile(filePath, 'utf8');
                    const fileLines = content.trim().split('\n').filter(line => line);
                    
                    fileLines.forEach(line => {
                        const logEntry = this.parseServiceLogLine(service, line);
                        if (logEntry) logs.push(logEntry);
                    });
                }
            } else {
                // Single file
                const content = await fs.readFile(logPath, 'utf8');
                const lines = content.trim().split('\n').filter(line => line);
                
                lines.forEach(line => {
                    const logEntry = this.parseServiceLogLine(service, line);
                    if (logEntry) logs.push(logEntry);
                });
            }

            return logs.slice(-limit);
        } catch (error) {
            console.error(`Failed to read service logs for ${service}:`, error);
            return [];
        }
    }

    parseLogLine(line) {
        try {
            // Expected format: timestamp [LEVEL] message meta
            const match = line.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z) \[(\w+)\] (.+?)( \{.*\})?$/);
            
            if (match) {
                const [, timestamp, level, message, metaStr] = match;
                let meta = {};
                
                if (metaStr) {
                    try {
                        meta = JSON.parse(metaStr.trim());
                    } catch (e) {
                        meta = { raw: metaStr.trim() };
                    }
                }

                return {
                    timestamp,
                    level,
                    message,
                    meta,
                    service: 'api'
                };
            }
        } catch (error) {
            console.error('Failed to parse log line:', error);
        }
        
        return null;
    }

    parseDockerLogLine(service, line) {
        try {
            // Docker log format includes timestamp prefix
            const timestampMatch = line.match(/^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)\s+(.+)$/);
            
            let timestamp = new Date().toISOString();
            let message = line;
            
            if (timestampMatch) {
                timestamp = timestampMatch[1];
                message = timestampMatch[2];
            }

            // Try to extract log level from message
            let level = 'INFO';
            const levelMatch = message.match(/\b(ERROR|WARN|INFO|DEBUG|TRACE)\b/i);
            if (levelMatch) {
                level = levelMatch[1].toUpperCase();
            }

            return {
                timestamp,
                level,
                message: message.trim(),
                meta: {},
                service
            };
        } catch (error) {
            return {
                timestamp: new Date().toISOString(),
                level: 'INFO',
                message: line,
                meta: {},
                service
            };
        }
    }

    parseServiceLogLine(service, line) {
        try {
            // Generic service log parsing
            // This would be customized for each service's log format
            
            return {
                timestamp: new Date().toISOString(),
                level: 'INFO',
                message: line.trim(),
                meta: {},
                service
            };
        } catch (error) {
            return null;
        }
    }

    deduplicateLogs(logs) {
        const seen = new Set();
        return logs.filter(log => {
            const key = `${log.timestamp}-${log.message}`;
            if (seen.has(key)) {
                return false;
            }
            seen.add(key);
            return true;
        });
    }

    subscribeClient(ws, options = {}) {
        const clientId = Math.random().toString(36).substr(2, 9);
        
        this.subscribers.set(clientId, {
            ws,
            options,
            lastSent: Date.now()
        });

        ws.on('close', () => {
            this.subscribers.delete(clientId);
        });

        // Send recent logs immediately
        this.sendRecentLogsToClient(ws, options);

        return clientId;
    }

    async sendRecentLogsToClient(ws, options = {}) {
        try {
            const logs = await this.getLogs({ ...options, limit: 50 });
            
            ws.send(JSON.stringify({
                type: 'log-history',
                data: logs,
                timestamp: new Date().toISOString()
            }));
        } catch (error) {
            console.error('Failed to send recent logs to client:', error);
        }
    }

    broadcastToSubscribers(type, data) {
        const message = JSON.stringify({
            type,
            data,
            timestamp: new Date().toISOString()
        });

        this.subscribers.forEach((subscriber, clientId) => {
            if (subscriber.ws.readyState === subscriber.ws.OPEN) {
                // Apply client-specific filters
                if (this.shouldSendToClient(data, subscriber.options)) {
                    subscriber.ws.send(message);
                    subscriber.lastSent = Date.now();
                }
            } else {
                this.subscribers.delete(clientId);
            }
        });
    }

    shouldSendToClient(logData, options) {
        // Apply filters based on client subscription options
        if (options.level && logData.level !== options.level.toUpperCase()) {
            return false;
        }

        if (options.service && logData.service !== options.service) {
            return false;
        }

        if (options.search) {
            const searchLower = options.search.toLowerCase();
            if (!logData.message.toLowerCase().includes(searchLower)) {
                return false;
            }
        }

        return true;
    }

    async getLogStatistics() {
        try {
            const stats = {
                totalEntries: this.logBuffer.length,
                levelDistribution: {},
                serviceDistribution: {},
                recentErrors: [],
                logFiles: {}
            };

            // Analyze buffer
            for (const entry of this.logBuffer) {
                stats.levelDistribution[entry.level] = (stats.levelDistribution[entry.level] || 0) + 1;
                stats.serviceDistribution[entry.service || 'api'] = (stats.serviceDistribution[entry.service || 'api'] || 0) + 1;
                
                if (entry.level === 'ERROR') {
                    stats.recentErrors.push({
                        timestamp: entry.timestamp,
                        message: entry.message,
                        meta: entry.meta
                    });
                }
            }

            // Get log file sizes
            for (const [type, logFile] of Object.entries(this.logFiles)) {
                try {
                    const stat = await fs.stat(logFile);
                    stats.logFiles[type] = {
                        size: stat.size,
                        modified: stat.mtime,
                        sizeFormatted: this.formatBytes(stat.size)
                    };
                } catch (error) {
                    stats.logFiles[type] = { error: error.message };
                }
            }

            stats.recentErrors = stats.recentErrors.slice(-10); // Last 10 errors

            return stats;
        } catch (error) {
            throw new Error('Failed to get log statistics: ' + error.message);
        }
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    async exportLogs(options = {}) {
        try {
            const {
                format = 'json',
                service = null,
                level = null,
                since = null,
                until = null
            } = options;

            const logs = await this.getLogs(options);
            
            let exported = '';
            
            if (format === 'json') {
                exported = JSON.stringify(logs, null, 2);
            } else if (format === 'csv') {
                const headers = 'timestamp,level,service,message,meta\n';
                const rows = logs.map(log => 
                    `"${log.timestamp}","${log.level}","${log.service || 'api'}","${log.message.replace(/"/g, '""')}","${JSON.stringify(log.meta).replace(/"/g, '""')}"`
                ).join('\n');
                exported = headers + rows;
            } else if (format === 'txt') {
                exported = logs.map(log => 
                    `${log.timestamp} [${log.level}] ${log.service ? `[${log.service}] ` : ''}${log.message}`
                ).join('\n');
            }

            return {
                format,
                count: logs.length,
                data: exported,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to export logs: ' + error.message);
        }
    }
}

module.exports = LogManager;