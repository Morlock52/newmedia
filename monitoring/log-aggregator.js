// Log Aggregator Service
// Collects logs from multiple sources and provides centralized access

const fs = require('fs').promises;
const path = require('path');
const { Tail } = require('tail');
const EventEmitter = require('events');
const { createLogger } = require('./comprehensive-logger');
const Docker = require('dockerode');
const chokidar = require('chokidar');

class LogAggregator extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            logDirectory: config.logDirectory || './logs',
            dockerLogs: config.dockerLogs !== false,
            fileLogs: config.fileLogs !== false,
            systemLogs: config.systemLogs || false,
            maxLogsPerSource: config.maxLogsPerSource || 10000,
            rotationInterval: config.rotationInterval || 86400000, // 24 hours
            sources: [
                { type: 'file', path: './logs/*.log', name: 'Application Logs' },
                { type: 'docker', container: 'jellyfin', name: 'Jellyfin' },
                { type: 'docker', container: 'sonarr', name: 'Sonarr' },
                { type: 'docker', container: 'radarr', name: 'Radarr' },
                { type: 'docker', container: 'prowlarr', name: 'Prowlarr' },
                { type: 'docker', container: 'qbittorrent', name: 'qBittorrent' },
                ...config.sources || []
            ],
            ...config
        };

        this.logger = createLogger({ service: 'log-aggregator' });
        this.docker = new Docker();
        this.logs = new Map(); // source -> logs[]
        this.tails = new Map(); // file -> Tail instance
        this.dockerStreams = new Map(); // container -> stream
        this.stats = new Map(); // source -> statistics
        
        this.initialize();
    }

    async initialize() {
        this.logger.info('Initializing log aggregator');
        
        // Create log directory if needed
        await fs.mkdir(this.config.logDirectory, { recursive: true });
        
        // Start collecting logs from all sources
        for (const source of this.config.sources) {
            await this.addSource(source);
        }
        
        // Set up rotation
        this.setupRotation();
        
        // Monitor for new log files
        if (this.config.fileLogs) {
            this.setupFileWatcher();
        }
    }

    async addSource(source) {
        try {
            switch (source.type) {
                case 'file':
                    await this.addFileSource(source);
                    break;
                case 'docker':
                    await this.addDockerSource(source);
                    break;
                case 'system':
                    await this.addSystemSource(source);
                    break;
                default:
                    this.logger.warn(`Unknown source type: ${source.type}`);
            }
            
            // Initialize stats for this source
            this.stats.set(source.name, {
                totalLogs: 0,
                errorCount: 0,
                warnCount: 0,
                lastUpdate: Date.now(),
                bytesProcessed: 0
            });
            
        } catch (error) {
            this.logger.error(`Failed to add source ${source.name}`, error);
        }
    }

    async addFileSource(source) {
        const files = await this.findFiles(source.path);
        
        for (const file of files) {
            if (this.tails.has(file)) continue;
            
            const tail = new Tail(file, {
                separator: /[\r]{0,1}\n/,
                fromBeginning: false,
                follow: true,
                logger: console
            });
            
            tail.on('line', (line) => {
                this.processLog(source.name, line, 'file');
            });
            
            tail.on('error', (error) => {
                this.logger.error(`Tail error for ${file}`, error);
            });
            
            this.tails.set(file, tail);
            this.logger.info(`Started tailing file: ${file}`);
        }
    }

    async addDockerSource(source) {
        try {
            const containers = await this.docker.listContainers({
                all: true,
                filters: { name: [source.container] }
            });
            
            if (containers.length === 0) {
                this.logger.warn(`Container not found: ${source.container}`);
                return;
            }
            
            const container = this.docker.getContainer(containers[0].Id);
            
            // Get recent logs
            const logs = await container.logs({
                stdout: true,
                stderr: true,
                follow: true,
                tail: 100,
                timestamps: true
            });
            
            // Process existing logs
            const existingLogs = logs.toString().split('\n').filter(line => line.trim());
            existingLogs.forEach(line => {
                this.processLog(source.name, line, 'docker');
            });
            
            // Set up stream for new logs
            const stream = await container.logs({
                stdout: true,
                stderr: true,
                follow: true,
                tail: 0,
                timestamps: true
            });
            
            stream.on('data', (chunk) => {
                const lines = chunk.toString().split('\n').filter(line => line.trim());
                lines.forEach(line => {
                    this.processLog(source.name, line, 'docker');
                });
            });
            
            stream.on('error', (error) => {
                this.logger.error(`Docker log stream error for ${source.container}`, error);
            });
            
            this.dockerStreams.set(source.container, stream);
            this.logger.info(`Started collecting Docker logs for: ${source.container}`);
            
        } catch (error) {
            this.logger.error(`Failed to add Docker source ${source.container}`, error);
        }
    }

    async addSystemSource(source) {
        // Platform-specific system log collection
        if (process.platform === 'linux') {
            // Use journalctl for systemd logs
            const { spawn } = require('child_process');
            const journalctl = spawn('journalctl', ['-f', '-o', 'json', '-n', '100']);
            
            journalctl.stdout.on('data', (data) => {
                const lines = data.toString().split('\n').filter(line => line.trim());
                lines.forEach(line => {
                    try {
                        const entry = JSON.parse(line);
                        this.processLog(source.name, entry, 'system');
                    } catch (e) {
                        // Not JSON, process as plain text
                        this.processLog(source.name, line, 'system');
                    }
                });
            });
            
            journalctl.stderr.on('data', (data) => {
                this.logger.error('Journalctl error:', data.toString());
            });
            
        } else if (process.platform === 'darwin') {
            // Use log command for macOS
            const { spawn } = require('child_process');
            const logCmd = spawn('log', ['stream', '--style', 'json']);
            
            logCmd.stdout.on('data', (data) => {
                const lines = data.toString().split('\n').filter(line => line.trim());
                lines.forEach(line => {
                    try {
                        const entry = JSON.parse(line);
                        this.processLog(source.name, entry, 'system');
                    } catch (e) {
                        this.processLog(source.name, line, 'system');
                    }
                });
            });
        }
    }

    processLog(sourceName, rawLog, sourceType) {
        const parsedLog = this.parseLog(rawLog, sourceType);
        
        if (!parsedLog) return;
        
        // Add source information
        parsedLog.source = sourceName;
        parsedLog.sourceType = sourceType;
        
        // Store log
        if (!this.logs.has(sourceName)) {
            this.logs.set(sourceName, []);
        }
        
        const sourceLogs = this.logs.get(sourceName);
        sourceLogs.push(parsedLog);
        
        // Trim logs if exceeded max
        if (sourceLogs.length > this.config.maxLogsPerSource) {
            sourceLogs.shift();
        }
        
        // Update stats
        const stats = this.stats.get(sourceName);
        if (stats) {
            stats.totalLogs++;
            stats.lastUpdate = Date.now();
            stats.bytesProcessed += Buffer.byteLength(JSON.stringify(parsedLog));
            
            if (parsedLog.level === 'error') stats.errorCount++;
            if (parsedLog.level === 'warn') stats.warnCount++;
        }
        
        // Emit log event
        this.emit('log', parsedLog);
        
        // Check for patterns that need alerting
        this.checkLogPatterns(parsedLog);
    }

    parseLog(rawLog, sourceType) {
        let parsed = {
            timestamp: new Date().toISOString(),
            raw: rawLog,
            level: 'info',
            message: rawLog
        };
        
        try {
            if (sourceType === 'docker') {
                // Docker logs often have timestamp prefix
                const match = rawLog.match(/^(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\s+(.*)$/);
                if (match) {
                    parsed.timestamp = match[1];
                    parsed.message = match[2];
                }
            }
            
            // Try to detect log level
            if (/\b(error|err|fatal|crit)\b/i.test(rawLog)) {
                parsed.level = 'error';
            } else if (/\b(warn|warning)\b/i.test(rawLog)) {
                parsed.level = 'warn';
            } else if (/\b(debug|trace)\b/i.test(rawLog)) {
                parsed.level = 'debug';
            }
            
            // Try to parse JSON logs
            if (rawLog.trim().startsWith('{')) {
                const jsonLog = JSON.parse(rawLog);
                parsed = { ...parsed, ...jsonLog };
            }
            
            // Extract common patterns
            parsed.patterns = this.extractPatterns(rawLog);
            
        } catch (error) {
            // Keep original if parsing fails
        }
        
        return parsed;
    }

    extractPatterns(log) {
        const patterns = {
            ipAddresses: [],
            urls: [],
            errors: [],
            statusCodes: [],
            durations: []
        };
        
        // IP addresses
        const ipRegex = /\b(?:\d{1,3}\.){3}\d{1,3}\b/g;
        patterns.ipAddresses = log.match(ipRegex) || [];
        
        // URLs
        const urlRegex = /https?:\/\/[^\s]+/g;
        patterns.urls = log.match(urlRegex) || [];
        
        // HTTP status codes
        const statusRegex = /\b[1-5]\d{2}\b/g;
        patterns.statusCodes = log.match(statusRegex) || [];
        
        // Durations (ms)
        const durationRegex = /(\d+(?:\.\d+)?)\s*ms/g;
        let match;
        while ((match = durationRegex.exec(log)) !== null) {
            patterns.durations.push(parseFloat(match[1]));
        }
        
        return patterns;
    }

    checkLogPatterns(log) {
        // Check for critical patterns that need immediate attention
        const criticalPatterns = [
            { pattern: /out of memory/i, alert: 'out-of-memory' },
            { pattern: /disk full/i, alert: 'disk-full' },
            { pattern: /connection refused/i, alert: 'connection-refused' },
            { pattern: /authentication failed/i, alert: 'auth-failed' },
            { pattern: /data corruption/i, alert: 'data-corruption' },
            { pattern: /segmentation fault/i, alert: 'segfault' },
            { pattern: /kernel panic/i, alert: 'kernel-panic' }
        ];
        
        for (const { pattern, alert } of criticalPatterns) {
            if (pattern.test(log.message)) {
                this.emit('critical-pattern', {
                    type: alert,
                    log,
                    pattern: pattern.toString()
                });
                
                this.logger.warn(`Critical pattern detected: ${alert}`, {
                    source: log.source,
                    message: log.message
                });
            }
        }
        
        // Check for high error rates
        const stats = this.stats.get(log.source);
        if (stats && stats.totalLogs > 100) {
            const errorRate = (stats.errorCount / stats.totalLogs) * 100;
            if (errorRate > 10) {
                this.emit('high-error-rate', {
                    source: log.source,
                    errorRate,
                    errorCount: stats.errorCount,
                    totalLogs: stats.totalLogs
                });
            }
        }
    }

    async findFiles(pattern) {
        const glob = require('glob');
        return new Promise((resolve, reject) => {
            glob(pattern, (err, files) => {
                if (err) reject(err);
                else resolve(files);
            });
        });
    }

    setupFileWatcher() {
        const watcher = chokidar.watch(this.config.sources
            .filter(s => s.type === 'file')
            .map(s => s.path), {
            ignored: /(^|[\/\\])\../,
            persistent: true
        });
        
        watcher.on('add', (path) => {
            this.logger.info(`New log file detected: ${path}`);
            // Find matching source and add it
            const source = this.config.sources.find(s => 
                s.type === 'file' && this.matchesPattern(path, s.path)
            );
            if (source) {
                this.addFileSource(source);
            }
        });
    }

    matchesPattern(path, pattern) {
        const minimatch = require('minimatch');
        return minimatch(path, pattern);
    }

    setupRotation() {
        setInterval(() => {
            this.rotateLogs();
        }, this.config.rotationInterval);
    }

    async rotateLogs() {
        this.logger.info('Rotating aggregated logs');
        
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const archiveDir = path.join(this.config.logDirectory, 'archive');
        await fs.mkdir(archiveDir, { recursive: true });
        
        for (const [source, logs] of this.logs.entries()) {
            if (logs.length === 0) continue;
            
            const filename = `${source.replace(/[^a-z0-9]/gi, '-')}-${timestamp}.json`;
            const filepath = path.join(archiveDir, filename);
            
            await fs.writeFile(filepath, JSON.stringify(logs, null, 2));
            
            // Clear logs after archiving
            this.logs.set(source, []);
            
            this.logger.info(`Archived ${logs.length} logs for ${source}`);
        }
    }

    // Query methods
    
    searchLogs(query = {}) {
        const {
            source,
            level,
            startTime,
            endTime,
            search,
            limit = 1000
        } = query;
        
        let results = [];
        
        // Get logs from specified sources or all
        const sources = source ? [source] : Array.from(this.logs.keys());
        
        for (const src of sources) {
            const logs = this.logs.get(src) || [];
            
            let filtered = logs;
            
            // Filter by level
            if (level) {
                filtered = filtered.filter(log => log.level === level);
            }
            
            // Filter by time range
            if (startTime) {
                filtered = filtered.filter(log => 
                    new Date(log.timestamp) >= new Date(startTime)
                );
            }
            
            if (endTime) {
                filtered = filtered.filter(log => 
                    new Date(log.timestamp) <= new Date(endTime)
                );
            }
            
            // Filter by search term
            if (search) {
                const searchLower = search.toLowerCase();
                filtered = filtered.filter(log => 
                    log.message.toLowerCase().includes(searchLower) ||
                    JSON.stringify(log).toLowerCase().includes(searchLower)
                );
            }
            
            results = results.concat(filtered);
        }
        
        // Sort by timestamp descending
        results.sort((a, b) => 
            new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()
        );
        
        // Apply limit
        return results.slice(0, limit);
    }

    getStats() {
        const stats = {};
        
        for (const [source, sourceStats] of this.stats.entries()) {
            stats[source] = {
                ...sourceStats,
                logsInMemory: (this.logs.get(source) || []).length
            };
        }
        
        return stats;
    }

    getSummary() {
        const summary = {
            sources: this.config.sources.length,
            totalLogs: 0,
            totalErrors: 0,
            totalWarnings: 0,
            activeTails: this.tails.size,
            activeDockerStreams: this.dockerStreams.size,
            memoryUsage: process.memoryUsage(),
            uptime: process.uptime()
        };
        
        for (const [, stats] of this.stats.entries()) {
            summary.totalLogs += stats.totalLogs;
            summary.totalErrors += stats.errorCount;
            summary.totalWarnings += stats.warnCount;
        }
        
        return summary;
    }

    async exportLogs(format = 'json', outputPath) {
        const allLogs = [];
        
        for (const [source, logs] of this.logs.entries()) {
            allLogs.push(...logs.map(log => ({ ...log, source })));
        }
        
        switch (format) {
            case 'json':
                await fs.writeFile(outputPath, JSON.stringify(allLogs, null, 2));
                break;
                
            case 'csv':
                const csv = this.convertToCSV(allLogs);
                await fs.writeFile(outputPath, csv);
                break;
                
            case 'ndjson':
                const ndjson = allLogs.map(log => JSON.stringify(log)).join('\n');
                await fs.writeFile(outputPath, ndjson);
                break;
                
            default:
                throw new Error(`Unsupported format: ${format}`);
        }
        
        this.logger.info(`Exported ${allLogs.length} logs to ${outputPath}`);
    }

    convertToCSV(logs) {
        if (logs.length === 0) return '';
        
        // Get all unique keys
        const keys = new Set();
        logs.forEach(log => Object.keys(log).forEach(key => keys.add(key)));
        
        const headers = Array.from(keys);
        const rows = [headers.join(',')];
        
        for (const log of logs) {
            const row = headers.map(key => {
                const value = log[key];
                if (value === undefined || value === null) return '';
                if (typeof value === 'object') return JSON.stringify(value);
                return `"${String(value).replace(/"/g, '""')}"`;
            });
            rows.push(row.join(','));
        }
        
        return rows.join('\n');
    }

    async cleanup() {
        this.logger.info('Cleaning up log aggregator');
        
        // Stop file tails
        for (const [, tail] of this.tails) {
            tail.unwatch();
        }
        
        // Close Docker streams
        for (const [, stream] of this.dockerStreams) {
            stream.destroy();
        }
        
        // Final rotation
        await this.rotateLogs();
    }
}

// Export
module.exports = LogAggregator;

// CLI usage
if (require.main === module) {
    const aggregator = new LogAggregator();
    
    // Graceful shutdown
    process.on('SIGTERM', async () => {
        await aggregator.cleanup();
        process.exit(0);
    });
    
    process.on('SIGINT', async () => {
        await aggregator.cleanup();
        process.exit(0);
    });
}