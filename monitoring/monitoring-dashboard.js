// Real-time Monitoring Dashboard Server
// Provides live monitoring, alerting, and visualization

const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const path = require('path');
const { createLogger } = require('./comprehensive-logger');
const os = require('os');
const si = require('systeminformation');
const Docker = require('dockerode');
const axios = require('axios');
const EventEmitter = require('events');

class MonitoringDashboard extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            port: config.port || 3005,
            updateInterval: config.updateInterval || 5000,
            alertThresholds: {
                cpu: 80,
                memory: 85,
                disk: 90,
                responseTime: 1000,
                errorRate: 5,
                ...config.alertThresholds
            },
            services: [
                { name: 'Jellyfin', url: 'http://localhost:8096/health', type: 'http' },
                { name: 'Sonarr', url: 'http://localhost:8989/api/v3/system/status', type: 'api', apiKey: process.env.SONARR_API_KEY },
                { name: 'Radarr', url: 'http://localhost:7878/api/v3/system/status', type: 'api', apiKey: process.env.RADARR_API_KEY },
                { name: 'Prowlarr', url: 'http://localhost:9696/api/v1/system/status', type: 'api', apiKey: process.env.PROWLARR_API_KEY },
                { name: 'qBittorrent', url: 'http://localhost:8080/api/v2/app/version', type: 'http' },
                ...config.services || []
            ],
            ...config
        };

        this.logger = createLogger({ service: 'monitoring-dashboard' });
        this.app = express();
        this.server = http.createServer(this.app);
        this.io = socketIo(this.server, {
            cors: { origin: "*", methods: ["GET", "POST"] }
        });
        this.docker = new Docker();
        
        // State management
        this.state = {
            system: {},
            services: {},
            containers: {},
            logs: [],
            alerts: [],
            metrics: {
                requests: 0,
                errors: 0,
                avgResponseTime: 0,
                uptime: 0
            }
        };

        this.alertHistory = [];
        this.metricsHistory = {
            cpu: [],
            memory: [],
            network: [],
            disk: []
        };

        this.setupRoutes();
        this.setupSocketHandlers();
        this.startMonitoring();
    }

    setupRoutes() {
        // Middleware
        this.app.use(express.json());
        this.app.use(express.static(path.join(__dirname, 'public')));
        
        // Logging middleware
        this.app.use((req, res, next) => {
            const start = Date.now();
            res.on('finish', () => {
                const duration = Date.now() - start;
                this.state.metrics.requests++;
                this.state.metrics.avgResponseTime = 
                    (this.state.metrics.avgResponseTime + duration) / 2;
                
                if (res.statusCode >= 400) {
                    this.state.metrics.errors++;
                }
            });
            next();
        });

        // API Routes
        this.app.get('/api/status', (req, res) => {
            res.json({
                status: 'healthy',
                uptime: process.uptime(),
                timestamp: new Date().toISOString(),
                state: this.state
            });
        });

        this.app.get('/api/metrics', (req, res) => {
            res.json({
                current: this.state,
                history: this.metricsHistory,
                alerts: this.alertHistory
            });
        });

        this.app.get('/api/logs', (req, res) => {
            const limit = parseInt(req.query.limit) || 100;
            const level = req.query.level;
            let logs = this.state.logs;
            
            if (level) {
                logs = logs.filter(log => log.level === level);
            }
            
            res.json(logs.slice(-limit));
        });

        this.app.post('/api/alerts/acknowledge/:id', (req, res) => {
            const alertId = req.params.id;
            const alert = this.state.alerts.find(a => a.id === alertId);
            
            if (alert) {
                alert.acknowledged = true;
                alert.acknowledgedAt = new Date().toISOString();
                this.logger.info('Alert acknowledged', { alertId });
                res.json({ success: true, alert });
            } else {
                res.status(404).json({ error: 'Alert not found' });
            }
        });

        this.app.get('/api/services/:name/logs', async (req, res) => {
            const serviceName = req.params.name;
            try {
                const logs = await this.getServiceLogs(serviceName);
                res.json({ service: serviceName, logs });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Dashboard HTML
        this.app.get('/', (req, res) => {
            res.sendFile(path.join(__dirname, 'monitoring-dashboard.html'));
        });
    }

    setupSocketHandlers() {
        this.io.on('connection', (socket) => {
            this.logger.info('Monitoring client connected', { 
                clientId: socket.id,
                address: socket.handshake.address 
            });

            // Send initial state
            socket.emit('initial-state', this.state);
            socket.emit('metrics-history', this.metricsHistory);

            // Handle client commands
            socket.on('refresh', () => {
                this.updateAllMetrics();
            });

            socket.on('restart-service', async (serviceName) => {
                try {
                    await this.restartService(serviceName);
                    socket.emit('service-restarted', { service: serviceName });
                } catch (error) {
                    socket.emit('error', { message: error.message });
                }
            });

            socket.on('get-logs', (filter) => {
                const logs = this.getFilteredLogs(filter);
                socket.emit('logs', logs);
            });

            socket.on('disconnect', () => {
                this.logger.info('Monitoring client disconnected', { 
                    clientId: socket.id 
                });
            });
        });
    }

    async startMonitoring() {
        this.logger.info('Starting monitoring system');
        
        // Initial update
        await this.updateAllMetrics();
        
        // Regular updates
        setInterval(() => {
            this.updateAllMetrics();
        }, this.config.updateInterval);

        // Metrics history collection (every minute)
        setInterval(() => {
            this.collectMetricsHistory();
        }, 60000);

        // Log collection
        this.setupLogCollection();
        
        // Start server
        this.server.listen(this.config.port, () => {
            this.logger.info(`Monitoring dashboard running on port ${this.config.port}`);
        });
    }

    async updateAllMetrics() {
        try {
            // System metrics
            await this.updateSystemMetrics();
            
            // Service health
            await this.updateServiceHealth();
            
            // Container metrics
            await this.updateContainerMetrics();
            
            // Check for alerts
            this.checkAlerts();
            
            // Emit updates to all connected clients
            this.io.emit('metrics-update', this.state);
            
        } catch (error) {
            this.logger.error('Error updating metrics', error);
        }
    }

    async updateSystemMetrics() {
        try {
            const [cpu, memory, disk, network, time] = await Promise.all([
                si.currentLoad(),
                si.mem(),
                si.fsSize(),
                si.networkStats(),
                si.time()
            ]);

            this.state.system = {
                cpu: {
                    usage: cpu.currentLoad,
                    cores: cpu.cpus.length,
                    temps: cpu.temps
                },
                memory: {
                    total: memory.total,
                    used: memory.used,
                    free: memory.free,
                    percentage: (memory.used / memory.total) * 100
                },
                disk: disk.map(d => ({
                    fs: d.fs,
                    size: d.size,
                    used: d.used,
                    available: d.available,
                    percentage: d.use
                })),
                network: {
                    rx: network[0]?.rx_sec || 0,
                    tx: network[0]?.tx_sec || 0,
                    total: (network[0]?.rx_sec || 0) + (network[0]?.tx_sec || 0)
                },
                uptime: time.uptime,
                timestamp: new Date().toISOString()
            };

            this.state.metrics.uptime = process.uptime();
            
        } catch (error) {
            this.logger.error('Error getting system metrics', error);
        }
    }

    async updateServiceHealth() {
        const healthChecks = this.config.services.map(async (service) => {
            const startTime = Date.now();
            
            try {
                let response;
                const config = {
                    timeout: 5000,
                    validateStatus: () => true
                };

                if (service.type === 'api' && service.apiKey) {
                    config.headers = { 'X-Api-Key': service.apiKey };
                }

                response = await axios.get(service.url, config);
                
                const responseTime = Date.now() - startTime;
                const healthy = response.status >= 200 && response.status < 300;

                this.state.services[service.name] = {
                    name: service.name,
                    url: service.url,
                    healthy,
                    status: response.status,
                    statusText: response.statusText,
                    responseTime,
                    lastCheck: new Date().toISOString(),
                    data: response.data
                };

                if (!healthy) {
                    this.createAlert('service-unhealthy', `${service.name} is unhealthy`, {
                        service: service.name,
                        status: response.status,
                        statusText: response.statusText
                    });
                }

            } catch (error) {
                const responseTime = Date.now() - startTime;
                
                this.state.services[service.name] = {
                    name: service.name,
                    url: service.url,
                    healthy: false,
                    error: error.message,
                    responseTime,
                    lastCheck: new Date().toISOString()
                };

                this.createAlert('service-down', `${service.name} is down`, {
                    service: service.name,
                    error: error.message
                });
            }
        });

        await Promise.all(healthChecks);
    }

    async updateContainerMetrics() {
        try {
            const containers = await this.docker.listContainers({ all: true });
            
            for (const containerInfo of containers) {
                const container = this.docker.getContainer(containerInfo.Id);
                const stats = await container.stats({ stream: false });
                
                const cpuDelta = stats.cpu_stats.cpu_usage.total_usage - 
                                stats.precpu_stats.cpu_usage.total_usage;
                const systemDelta = stats.cpu_stats.system_cpu_usage - 
                                   stats.precpu_stats.system_cpu_usage;
                const cpuPercent = (cpuDelta / systemDelta) * 100;
                
                const memoryUsage = stats.memory_stats.usage;
                const memoryLimit = stats.memory_stats.limit;
                const memoryPercent = (memoryUsage / memoryLimit) * 100;

                this.state.containers[containerInfo.Names[0]] = {
                    id: containerInfo.Id,
                    name: containerInfo.Names[0],
                    image: containerInfo.Image,
                    state: containerInfo.State,
                    status: containerInfo.Status,
                    cpu: {
                        percent: cpuPercent,
                        usage: cpuDelta
                    },
                    memory: {
                        usage: memoryUsage,
                        limit: memoryLimit,
                        percent: memoryPercent
                    },
                    network: {
                        rx: stats.networks?.eth0?.rx_bytes || 0,
                        tx: stats.networks?.eth0?.tx_bytes || 0
                    }
                };
            }
        } catch (error) {
            this.logger.error('Error getting container metrics', error);
        }
    }

    checkAlerts() {
        // CPU Alert
        if (this.state.system.cpu?.usage > this.config.alertThresholds.cpu) {
            this.createAlert('high-cpu', 'High CPU usage detected', {
                current: this.state.system.cpu.usage,
                threshold: this.config.alertThresholds.cpu
            });
        }

        // Memory Alert
        if (this.state.system.memory?.percentage > this.config.alertThresholds.memory) {
            this.createAlert('high-memory', 'High memory usage detected', {
                current: this.state.system.memory.percentage,
                threshold: this.config.alertThresholds.memory
            });
        }

        // Disk Alert
        for (const disk of this.state.system.disk || []) {
            if (disk.percentage > this.config.alertThresholds.disk) {
                this.createAlert('high-disk', `High disk usage on ${disk.fs}`, {
                    filesystem: disk.fs,
                    current: disk.percentage,
                    threshold: this.config.alertThresholds.disk
                });
            }
        }

        // Error Rate Alert
        const errorRate = (this.state.metrics.errors / this.state.metrics.requests) * 100;
        if (errorRate > this.config.alertThresholds.errorRate) {
            this.createAlert('high-error-rate', 'High error rate detected', {
                current: errorRate,
                threshold: this.config.alertThresholds.errorRate
            });
        }

        // Response Time Alert
        if (this.state.metrics.avgResponseTime > this.config.alertThresholds.responseTime) {
            this.createAlert('slow-response', 'Slow response times detected', {
                current: this.state.metrics.avgResponseTime,
                threshold: this.config.alertThresholds.responseTime
            });
        }
    }

    createAlert(type, message, details = {}) {
        const alertId = `${type}-${Date.now()}`;
        
        // Check if similar alert already exists
        const existingAlert = this.state.alerts.find(a => 
            a.type === type && !a.resolved && 
            (Date.now() - new Date(a.timestamp).getTime()) < 300000 // 5 minutes
        );

        if (existingAlert) {
            existingAlert.count = (existingAlert.count || 1) + 1;
            existingAlert.lastOccurrence = new Date().toISOString();
            return;
        }

        const alert = {
            id: alertId,
            type,
            message,
            details,
            severity: this.getAlertSeverity(type),
            timestamp: new Date().toISOString(),
            acknowledged: false,
            resolved: false
        };

        this.state.alerts.unshift(alert);
        this.alertHistory.unshift(alert);
        
        // Keep only last 100 alerts in state
        if (this.state.alerts.length > 100) {
            this.state.alerts = this.state.alerts.slice(0, 100);
        }

        // Emit alert to connected clients
        this.io.emit('alert', alert);
        
        // Log alert
        this.logger.warn('Alert created', alert);
        
        // Auto-resolve certain alerts after conditions improve
        setTimeout(() => {
            this.checkAlertResolution(alertId);
        }, 60000); // Check after 1 minute
    }

    getAlertSeverity(type) {
        const severityMap = {
            'service-down': 'critical',
            'high-cpu': 'warning',
            'high-memory': 'warning',
            'high-disk': 'warning',
            'high-error-rate': 'critical',
            'slow-response': 'warning',
            'service-unhealthy': 'warning'
        };
        
        return severityMap[type] || 'info';
    }

    checkAlertResolution(alertId) {
        const alert = this.state.alerts.find(a => a.id === alertId);
        if (!alert || alert.resolved) return;

        let resolved = false;

        switch (alert.type) {
            case 'high-cpu':
                resolved = this.state.system.cpu?.usage < this.config.alertThresholds.cpu - 10;
                break;
            case 'high-memory':
                resolved = this.state.system.memory?.percentage < this.config.alertThresholds.memory - 10;
                break;
            case 'service-down':
            case 'service-unhealthy':
                const serviceName = alert.details.service;
                resolved = this.state.services[serviceName]?.healthy === true;
                break;
        }

        if (resolved) {
            alert.resolved = true;
            alert.resolvedAt = new Date().toISOString();
            this.io.emit('alert-resolved', alert);
            this.logger.info('Alert resolved', { alertId });
        }
    }

    collectMetricsHistory() {
        // Store current metrics in history
        const timestamp = new Date().toISOString();
        
        // CPU history
        this.metricsHistory.cpu.push({
            timestamp,
            value: this.state.system.cpu?.usage || 0
        });

        // Memory history
        this.metricsHistory.memory.push({
            timestamp,
            value: this.state.system.memory?.percentage || 0
        });

        // Network history
        this.metricsHistory.network.push({
            timestamp,
            rx: this.state.system.network?.rx || 0,
            tx: this.state.system.network?.tx || 0
        });

        // Disk history (primary disk)
        const primaryDisk = this.state.system.disk?.[0];
        if (primaryDisk) {
            this.metricsHistory.disk.push({
                timestamp,
                value: primaryDisk.percentage
            });
        }

        // Keep only last 24 hours of history (1440 minutes)
        const maxEntries = 1440;
        Object.keys(this.metricsHistory).forEach(metric => {
            if (this.metricsHistory[metric].length > maxEntries) {
                this.metricsHistory[metric] = this.metricsHistory[metric].slice(-maxEntries);
            }
        });
    }

    setupLogCollection() {
        // Intercept logger to capture logs
        const originalLog = this.logger.log.bind(this.logger);
        this.logger.log = (level, message, meta) => {
            // Call original logger
            originalLog(level, message, meta);
            
            // Store in state
            const logEntry = {
                timestamp: new Date().toISOString(),
                level,
                message,
                meta
            };
            
            this.state.logs.unshift(logEntry);
            
            // Keep only last 1000 logs
            if (this.state.logs.length > 1000) {
                this.state.logs = this.state.logs.slice(0, 1000);
            }
            
            // Emit to connected clients
            this.io.emit('log', logEntry);
        };
    }

    async getServiceLogs(serviceName) {
        try {
            const container = await this.findContainerByService(serviceName);
            if (!container) {
                throw new Error(`Container for ${serviceName} not found`);
            }

            const logs = await container.logs({
                stdout: true,
                stderr: true,
                tail: 100,
                timestamps: true
            });

            return logs.toString().split('\n').filter(line => line.trim());
        } catch (error) {
            this.logger.error('Error getting service logs', error);
            throw error;
        }
    }

    async findContainerByService(serviceName) {
        const containers = await this.docker.listContainers();
        const container = containers.find(c => 
            c.Names.some(name => name.toLowerCase().includes(serviceName.toLowerCase()))
        );
        
        return container ? this.docker.getContainer(container.Id) : null;
    }

    async restartService(serviceName) {
        try {
            const container = await this.findContainerByService(serviceName);
            if (!container) {
                throw new Error(`Container for ${serviceName} not found`);
            }

            await container.restart();
            this.logger.info(`Service ${serviceName} restarted`);
            
            // Create audit log
            this.logger.logAudit('service-restart', 'system', 'container', serviceName, {
                reason: 'Manual restart from monitoring dashboard'
            });
            
        } catch (error) {
            this.logger.error('Error restarting service', error);
            throw error;
        }
    }

    getFilteredLogs(filter = {}) {
        let logs = this.state.logs;
        
        if (filter.level) {
            logs = logs.filter(log => log.level === filter.level);
        }
        
        if (filter.service) {
            logs = logs.filter(log => 
                log.meta?.service === filter.service ||
                log.message.toLowerCase().includes(filter.service.toLowerCase())
            );
        }
        
        if (filter.startTime) {
            logs = logs.filter(log => 
                new Date(log.timestamp) >= new Date(filter.startTime)
            );
        }
        
        if (filter.endTime) {
            logs = logs.filter(log => 
                new Date(log.timestamp) <= new Date(filter.endTime)
            );
        }
        
        if (filter.search) {
            const searchLower = filter.search.toLowerCase();
            logs = logs.filter(log => 
                log.message.toLowerCase().includes(searchLower) ||
                JSON.stringify(log.meta).toLowerCase().includes(searchLower)
            );
        }
        
        return logs.slice(0, filter.limit || 100);
    }

    async shutdown() {
        this.logger.info('Shutting down monitoring dashboard');
        
        // Close socket connections
        this.io.close();
        
        // Close server
        this.server.close();
        
        // Shutdown logger
        await this.logger.shutdown();
    }
}

// Start monitoring dashboard if run directly
if (require.main === module) {
    const dashboard = new MonitoringDashboard();
    
    // Graceful shutdown
    process.on('SIGTERM', async () => {
        await dashboard.shutdown();
        process.exit(0);
    });
    
    process.on('SIGINT', async () => {
        await dashboard.shutdown();
        process.exit(0);
    });
}

module.exports = MonitoringDashboard;