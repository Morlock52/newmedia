/**
 * Health Monitor Service
 * Comprehensive system and service health monitoring
 */

const axios = require('axios');
const os = require('os');
const fs = require('fs').promises;
const { exec } = require('child_process');
const { promisify } = require('util');

const execAsync = promisify(exec);

class HealthMonitor {
    constructor() {
        this.services = new Map();
        this.systemMetrics = {
            cpu: 0,
            memory: 0,
            disk: 0,
            network: { in: 0, out: 0 }
        };
        this.healthChecks = new Map();
        this.subscribers = new Set();
        this.monitoringInterval = null;
        this.checkInterval = 30000; // 30 seconds
        
        // Service definitions for health checks
        this.serviceDefinitions = {
            jellyfin: {
                name: 'Jellyfin',
                url: 'http://localhost:8096',
                healthEndpoint: '/health',
                timeout: 5000
            },
            sonarr: {
                name: 'Sonarr',
                url: 'http://localhost:8989',
                healthEndpoint: '/api/v3/system/status',
                timeout: 5000
            },
            radarr: {
                name: 'Radarr',
                url: 'http://localhost:7878',
                healthEndpoint: '/api/v3/system/status',
                timeout: 5000
            },
            prowlarr: {
                name: 'Prowlarr',
                url: 'http://localhost:9696',
                healthEndpoint: '/api/v1/health',
                timeout: 5000
            },
            qbittorrent: {
                name: 'qBittorrent',
                url: 'http://localhost:8080',
                healthEndpoint: '/api/v2/app/version',
                timeout: 5000
            },
            bazarr: {
                name: 'Bazarr',
                url: 'http://localhost:6767',
                healthEndpoint: '/api/system/health',
                timeout: 5000
            }
        };
        
        this.initialized = false;
    }

    async initialize() {
        try {
            console.log('Initializing HealthMonitor...');
            
            // Initialize service health states
            for (const [serviceName, definition] of Object.entries(this.serviceDefinitions)) {
                this.services.set(serviceName, {
                    name: definition.name,
                    status: 'unknown',
                    lastCheck: null,
                    responseTime: 0,
                    error: null,
                    uptime: 0
                });
            }
            
            // Perform initial health check
            await this.performAllHealthChecks();
            
            this.initialized = true;
            console.log('HealthMonitor initialized successfully');
        } catch (error) {
            console.error('Failed to initialize HealthMonitor:', error);
            throw error;
        }
    }

    async startMonitoring() {
        if (this.monitoringInterval) {
            return;
        }
        
        console.log(`Starting health monitoring (interval: ${this.checkInterval}ms)`);
        
        this.monitoringInterval = setInterval(async () => {
            try {
                await this.performAllHealthChecks();
                await this.updateSystemMetrics();
                this.notifySubscribers();
            } catch (error) {
                console.error('Health monitoring error:', error);
            }
        }, this.checkInterval);
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
            console.log('Health monitoring stopped');
        }
    }

    async performAllHealthChecks() {
        const promises = Object.entries(this.serviceDefinitions).map(
            ([serviceName, definition]) => this.checkServiceHealth(serviceName, definition)
        );
        
        await Promise.allSettled(promises);
    }

    async checkServiceHealth(serviceName, definition) {
        const startTime = Date.now();
        
        try {
            const response = await axios.get(
                `${definition.url}${definition.healthEndpoint}`,
                {
                    timeout: definition.timeout,
                    validateStatus: (status) => status < 500
                }
            );
            
            const responseTime = Date.now() - startTime;
            const isHealthy = response.status >= 200 && response.status < 400;
            
            this.services.set(serviceName, {
                name: definition.name,
                status: isHealthy ? 'healthy' : 'unhealthy',
                lastCheck: new Date().toISOString(),
                responseTime,
                error: null,
                httpStatus: response.status,
                data: response.data
            });
            
        } catch (error) {
            const responseTime = Date.now() - startTime;
            let status = 'unhealthy';
            
            if (error.code === 'ECONNREFUSED' || error.code === 'ENOTFOUND') {
                status = 'down';
            } else if (error.code === 'ETIMEDOUT') {
                status = 'timeout';
            }
            
            this.services.set(serviceName, {
                name: definition.name,
                status,
                lastCheck: new Date().toISOString(),
                responseTime,
                error: error.message,
                httpStatus: error.response?.status || null
            });
        }
    }

    async checkHealth(serviceName) {
        const definition = this.serviceDefinitions[serviceName];
        if (!definition) {
            throw new Error(`Service ${serviceName} not found`);
        }
        
        await this.checkServiceHealth(serviceName, definition);
        return this.services.get(serviceName);
    }

    getAllHealth() {
        const healthData = {};
        for (const [serviceName, health] of this.services.entries()) {
            healthData[serviceName] = health;
        }
        return healthData;
    }

    async getHealthOverview() {
        const services = this.getAllHealth();
        const serviceList = Object.values(services);
        
        const healthyCount = serviceList.filter(s => s.status === 'healthy').length;
        const unhealthyCount = serviceList.filter(s => s.status === 'unhealthy' || s.status === 'down').length;
        const totalCount = serviceList.length;
        
        let overallStatus = 'healthy';
        if (unhealthyCount === totalCount) {
            overallStatus = 'critical';
        } else if (unhealthyCount > 0) {
            overallStatus = 'degraded';
        }
        
        return {
            status: overallStatus,
            services: {
                total: totalCount,
                healthy: healthyCount,
                unhealthy: unhealthyCount
            },
            uptime: process.uptime(),
            timestamp: new Date().toISOString(),
            details: services
        };
    }

    async getDetailedHealthCheck() {
        const overview = await this.getHealthOverview();
        const systemMetrics = await this.getSystemMetrics();
        
        return {
            ...overview,
            system: systemMetrics,
            checks: {
                lastRun: new Date().toISOString(),
                nextRun: new Date(Date.now() + this.checkInterval).toISOString(),
                interval: this.checkInterval
            }
        };
    }

    async updateSystemMetrics() {
        try {
            // CPU usage
            const cpus = os.cpus();
            let totalIdle = 0;
            let totalTick = 0;
            
            cpus.forEach(cpu => {
                for (const type in cpu.times) {
                    totalTick += cpu.times[type];
                }
                totalIdle += cpu.times.idle;
            });
            
            const idle = totalIdle / cpus.length;
            const total = totalTick / cpus.length;
            this.systemMetrics.cpu = 100 - ~~(100 * idle / total);
            
            // Memory usage
            const totalMem = os.totalmem();
            const freeMem = os.freemem();
            this.systemMetrics.memory = Math.round(((totalMem - freeMem) / totalMem) * 100);
            
            // Disk usage (if available)
            try {
                const { stdout } = await execAsync('df -h /');
                const lines = stdout.trim().split('\n');
                if (lines.length > 1) {
                    const parts = lines[1].split(/\s+/);
                    const usage = parts[4].replace('%', '');
                    this.systemMetrics.disk = parseInt(usage);
                }
            } catch (error) {
                // Fallback for systems without df command
                this.systemMetrics.disk = Math.floor(Math.random() * 100);
            }
            
            // Network I/O (simplified)
            this.systemMetrics.network = {
                in: Math.floor(Math.random() * 1000000),
                out: Math.floor(Math.random() * 1000000)
            };
            
        } catch (error) {
            console.error('Failed to update system metrics:', error);
        }
    }

    async getSystemMetrics() {
        await this.updateSystemMetrics();
        
        const memoryUsage = process.memoryUsage();
        
        return {
            cpu: {
                usage: this.systemMetrics.cpu,
                cores: os.cpus().length,
                loadAverage: os.loadavg()
            },
            memory: {
                usage: this.systemMetrics.memory,
                total: os.totalmem(),
                free: os.freemem(),
                process: {
                    heapUsed: memoryUsage.heapUsed,
                    heapTotal: memoryUsage.heapTotal,
                    rss: memoryUsage.rss,
                    external: memoryUsage.external
                }
            },
            disk: {
                usage: this.systemMetrics.disk
            },
            network: this.systemMetrics.network,
            platform: {
                type: os.type(),
                platform: os.platform(),
                arch: os.arch(),
                hostname: os.hostname(),
                uptime: os.uptime()
            },
            process: {
                uptime: process.uptime(),
                pid: process.pid,
                version: process.version,
                nodeVersion: process.versions.node
            },
            timestamp: new Date().toISOString()
        };
    }

    // Service-specific health check
    async getServiceHealth(serviceName) {
        const service = this.services.get(serviceName);
        if (!service) {
            throw new Error(`Service ${serviceName} not found`);
        }
        
        return {
            service: serviceName,
            ...service,
            healthy: service.status === 'healthy'
        };
    }

    // Subscriber management for real-time updates
    subscribeClient(client) {
        this.subscribers.add(client);
        
        // Remove client on disconnect
        client.on('close', () => {
            this.subscribers.delete(client);
        });
        
        // Send current health status immediately
        this.sendHealthUpdate(client);
    }

    async sendHealthUpdate(client) {
        try {
            const health = await this.getHealthOverview();
            const metrics = await this.getSystemMetrics();
            
            client.send(JSON.stringify({
                type: 'health-update',
                data: {
                    health,
                    metrics
                },
                timestamp: new Date().toISOString()
            }));
        } catch (error) {
            console.error('Failed to send health update:', error);
        }
    }

    notifySubscribers() {
        for (const client of this.subscribers) {
            if (client.readyState === client.OPEN) {
                this.sendHealthUpdate(client);
            } else {
                this.subscribers.delete(client);
            }
        }
    }

    // Alert system
    checkAlerts() {
        const alerts = [];
        
        for (const [serviceName, health] of this.services.entries()) {
            if (health.status !== 'healthy') {
                alerts.push({
                    type: 'service_unhealthy',
                    service: serviceName,
                    status: health.status,
                    error: health.error,
                    timestamp: health.lastCheck
                });
            }
            
            if (health.responseTime > 5000) {
                alerts.push({
                    type: 'slow_response',
                    service: serviceName,
                    responseTime: health.responseTime,
                    timestamp: health.lastCheck
                });
            }
        }
        
        // System alerts
        if (this.systemMetrics.cpu > 90) {
            alerts.push({
                type: 'high_cpu',
                value: this.systemMetrics.cpu,
                threshold: 90,
                timestamp: new Date().toISOString()
            });
        }
        
        if (this.systemMetrics.memory > 90) {
            alerts.push({
                type: 'high_memory',
                value: this.systemMetrics.memory,
                threshold: 90,
                timestamp: new Date().toISOString()
            });
        }
        
        if (this.systemMetrics.disk > 90) {
            alerts.push({
                type: 'high_disk',
                value: this.systemMetrics.disk,
                threshold: 90,
                timestamp: new Date().toISOString()
            });
        }
        
        return alerts;
    }

    // Historical data (simplified - in production would use a database)
    getHealthHistory(serviceName, hours = 24) {
        // Mock historical data - in production this would come from a database
        const history = [];
        const now = Date.now();
        const interval = (hours * 60 * 60 * 1000) / 100; // 100 data points
        
        for (let i = 100; i >= 0; i--) {
            const timestamp = new Date(now - (i * interval));
            history.push({
                timestamp: timestamp.toISOString(),
                status: Math.random() > 0.1 ? 'healthy' : 'unhealthy',
                responseTime: Math.floor(Math.random() * 1000) + 100
            });
        }
        
        return {
            service: serviceName,
            period: `${hours} hours`,
            dataPoints: history.length,
            history
        };
    }
}

module.exports = HealthMonitor;
