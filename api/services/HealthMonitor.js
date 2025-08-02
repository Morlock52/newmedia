/**
 * Health Monitor Service
 * Comprehensive system and service health monitoring with metrics collection
 */

const { exec } = require('child_process');
const { promisify } = require('util');
const os = require('os');
const fs = require('fs').promises;

const execAsync = promisify(exec);

class HealthMonitor {
    constructor() {
        this.healthCache = new Map();
        this.metricsHistory = [];
        this.maxHistorySize = 1000;
        this.monitoringInterval = null;
        this.subscribers = new Set();
        
        // Health check endpoints for different services
        this.healthEndpoints = {
            jellyfin: { port: 8096, path: '/health', timeout: 5000 },
            sonarr: { port: 8989, path: '/api/v3/health', timeout: 5000 },
            radarr: { port: 7878, path: '/api/v3/health', timeout: 5000 },
            lidarr: { port: 8686, path: '/api/v1/health', timeout: 5000 },
            prowlarr: { port: 9696, path: '/api/v1/health', timeout: 5000 },
            bazarr: { port: 6767, path: '/api/system/health', timeout: 5000 },
            qbittorrent: { port: 8080, path: '/api/v2/app/version', timeout: 5000 },
            sabnzbd: { port: 8081, path: '/api', timeout: 5000 },
            overseerr: { port: 5055, path: '/api/v1/status', timeout: 5000 },
            tautulli: { port: 8181, path: '/api/v2', timeout: 5000 },
            prometheus: { port: 9090, path: '/-/healthy', timeout: 5000 },
            grafana: { port: 3000, path: '/api/health', timeout: 5000 },
            homepage: { port: 3001, path: '/api/config', timeout: 5000 },
            portainer: { port: 9000, path: '/api/status', timeout: 5000 },
            traefik: { port: 8082, path: '/ping', timeout: 5000 }
        };

        // System thresholds for alerts
        this.thresholds = {
            cpu: { warning: 70, critical: 90 },
            memory: { warning: 80, critical: 95 },
            disk: { warning: 85, critical: 95 },
            temperature: { warning: 75, critical: 85 },
            responseTime: { warning: 2000, critical: 5000 }
        };
    }

    async initialize() {
        try {
            console.log('HealthMonitor initialized successfully');
        } catch (error) {
            console.error('Failed to initialize HealthMonitor:', error);
            throw error;
        }
    }

    async getHealthOverview() {
        try {
            const [systemHealth, servicesHealth] = await Promise.all([
                this.getSystemHealth(),
                this.getServicesHealth()
            ]);

            const overview = {
                overall: this.calculateOverallHealth(systemHealth, servicesHealth),
                system: systemHealth,
                services: servicesHealth,
                timestamp: new Date().toISOString()
            };

            // Cache the result
            this.healthCache.set('overview', {
                data: overview,
                timestamp: Date.now()
            });

            return overview;
        } catch (error) {
            throw new Error('Failed to get health overview: ' + error.message);
        }
    }

    async getSystemHealth() {
        try {
            const [cpuInfo, memoryInfo, diskInfo, networkInfo, processInfo] = await Promise.all([
                this.getCPUInfo(),
                this.getMemoryInfo(),
                this.getDiskInfo(),
                this.getNetworkInfo(),
                this.getProcessInfo()
            ]);

            return {
                cpu: cpuInfo,
                memory: memoryInfo,
                disk: diskInfo,
                network: networkInfo,
                processes: processInfo,
                uptime: os.uptime(),
                loadAverage: os.loadavg(),
                platform: os.platform(),
                arch: os.arch(),
                nodeVersion: process.version,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to get system health: ' + error.message);
        }
    }

    async getCPUInfo() {
        try {
            const cpus = os.cpus();
            const numCPUs = cpus.length;
            
            // Get CPU usage
            let cpuUsage = 0;
            if (os.platform() === 'linux') {
                try {
                    const { stdout } = await execAsync("top -bn1 | grep 'Cpu(s)' | awk '{print $2}' | awk -F'%' '{print $1}'");
                    cpuUsage = parseFloat(stdout.trim()) || 0;
                } catch (error) {
                    // Fallback to load average approximation
                    cpuUsage = (os.loadavg()[0] / numCPUs) * 100;
                }
            } else {
                // Fallback for non-Linux systems
                cpuUsage = (os.loadavg()[0] / numCPUs) * 100;
            }

            const status = this.getHealthStatus(cpuUsage, this.thresholds.cpu);

            return {
                usage: Math.round(cpuUsage * 100) / 100,
                cores: numCPUs,
                model: cpus[0].model,
                speed: cpus[0].speed,
                status,
                loadAverage: os.loadavg()
            };
        } catch (error) {
            return {
                usage: 0,
                cores: os.cpus().length,
                status: 'unknown',
                error: error.message
            };
        }
    }

    async getMemoryInfo() {
        const totalMemory = os.totalmem();
        const freeMemory = os.freemem();
        const usedMemory = totalMemory - freeMemory;
        const usagePercent = (usedMemory / totalMemory) * 100;
        
        const status = this.getHealthStatus(usagePercent, this.thresholds.memory);

        return {
            total: totalMemory,
            free: freeMemory,
            used: usedMemory,
            usage: Math.round(usagePercent * 100) / 100,
            status,
            formatted: {
                total: this.formatBytes(totalMemory),
                free: this.formatBytes(freeMemory),
                used: this.formatBytes(usedMemory)
            }
        };
    }

    async getDiskInfo() {
        try {
            const disks = [];
            
            if (os.platform() === 'linux') {
                const { stdout } = await execAsync("df -h | grep -E '^/dev/' | awk '{print $1,$2,$3,$4,$5,$6}'");
                const lines = stdout.trim().split('\n');
                
                for (const line of lines) {
                    const [device, size, used, available, usage, mountpoint] = line.split(/\s+/);
                    const usagePercent = parseInt(usage.replace('%', ''));
                    
                    disks.push({
                        device,
                        size,
                        used,
                        available,
                        usage: usagePercent,
                        mountpoint,
                        status: this.getHealthStatus(usagePercent, this.thresholds.disk)
                    });
                }
            } else {
                // Fallback for non-Linux systems
                disks.push({
                    device: 'Unknown',
                    size: 'Unknown',
                    used: 'Unknown',
                    available: 'Unknown',
                    usage: 0,
                    mountpoint: '/',
                    status: 'unknown'
                });
            }

            return {
                disks,
                overall: disks.length > 0 ? Math.max(...disks.map(d => d.usage)) : 0
            };
        } catch (error) {
            return {
                disks: [],
                overall: 0,
                error: error.message
            };
        }
    }

    async getNetworkInfo() {
        try {
            const interfaces = os.networkInterfaces();
            const networkStats = {};

            // Get basic interface information
            for (const [name, addresses] of Object.entries(interfaces)) {
                if (addresses) {
                    const ipv4 = addresses.find(addr => addr.family === 'IPv4' && !addr.internal);
                    if (ipv4) {
                        networkStats[name] = {
                            address: ipv4.address,
                            netmask: ipv4.netmask,
                            mac: ipv4.mac
                        };
                    }
                }
            }

            // Try to get network statistics on Linux
            let rxBytes = 0, txBytes = 0;
            if (os.platform() === 'linux') {
                try {
                    const { stdout } = await execAsync("cat /proc/net/dev | tail -n +3 | awk '{rx+=$2; tx+=$10} END {print rx,tx}'");
                    const [rx, tx] = stdout.trim().split(' ').map(Number);
                    rxBytes = rx || 0;
                    txBytes = tx || 0;
                } catch (error) {
                    // Network stats not available
                }
            }

            return {
                interfaces: networkStats,
                stats: {
                    rxBytes,
                    txBytes,
                    rxFormatted: this.formatBytes(rxBytes),
                    txFormatted: this.formatBytes(txBytes)
                }
            };
        } catch (error) {
            return {
                interfaces: {},
                stats: { rxBytes: 0, txBytes: 0 },
                error: error.message
            };
        }
    }

    async getProcessInfo() {
        try {
            let dockerProcesses = 0;
            let totalProcesses = 0;

            if (os.platform() === 'linux') {
                try {
                    const { stdout: dockerPs } = await execAsync("ps aux | grep -c '[d]ocker'");
                    dockerProcesses = parseInt(dockerPs.trim()) || 0;
                    
                    const { stdout: totalPs } = await execAsync("ps aux | wc -l");
                    totalProcesses = parseInt(totalPs.trim()) - 1 || 0; // Subtract header line
                } catch (error) {
                    // Process counting failed
                }
            }

            return {
                total: totalProcesses,
                docker: dockerProcesses,
                node: {
                    pid: process.pid,
                    uptime: process.uptime(),
                    memoryUsage: process.memoryUsage()
                }
            };
        } catch (error) {
            return {
                total: 0,
                docker: 0,
                node: {
                    pid: process.pid,
                    uptime: process.uptime(),
                    memoryUsage: process.memoryUsage()
                },
                error: error.message
            };
        }
    }

    async getServicesHealth() {
        const services = [];
        const healthChecks = [];

        // Perform health checks for all configured services
        for (const [serviceName, config] of Object.entries(this.healthEndpoints)) {
            healthChecks.push(this.performServiceHealthCheck(serviceName, config));
        }

        const results = await Promise.allSettled(healthChecks);
        
        results.forEach((result, index) => {
            const serviceName = Object.keys(this.healthEndpoints)[index];
            if (result.status === 'fulfilled') {
                services.push(result.value);
            } else {
                services.push({
                    service: serviceName,
                    status: 'error',
                    error: result.reason.message,
                    timestamp: new Date().toISOString()
                });
            }
        });

        return {
            total: services.length,
            healthy: services.filter(s => s.status === 'healthy').length,
            unhealthy: services.filter(s => s.status === 'unhealthy').length,
            unknown: services.filter(s => s.status === 'unknown' || s.status === 'error').length,
            services
        };
    }

    async performServiceHealthCheck(serviceName, config) {
        const startTime = Date.now();
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), config.timeout);

            const url = `http://localhost:${config.port}${config.path}`;
            const response = await fetch(url, {
                method: 'GET',
                signal: controller.signal,
                headers: { 'Accept': 'application/json' }
            });

            clearTimeout(timeoutId);
            const responseTime = Date.now() - startTime;
            const responseTimeStatus = this.getHealthStatus(responseTime, this.thresholds.responseTime);

            let healthData = null;
            try {
                healthData = await response.json();
            } catch (e) {
                // Some endpoints don't return JSON
            }

            const status = response.ok ? 'healthy' : 'unhealthy';

            return {
                service: serviceName,
                status,
                responseTime,
                responseTimeStatus,
                httpStatus: response.status,
                url,
                data: healthData,
                timestamp: new Date().toISOString()
            };

        } catch (error) {
            const responseTime = Date.now() - startTime;
            
            let status = 'unknown';
            if (error.name === 'AbortError') {
                status = 'timeout';
            } else if (error.code === 'ECONNREFUSED') {
                status = 'unreachable';
            }

            return {
                service: serviceName,
                status,
                responseTime,
                error: error.message,
                url: `http://localhost:${config.port}${config.path}`,
                timestamp: new Date().toISOString()
            };
        }
    }

    async getDetailedHealthCheck() {
        const overview = await this.getHealthOverview();
        
        // Add additional detailed information
        const detailed = {
            ...overview,
            docker: await this.getDockerHealth(),
            filesystem: await this.getFilesystemHealth(),
            ports: await this.getPortsHealth(),
            certificates: await this.getCertificatesHealth()
        };

        return detailed;
    }

    async getDockerHealth() {
        try {
            const [containerStats, imageStats, volumeStats, networkStats] = await Promise.all([
                this.getDockerContainerStats(),
                this.getDockerImageStats(),  
                this.getDockerVolumeStats(),
                this.getDockerNetworkStats()
            ]);

            return {
                containers: containerStats,
                images: imageStats,
                volumes: volumeStats,
                networks: networkStats,
                daemon: await this.checkDockerDaemon()
            };
        } catch (error) {
            return {
                error: error.message,
                available: false
            };
        }
    }

    async getDockerContainerStats() {
        try {
            const { stdout } = await execAsync('docker ps --format "table {{.Names}},{{.Status}},{{.Ports}}" | tail -n +2');
            const containers = stdout.trim().split('\n').map(line => {
                const [name, status, ports] = line.split(',');
                return { name: name.trim(), status: status.trim(), ports: ports.trim() };
            });

            return {
                total: containers.length,
                running: containers.filter(c => c.status.includes('Up')).length,
                containers
            };
        } catch (error) {
            return { total: 0, running: 0, containers: [], error: error.message };
        }
    }

    async getDockerImageStats() {
        try {
            const { stdout } = await execAsync('docker images --format "table {{.Repository}},{{.Tag}},{{.Size}}" | tail -n +2');
            const images = stdout.trim().split('\n').map(line => {
                const [repository, tag, size] = line.split(',');
                return { repository: repository.trim(), tag: tag.trim(), size: size.trim() };
            });

            return {
                total: images.length,
                images
            };
        } catch (error) {
            return { total: 0, images: [], error: error.message };
        }
    }

    async getDockerVolumeStats() {
        try {
            const { stdout } = await execAsync('docker volume ls --format "table {{.Name}},{{.Driver}}" | tail -n +2');
            const volumes = stdout.trim().split('\n').map(line => {
                const [name, driver] = line.split(',');
                return { name: name.trim(), driver: driver.trim() };
            });

            return {
                total: volumes.length,
                volumes
            };
        } catch (error) {
            return { total: 0, volumes: [], error: error.message };
        }
    }

    async getDockerNetworkStats() {
        try {
            const { stdout } = await execAsync('docker network ls --format "table {{.Name}},{{.Driver}}" | tail -n +2');
            const networks = stdout.trim().split('\n').map(line => {
                const [name, driver] = line.split(',');
                return { name: name.trim(), driver: driver.trim() };
            });

            return {
                total: networks.length,
                networks
            };
        } catch (error) {
            return { total: 0, networks: [], error: error.message };
        }
    }

    async checkDockerDaemon() {
        try {
            await execAsync('docker info');
            return { status: 'healthy', message: 'Docker daemon is running' };
        } catch (error) {
            return { status: 'unhealthy', message: 'Docker daemon is not accessible', error: error.message };
        }
    }

    async getFilesystemHealth() {
        try {
            // Check important directories
            const importantPaths = [
                './config',
                './media-data', 
                './downloads',
                './secrets'
            ];

            const pathChecks = await Promise.all(
                importantPaths.map(async (path) => {
                    try {
                        const stats = await fs.stat(path);
                        return {
                            path,
                            exists: true,
                            isDirectory: stats.isDirectory(),
                            size: stats.size,
                            modified: stats.mtime,
                            permissions: stats.mode.toString(8)
                        };
                    } catch (error) {
                        return {
                            path,
                            exists: false,
                            error: error.message
                        };
                    }
                })
            );

            return {
                paths: pathChecks,
                healthy: pathChecks.filter(p => p.exists).length,
                total: pathChecks.length
            };
        } catch (error) {
            return { error: error.message };
        }
    }

    async getPortsHealth() {
        const ports = [8096, 8989, 7878, 8686, 9696, 6767, 8080, 8081, 5055, 8181, 9090, 3000, 3001, 9000, 8082];
        const portChecks = [];

        for (const port of ports) {
            portChecks.push(this.checkPort(port));
        }

        const results = await Promise.allSettled(portChecks);
        const portStatuses = results.map((result, index) => ({
            port: ports[index],
            status: result.status === 'fulfilled' ? result.value : 'error',
            error: result.status === 'rejected' ? result.reason.message : null
        }));

        return {
            total: ports.length,
            open: portStatuses.filter(p => p.status === 'open').length,
            closed: portStatuses.filter(p => p.status === 'closed').length,
            ports: portStatuses
        };
    }

    async checkPort(port) {
        return new Promise((resolve, reject) => {
            const net = require('net');
            const socket = new net.Socket();
            
            socket.setTimeout(1000);
            
            socket.on('connect', () => {
                socket.destroy();
                resolve('open');
            });
            
            socket.on('timeout', () => {
                socket.destroy();
                resolve('closed');
            });
            
            socket.on('error', () => {
                resolve('closed');
            });
            
            socket.connect(port, 'localhost');
        });
    }

    async getCertificatesHealth() {
        // Check for SSL certificates
        try {
            const certPaths = [
                './config/traefik/acme.json',
                './secrets/cert.pem',
                './secrets/key.pem'
            ];

            const certChecks = await Promise.all(
                certPaths.map(async (path) => {
                    try {
                        const stats = await fs.stat(path);
                        return {
                            path,
                            exists: true,
                            size: stats.size,
                            modified: stats.mtime
                        };
                    } catch (error) {
                        return {
                            path,
                            exists: false
                        };
                    }
                })
            );

            return {
                certificates: certChecks,
                total: certChecks.length,
                present: certChecks.filter(c => c.exists).length
            };
        } catch (error) {
            return { error: error.message };
        }
    }

    async getSystemMetrics() {
        const metrics = {
            timestamp: new Date().toISOString(),
            system: await this.getSystemHealth(),
            services: await this.getServicesHealth()
        };

        // Add to history
        this.metricsHistory.push(metrics);
        if (this.metricsHistory.length > this.maxHistorySize) {
            this.metricsHistory.shift();
        }

        return {
            current: metrics,
            history: this.metricsHistory.slice(-100), // Return last 100 entries
            trends: this.calculateTrends()
        };
    }

    calculateTrends() {
        if (this.metricsHistory.length < 2) {
            return null;
        }

        const recent = this.metricsHistory.slice(-10);
        const cpuTrend = this.calculateTrendFor(recent, 'system.cpu.usage');
        const memoryTrend = this.calculateTrendFor(recent, 'system.memory.usage');

        return {
            cpu: cpuTrend,
            memory: memoryTrend
        };
    }

    calculateTrendFor(data, path) {
        const values = data.map(item => this.getValueByPath(item, path)).filter(v => v !== undefined);
        
        if (values.length < 2) return null;

        const first = values[0];
        const last = values[values.length - 1];
        const change = last - first;
        const percentChange = (change / first) * 100;

        return {
            direction: change > 0 ? 'up' : change < 0 ? 'down' : 'stable',
            change: Math.round(change * 100) / 100,
            percentChange: Math.round(percentChange * 100) / 100
        };
    }

    getValueByPath(obj, path) {
        return path.split('.').reduce((current, key) => current && current[key], obj);
    }

    calculateOverallHealth(systemHealth, servicesHealth) {
        let score = 100;
        
        // Deduct points for system issues
        if (systemHealth.cpu.usage > this.thresholds.cpu.critical) score -= 30;
        else if (systemHealth.cpu.usage > this.thresholds.cpu.warning) score -= 15;
        
        if (systemHealth.memory.usage > this.thresholds.memory.critical) score -= 30;
        else if (systemHealth.memory.usage > this.thresholds.memory.warning) score -= 15;
        
        if (systemHealth.disk.overall > this.thresholds.disk.critical) score -= 20;
        else if (systemHealth.disk.overall > this.thresholds.disk.warning) score -= 10;
        
        // Deduct points for service issues
        const unhealthyServices = servicesHealth.unhealthy + servicesHealth.unknown;
        const serviceHealthPercent = ((servicesHealth.total - unhealthyServices) / servicesHealth.total) * 100;
        score = Math.min(score, serviceHealthPercent);
        
        let status = 'healthy';
        if (score < 60) status = 'critical';
        else if (score < 80) status = 'warning';
        
        return {
            score: Math.max(0, Math.round(score)),
            status
        };
    }

    getHealthStatus(value, thresholds) {
        if (value >= thresholds.critical) return 'critical';
        if (value >= thresholds.warning) return 'warning';
        return 'healthy';
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }

    startMonitoring(interval = 30000) {
        this.stopMonitoring();
        
        this.monitoringInterval = setInterval(async () => {
            try {
                const overview = await this.getHealthOverview();
                
                // Broadcast to subscribers
                this.broadcastToSubscribers('health-update', overview);
                
            } catch (error) {
                console.error('Health monitoring error:', error);
            }
        }, interval);
        
        console.log(`Health monitoring started with ${interval}ms interval`);
    }

    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
            console.log('Health monitoring stopped');
        }
    }

    subscribeClient(client) {
        this.subscribers.add(client);
        
        client.on('close', () => {
            this.subscribers.delete(client);
        });
    }

    broadcastToSubscribers(type, data) {
        const message = JSON.stringify({
            type,
            data,
            timestamp: new Date().toISOString()
        });

        this.subscribers.forEach(client => {
            if (client.readyState === client.OPEN) {
                client.send(message);
            }
        });
    }
}

module.exports = HealthMonitor;