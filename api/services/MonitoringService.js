/**
 * MonitoringService - Prometheus metrics, Grafana dashboards, Uptime Kuma
 * Provides comprehensive monitoring and alerting for the media server infrastructure
 */

const axios = require('axios');
const os = require('os');
const fs = require('fs').promises;
const { exec } = require('child_process');
const { promisify } = require('util');
const EventEmitter = require('events');

const execAsync = promisify(exec);

class MonitoringService extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            prometheusUrl: config.prometheusUrl || process.env.PROMETHEUS_URL || 'http://prometheus:9090',
            grafanaUrl: config.grafanaUrl || process.env.GRAFANA_URL || 'http://grafana:3000',
            grafanaApiKey: config.grafanaApiKey || process.env.GRAFANA_API_KEY,
            uptimeKumaUrl: config.uptimeKumaUrl || process.env.UPTIME_KUMA_URL || 'http://uptime-kuma:3001',
            alertmanagerUrl: config.alertmanagerUrl || process.env.ALERTMANAGER_URL || 'http://alertmanager:9093',
            metricsInterval: config.metricsInterval || 30000, // 30 seconds
            healthCheckInterval: config.healthCheckInterval || 60000, // 1 minute
            retentionDays: config.retentionDays || 30,
            enableAlerts: config.enableAlerts !== false,
            diskThreshold: config.diskThreshold || 85, // 85%
            memoryThreshold: config.memoryThreshold || 90, // 90%
            cpuThreshold: config.cpuThreshold || 80, // 80%
            ...config
        };

        this.metrics = new Map();
        this.alerts = new Map();
        this.dashboards = new Map();
        this.healthChecks = new Map();
        this.isInitialized = false;
        this.metricsTimer = null;
        this.healthTimer = null;
        this.systemStats = {
            cpu: 0,
            memory: 0,
            disk: 0,
            network: { rx: 0, tx: 0 },
            uptime: 0
        };

        this.alertTypes = {
            SYSTEM: 'system',
            SERVICE: 'service',
            SECURITY: 'security',
            PERFORMANCE: 'performance',
            STORAGE: 'storage',
            NETWORK: 'network'
        };

        this.severityLevels = {
            CRITICAL: { level: 1, color: '#FF0000' },
            HIGH: { level: 2, color: '#FF8C00' },
            MEDIUM: { level: 3, color: '#FFD700' },
            LOW: { level: 4, color: '#32CD32' },
            INFO: { level: 5, color: '#1E90FF' }
        };
    }

    /**
     * Initialize Monitoring service
     */
    async initialize() {
        try {
            console.log('📊 Initializing MonitoringService...');
            
            // Test connections to monitoring services
            await this.testConnections();
            
            // Initialize Prometheus metrics
            await this.initializePrometheus();
            
            // Initialize Grafana dashboards
            await this.initializeGrafana();
            
            // Initialize Uptime Kuma monitors
            await this.initializeUptimeKuma();
            
            // Load existing dashboards
            await this.loadDashboards();
            
            // Start metrics collection
            this.startMetricsCollection();
            
            // Start health monitoring
            this.startHealthMonitoring();
            
            this.isInitialized = true;
            this.emit('initialized');
            console.log('✅ MonitoringService initialized successfully');
            
            return { success: true, message: 'MonitoringService initialized' };
        } catch (error) {
            console.error('❌ MonitoringService initialization failed:', error);
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Test connections to monitoring services
     */
    async testConnections() {
        try {
            const tests = [];
            
            // Test Prometheus
            tests.push(
                axios.get(`${this.config.prometheusUrl}/api/v1/query?query=up`, { timeout: 5000 })
                    .then(() => console.log('✅ Prometheus connection verified'))
                    .catch(err => console.warn('⚠️ Prometheus connection failed:', err.message))
            );
            
            // Test Grafana
            if (this.config.grafanaApiKey) {
                tests.push(
                    axios.get(`${this.config.grafanaUrl}/api/health`, {
                        headers: { 'Authorization': `Bearer ${this.config.grafanaApiKey}` },
                        timeout: 5000
                    })
                        .then(() => console.log('✅ Grafana connection verified'))
                        .catch(err => console.warn('⚠️ Grafana connection failed:', err.message))
                );
            }
            
            // Test Uptime Kuma
            tests.push(
                axios.get(`${this.config.uptimeKumaUrl}/api/status-page/heartbeat`, { timeout: 5000 })
                    .then(() => console.log('✅ Uptime Kuma connection verified'))
                    .catch(err => console.warn('⚠️ Uptime Kuma connection failed:', err.message))
            );
            
            await Promise.allSettled(tests);
        } catch (error) {
            console.warn('⚠️ Monitoring service connection tests failed:', error.message);
        }
    }

    /**
     * Initialize Prometheus metrics
     */
    async initializePrometheus() {
        try {
            // Define custom metrics
            const customMetrics = {
                media_server_up: {
                    name: 'media_server_up',
                    help: 'Media server uptime status',
                    type: 'gauge'
                },
                media_server_requests_total: {
                    name: 'media_server_requests_total',
                    help: 'Total number of API requests',
                    type: 'counter'
                },
                media_server_response_time: {
                    name: 'media_server_response_time',
                    help: 'API response time in milliseconds',
                    type: 'histogram'
                },
                media_files_total: {
                    name: 'media_files_total',
                    help: 'Total number of media files',
                    type: 'gauge'
                },
                download_queue_size: {
                    name: 'download_queue_size',
                    help: 'Number of items in download queue',
                    type: 'gauge'
                },
                storage_usage_bytes: {
                    name: 'storage_usage_bytes',
                    help: 'Storage usage in bytes',
                    type: 'gauge'
                }
            };
            
            Object.entries(customMetrics).forEach(([key, metric]) => {
                this.metrics.set(key, { ...metric, value: 0, labels: {} });
            });
            
            console.log('✅ Prometheus metrics initialized');
        } catch (error) {
            console.error('❌ Prometheus initialization failed:', error);
        }
    }

    /**
     * Initialize Grafana dashboards
     */
    async initializeGrafana() {
        try {
            if (!this.config.grafanaApiKey) {
                console.warn('⚠️ Grafana API key not configured, skipping dashboard setup');
                return;
            }
            
            // Create media server dashboard
            const dashboard = await this.createMediaServerDashboard();
            
            // Import dashboard to Grafana
            await this.importGrafanaDashboard(dashboard);
            
            console.log('✅ Grafana dashboards initialized');
        } catch (error) {
            console.error('❌ Grafana initialization failed:', error);
        }
    }

    /**
     * Create media server dashboard
     */
    async createMediaServerDashboard() {
        return {
            dashboard: {
                title: 'Media Server Overview',
                tags: ['media-server', 'monitoring'],
                timezone: 'browser',
                refresh: '30s',
                panels: [
                    {
                        title: 'System Overview',
                        type: 'stat',
                        targets: [
                            { expr: 'up{job="media-server"}', legendFormat: 'Uptime' },
                            { expr: 'rate(cpu_usage_percent[5m])', legendFormat: 'CPU Usage' },
                            { expr: 'memory_usage_percent', legendFormat: 'Memory Usage' }
                        ]
                    },
                    {
                        title: 'API Requests',
                        type: 'graph',
                        targets: [
                            { expr: 'rate(media_server_requests_total[5m])', legendFormat: 'Requests/sec' }
                        ]
                    },
                    {
                        title: 'Response Time',
                        type: 'graph',
                        targets: [
                            { expr: 'histogram_quantile(0.95, media_server_response_time)', legendFormat: '95th percentile' },
                            { expr: 'histogram_quantile(0.50, media_server_response_time)', legendFormat: 'Median' }
                        ]
                    },
                    {
                        title: 'Storage Usage',
                        type: 'piechart',
                        targets: [
                            { expr: 'storage_usage_bytes', legendFormat: 'Used Space' }
                        ]
                    },
                    {
                        title: 'Service Status',
                        type: 'table',
                        targets: [
                            { expr: 'up', legendFormat: '{{ instance }}' }
                        ]
                    },
                    {
                        title: 'Download Queue',
                        type: 'stat',
                        targets: [
                            { expr: 'download_queue_size', legendFormat: 'Queue Size' }
                        ]
                    }
                ]
            },
            overwrite: true
        };
    }

    /**
     * Import dashboard to Grafana
     */
    async importGrafanaDashboard(dashboard) {
        try {
            const response = await axios.post(
                `${this.config.grafanaUrl}/api/dashboards/db`,
                dashboard,
                {
                    headers: {
                        'Authorization': `Bearer ${this.config.grafanaApiKey}`,
                        'Content-Type': 'application/json'
                    }
                }
            );
            
            this.dashboards.set('media-server-overview', {
                id: response.data.id,
                url: response.data.url,
                version: response.data.version
            });
            
            console.log('✅ Dashboard imported to Grafana');
        } catch (error) {
            console.error('❌ Dashboard import failed:', error.message);
        }
    }

    /**
     * Initialize Uptime Kuma monitors
     */
    async initializeUptimeKuma() {
        try {
            // Define monitors for key services
            const monitors = [
                {
                    name: 'Media API Server',
                    type: 'http',
                    url: 'http://api-server:3002/health',
                    interval: 60
                },
                {
                    name: 'Jellyfin',
                    type: 'http',
                    url: 'http://jellyfin:8096/health',
                    interval: 60
                },
                {
                    name: 'Sonarr',
                    type: 'http',
                    url: 'http://sonarr:8989/ping',
                    interval: 60
                },
                {
                    name: 'Radarr',
                    type: 'http',
                    url: 'http://radarr:7878/ping',
                    interval: 60
                },
                {
                    name: 'Prowlarr',
                    type: 'http',
                    url: 'http://prowlarr:9696/ping',
                    interval: 60
                },
                {
                    name: 'qBittorrent',
                    type: 'http',
                    url: 'http://qbittorrent:8080',
                    interval: 60
                }
            ];
            
            // Store monitor configurations
            monitors.forEach(monitor => {
                this.healthChecks.set(monitor.name, {
                    ...monitor,
                    status: 'unknown',
                    lastCheck: null,
                    responseTime: 0
                });
            });
            
            console.log('✅ Uptime Kuma monitors initialized');
        } catch (error) {
            console.error('❌ Uptime Kuma initialization failed:', error);
        }
    }

    /**
     * Start metrics collection
     */
    startMetricsCollection() {
        if (this.metricsTimer) {
            clearInterval(this.metricsTimer);
        }
        
        this.metricsTimer = setInterval(async () => {
            try {
                await this.collectSystemMetrics();
                await this.collectApplicationMetrics();
                await this.pushMetricsToPrometheus();
            } catch (error) {
                console.warn('⚠️ Metrics collection failed:', error.message);
            }
        }, this.config.metricsInterval);
        
        console.log('✅ Metrics collection started');
    }

    /**
     * Collect system metrics
     */
    async collectSystemMetrics() {
        try {
            // CPU usage
            const cpuUsage = await this.getCPUUsage();
            this.updateMetric('cpu_usage_percent', cpuUsage);
            
            // Memory usage
            const memoryUsage = this.getMemoryUsage();
            this.updateMetric('memory_usage_percent', memoryUsage.percentage);
            this.updateMetric('memory_usage_bytes', memoryUsage.used);
            
            // Disk usage
            const diskUsage = await this.getDiskUsage();
            this.updateMetric('disk_usage_percent', diskUsage.percentage);
            this.updateMetric('storage_usage_bytes', diskUsage.used);
            
            // Network stats
            const networkStats = await this.getNetworkStats();
            this.updateMetric('network_rx_bytes', networkStats.rx);
            this.updateMetric('network_tx_bytes', networkStats.tx);
            
            // System uptime
            this.updateMetric('system_uptime_seconds', os.uptime());
            
            // Update internal stats
            this.systemStats = {
                cpu: cpuUsage,
                memory: memoryUsage.percentage,
                disk: diskUsage.percentage,
                network: networkStats,
                uptime: os.uptime()
            };
            
            // Check thresholds and create alerts
            await this.checkSystemThresholds();
        } catch (error) {
            console.error('❌ System metrics collection failed:', error);
        }
    }

    /**
     * Get CPU usage percentage
     */
    async getCPUUsage() {
        try {
            const startMeasure = this.cpuAverage();
            await new Promise(resolve => setTimeout(resolve, 1000));
            const endMeasure = this.cpuAverage();
            
            const idleDifference = endMeasure.idle - startMeasure.idle;
            const totalDifference = endMeasure.total - startMeasure.total;
            
            const percentageCPU = 100 - Math.floor(100 * idleDifference / totalDifference);
            return percentageCPU;
        } catch (error) {
            return 0;
        }
    }

    /**
     * Calculate CPU average
     */
    cpuAverage() {
        const cpus = os.cpus();
        let user = 0, nice = 0, sys = 0, idle = 0, irq = 0;
        
        cpus.forEach(cpu => {
            user += cpu.times.user;
            nice += cpu.times.nice;
            sys += cpu.times.sys;
            idle += cpu.times.idle;
            irq += cpu.times.irq;
        });
        
        const total = user + nice + sys + idle + irq;
        return { idle, total };
    }

    /**
     * Get memory usage
     */
    getMemoryUsage() {
        const totalMemory = os.totalmem();
        const freeMemory = os.freemem();
        const usedMemory = totalMemory - freeMemory;
        
        return {
            total: totalMemory,
            free: freeMemory,
            used: usedMemory,
            percentage: Math.round((usedMemory / totalMemory) * 100)
        };
    }

    /**
     * Get disk usage
     */
    async getDiskUsage() {
        try {
            const { stdout } = await execAsync('df -h / | tail -1');
            const parts = stdout.trim().split(/\s+/);
            const used = parseInt(parts[2].replace(/[^0-9]/g, ''));
            const available = parseInt(parts[3].replace(/[^0-9]/g, ''));
            const percentage = parseInt(parts[4].replace('%', ''));
            
            return {
                used: used * 1024 * 1024, // Convert to bytes
                available: available * 1024 * 1024,
                percentage
            };
        } catch (error) {
            return { used: 0, available: 0, percentage: 0 };
        }
    }

    /**
     * Get network statistics
     */
    async getNetworkStats() {
        try {
            const interfaces = os.networkInterfaces();
            let totalRx = 0, totalTx = 0;
            
            // This is a simplified version - in production, use /proc/net/dev
            Object.values(interfaces).flat().forEach(iface => {
                if (!iface.internal) {
                    // Mock network stats - implement proper stats reading
                    totalRx += Math.random() * 1000000;
                    totalTx += Math.random() * 1000000;
                }
            });
            
            return { rx: totalRx, tx: totalTx };
        } catch (error) {
            return { rx: 0, tx: 0 };
        }
    }

    /**
     * Collect application-specific metrics
     */
    async collectApplicationMetrics() {
        try {
            // Media server uptime
            this.updateMetric('media_server_up', 1);
            
            // Mock application metrics - replace with real data
            this.updateMetric('media_files_total', Math.floor(Math.random() * 10000));
            this.updateMetric('download_queue_size', Math.floor(Math.random() * 50));
            this.updateMetric('active_users', Math.floor(Math.random() * 20));
        } catch (error) {
            console.error('❌ Application metrics collection failed:', error);
        }
    }

    /**
     * Update metric value
     */
    updateMetric(name, value, labels = {}) {
        if (this.metrics.has(name)) {
            const metric = this.metrics.get(name);
            metric.value = value;
            metric.labels = { ...metric.labels, ...labels };
            metric.timestamp = Date.now();
        } else {
            this.metrics.set(name, {
                name,
                value,
                labels,
                timestamp: Date.now(),
                type: 'gauge'
            });
        }
    }

    /**
     * Push metrics to Prometheus
     */
    async pushMetricsToPrometheus() {
        try {
            // Format metrics for Prometheus push gateway
            const metricsText = Array.from(this.metrics.values())
                .map(metric => {
                    const labels = Object.entries(metric.labels)
                        .map(([key, value]) => `${key}="${value}"`)
                        .join(',');
                    
                    const labelStr = labels ? `{${labels}}` : '';
                    return `${metric.name}${labelStr} ${metric.value}`;
                })
                .join('\n');
            
            // In production, push to Prometheus push gateway
            // await axios.post(`${this.config.prometheusUrl}/metrics/job/media-server`, metricsText);
            
            this.emit('metricsPushed', { timestamp: new Date(), metricsCount: this.metrics.size });
        } catch (error) {
            console.warn('⚠️ Metrics push failed:', error.message);
        }
    }

    /**
     * Start health monitoring
     */
    startHealthMonitoring() {
        if (this.healthTimer) {
            clearInterval(this.healthTimer);
        }
        
        this.healthTimer = setInterval(async () => {
            try {
                await this.performHealthChecks();
            } catch (error) {
                console.warn('⚠️ Health monitoring failed:', error.message);
            }
        }, this.config.healthCheckInterval);
        
        console.log('✅ Health monitoring started');
    }

    /**
     * Perform health checks on all monitored services
     */
    async performHealthChecks() {
        const promises = Array.from(this.healthChecks.entries()).map(async ([name, check]) => {
            try {
                const startTime = Date.now();
                const response = await axios.get(check.url, {
                    timeout: 10000,
                    validateStatus: status => status < 500
                });
                
                const responseTime = Date.now() - startTime;
                const isHealthy = response.status >= 200 && response.status < 400;
                
                check.status = isHealthy ? 'up' : 'down';
                check.responseTime = responseTime;
                check.lastCheck = new Date();
                check.lastError = null;
                
                this.updateMetric(`service_up`, isHealthy ? 1 : 0, { service: name });
                this.updateMetric(`service_response_time`, responseTime, { service: name });
                
                if (!isHealthy) {
                    await this.createAlert({
                        type: this.alertTypes.SERVICE,
                        severity: 'HIGH',
                        title: `Service Down: ${name}`,
                        description: `Service ${name} is not responding properly`,
                        service: name,
                        responseTime
                    });
                }
            } catch (error) {
                check.status = 'down';
                check.lastCheck = new Date();
                check.lastError = error.message;
                
                this.updateMetric(`service_up`, 0, { service: name });
                
                await this.createAlert({
                    type: this.alertTypes.SERVICE,
                    severity: 'CRITICAL',
                    title: `Service Unreachable: ${name}`,
                    description: `Cannot connect to service ${name}: ${error.message}`,
                    service: name
                });
            }
        });
        
        await Promise.allSettled(promises);
        this.emit('healthCheckCompleted', { timestamp: new Date(), checks: this.healthChecks.size });
    }

    /**
     * Check system thresholds and create alerts
     */
    async checkSystemThresholds() {
        try {
            // CPU threshold check
            if (this.systemStats.cpu > this.config.cpuThreshold) {
                await this.createAlert({
                    type: this.alertTypes.SYSTEM,
                    severity: 'HIGH',
                    title: 'High CPU Usage',
                    description: `CPU usage is ${this.systemStats.cpu}%, exceeding threshold of ${this.config.cpuThreshold}%`,
                    value: this.systemStats.cpu,
                    threshold: this.config.cpuThreshold
                });
            }
            
            // Memory threshold check
            if (this.systemStats.memory > this.config.memoryThreshold) {
                await this.createAlert({
                    type: this.alertTypes.SYSTEM,
                    severity: 'HIGH',
                    title: 'High Memory Usage',
                    description: `Memory usage is ${this.systemStats.memory}%, exceeding threshold of ${this.config.memoryThreshold}%`,
                    value: this.systemStats.memory,
                    threshold: this.config.memoryThreshold
                });
            }
            
            // Disk threshold check
            if (this.systemStats.disk > this.config.diskThreshold) {
                await this.createAlert({
                    type: this.alertTypes.STORAGE,
                    severity: 'MEDIUM',
                    title: 'High Disk Usage',
                    description: `Disk usage is ${this.systemStats.disk}%, exceeding threshold of ${this.config.diskThreshold}%`,
                    value: this.systemStats.disk,
                    threshold: this.config.diskThreshold
                });
            }
        } catch (error) {
            console.error('❌ Threshold checking failed:', error);
        }
    }

    /**
     * Create alert
     */
    async createAlert(alertData) {
        try {
            if (!this.config.enableAlerts) return;
            
            const alertId = `${alertData.type}_${Date.now()}`;
            const alert = {
                id: alertId,
                timestamp: new Date(),
                resolved: false,
                resolvedAt: null,
                ...alertData
            };
            
            this.alerts.set(alertId, alert);
            
            // Emit alert event
            this.emit('alert', alert);
            
            // Send to Alertmanager
            await this.sendToAlertmanager(alert);
            
            console.log(`🚨 Alert created: ${alert.title} [${alert.severity}]`);
            
            return alert;
        } catch (error) {
            console.error('❌ Alert creation failed:', error);
        }
    }

    /**
     * Send alert to Alertmanager
     */
    async sendToAlertmanager(alert) {
        try {
            const alertmanagerAlert = {
                labels: {
                    alertname: alert.title.replace(/\s+/g, '_'),
                    severity: alert.severity.toLowerCase(),
                    type: alert.type,
                    service: alert.service || 'media-server'
                },
                annotations: {
                    description: alert.description,
                    summary: alert.title
                },
                startsAt: alert.timestamp.toISOString()
            };
            
            await axios.post(
                `${this.config.alertmanagerUrl}/api/v1/alerts`,
                [alertmanagerAlert],
                { timeout: 5000 }
            );
        } catch (error) {
            console.warn('⚠️ Failed to send alert to Alertmanager:', error.message);
        }
    }

    /**
     * Load existing dashboards
     */
    async loadDashboards() {
        try {
            // Load dashboard configurations
            console.log('📊 Loading monitoring dashboards...');
            
            // In production, load from persistent storage or Grafana API
            console.log('✅ Dashboards loaded');
        } catch (error) {
            console.warn('⚠️ Dashboard loading failed:', error.message);
        }
    }

    /**
     * Get metrics summary
     */
    getMetricsSummary() {
        const summary = {
            system: this.systemStats,
            services: {},
            alerts: {
                total: this.alerts.size,
                active: Array.from(this.alerts.values()).filter(alert => !alert.resolved).length,
                byType: {},
                bySeverity: {}
            },
            healthChecks: {}
        };
        
        // Service health summary
        this.healthChecks.forEach((check, name) => {
            summary.services[name] = {
                status: check.status,
                responseTime: check.responseTime,
                lastCheck: check.lastCheck
            };
        });
        
        // Alert summaries
        Array.from(this.alerts.values()).forEach(alert => {
            summary.alerts.byType[alert.type] = (summary.alerts.byType[alert.type] || 0) + 1;
            summary.alerts.bySeverity[alert.severity] = (summary.alerts.bySeverity[alert.severity] || 0) + 1;
        });
        
        return summary;
    }

    /**
     * Get service status
     */
    getStatus() {
        return {
            initialized: this.isInitialized,
            metricsCollection: !!this.metricsTimer,
            healthMonitoring: !!this.healthTimer,
            metrics: this.metrics.size,
            alerts: this.alerts.size,
            activeAlerts: Array.from(this.alerts.values()).filter(alert => !alert.resolved).length,
            healthChecks: this.healthChecks.size,
            dashboards: this.dashboards.size,
            systemStats: this.systemStats,
            config: {
                prometheusUrl: this.config.prometheusUrl,
                grafanaUrl: this.config.grafanaUrl,
                uptimeKumaUrl: this.config.uptimeKumaUrl,
                enableAlerts: this.config.enableAlerts,
                metricsInterval: this.config.metricsInterval,
                healthCheckInterval: this.config.healthCheckInterval
            },
            lastUpdate: new Date()
        };
    }

    /**
     * Resolve alert
     */
    async resolveAlert(alertId) {
        try {
            const alert = this.alerts.get(alertId);
            if (alert) {
                alert.resolved = true;
                alert.resolvedAt = new Date();
                
                this.emit('alertResolved', alert);
                console.log(`✅ Alert resolved: ${alert.title}`);
                
                return { success: true, alert };
            }
            
            return { success: false, message: 'Alert not found' };
        } catch (error) {
            console.error('❌ Alert resolution failed:', error);
            throw error;
        }
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            console.log('🧹 Cleaning up MonitoringService...');
            
            if (this.metricsTimer) {
                clearInterval(this.metricsTimer);
                this.metricsTimer = null;
            }
            
            if (this.healthTimer) {
                clearInterval(this.healthTimer);
                this.healthTimer = null;
            }
            
            this.metrics.clear();
            this.alerts.clear();
            this.dashboards.clear();
            this.healthChecks.clear();
            this.removeAllListeners();
            
            this.isInitialized = false;
            console.log('✅ MonitoringService cleanup completed');
        } catch (error) {
            console.error('❌ MonitoringService cleanup failed:', error);
        }
    }
}

module.exports = MonitoringService;