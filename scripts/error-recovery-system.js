#!/usr/bin/env node

/**
 * Comprehensive Error Recovery System for Media Server Infrastructure
 * Implements circuit breakers, retry logic, health checks, and self-healing mechanisms
 */

const fs = require('fs').promises;
const path = require('path');
const { spawn, exec } = require('child_process');
const { promisify } = require('util');
const execAsync = promisify(exec);

class CircuitBreaker {
    constructor(name, options = {}) {
        this.name = name;
        this.failureThreshold = options.failureThreshold || 5;
        this.recoveryTimeout = options.recoveryTimeout || 60000; // 1 minute
        this.monitorTimeout = options.monitorTimeout || 10000; // 10 seconds
        
        this.state = 'CLOSED'; // CLOSED, OPEN, HALF_OPEN
        this.failureCount = 0;
        this.lastFailureTime = null;
        this.nextAttempt = null;
        
        this.metrics = {
            totalRequests: 0,
            failedRequests: 0,
            successRequests: 0,
            circuitOpenCount: 0
        };
    }

    async execute(operation, fallback = null) {
        this.metrics.totalRequests++;
        
        if (this.state === 'OPEN') {
            if (Date.now() < this.nextAttempt) {
                console.log(`[${this.name}] Circuit OPEN - using fallback`);
                return fallback ? await fallback() : null;
            } else {
                this.state = 'HALF_OPEN';
                console.log(`[${this.name}] Circuit HALF_OPEN - attempting recovery`);
            }
        }

        try {
            const result = await this.executeWithTimeout(operation);
            this.onSuccess();
            return result;
        } catch (error) {
            this.onFailure(error);
            if (fallback) {
                console.log(`[${this.name}] Using fallback due to: ${error.message}`);
                return await fallback();
            }
            throw error;
        }
    }

    async executeWithTimeout(operation) {
        return new Promise((resolve, reject) => {
            const timeout = setTimeout(() => {
                reject(new Error(`Operation timeout after ${this.monitorTimeout}ms`));
            }, this.monitorTimeout);

            Promise.resolve(operation())
                .then(result => {
                    clearTimeout(timeout);
                    resolve(result);
                })
                .catch(error => {
                    clearTimeout(timeout);
                    reject(error);
                });
        });
    }

    onSuccess() {
        this.failureCount = 0;
        if (this.state === 'HALF_OPEN') {
            this.state = 'CLOSED';
            console.log(`[${this.name}] Circuit CLOSED - recovery successful`);
        }
        this.metrics.successRequests++;
    }

    onFailure(error) {
        this.failureCount++;
        this.lastFailureTime = Date.now();
        this.metrics.failedRequests++;
        
        console.error(`[${this.name}] Failure ${this.failureCount}/${this.failureThreshold}: ${error.message}`);

        if (this.failureCount >= this.failureThreshold) {
            this.state = 'OPEN';
            this.nextAttempt = Date.now() + this.recoveryTimeout;
            this.metrics.circuitOpenCount++;
            console.error(`[${this.name}] Circuit OPEN - recovery in ${this.recoveryTimeout/1000}s`);
        }
    }

    getStatus() {
        return {
            name: this.name,
            state: this.state,
            failureCount: this.failureCount,
            metrics: this.metrics,
            nextAttempt: this.nextAttempt ? new Date(this.nextAttempt).toISOString() : null
        };
    }
}

class RetryManager {
    static async withExponentialBackoff(operation, options = {}) {
        const maxRetries = options.maxRetries || 3;
        const baseDelay = options.baseDelay || 1000;
        const maxDelay = options.maxDelay || 30000;
        const backoffFactor = options.backoffFactor || 2;
        
        let lastError;
        
        for (let attempt = 0; attempt <= maxRetries; attempt++) {
            try {
                return await operation();
            } catch (error) {
                lastError = error;
                
                if (attempt === maxRetries) {
                    throw new Error(`Max retries (${maxRetries}) exceeded. Last error: ${error.message}`);
                }
                
                const delay = Math.min(baseDelay * Math.pow(backoffFactor, attempt), maxDelay);
                const jitter = Math.random() * delay * 0.1; // Add 10% jitter
                const actualDelay = delay + jitter;
                
                console.warn(`Attempt ${attempt + 1} failed: ${error.message}. Retrying in ${Math.round(actualDelay)}ms...`);
                await new Promise(resolve => setTimeout(resolve, actualDelay));
            }
        }
    }
}

class HealthChecker {
    constructor() {
        this.services = new Map();
        this.circuitBreakers = new Map();
        this.healthHistory = new Map();
        this.alertThresholds = {
            errorRate: 0.1, // 10% error rate
            responseTime: 5000, // 5 seconds
            consecutiveFailures: 3
        };
    }

    registerService(name, config) {
        this.services.set(name, {
            name,
            url: config.url,
            healthEndpoint: config.healthEndpoint || '/health',
            timeout: config.timeout || 10000,
            interval: config.interval || 30000,
            dependencies: config.dependencies || [],
            restartCommand: config.restartCommand,
            fallbackActions: config.fallbackActions || []
        });

        this.circuitBreakers.set(name, new CircuitBreaker(name, {
            failureThreshold: config.failureThreshold || 5,
            recoveryTimeout: config.recoveryTimeout || 60000
        }));

        this.healthHistory.set(name, []);
    }

    async checkHealth(serviceName) {
        const service = this.services.get(serviceName);
        if (!service) {
            throw new Error(`Service ${serviceName} not registered`);
        }

        const circuitBreaker = this.circuitBreakers.get(serviceName);
        const startTime = Date.now();

        return circuitBreaker.execute(
            async () => {
                const response = await this.makeHealthRequest(service);
                const responseTime = Date.now() - startTime;
                
                this.recordHealthCheck(serviceName, true, responseTime, null);
                return { healthy: true, responseTime, service: serviceName };
            },
            async () => {
                // Fallback: attempt service recovery
                console.log(`[${serviceName}] Attempting recovery...`);
                await this.attemptRecovery(service);
                return { healthy: false, recovered: true, service: serviceName };
            }
        );
    }

    async makeHealthRequest(service) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), service.timeout);

        try {
            const response = await fetch(`${service.url}${service.healthEndpoint}`, {
                signal: controller.signal,
                headers: { 'User-Agent': 'MediaServer-HealthChecker/1.0' }
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            return await response.json();
        } catch (error) {
            clearTimeout(timeoutId);
            throw error;
        }
    }

    async attemptRecovery(service) {
        console.log(`[${service.name}] Starting recovery procedures...`);
        
        // Execute fallback actions in order
        for (const action of service.fallbackActions) {
            try {
                await this.executeFallbackAction(service, action);
                console.log(`[${service.name}] Recovery action '${action.type}' completed`);
            } catch (error) {
                console.error(`[${service.name}] Recovery action '${action.type}' failed: ${error.message}`);
            }
        }

        // Restart service if configured
        if (service.restartCommand) {
            try {
                console.log(`[${service.name}] Executing restart command: ${service.restartCommand}`);
                await execAsync(service.restartCommand);
                console.log(`[${service.name}] Service restarted successfully`);
                
                // Wait for service to come up
                await new Promise(resolve => setTimeout(resolve, 10000));
            } catch (error) {
                console.error(`[${service.name}] Restart failed: ${error.message}`);
                throw error;
            }
        }
    }

    async executeFallbackAction(service, action) {
        switch (action.type) {
            case 'clear_cache':
                await this.clearServiceCache(service, action.path);
                break;
            case 'restart_container':
                await this.restartContainer(action.container);
                break;
            case 'scale_service':
                await this.scaleService(action.service, action.replicas);
                break;
            case 'flush_dns':
                await execAsync('docker exec -it nginx-proxy-manager nginx -s reload');
                break;
            case 'clean_temp':
                await this.cleanTempFiles(action.paths);
                break;
            default:
                console.warn(`[${service.name}] Unknown fallback action: ${action.type}`);
        }
    }

    async clearServiceCache(service, cachePath) {
        if (cachePath) {
            try {
                await execAsync(`docker exec -it ${service.name} rm -rf ${cachePath}/*`);
                console.log(`[${service.name}] Cache cleared: ${cachePath}`);
            } catch (error) {
                console.error(`[${service.name}] Failed to clear cache: ${error.message}`);
            }
        }
    }

    async restartContainer(containerName) {
        try {
            await execAsync(`docker restart ${containerName}`);
            console.log(`Container ${containerName} restarted`);
        } catch (error) {
            console.error(`Failed to restart container ${containerName}: ${error.message}`);
            throw error;
        }
    }

    async scaleService(serviceName, replicas) {
        try {
            await execAsync(`docker-compose up -d --scale ${serviceName}=${replicas}`);
            console.log(`Service ${serviceName} scaled to ${replicas} replicas`);
        } catch (error) {
            console.error(`Failed to scale service ${serviceName}: ${error.message}`);
            throw error;
        }
    }

    async cleanTempFiles(paths) {
        for (const path of paths) {
            try {
                await execAsync(`find ${path} -type f -name "*.tmp" -delete`);
                console.log(`Cleaned temp files in ${path}`);
            } catch (error) {
                console.error(`Failed to clean ${path}: ${error.message}`);
            }
        }
    }

    recordHealthCheck(serviceName, healthy, responseTime, error) {
        const history = this.healthHistory.get(serviceName) || [];
        const record = {
            timestamp: new Date().toISOString(),
            healthy,
            responseTime,
            error: error ? error.message : null
        };

        history.push(record);
        
        // Keep only last 100 records
        if (history.length > 100) {
            history.shift();
        }
        
        this.healthHistory.set(serviceName, history);
        this.checkAlertConditions(serviceName, history);
    }

    checkAlertConditions(serviceName, history) {
        if (history.length < 3) return;

        const recent = history.slice(-10);
        const failureRate = recent.filter(r => !r.healthy).length / recent.length;
        const avgResponseTime = recent
            .filter(r => r.healthy && r.responseTime)
            .reduce((sum, r) => sum + r.responseTime, 0) / recent.length;

        // Check consecutive failures
        const consecutiveFailures = this.getConsecutiveFailures(history);
        
        if (consecutiveFailures >= this.alertThresholds.consecutiveFailures) {
            this.sendAlert(serviceName, 'CONSECUTIVE_FAILURES', {
                count: consecutiveFailures,
                threshold: this.alertThresholds.consecutiveFailures
            });
        }

        if (failureRate > this.alertThresholds.errorRate) {
            this.sendAlert(serviceName, 'HIGH_ERROR_RATE', {
                rate: failureRate,
                threshold: this.alertThresholds.errorRate
            });
        }

        if (avgResponseTime > this.alertThresholds.responseTime) {
            this.sendAlert(serviceName, 'SLOW_RESPONSE', {
                avgTime: avgResponseTime,
                threshold: this.alertThresholds.responseTime
            });
        }
    }

    getConsecutiveFailures(history) {
        let count = 0;
        for (let i = history.length - 1; i >= 0; i--) {
            if (history[i].healthy) break;
            count++;
        }
        return count;
    }

    async sendAlert(serviceName, alertType, data) {
        const alert = {
            timestamp: new Date().toISOString(),
            service: serviceName,
            type: alertType,
            data,
            severity: this.getAlertSeverity(alertType)
        };

        console.error(`🚨 ALERT [${alert.severity}]: ${serviceName} - ${alertType}`, data);
        
        // Store alert for dashboard
        await this.storeAlert(alert);
        
        // Send notification if configured
        await this.sendNotification(alert);
    }

    getAlertSeverity(alertType) {
        const severityMap = {
            'CONSECUTIVE_FAILURES': 'CRITICAL',
            'HIGH_ERROR_RATE': 'HIGH',
            'SLOW_RESPONSE': 'MEDIUM',
            'RECOVERY_SUCCESS': 'INFO'
        };
        return severityMap[alertType] || 'LOW';
    }

    async storeAlert(alert) {
        try {
            const alertsFile = '/config/alerts.json';
            let alerts = [];
            
            try {
                const content = await fs.readFile(alertsFile, 'utf8');
                alerts = JSON.parse(content);
            } catch (error) {
                // File doesn't exist or is invalid, start fresh
            }
            
            alerts.push(alert);
            
            // Keep only last 1000 alerts
            if (alerts.length > 1000) {
                alerts = alerts.slice(-1000);
            }
            
            await fs.writeFile(alertsFile, JSON.stringify(alerts, null, 2));
        } catch (error) {
            console.error('Failed to store alert:', error.message);
        }
    }

    async sendNotification(alert) {
        // Implement notification logic (webhook, email, Slack, etc.)
        // This is a placeholder for notification integrations
        console.log(`📧 Notification sent for ${alert.service}: ${alert.type}`);
    }

    async startMonitoring() {
        console.log('🔍 Starting health monitoring...');
        
        for (const [serviceName, service] of this.services) {
            this.startServiceMonitoring(serviceName, service);
        }
    }

    startServiceMonitoring(serviceName, service) {
        const monitor = async () => {
            try {
                await this.checkHealth(serviceName);
            } catch (error) {
                console.error(`[${serviceName}] Health check failed: ${error.message}`);
            }
        };

        // Initial check
        monitor();
        
        // Schedule recurring checks
        setInterval(monitor, service.interval);
        
        console.log(`✅ Monitoring started for ${serviceName} (interval: ${service.interval}ms)`);
    }

    getOverallStatus() {
        const status = {
            timestamp: new Date().toISOString(),
            services: {},
            circuitBreakers: {},
            summary: {
                total: this.services.size,
                healthy: 0,
                unhealthy: 0,
                unknown: 0
            }
        };

        for (const [name, service] of this.services) {
            const history = this.healthHistory.get(name) || [];
            const lastCheck = history[history.length - 1];
            const circuitBreaker = this.circuitBreakers.get(name);
            
            const serviceStatus = {
                healthy: lastCheck?.healthy || false,
                lastCheck: lastCheck?.timestamp || null,
                responseTime: lastCheck?.responseTime || null,
                circuitState: circuitBreaker.state,
                failureCount: circuitBreaker.failureCount
            };
            
            status.services[name] = serviceStatus;
            status.circuitBreakers[name] = circuitBreaker.getStatus();
            
            if (serviceStatus.healthy) {
                status.summary.healthy++;
            } else if (lastCheck) {
                status.summary.unhealthy++;
            } else {
                status.summary.unknown++;
            }
        }

        return status;
    }
}

// Configuration for media server services
const serviceConfigs = {
    jellyfin: {
        url: 'http://localhost:8096',
        healthEndpoint: '/health',
        timeout: 10000,
        interval: 30000,
        failureThreshold: 3,
        recoveryTimeout: 60000,
        restartCommand: 'docker restart jellyfin',
        fallbackActions: [
            { type: 'clear_cache', path: '/config/jellyfin/cache' },
            { type: 'restart_container', container: 'jellyfin' }
        ]
    },
    sonarr: {
        url: 'http://localhost:8989',
        healthEndpoint: '/ping',
        timeout: 8000,
        interval: 30000,
        failureThreshold: 3,
        recoveryTimeout: 60000,
        restartCommand: 'docker restart sonarr',
        fallbackActions: [
            { type: 'restart_container', container: 'sonarr' }
        ]
    },
    radarr: {
        url: 'http://localhost:7878',
        healthEndpoint: '/ping',
        timeout: 8000,
        interval: 30000,
        failureThreshold: 3,
        recoveryTimeout: 60000,
        restartCommand: 'docker restart radarr',
        fallbackActions: [
            { type: 'restart_container', container: 'radarr' }
        ]
    },
    prowlarr: {
        url: 'http://localhost:9696',
        healthEndpoint: '/ping',
        timeout: 8000,
        interval: 30000,
        failureThreshold: 3,
        recoveryTimeout: 60000,
        restartCommand: 'docker restart prowlarr',
        fallbackActions: [
            { type: 'restart_container', container: 'prowlarr' }
        ]
    },
    qbittorrent: {
        url: 'http://localhost:8080',
        healthEndpoint: '/api/v2/app/version',
        timeout: 8000,
        interval: 30000,
        failureThreshold: 3,
        recoveryTimeout: 60000,
        restartCommand: 'docker restart qbittorrent',
        fallbackActions: [
            { type: 'restart_container', container: 'qbittorrent' }
        ]
    },
    plex: {
        url: 'http://localhost:32400',
        healthEndpoint: '/identity',
        timeout: 10000,
        interval: 30000,
        failureThreshold: 3,
        recoveryTimeout: 60000,
        restartCommand: 'docker restart plex',
        fallbackActions: [
            { type: 'clear_cache', path: '/config/plex/Library/Application Support/Plex Media Server/Cache' },
            { type: 'restart_container', container: 'plex' }
        ]
    }
};

// Main execution
async function main() {
    const healthChecker = new HealthChecker();
    
    // Register all services
    for (const [name, config] of Object.entries(serviceConfigs)) {
        healthChecker.registerService(name, config);
    }
    
    // Start monitoring
    await healthChecker.startMonitoring();
    
    // API server for status dashboard
    const express = require('express');
    const app = express();
    const port = process.env.HEALTH_CHECK_PORT || 3010;
    
    app.use(express.json());
    
    app.get('/health', (req, res) => {
        res.json({ status: 'ok', service: 'error-recovery-system' });
    });
    
    app.get('/status', (req, res) => {
        res.json(healthChecker.getOverallStatus());
    });
    
    app.get('/circuit-breakers', (req, res) => {
        const breakers = {};
        for (const [name, breaker] of healthChecker.circuitBreakers) {
            breakers[name] = breaker.getStatus();
        }
        res.json(breakers);
    });
    
    app.listen(port, () => {
        console.log(`🔧 Error Recovery System API listening on port ${port}`);
        console.log(`📊 Status dashboard: http://localhost:${port}/status`);
        console.log(`⚡ Circuit breakers: http://localhost:${port}/circuit-breakers`);
    });
    
    // Graceful shutdown
    process.on('SIGTERM', () => {
        console.log('Received SIGTERM, shutting down gracefully...');
        process.exit(0);
    });
    
    process.on('SIGINT', () => {
        console.log('Received SIGINT, shutting down gracefully...');
        process.exit(0);
    });
}

if (require.main === module) {
    main().catch(error => {
        console.error('Fatal error:', error);
        process.exit(1);
    });
}

module.exports = {
    CircuitBreaker,
    RetryManager,
    HealthChecker
};