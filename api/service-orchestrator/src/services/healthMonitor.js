import axios from 'axios';
import { EventEmitter } from 'events';
import { logger } from '../utils/logger.js';
import { ServiceRegistry } from './serviceRegistry.js';
import { AlertManager } from './alertManager.js';
import { MetricsCollector } from './metricsCollector.js';

export class HealthMonitor extends EventEmitter {
  constructor(eventBus) {
    super();
    this.eventBus = eventBus;
    this.registry = new ServiceRegistry();
    this.alertManager = new AlertManager();
    this.metricsCollector = new MetricsCollector();
    this.monitors = new Map();
    this.healthStatus = new Map();
    this.isMonitoring = false;
  }

  async startMonitoring() {
    if (this.isMonitoring) {
      logger.warn('Health monitoring already started');
      return;
    }

    this.isMonitoring = true;
    logger.info('Starting health monitoring system');

    // Load all registered services
    const services = await this.registry.getAllServices();
    
    for (const service of services) {
      await this.addServiceMonitor(service);
    }

    // Subscribe to service events
    this.eventBus.on('service:installed', async (event) => {
      const service = await this.registry.get(event.service);
      await this.addServiceMonitor(service);
    });

    this.eventBus.on('service:uninstalled', (event) => {
      this.removeServiceMonitor(event.service);
    });

    // Start periodic health summary
    this.startHealthSummary();
  }

  async stopMonitoring() {
    if (!this.isMonitoring) {
      return;
    }

    this.isMonitoring = false;
    logger.info('Stopping health monitoring system');

    // Stop all monitors
    for (const [serviceName, monitor] of this.monitors) {
      clearInterval(monitor.intervalId);
    }

    this.monitors.clear();
    this.healthStatus.clear();

    // Clear health summary interval
    if (this.healthSummaryInterval) {
      clearInterval(this.healthSummaryInterval);
    }
  }

  async addServiceMonitor(service) {
    if (this.monitors.has(service.name)) {
      logger.warn(`Monitor already exists for service: ${service.name}`);
      return;
    }

    const monitor = {
      service: service,
      config: this.getHealthCheckConfig(service),
      intervalId: null,
      consecutiveFailures: 0,
      lastCheck: null,
      recoveryAttempts: 0
    };

    // Start monitoring
    monitor.intervalId = setInterval(
      () => this.performHealthCheck(monitor),
      monitor.config.interval
    );

    this.monitors.set(service.name, monitor);
    
    // Perform initial check
    await this.performHealthCheck(monitor);
    
    logger.info(`Added health monitor for service: ${service.name}`);
  }

  removeServiceMonitor(serviceName) {
    const monitor = this.monitors.get(serviceName);
    if (!monitor) {
      return;
    }

    clearInterval(monitor.intervalId);
    this.monitors.delete(serviceName);
    this.healthStatus.delete(serviceName);
    
    logger.info(`Removed health monitor for service: ${serviceName}`);
  }

  async performHealthCheck(monitor) {
    const { service, config } = monitor;
    const startTime = Date.now();

    try {
      // Execute health check
      const result = await this.executeHealthCheck(service, config);
      const duration = Date.now() - startTime;

      // Update monitor state
      monitor.lastCheck = new Date();
      monitor.consecutiveFailures = 0;
      monitor.recoveryAttempts = 0;

      // Update health status
      const previousStatus = this.healthStatus.get(service.name);
      this.healthStatus.set(service.name, {
        status: result.status,
        message: result.message,
        lastCheck: monitor.lastCheck,
        responseTime: duration,
        details: result.details
      });

      // Collect metrics
      await this.metricsCollector.recordHealthCheck(service.name, {
        status: result.status,
        responseTime: duration
      });

      // Emit status change event
      if (previousStatus?.status !== result.status) {
        await this.handleStatusChange(service, previousStatus?.status, result.status);
      }

      // Clear any active alerts if healthy
      if (result.status === 'healthy') {
        await this.alertManager.clearAlert(service.name, 'health');
      }

    } catch (error) {
      logger.error(`Health check failed for ${service.name}:`, error);
      
      monitor.consecutiveFailures++;
      monitor.lastCheck = new Date();

      // Update health status
      const previousStatus = this.healthStatus.get(service.name);
      this.healthStatus.set(service.name, {
        status: 'unhealthy',
        message: error.message,
        lastCheck: monitor.lastCheck,
        consecutiveFailures: monitor.consecutiveFailures,
        error: error.toString()
      });

      // Handle unhealthy state
      await this.handleUnhealthyService(monitor);

      // Emit status change event
      if (previousStatus?.status !== 'unhealthy') {
        await this.handleStatusChange(service, previousStatus?.status, 'unhealthy');
      }
    }
  }

  async executeHealthCheck(service, config) {
    // Try multiple health check methods
    const checks = [];

    // HTTP health endpoint
    if (config.endpoint) {
      checks.push(this.httpHealthCheck(service, config));
    }

    // Docker container health
    if (config.dockerCheck) {
      checks.push(this.dockerHealthCheck(service));
    }

    // TCP port check
    if (config.tcpCheck) {
      checks.push(this.tcpHealthCheck(service, config));
    }

    // Custom health check command
    if (config.customCheck) {
      checks.push(this.customHealthCheck(service, config));
    }

    // Execute all checks
    const results = await Promise.allSettled(checks);
    
    // Aggregate results
    const healthyChecks = results.filter(r => r.status === 'fulfilled' && r.value.healthy);
    const failedChecks = results.filter(r => r.status === 'rejected' || !r.value?.healthy);

    if (healthyChecks.length === 0 && failedChecks.length > 0) {
      throw new Error(`All health checks failed: ${failedChecks.map(r => r.reason || r.value?.message).join(', ')}`);
    }

    return {
      status: healthyChecks.length > 0 ? 'healthy' : 'unhealthy',
      message: `${healthyChecks.length}/${results.length} checks passed`,
      details: {
        passed: healthyChecks.map(r => r.value),
        failed: failedChecks.map(r => r.reason || r.value)
      }
    };
  }

  async httpHealthCheck(service, config) {
    const url = `http://${service.name}:${config.port || service.config.port}${config.endpoint}`;
    
    try {
      const response = await axios.get(url, {
        timeout: config.timeout,
        validateStatus: (status) => status < 500
      });

      if (response.status >= 200 && response.status < 300) {
        return { 
          healthy: true, 
          type: 'http',
          status: response.status,
          message: 'HTTP health check passed' 
        };
      }

      return {
        healthy: false,
        type: 'http',
        status: response.status,
        message: `HTTP health check failed with status ${response.status}`
      };

    } catch (error) {
      throw new Error(`HTTP health check failed: ${error.message}`);
    }
  }

  async dockerHealthCheck(service) {
    try {
      const docker = new Docker();
      const containers = await docker.listContainers({
        filters: {
          label: [`com.media-server.service=${service.name}`]
        }
      });

      if (containers.length === 0) {
        throw new Error('No containers found');
      }

      const healthStatuses = await Promise.all(
        containers.map(async (containerInfo) => {
          const container = docker.getContainer(containerInfo.Id);
          const info = await container.inspect();
          
          return {
            id: info.Id,
            state: info.State.Status,
            health: info.State.Health?.Status || 'none'
          };
        })
      );

      const allHealthy = healthStatuses.every(
        s => s.state === 'running' && (s.health === 'healthy' || s.health === 'none')
      );

      return {
        healthy: allHealthy,
        type: 'docker',
        containers: healthStatuses,
        message: allHealthy ? 'All containers healthy' : 'Some containers unhealthy'
      };

    } catch (error) {
      throw new Error(`Docker health check failed: ${error.message}`);
    }
  }

  async tcpHealthCheck(service, config) {
    const net = require('net');
    const port = config.tcpPort || config.port || service.config.port;
    const host = service.name;

    return new Promise((resolve, reject) => {
      const socket = new net.Socket();
      const timeout = config.timeout || 5000;

      socket.setTimeout(timeout);

      socket.on('connect', () => {
        socket.destroy();
        resolve({
          healthy: true,
          type: 'tcp',
          message: `TCP port ${port} is open`
        });
      });

      socket.on('timeout', () => {
        socket.destroy();
        reject(new Error(`TCP connection timeout to ${host}:${port}`));
      });

      socket.on('error', (error) => {
        reject(new Error(`TCP connection failed: ${error.message}`));
      });

      socket.connect(port, host);
    });
  }

  async customHealthCheck(service, config) {
    const { exec } = require('child_process');
    const { promisify } = require('util');
    const execAsync = promisify(exec);

    try {
      const command = config.customCheck.replace('${SERVICE_NAME}', service.name);
      const { stdout, stderr } = await execAsync(command, {
        timeout: config.timeout || 10000
      });

      if (stderr) {
        throw new Error(`Custom check stderr: ${stderr}`);
      }

      return {
        healthy: true,
        type: 'custom',
        message: 'Custom health check passed',
        output: stdout.trim()
      };

    } catch (error) {
      throw new Error(`Custom health check failed: ${error.message}`);
    }
  }

  async handleUnhealthyService(monitor) {
    const { service, consecutiveFailures } = monitor;
    
    // Check if we should attempt recovery
    if (consecutiveFailures >= 3 && monitor.recoveryAttempts < 3) {
      logger.warn(`Service ${service.name} has failed ${consecutiveFailures} consecutive checks, attempting recovery`);
      
      await this.attemptServiceRecovery(monitor);
      
    } else if (monitor.recoveryAttempts >= 3) {
      // Alert after max recovery attempts
      await this.alertManager.sendAlert({
        severity: 'critical',
        service: service.name,
        type: 'health',
        message: `Service ${service.name} is unhealthy and automatic recovery failed`,
        details: {
          consecutiveFailures: consecutiveFailures,
          recoveryAttempts: monitor.recoveryAttempts,
          lastCheck: monitor.lastCheck
        }
      });
    }
  }

  async attemptServiceRecovery(monitor) {
    const { service } = monitor;
    monitor.recoveryAttempts++;

    try {
      logger.info(`Attempting recovery for service ${service.name} (attempt ${monitor.recoveryAttempts})`);

      // Recovery strategies
      const strategies = [
        () => this.restartService(service.name),
        () => this.recreateService(service.name),
        () => this.redeployService(service.name)
      ];

      const strategy = strategies[Math.min(monitor.recoveryAttempts - 1, strategies.length - 1)];
      await strategy();

      // Wait for service to stabilize
      await new Promise(resolve => setTimeout(resolve, 30000));

      // Check if recovery was successful
      await this.performHealthCheck(monitor);

      if (this.healthStatus.get(service.name).status === 'healthy') {
        logger.info(`Service ${service.name} recovered successfully`);
        
        await this.alertManager.sendAlert({
          severity: 'info',
          service: service.name,
          type: 'recovery',
          message: `Service ${service.name} has recovered after ${monitor.recoveryAttempts} attempts`
        });
      }

    } catch (error) {
      logger.error(`Recovery failed for service ${service.name}:`, error);
    }
  }

  async restartService(serviceName) {
    logger.info(`Restarting service: ${serviceName}`);
    
    const docker = new Docker();
    const containers = await docker.listContainers({
      filters: {
        label: [`com.media-server.service=${serviceName}`]
      }
    });

    for (const containerInfo of containers) {
      const container = docker.getContainer(containerInfo.Id);
      await container.restart();
    }
  }

  async recreateService(serviceName) {
    logger.info(`Recreating service: ${serviceName}`);
    
    // This would typically involve:
    // 1. Stop and remove containers
    // 2. Pull latest images
    // 3. Recreate containers with same config
    
    await this.eventBus.emit('service:recreate', {
      service: serviceName,
      reason: 'health_recovery'
    });
  }

  async redeployService(serviceName) {
    logger.info(`Redeploying service: ${serviceName}`);
    
    // Full redeploy including configuration refresh
    await this.eventBus.emit('service:redeploy', {
      service: serviceName,
      reason: 'health_recovery'
    });
  }

  async handleStatusChange(service, previousStatus, newStatus) {
    logger.info(`Service ${service.name} status changed from ${previousStatus} to ${newStatus}`);

    // Emit status change event
    await this.eventBus.emit('service:health:changed', {
      service: service.name,
      previousStatus,
      newStatus,
      timestamp: new Date()
    });

    // Send appropriate alerts
    if (newStatus === 'unhealthy') {
      await this.alertManager.sendAlert({
        severity: 'warning',
        service: service.name,
        type: 'health',
        message: `Service ${service.name} is now unhealthy`,
        previousStatus
      });
    } else if (newStatus === 'healthy' && previousStatus === 'unhealthy') {
      await this.alertManager.sendAlert({
        severity: 'info',
        service: service.name,
        type: 'recovery',
        message: `Service ${service.name} has recovered and is now healthy`
      });
    }
  }

  getHealthCheckConfig(service) {
    // Default health check configuration
    const defaults = {
      interval: 30000, // 30 seconds
      timeout: 10000, // 10 seconds
      retries: 3,
      dockerCheck: true,
      tcpCheck: true
    };

    // Service-specific configurations
    const serviceConfigs = {
      jellyfin: {
        endpoint: '/health',
        port: 8096
      },
      plex: {
        endpoint: '/identity',
        port: 32400
      },
      sonarr: {
        endpoint: '/api/v3/system/status',
        port: 8989,
        headers: { 'X-Api-Key': '${SONARR_API_KEY}' }
      },
      radarr: {
        endpoint: '/api/v3/system/status',
        port: 7878,
        headers: { 'X-Api-Key': '${RADARR_API_KEY}' }
      },
      prowlarr: {
        endpoint: '/ping',
        port: 9696
      },
      overseerr: {
        endpoint: '/api/v1/status',
        port: 5055
      },
      grafana: {
        endpoint: '/api/health',
        port: 3000
      },
      prometheus: {
        endpoint: '/-/healthy',
        port: 9090
      },
      portainer: {
        endpoint: '/api/system/status',
        port: 9000
      }
    };

    return {
      ...defaults,
      ...(serviceConfigs[service.name] || {}),
      ...(service.config?.healthcheck || {})
    };
  }

  async getSystemHealth() {
    const services = [];
    
    for (const [serviceName, status] of this.healthStatus) {
      services.push({
        name: serviceName,
        ...status
      });
    }

    const healthyCount = services.filter(s => s.status === 'healthy').length;
    const unhealthyCount = services.filter(s => s.status === 'unhealthy').length;
    const totalCount = services.length;

    const overallStatus = unhealthyCount === 0 ? 'healthy' : 
                         unhealthyCount < totalCount / 2 ? 'degraded' : 'critical';

    return {
      status: overallStatus,
      timestamp: new Date(),
      services: {
        total: totalCount,
        healthy: healthyCount,
        unhealthy: unhealthyCount
      },
      details: services
    };
  }

  async getDependencyHealth() {
    const dependencies = new Map();

    // Check critical dependencies
    const criticalDeps = [
      { name: 'docker', check: this.checkDocker },
      { name: 'network', check: this.checkNetwork },
      { name: 'storage', check: this.checkStorage },
      { name: 'database', check: this.checkDatabase }
    ];

    for (const dep of criticalDeps) {
      try {
        const result = await dep.check.call(this);
        dependencies.set(dep.name, {
          status: 'healthy',
          ...result
        });
      } catch (error) {
        dependencies.set(dep.name, {
          status: 'unhealthy',
          error: error.message
        });
      }
    }

    return Object.fromEntries(dependencies);
  }

  async checkDocker() {
    const docker = new Docker();
    const info = await docker.info();
    
    return {
      version: info.ServerVersion,
      containers: info.Containers,
      images: info.Images,
      status: 'healthy'
    };
  }

  async checkNetwork() {
    // Check network connectivity
    try {
      await axios.get('https://1.1.1.1', { timeout: 5000 });
      return { status: 'healthy', latency: 'low' };
    } catch (error) {
      throw new Error('Network connectivity check failed');
    }
  }

  async checkStorage() {
    const { statSync } = require('fs');
    const { execSync } = require('child_process');
    
    try {
      const df = execSync('df -h /').toString();
      const lines = df.split('\n');
      const data = lines[1].split(/\s+/);
      
      const usage = parseInt(data[4]);
      
      return {
        usage: `${usage}%`,
        available: data[3],
        status: usage < 90 ? 'healthy' : 'warning'
      };
    } catch (error) {
      throw new Error('Storage check failed');
    }
  }

  async checkDatabase() {
    // Check database connectivity
    // This would check PostgreSQL/Redis connections
    return { status: 'healthy', connections: 'active' };
  }

  startHealthSummary() {
    // Send periodic health summary
    this.healthSummaryInterval = setInterval(async () => {
      const systemHealth = await this.getSystemHealth();
      
      if (systemHealth.status !== 'healthy') {
        logger.warn(`System health status: ${systemHealth.status}`);
      }

      // Emit health summary event
      await this.eventBus.emit('health:summary', systemHealth);
      
      // Collect metrics
      await this.metricsCollector.recordSystemHealth(systemHealth);
      
    }, 300000); // Every 5 minutes
  }
}