import Docker from 'dockerode';
import { promisify } from 'util';
import { exec } from 'child_process';
import path from 'path';
import fs from 'fs/promises';
import yaml from 'yaml';
import { logger } from '../utils/logger.js';
import { ServiceRegistry } from './serviceRegistry.js';
import { DependencyResolver } from './dependencyResolver.js';
import { ServiceValidator } from './serviceValidator.js';

const execAsync = promisify(exec);

export class ServiceOrchestrator {
  constructor(eventBus, configManager) {
    this.docker = new Docker();
    this.eventBus = eventBus;
    this.configManager = configManager;
    this.registry = new ServiceRegistry();
    this.resolver = new DependencyResolver();
    this.validator = new ServiceValidator();
    this.services = new Map();
  }

  async installService(serviceSpec) {
    try {
      logger.info(`Installing service: ${serviceSpec.name}`);
      
      // Validate service specification
      const validation = await this.validator.validate(serviceSpec);
      if (!validation.valid) {
        throw new Error(`Invalid service spec: ${validation.errors.join(', ')}`);
      }

      // Check if service already exists
      if (await this.registry.exists(serviceSpec.name)) {
        throw new Error(`Service ${serviceSpec.name} already installed`);
      }

      // Resolve dependencies
      const dependencies = await this.resolver.resolve(serviceSpec);
      logger.info(`Resolved dependencies: ${dependencies.join(', ')}`);

      // Install missing dependencies
      for (const dep of dependencies) {
        if (!await this.registry.exists(dep)) {
          await this.installDependency(dep);
        }
      }

      // Prepare environment
      const environment = await this.prepareEnvironment(serviceSpec);

      // Pull Docker images
      await this.pullImages(serviceSpec.images || [serviceSpec.image]);

      // Create volumes and networks
      await this.createResources(serviceSpec);

      // Deploy service
      const service = await this.deployService(serviceSpec, environment);

      // Register service
      await this.registry.register(service);

      // Configure integrations
      await this.configureIntegrations(service);

      // Emit event
      await this.eventBus.emit('service:installed', {
        service: service.name,
        version: service.version,
        status: 'installed'
      });

      logger.info(`Service ${serviceSpec.name} installed successfully`);
      return service;

    } catch (error) {
      logger.error(`Failed to install service ${serviceSpec.name}:`, error);
      
      // Rollback on failure
      await this.rollbackInstallation(serviceSpec.name);
      
      throw error;
    }
  }

  async uninstallService(serviceName) {
    try {
      logger.info(`Uninstalling service: ${serviceName}`);

      // Get service info
      const service = await this.registry.get(serviceName);
      if (!service) {
        throw new Error(`Service ${serviceName} not found`);
      }

      // Check for dependent services
      const dependents = await this.resolver.getDependents(serviceName);
      if (dependents.length > 0) {
        throw new Error(`Cannot uninstall ${serviceName}: required by ${dependents.join(', ')}`);
      }

      // Stop service
      await this.stopService(serviceName);

      // Remove containers
      await this.removeContainers(serviceName);

      // Clean up resources
      await this.cleanupResources(service);

      // Unregister service
      await this.registry.unregister(serviceName);

      // Emit event
      await this.eventBus.emit('service:uninstalled', {
        service: serviceName,
        status: 'uninstalled'
      });

      logger.info(`Service ${serviceName} uninstalled successfully`);
      return true;

    } catch (error) {
      logger.error(`Failed to uninstall service ${serviceName}:`, error);
      throw error;
    }
  }

  async startService(serviceName) {
    try {
      logger.info(`Starting service: ${serviceName}`);

      const service = await this.registry.get(serviceName);
      if (!service) {
        throw new Error(`Service ${serviceName} not found`);
      }

      // Check dependencies are running
      await this.ensureDependenciesRunning(service);

      // Start containers
      const containers = await this.getServiceContainers(serviceName);
      for (const container of containers) {
        await container.start();
      }

      // Wait for health check
      await this.waitForHealthy(serviceName);

      // Update status
      await this.registry.updateStatus(serviceName, 'running');

      // Emit event
      await this.eventBus.emit('service:started', {
        service: serviceName,
        status: 'running'
      });

      logger.info(`Service ${serviceName} started successfully`);
      return service;

    } catch (error) {
      logger.error(`Failed to start service ${serviceName}:`, error);
      throw error;
    }
  }

  async stopService(serviceName) {
    try {
      logger.info(`Stopping service: ${serviceName}`);

      const service = await this.registry.get(serviceName);
      if (!service) {
        throw new Error(`Service ${serviceName} not found`);
      }

      // Check for dependent services
      const runningDependents = await this.getRunningDependents(serviceName);
      if (runningDependents.length > 0) {
        throw new Error(`Cannot stop ${serviceName}: required by running services ${runningDependents.join(', ')}`);
      }

      // Stop containers
      const containers = await this.getServiceContainers(serviceName);
      for (const container of containers) {
        await container.stop({ t: 30 }); // 30 second timeout
      }

      // Update status
      await this.registry.updateStatus(serviceName, 'stopped');

      // Emit event
      await this.eventBus.emit('service:stopped', {
        service: serviceName,
        status: 'stopped'
      });

      logger.info(`Service ${serviceName} stopped successfully`);
      return service;

    } catch (error) {
      logger.error(`Failed to stop service ${serviceName}:`, error);
      throw error;
    }
  }

  async restartService(serviceName) {
    await this.stopService(serviceName);
    await this.startService(serviceName);
  }

  async updateService(serviceName, newConfig) {
    try {
      logger.info(`Updating service: ${serviceName}`);

      const service = await this.registry.get(serviceName);
      if (!service) {
        throw new Error(`Service ${serviceName} not found`);
      }

      // Validate new configuration
      const validation = await this.validator.validateConfig(newConfig);
      if (!validation.valid) {
        throw new Error(`Invalid configuration: ${validation.errors.join(', ')}`);
      }

      // Backup current configuration
      const backup = await this.configManager.backupConfig(serviceName);

      try {
        // Update configuration
        await this.configManager.updateConfig(serviceName, newConfig);

        // Recreate containers with new config
        await this.recreateContainers(serviceName, newConfig);

        // Update registry
        await this.registry.update(serviceName, { config: newConfig });

        // Emit event
        await this.eventBus.emit('service:updated', {
          service: serviceName,
          status: 'updated'
        });

        logger.info(`Service ${serviceName} updated successfully`);
        return service;

      } catch (error) {
        // Rollback on failure
        logger.error(`Update failed, rolling back configuration`);
        await this.configManager.restoreConfig(serviceName, backup);
        throw error;
      }

    } catch (error) {
      logger.error(`Failed to update service ${serviceName}:`, error);
      throw error;
    }
  }

  async getServiceStatus(serviceName) {
    const service = await this.registry.get(serviceName);
    if (!service) {
      throw new Error(`Service ${serviceName} not found`);
    }

    const containers = await this.getServiceContainers(serviceName);
    const containerStatuses = await Promise.all(
      containers.map(async (container) => {
        const info = await container.inspect();
        return {
          id: info.Id,
          name: info.Name,
          status: info.State.Status,
          health: info.State.Health?.Status || 'none',
          uptime: info.State.StartedAt,
          restartCount: info.RestartCount
        };
      })
    );

    return {
      service: service.name,
      version: service.version,
      status: service.status,
      containers: containerStatuses,
      dependencies: await this.resolver.resolve({ name: serviceName }),
      lastUpdated: service.lastUpdated
    };
  }

  async getServiceLogs(serviceName, options = {}) {
    const containers = await this.getServiceContainers(serviceName);
    const logs = {};

    for (const container of containers) {
      const stream = await container.logs({
        stdout: true,
        stderr: true,
        timestamps: true,
        tail: options.tail || 100,
        since: options.since || 0
      });

      logs[container.id] = stream.toString();
    }

    return logs;
  }

  async getServiceMetrics(serviceName) {
    const containers = await this.getServiceContainers(serviceName);
    const metrics = {};

    for (const container of containers) {
      const stats = await container.stats({ stream: false });
      
      metrics[container.id] = {
        cpu: this.calculateCPUPercent(stats),
        memory: {
          usage: stats.memory_stats.usage,
          limit: stats.memory_stats.limit,
          percent: (stats.memory_stats.usage / stats.memory_stats.limit) * 100
        },
        network: {
          rx_bytes: stats.networks?.eth0?.rx_bytes || 0,
          tx_bytes: stats.networks?.eth0?.tx_bytes || 0
        },
        disk: {
          read_bytes: stats.blkio_stats?.io_service_bytes_recursive?.[0]?.value || 0,
          write_bytes: stats.blkio_stats?.io_service_bytes_recursive?.[1]?.value || 0
        }
      };
    }

    return metrics;
  }

  // Private helper methods

  async pullImages(images) {
    for (const image of images) {
      logger.info(`Pulling Docker image: ${image}`);
      
      await new Promise((resolve, reject) => {
        this.docker.pull(image, (err, stream) => {
          if (err) return reject(err);
          
          this.docker.modem.followProgress(stream, (err, output) => {
            if (err) return reject(err);
            resolve(output);
          });
        });
      });
    }
  }

  async createResources(serviceSpec) {
    // Create networks
    if (serviceSpec.networks) {
      for (const networkName of serviceSpec.networks) {
        try {
          await this.docker.createNetwork({
            Name: networkName,
            Driver: 'bridge'
          });
          logger.info(`Created network: ${networkName}`);
        } catch (error) {
          if (!error.message.includes('already exists')) {
            throw error;
          }
        }
      }
    }

    // Create volumes
    if (serviceSpec.volumes) {
      for (const volume of serviceSpec.volumes) {
        try {
          await this.docker.createVolume({
            Name: volume.name,
            Driver: volume.driver || 'local'
          });
          logger.info(`Created volume: ${volume.name}`);
        } catch (error) {
          if (!error.message.includes('already exists')) {
            throw error;
          }
        }
      }
    }
  }

  async deployService(serviceSpec, environment) {
    const containerConfig = {
      Image: serviceSpec.image,
      name: serviceSpec.containerName || serviceSpec.name,
      Env: Object.entries(environment).map(([k, v]) => `${k}=${v}`),
      Labels: {
        'com.media-server.service': serviceSpec.name,
        'com.media-server.version': serviceSpec.version || 'latest'
      },
      HostConfig: {
        RestartPolicy: {
          Name: serviceSpec.restart || 'unless-stopped'
        }
      }
    };

    // Add port mappings
    if (serviceSpec.ports) {
      containerConfig.ExposedPorts = {};
      containerConfig.HostConfig.PortBindings = {};
      
      for (const port of serviceSpec.ports) {
        const [hostPort, containerPort] = port.split(':');
        containerConfig.ExposedPorts[`${containerPort}/tcp`] = {};
        containerConfig.HostConfig.PortBindings[`${containerPort}/tcp`] = [
          { HostPort: hostPort }
        ];
      }
    }

    // Add volume mounts
    if (serviceSpec.volumes) {
      containerConfig.HostConfig.Binds = serviceSpec.volumes.map(v => {
        if (typeof v === 'string') return v;
        return `${v.source}:${v.target}:${v.mode || 'rw'}`;
      });
    }

    // Add networks
    if (serviceSpec.networks) {
      containerConfig.NetworkingConfig = {
        EndpointsConfig: {}
      };
      
      for (const network of serviceSpec.networks) {
        containerConfig.NetworkingConfig.EndpointsConfig[network] = {};
      }
    }

    // Create and start container
    const container = await this.docker.createContainer(containerConfig);
    await container.start();

    return {
      name: serviceSpec.name,
      version: serviceSpec.version || 'latest',
      containerId: container.id,
      status: 'running',
      config: serviceSpec,
      environment: environment
    };
  }

  async prepareEnvironment(serviceSpec) {
    // Get base environment
    const baseEnv = await this.configManager.getEnvironment(serviceSpec.name);
    
    // Merge with service-specific environment
    const environment = {
      ...baseEnv,
      ...(serviceSpec.environment || {})
    };

    // Resolve environment variable references
    for (const [key, value] of Object.entries(environment)) {
      if (typeof value === 'string' && value.startsWith('${') && value.endsWith('}')) {
        const varName = value.slice(2, -1);
        environment[key] = process.env[varName] || '';
      }
    }

    return environment;
  }

  async getServiceContainers(serviceName) {
    const containers = await this.docker.listContainers({
      all: true,
      filters: {
        label: [`com.media-server.service=${serviceName}`]
      }
    });

    return Promise.all(
      containers.map(c => this.docker.getContainer(c.Id))
    );
  }

  async ensureDependenciesRunning(service) {
    const dependencies = await this.resolver.resolve(service);
    
    for (const dep of dependencies) {
      const depService = await this.registry.get(dep);
      if (!depService || depService.status !== 'running') {
        logger.info(`Starting dependency: ${dep}`);
        await this.startService(dep);
      }
    }
  }

  async waitForHealthy(serviceName, timeout = 60000) {
    const startTime = Date.now();
    
    while (Date.now() - startTime < timeout) {
      const containers = await this.getServiceContainers(serviceName);
      
      const allHealthy = await Promise.all(
        containers.map(async (container) => {
          const info = await container.inspect();
          return info.State.Health?.Status === 'healthy' || !info.Config.Healthcheck;
        })
      );

      if (allHealthy.every(h => h)) {
        return true;
      }

      await new Promise(resolve => setTimeout(resolve, 2000));
    }

    throw new Error(`Service ${serviceName} failed to become healthy within ${timeout}ms`);
  }

  calculateCPUPercent(stats) {
    const cpuDelta = stats.cpu_stats.cpu_usage.total_usage - 
                     stats.precpu_stats.cpu_usage.total_usage;
    const systemDelta = stats.cpu_stats.system_cpu_usage - 
                        stats.precpu_stats.system_cpu_usage;
    const cpuCount = stats.cpu_stats.online_cpus || 1;

    if (systemDelta > 0 && cpuDelta > 0) {
      return (cpuDelta / systemDelta) * cpuCount * 100;
    }
    
    return 0;
  }

  async rollbackInstallation(serviceName) {
    try {
      logger.info(`Rolling back installation of ${serviceName}`);
      
      // Remove containers
      await this.removeContainers(serviceName);
      
      // Clean up resources
      const service = await this.registry.get(serviceName);
      if (service) {
        await this.cleanupResources(service);
        await this.registry.unregister(serviceName);
      }
      
    } catch (error) {
      logger.error(`Failed to rollback installation:`, error);
    }
  }

  async removeContainers(serviceName) {
    const containers = await this.getServiceContainers(serviceName);
    
    for (const container of containers) {
      try {
        await container.stop({ t: 10 });
        await container.remove({ v: true });
      } catch (error) {
        if (!error.message.includes('No such container')) {
          throw error;
        }
      }
    }
  }

  async cleanupResources(service) {
    // Clean up volumes if marked for removal
    if (service.config?.volumes) {
      for (const volume of service.config.volumes) {
        if (volume.removeOnUninstall) {
          try {
            const dockerVolume = this.docker.getVolume(volume.name);
            await dockerVolume.remove();
          } catch (error) {
            logger.warn(`Failed to remove volume ${volume.name}:`, error.message);
          }
        }
      }
    }
  }

  async configureIntegrations(service) {
    // Configure service integrations based on type
    if (service.config?.integrations) {
      for (const integration of service.config.integrations) {
        await this.configureIntegration(service.name, integration);
      }
    }
  }

  async configureIntegration(serviceName, integration) {
    logger.info(`Configuring integration between ${serviceName} and ${integration.target}`);
    
    // Implementation depends on integration type
    switch (integration.type) {
      case 'api':
        await this.configureAPIIntegration(serviceName, integration);
        break;
      case 'webhook':
        await this.configureWebhookIntegration(serviceName, integration);
        break;
      case 'database':
        await this.configureDatabaseIntegration(serviceName, integration);
        break;
      default:
        logger.warn(`Unknown integration type: ${integration.type}`);
    }
  }

  async configureAPIIntegration(serviceName, integration) {
    // Configure API integration between services
    const sourceService = await this.registry.get(serviceName);
    const targetService = await this.registry.get(integration.target);
    
    if (!targetService) {
      logger.warn(`Target service ${integration.target} not found for integration`);
      return;
    }

    // Update source service configuration with target API details
    await this.configManager.updateConfig(serviceName, {
      integrations: {
        [integration.target]: {
          url: `http://${integration.target}:${targetService.config.port || 80}`,
          apiKey: await this.configManager.generateAPIKey(serviceName, integration.target)
        }
      }
    });
  }

  async getRunningDependents(serviceName) {
    const dependents = await this.resolver.getDependents(serviceName);
    const running = [];
    
    for (const dep of dependents) {
      const service = await this.registry.get(dep);
      if (service && service.status === 'running') {
        running.push(dep);
      }
    }
    
    return running;
  }

  async recreateContainers(serviceName, newConfig) {
    // Stop and remove existing containers
    await this.stopService(serviceName);
    await this.removeContainers(serviceName);
    
    // Deploy with new configuration
    const environment = await this.prepareEnvironment(newConfig);
    await this.deployService(newConfig, environment);
    
    // Start the service
    await this.startService(serviceName);
  }
}