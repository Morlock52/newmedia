/**
 * Docker Manager Service
 * Comprehensive Docker Compose service management with profile support
 */

const { exec, spawn } = require('child_process');
const { promisify } = require('util');
const fs = require('fs').promises;
const path = require('path');
const yaml = require('js-yaml');

const execAsync = promisify(exec);

class DockerManager {
    constructor() {
        this.projectPath = process.env.DOCKER_PROJECT_PATH || path.join(__dirname, '../../');
        this.composeFile = process.env.DOCKER_COMPOSE_FILE || 'docker-compose.yml';
        this.serviceCache = new Map();
        this.cacheTimeout = 30000; // 30 seconds
        
        // Service profiles configuration
        this.profiles = {
            minimal: ['jellyfin', 'homepage'],
            media: ['jellyfin', 'sonarr', 'radarr', 'bazarr', 'homepage'],
            download: ['qbittorrent', 'vpn', 'sabnzbd'],
            monitoring: ['prometheus', 'grafana', 'tautulli'],
            full: [] // All services
        };
        
        // Service dependencies
        this.dependencies = {
            'qbittorrent': ['vpn'],
            'sonarr': ['prowlarr'],
            'radarr': ['prowlarr'],
            'bazarr': ['sonarr', 'radarr']
        };
    }

    async initialize() {
        try {
            // Verify Docker is available
            await this.verifyDocker();
            
            // Load compose configuration
            await this.loadComposeConfiguration();
            
            // Initialize service definitions
            await this.initializeServiceDefinitions();
            
            console.log('DockerManager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize DockerManager:', error);
            throw error;
        }
    }

    async verifyDocker() {
        try {
            const { stdout } = await execAsync('docker --version');
            console.log('Docker version:', stdout.trim());
            
            const { stdout: composeVersion } = await execAsync('docker compose version');
            console.log('Docker Compose version:', composeVersion.trim());
        } catch (error) {
            throw new Error('Docker or Docker Compose not available: ' + error.message);
        }
    }

    async loadComposeConfiguration() {
        try {
            const composePath = path.join(this.projectPath, this.composeFile);
            const composeContent = await fs.readFile(composePath, 'utf8');
            this.composeConfig = yaml.load(composeContent);
            
            // Extract service names
            this.availableServices = Object.keys(this.composeConfig.services || {});
            console.log('Available services:', this.availableServices);
        } catch (error) {
            throw new Error('Failed to load Docker Compose configuration: ' + error.message);
        }
    }

    async initializeServiceDefinitions() {
        // Enhanced service definitions with health check endpoints
        this.serviceDefinitions = {
            jellyfin: {
                name: 'Jellyfin',
                description: 'Media Server',
                category: 'media',
                port: 8096,
                healthEndpoint: '/health',
                icon: '🎬',
                webUrl: 'http://localhost:8096',
                priority: 1
            },
            sonarr: {
                name: 'Sonarr',
                description: 'TV Show Manager',
                category: 'arr',
                port: 8989,
                healthEndpoint: '/api/v3/health',
                icon: '📺',
                webUrl: 'http://localhost:8989',
                priority: 2
            },
            radarr: {
                name: 'Radarr',
                description: 'Movie Manager',
                category: 'arr',
                port: 7878,
                healthEndpoint: '/api/v3/health',
                icon: '🍿',
                webUrl: 'http://localhost:7878',
                priority: 2
            },
            lidarr: {
                name: 'Lidarr',
                description: 'Music Manager',
                category: 'arr',
                port: 8686,
                healthEndpoint: '/api/v1/health',
                icon: '🎵',
                webUrl: 'http://localhost:8686',
                priority: 3
            },
            prowlarr: {
                name: 'Prowlarr',
                description: 'Indexer Manager',
                category: 'arr',
                port: 9696,
                healthEndpoint: '/api/v1/health',
                icon: '🔍',
                webUrl: 'http://localhost:9696',
                priority: 1
            },
            bazarr: {
                name: 'Bazarr',
                description: 'Subtitle Manager',
                category: 'arr',
                port: 6767,
                healthEndpoint: '/api/system/health',
                icon: '📝',
                webUrl: 'http://localhost:6767',
                priority: 3
            },
            qbittorrent: {
                name: 'qBittorrent',
                description: 'Torrent Client',
                category: 'download',
                port: 8080,
                healthEndpoint: '/api/v2/app/version',
                icon: '⬇️',
                webUrl: 'http://localhost:8080',
                priority: 2
            },
            sabnzbd: {
                name: 'SABnzbd',
                description: 'Usenet Client',
                category: 'download',
                port: 8081,
                healthEndpoint: '/api',
                icon: '📰',
                webUrl: 'http://localhost:8081',
                priority: 2
            },
            overseerr: {
                name: 'Overseerr',
                description: 'Request Management',
                category: 'request',
                port: 5055,
                healthEndpoint: '/api/v1/status',
                icon: '📝',
                webUrl: 'http://localhost:5055',
                priority: 3
            },
            tautulli: {
                name: 'Tautulli',
                description: 'Plex/Jellyfin Analytics',
                category: 'monitoring',
                port: 8181,
                healthEndpoint: '/api/v2',
                icon: '📊',
                webUrl: 'http://localhost:8181',
                priority: 3
            },
            prometheus: {
                name: 'Prometheus',
                description: 'Metrics Collection',
                category: 'monitoring',
                port: 9090,
                healthEndpoint: '/-/healthy',
                icon: '📈',
                webUrl: 'http://localhost:9090',
                priority: 4
            },
            grafana: {
                name: 'Grafana',
                description: 'Metrics Visualization',
                category: 'monitoring',
                port: 3000,
                healthEndpoint: '/api/health',
                icon: '📊',
                webUrl: 'http://localhost:3000',
                priority: 4
            },
            homepage: {
                name: 'Homepage',
                description: 'Dashboard',
                category: 'management',
                port: 3001,
                healthEndpoint: '/api/config',
                icon: '🏠',
                webUrl: 'http://localhost:3001',
                priority: 1
            },
            portainer: {
                name: 'Portainer',
                description: 'Container Management',
                category: 'management',
                port: 9000,
                healthEndpoint: '/api/status',
                icon: '🐳',
                webUrl: 'http://localhost:9000',
                priority: 4
            },
            traefik: {
                name: 'Traefik',
                description: 'Reverse Proxy',
                category: 'network',
                port: 8082,
                healthEndpoint: '/ping',
                icon: '🔀',
                webUrl: 'http://localhost:8082',
                priority: 1
            }
        };
    }

    async getAllServices() {
        try {
            const services = [];
            
            for (const serviceName of this.availableServices) {
                const status = await this.getServiceStatus(serviceName);
                services.push(status);
            }
            
            // Sort by priority and name
            services.sort((a, b) => {
                if (a.priority !== b.priority) {
                    return a.priority - b.priority;
                }
                return a.name.localeCompare(b.name);
            });
            
            return services;
        } catch (error) {
            throw new Error('Failed to get services: ' + error.message);
        }
    }

    async getServiceStatus(serviceName) {
        // Check cache first
        const cached = this.serviceCache.get(serviceName);
        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
            return cached.data;
        }

        try {
            const serviceDefinition = this.serviceDefinitions[serviceName] || {
                name: serviceName.charAt(0).toUpperCase() + serviceName.slice(1),
                description: 'Service',
                category: 'unknown',
                port: null,
                healthEndpoint: null,
                icon: '⚙️',
                priority: 5
            };

            // Get container status
            const { stdout } = await execAsync(
                `docker compose -f ${this.composeFile} ps --format json ${serviceName}`,
                { cwd: this.projectPath }
            );

            let containerInfo = {};
            if (stdout.trim()) {
                containerInfo = JSON.parse(stdout.trim());
            }

            const isRunning = containerInfo.State === 'running';
            
            // Get detailed container stats if running
            let stats = null;
            if (isRunning) {
                stats = await this.getContainerStats(containerInfo.Name);
            }

            // Perform health check if running and has health endpoint
            let healthCheck = null;
            if (isRunning && serviceDefinition.healthEndpoint) {
                healthCheck = await this.performHealthCheck(serviceDefinition);
            }

            const status = {
                service: serviceName,
                ...serviceDefinition,
                status: containerInfo.State || 'unknown',
                running: isRunning,
                containerId: containerInfo.ID || null,
                containerName: containerInfo.Name || null,
                image: containerInfo.Image || null,
                ports: containerInfo.Publishers || [],
                stats,
                healthCheck,
                lastChecked: new Date().toISOString()
            };

            // Cache the result
            this.serviceCache.set(serviceName, {
                data: status,
                timestamp: Date.now()
            });

            return status;
        } catch (error) {
            console.error(`Failed to get status for ${serviceName}:`, error);
            
            // Return fallback status
            return {
                service: serviceName,
                name: serviceName.charAt(0).toUpperCase() + serviceName.slice(1),
                description: 'Service status unavailable',
                category: 'unknown',
                status: 'unknown',
                running: false,
                error: error.message,
                lastChecked: new Date().toISOString()
            };
        }
    }

    async getContainerStats(containerName) {
        try {
            const { stdout } = await execAsync(
                `docker stats ${containerName} --no-stream --format "table {{.CPUPerc}},{{.MemUsage}},{{.NetIO}},{{.BlockIO}}"`
            );
            
            const lines = stdout.trim().split('\n');
            if (lines.length > 1) {
                const data = lines[1].split(',');
                return {
                    cpu: data[0] || '0%',
                    memory: data[1] || '0B / 0B',
                    network: data[2] || '0B / 0B',
                    disk: data[3] || '0B / 0B'
                };
            }
        } catch (error) {
            console.error(`Failed to get stats for ${containerName}:`, error);
        }
        
        return null;
    }

    async performHealthCheck(serviceDefinition) {
        const startTime = Date.now();
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            const url = `http://localhost:${serviceDefinition.port}${serviceDefinition.healthEndpoint}`;
            const response = await fetch(url, {
                method: 'GET',
                signal: controller.signal,
                headers: { 'Accept': 'application/json' }
            });

            clearTimeout(timeoutId);
            const responseTime = Date.now() - startTime;

            if (response.ok) {
                let data = null;
                try {
                    data = await response.json();
                } catch (e) {
                    // Some endpoints don't return JSON
                }

                return {
                    status: 'healthy',
                    responseTime,
                    httpStatus: response.status,
                    data
                };
            } else {
                return {
                    status: 'unhealthy',
                    responseTime,
                    httpStatus: response.status,
                    error: `HTTP ${response.status}`
                };
            }
        } catch (error) {
            const responseTime = Date.now() - startTime;
            
            return {
                status: error.name === 'AbortError' ? 'timeout' : 'unreachable',
                responseTime,
                error: error.message
            };
        }
    }

    async startServices(serviceNames = [], profile = null) {
        try {
            let command = `docker compose -f ${this.composeFile}`;
            
            // Add profile if specified
            if (profile && this.profiles[profile]) {
                const profileServices = this.profiles[profile];
                serviceNames = profileServices.length > 0 ? profileServices : this.availableServices;
            }
            
            // If specific services are provided, start only those
            if (serviceNames.length > 0) {
                // Resolve dependencies
                const resolvedServices = this.resolveDependencies(serviceNames);
                command += ` up -d ${resolvedServices.join(' ')}`;
            } else {
                command += ' up -d';
            }

            console.log('Executing:', command);
            const { stdout, stderr } = await execAsync(command, { cwd: this.projectPath });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return {
                success: true,
                services: serviceNames,
                profile,
                stdout,
                stderr,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to start services: ' + error.message);
        }
    }

    async stopServices(serviceNames = []) {
        try {
            let command = `docker compose -f ${this.composeFile}`;
            
            if (serviceNames.length > 0) {
                command += ` stop ${serviceNames.join(' ')}`;
            } else {
                command += ' down';
            }

            console.log('Executing:', command);
            const { stdout, stderr } = await execAsync(command, { cwd: this.projectPath });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return {
                success: true,
                services: serviceNames,
                stdout,
                stderr,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to stop services: ' + error.message);
        }
    }

    async restartServices(serviceNames = []) {
        try {
            let command = `docker compose -f ${this.composeFile}`;
            
            if (serviceNames.length > 0) {
                command += ` restart ${serviceNames.join(' ')}`;
            } else {
                command += ' restart';
            }

            console.log('Executing:', command);
            const { stdout, stderr } = await execAsync(command, { cwd: this.projectPath });
            
            // Clear cache to force refresh
            this.clearCache();
            
            return {
                success: true,
                services: serviceNames,
                stdout,
                stderr,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to restart services: ' + error.message);
        }
    }

    async getServiceLogs(serviceName, options = {}) {
        try {
            const { lines = 100, follow = false } = options;
            
            let command = `docker compose -f ${this.composeFile} logs --tail ${lines}`;
            if (follow) {
                command += ' -f';
            }
            command += ` ${serviceName}`;

            const { stdout } = await execAsync(command, { cwd: this.projectPath });
            
            return {
                service: serviceName,
                logs: stdout.split('\n').filter(line => line.trim()),
                lines,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error(`Failed to get logs for ${serviceName}: ` + error.message);
        }
    }

    resolveDependencies(serviceNames) {
        const resolved = new Set();
        const toProcess = [...serviceNames];
        
        while (toProcess.length > 0) {
            const service = toProcess.pop();
            resolved.add(service);
            
            // Add dependencies
            if (this.dependencies[service]) {
                for (const dep of this.dependencies[service]) {
                    if (!resolved.has(dep)) {
                        toProcess.push(dep);
                    }
                }
            }
        }
        
        return Array.from(resolved);
    }

    getProfiles() {
        return this.profiles;
    }

    clearCache() {
        this.serviceCache.clear();
    }

    async getComposeStatus() {
        try {
            const { stdout } = await execAsync(
                `docker compose -f ${this.composeFile} ps --format json`,
                { cwd: this.projectPath }
            );
            
            const containers = stdout.trim().split('\n')
                .filter(line => line.trim())
                .map(line => JSON.parse(line));
            
            return {
                totalContainers: containers.length,
                runningContainers: containers.filter(c => c.State === 'running').length,
                containers,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to get compose status: ' + error.message);
        }
    }
}

module.exports = DockerManager;