/**
 * Docker Service Integration Client
 * Modern API integration for Docker service management with fallbacks
 */

class DockerServiceClient {
    constructor(options = {}) {
        this.baseURL = options.baseURL || 'http://localhost:3000';
        this.configAPI = new ConfigServerAPIClient({ baseURL: this.baseURL });
        
        // Service definitions with fallback data
        this.services = {
            jellyfin: {
                name: 'Jellyfin',
                description: 'Media Server',
                port: 8096,
                healthEndpoint: '/health',
                icon: '🎬'
            },
            sonarr: {
                name: 'Sonarr',
                description: 'TV Show Manager',
                port: 8989,
                healthEndpoint: '/api/v3/health',
                icon: '📺'
            },
            radarr: {
                name: 'Radarr',
                description: 'Movie Manager',
                port: 7878,
                healthEndpoint: '/api/v3/health',
                icon: '🍿'
            },
            prowlarr: {
                name: 'Prowlarr',
                description: 'Indexer Manager',
                port: 9696,
                healthEndpoint: '/api/v1/health',
                icon: '🔍'
            },
            bazarr: {
                name: 'Bazarr',
                description: 'Subtitle Manager',
                port: 6767,
                healthEndpoint: '/api/system/health',
                icon: '📝'
            },
            qbittorrent: {
                name: 'qBittorrent',
                description: 'Torrent Client',
                port: 8080,
                healthEndpoint: '/api/v2/app/version',
                icon: '⬇️'
            }
        };

        // Cache for service status
        this.statusCache = new Map();
        this.cacheTimeout = 30000; // 30 seconds
        
        // Event emitter
        this.events = new Map();
    }

    on(event, callback) {
        if (!this.events.has(event)) {
            this.events.set(event, []);
        }
        this.events.get(event).push(callback);
    }

    emit(event, data) {
        const callbacks = this.events.get(event);
        if (callbacks) {
            callbacks.forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event handler for ${event}:`, error);
                }
            });
        }
    }

    /**
     * Get all services with their status
     */
    async getAllServices() {
        try {
            const response = await this.configAPI.getServices();
            
            if (response.fallback) {
                console.warn('Using fallback service data');
                return this.getFallbackServices();
            }

            // Enhance with detailed health checks
            const services = response.data.services || [];
            const enhancedServices = await Promise.all(
                services.map(service => this.enhanceServiceStatus(service))
            );

            this.emit('services-updated', enhancedServices);
            return enhancedServices;

        } catch (error) {
            console.error('Failed to get services:', error);
            return this.getFallbackServices();
        }
    }

    /**
     * Get individual service status
     */
    async getServiceStatus(serviceName) {
        // Check cache first
        const cached = this.statusCache.get(serviceName);
        if (cached && Date.now() - cached.timestamp < this.cacheTimeout) {
            return cached.data;
        }

        try {
            const response = await this.configAPI.getServiceStatus(serviceName);
            
            if (response.fallback) {
                return this.getFallbackServiceStatus(serviceName);
            }

            const enhancedStatus = await this.enhanceServiceStatus(response.data);
            
            // Cache the result
            this.statusCache.set(serviceName, {
                data: enhancedStatus,
                timestamp: Date.now()
            });

            this.emit('service-status-updated', { service: serviceName, status: enhancedStatus });
            return enhancedStatus;

        } catch (error) {
            console.error(`Failed to get status for ${serviceName}:`, error);
            return this.getFallbackServiceStatus(serviceName);
        }
    }

    /**
     * Enhance service status with additional health checks
     */
    async enhanceServiceStatus(serviceData) {
        const serviceName = serviceData.service || serviceData.name;
        const serviceConfig = this.services[serviceName];
        
        if (!serviceConfig) {
            return {
                ...serviceData,
                enhanced: false,
                healthStatus: 'unknown'
            };
        }

        // Perform direct health check if service is running
        let healthStatus = 'unknown';
        let responseTime = null;
        let version = null;

        if (serviceData.running || serviceData.status === 'running') {
            const healthResult = await this.performHealthCheck(serviceConfig);
            healthStatus = healthResult.status;
            responseTime = healthResult.responseTime;
            version = healthResult.version;
        }

        return {
            ...serviceData,
            ...serviceConfig,
            healthStatus,
            responseTime,
            version,
            enhanced: true,
            lastChecked: new Date().toISOString()
        };
    }

    /**
     * Perform direct health check on service
     */
    async performHealthCheck(serviceConfig) {
        const startTime = Date.now();
        
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000); // 5 second timeout

            const url = `http://localhost:${serviceConfig.port}${serviceConfig.healthEndpoint}`;
            const response = await fetch(url, {
                method: 'GET',
                signal: controller.signal,
                headers: {
                    'Accept': 'application/json'
                }
            });

            clearTimeout(timeoutId);
            const responseTime = Date.now() - startTime;

            if (response.ok) {
                let version = null;
                try {
                    const data = await response.json();
                    version = data.version || data.appVersion || null;
                } catch (e) {
                    // Some endpoints don't return JSON
                }

                return {
                    status: 'healthy',
                    responseTime,
                    version
                };
            } else {
                return {
                    status: 'unhealthy',
                    responseTime,
                    error: `HTTP ${response.status}`
                };
            }

        } catch (error) {
            const responseTime = Date.now() - startTime;
            
            if (error.name === 'AbortError') {
                return {
                    status: 'timeout',
                    responseTime,
                    error: 'Health check timeout'
                };
            }

            return {
                status: 'unreachable',
                responseTime,
                error: error.message
            };
        }
    }

    /**
     * Start services
     */
    async startServices(serviceNames = []) {
        try {
            const response = await this.configAPI.startServices(serviceNames);
            
            this.emit('services-started', { 
                services: serviceNames.length ? serviceNames : 'all',
                success: true 
            });

            // Clear cache to force refresh
            this.clearCache();
            
            return response;

        } catch (error) {
            this.emit('services-started', { 
                services: serviceNames.length ? serviceNames : 'all',
                success: false,
                error: error.message 
            });
            throw error;
        }
    }

    /**
     * Stop services
     */
    async stopServices(serviceNames = []) {
        try {
            const response = await this.configAPI.stopServices(serviceNames);
            
            this.emit('services-stopped', { 
                services: serviceNames.length ? serviceNames : 'all',
                success: true 
            });

            // Clear cache to force refresh
            this.clearCache();
            
            return response;

        } catch (error) {
            this.emit('services-stopped', { 
                services: serviceNames.length ? serviceNames : 'all',
                success: false,
                error: error.message 
            });
            throw error;
        }
    }

    /**
     * Restart services
     */
    async restartServices(serviceNames = []) {
        try {
            const response = await this.configAPI.restartServices(serviceNames);
            
            this.emit('services-restarted', { 
                services: serviceNames.length ? serviceNames : 'all',
                success: true 
            });

            // Clear cache to force refresh
            this.clearCache();
            
            return response;

        } catch (error) {
            this.emit('services-restarted', { 
                services: serviceNames.length ? serviceNames : 'all',
                success: false,
                error: error.message 
            });
            throw error;
        }
    }

    /**
     * Get fallback services data
     */
    getFallbackServices() {
        return Object.keys(this.services).map(key => ({
            service: key,
            ...this.services[key],
            status: 'unknown',
            running: false,
            message: 'Service status unavailable - using fallback data',
            healthStatus: 'unknown',
            fallback: true,
            lastChecked: new Date().toISOString()
        }));
    }

    /**
     * Get fallback status for individual service
     */
    getFallbackServiceStatus(serviceName) {
        const serviceConfig = this.services[serviceName];
        
        return {
            service: serviceName,
            ...serviceConfig,
            status: 'unknown',
            running: false,
            message: 'Service status unavailable - using fallback data',
            healthStatus: 'unknown',
            fallback: true,
            lastChecked: new Date().toISOString()
        };
    }

    /**
     * Clear status cache
     */
    clearCache() {
        this.statusCache.clear();
        this.emit('cache-cleared', { timestamp: Date.now() });
    }

    /**
     * Start monitoring services periodically
     */
    startMonitoring(interval = 60000) { // 1 minute default
        this.stopMonitoring(); // Stop any existing monitoring
        
        this.monitoringInterval = setInterval(async () => {
            try {
                const services = await this.getAllServices();
                this.emit('monitoring-update', { 
                    services, 
                    timestamp: Date.now() 
                });
            } catch (error) {
                console.error('Monitoring error:', error);
                this.emit('monitoring-error', { 
                    error: error.message, 
                    timestamp: Date.now() 
                });
            }
        }, interval);

        this.emit('monitoring-started', { interval });
    }

    /**
     * Stop monitoring
     */
    stopMonitoring() {
        if (this.monitoringInterval) {
            clearInterval(this.monitoringInterval);
            this.monitoringInterval = null;
            this.emit('monitoring-stopped', { timestamp: Date.now() });
        }
    }

    /**
     * Get service URLs for direct access
     */
    getServiceUrls() {
        const urls = {};
        
        Object.keys(this.services).forEach(key => {
            const service = this.services[key];
            urls[key] = {
                name: service.name,
                url: `http://localhost:${service.port}`,
                description: service.description,
                icon: service.icon
            };
        });

        return urls;
    }

    /**
     * Generate service health report
     */
    async generateHealthReport() {
        const services = await this.getAllServices();
        
        const report = {
            timestamp: new Date().toISOString(),
            totalServices: services.length,
            runningServices: services.filter(s => s.running).length,
            healthyServices: services.filter(s => s.healthStatus === 'healthy').length,
            services: services.map(service => ({
                name: service.name,
                status: service.status,
                running: service.running,
                healthStatus: service.healthStatus,
                responseTime: service.responseTime,
                version: service.version,
                lastChecked: service.lastChecked
            }))
        };

        this.emit('health-report-generated', report);
        return report;
    }
}

// Export for browser
if (typeof window !== 'undefined') {
    window.DockerServiceClient = DockerServiceClient;
}

// Export for Node.js
if (typeof module !== 'undefined' && module.exports) {
    module.exports = DockerServiceClient;
}