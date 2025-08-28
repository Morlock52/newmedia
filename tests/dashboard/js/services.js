// Services Management Module

class ServicesManager {
    constructor() {
        this.services = new Map();
        this.healthCheckInterval = null;
        this.init();
    }

    init() {
        this.loadServiceConfigurations();
        this.startHealthChecks();
    }

    loadServiceConfigurations() {
        const serviceConfigs = [
            {
                id: 'plex',
                name: 'Plex Media Server',
                url: 'http://localhost:32400',
                icon: '🎬',
                description: 'Personal media streaming service',
                category: 'Media Server',
                ports: [32400],
                healthEndpoint: '/web/index.html',
                configurable: true,
                restartable: true
            },
            {
                id: 'sonarr',
                name: 'Sonarr',
                url: 'http://localhost:8989',
                icon: '📺',
                description: 'TV show collection manager',
                category: 'Content Manager',
                ports: [8989],
                healthEndpoint: '/api/v3/system/status',
                configurable: true,
                restartable: true
            },
            {
                id: 'radarr',
                name: 'Radarr',
                url: 'http://localhost:7878',
                icon: '🎥',
                description: 'Movie collection manager',
                category: 'Content Manager',
                ports: [7878],
                healthEndpoint: '/api/v3/system/status',
                configurable: true,
                restartable: true
            },
            {
                id: 'lidarr',
                name: 'Lidarr',
                url: 'http://localhost:8687',
                icon: '🎵',
                description: 'Music collection manager',
                category: 'Content Manager',
                ports: [8687],
                healthEndpoint: '/api/v1/system/status',
                configurable: true,
                restartable: true
            },
            {
                id: 'prowlarr',
                name: 'Prowlarr',
                url: 'http://localhost:9696',
                icon: '🔍',
                description: 'Indexer manager',
                category: 'Download Client',
                ports: [9696],
                healthEndpoint: '/api/v1/system/status',
                configurable: true,
                restartable: true
            },
            {
                id: 'bazarr',
                name: 'Bazarr',
                url: 'http://localhost:6767',
                icon: '📝',
                description: 'Subtitle manager',
                category: 'Utility',
                ports: [6767],
                healthEndpoint: '/api/system/status',
                configurable: true,
                restartable: true
            },
            {
                id: 'overseerr',
                name: 'Overseerr',
                url: 'http://localhost:5055',
                icon: '📋',
                description: 'Request management',
                category: 'Request Manager',
                ports: [5055],
                healthEndpoint: '/api/v1/status',
                configurable: true,
                restartable: true
            },
            {
                id: 'tautulli',
                name: 'Tautulli',
                url: 'http://localhost:8181',
                icon: '📊',
                description: 'Plex monitoring and statistics',
                category: 'Monitoring',
                ports: [8181],
                healthEndpoint: '/api/v2?apikey=&cmd=arnold',
                configurable: true,
                restartable: true
            },
            {
                id: 'qbittorrent',
                name: 'qBittorrent',
                url: 'http://localhost:8080',
                icon: '⬇️',
                description: 'BitTorrent client',
                category: 'Download Client',
                ports: [8080],
                healthEndpoint: '/api/v2/app/version',
                configurable: true,
                restartable: true
            },
            {
                id: 'sabnzbd',
                name: 'SABnzbd',
                url: 'http://localhost:8082',
                icon: '📦',
                description: 'Usenet downloader',
                category: 'Download Client',
                ports: [8082],
                healthEndpoint: '/api?mode=version',
                configurable: true,
                restartable: true
            }
        ];

        serviceConfigs.forEach(config => {
            this.services.set(config.id, {
                ...config,
                status: 'unknown',
                lastChecked: null,
                responseTime: null,
                uptime: 0,
                errors: [],
                metrics: {
                    cpu: 0,
                    memory: 0,
                    network: 0
                }
            });
        });
    }

    startHealthChecks() {
        this.checkAllServices();
        this.healthCheckInterval = setInterval(() => {
            this.checkAllServices();
        }, 30000); // Check every 30 seconds
    }

    async checkAllServices() {
        const promises = Array.from(this.services.keys()).map(serviceId => 
            this.checkServiceHealth(serviceId)
        );
        
        await Promise.allSettled(promises);
        this.updateServicesDisplay();
    }

    async checkServiceHealth(serviceId) {
        const service = this.services.get(serviceId);
        if (!service) return;

        const startTime = Date.now();
        
        try {
            const response = await this.pingService(service);
            const responseTime = Date.now() - startTime;
            
            service.status = response.ok ? 'online' : 'offline';
            service.responseTime = responseTime;
            service.lastChecked = new Date();
            service.uptime = response.ok ? service.uptime + 30 : 0;
            
            if (response.ok && response.data) {
                this.updateServiceMetrics(serviceId, response.data);
            }
            
            // Clear errors if service is back online
            if (response.ok && service.errors.length > 0) {
                service.errors = [];
            }
            
        } catch (error) {
            service.status = 'offline';
            service.responseTime = Date.now() - startTime;
            service.lastChecked = new Date();
            service.uptime = 0;
            
            this.addServiceError(serviceId, error.message);
        }
    }

    async pingService(service) {
        const controller = new AbortController();
        const timeoutId = setTimeout(() => controller.abort(), 5000);
        
        try {
            const response = await fetch(`/api/services/${service.id}/health`, {
                method: 'GET',
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            clearTimeout(timeoutId);
            
            if (response.ok) {
                const data = await response.json();
                return { ok: true, data };
            } else {
                return { ok: false, status: response.status };
            }
        } catch (error) {
            clearTimeout(timeoutId);
            
            if (error.name === 'AbortError') {
                throw new Error('Service timeout');
            }
            throw new Error(`Connection failed: ${error.message}`);
        }
    }

    updateServiceMetrics(serviceId, data) {
        const service = this.services.get(serviceId);
        if (!service) return;

        if (data.metrics) {
            service.metrics = {
                cpu: data.metrics.cpu || 0,
                memory: data.metrics.memory || 0,
                network: data.metrics.network || 0
            };
        }
    }

    addServiceError(serviceId, errorMessage) {
        const service = this.services.get(serviceId);
        if (!service) return;

        const error = {
            timestamp: new Date(),
            message: errorMessage
        };

        service.errors.unshift(error);
        
        // Keep only last 10 errors
        if (service.errors.length > 10) {
            service.errors = service.errors.slice(0, 10);
        }

        // Send notification for critical services
        if (['plex', 'sonarr', 'radarr'].includes(serviceId)) {
            window.dashboard?.showNotification(
                `${service.name} is experiencing issues: ${errorMessage}`,
                'warning'
            );
        }
    }

    updateServicesDisplay() {
        // Update main dashboard services grid
        this.renderServicesGrid();
        
        // Update detailed services page if active
        if (window.dashboard?.currentPage === 'services') {
            this.renderDetailedServices();
        }
    }

    renderServicesGrid() {
        const container = document.getElementById('services-grid');
        if (!container) return;

        const html = Array.from(this.services.values()).slice(0, 8).map(service => `
            <div class="service-card p-4 bg-glass rounded-xl border border-glass-border hover:border-neon-${this.getStatusColor(service.status)}/50 transition-all cursor-pointer"
                 onclick="servicesManager.showServiceDetails('${service.id}')">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-2">
                        <span class="text-lg">${service.icon}</span>
                        <span class="font-medium">${service.name}</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <div class="w-2 h-2 bg-neon-${this.getStatusColor(service.status)} rounded-full ${service.status === 'online' ? 'animate-pulse' : ''}"></div>
                        ${service.responseTime ? `<span class="text-xs text-gray-400">${service.responseTime}ms</span>` : ''}
                    </div>
                </div>
                <div class="flex items-center justify-between">
                    <p class="text-sm text-gray-400 capitalize">${service.status}</p>
                    ${service.status === 'online' ? `<span class="text-xs text-neon-green">✓</span>` : ''}
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    renderDetailedServices() {
        const container = document.getElementById('services-detailed');
        if (!container) return;

        const html = Array.from(this.services.values()).map(service => `
            <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border hover:border-neon-${this.getStatusColor(service.status)}/50 transition-all">
                <div class="flex items-start justify-between mb-4">
                    <div class="flex items-center space-x-3">
                        <span class="text-2xl">${service.icon}</span>
                        <div>
                            <h3 class="font-semibold text-lg">${service.name}</h3>
                            <p class="text-sm text-gray-400">${service.description}</p>
                            <p class="text-xs text-gray-500">${service.category}</p>
                        </div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="status-badge status-${service.status}">
                            <div class="w-2 h-2 bg-current rounded-full"></div>
                            ${service.status}
                        </span>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-4 mb-4">
                    <div>
                        <div class="text-xs text-gray-400">Response Time</div>
                        <div class="font-medium">${service.responseTime ? service.responseTime + 'ms' : 'N/A'}</div>
                    </div>
                    <div>
                        <div class="text-xs text-gray-400">Uptime</div>
                        <div class="font-medium">${this.formatUptime(service.uptime)}</div>
                    </div>
                    <div>
                        <div class="text-xs text-gray-400">Last Checked</div>
                        <div class="font-medium">${service.lastChecked ? this.formatTime(service.lastChecked) : 'Never'}</div>
                    </div>
                    <div>
                        <div class="text-xs text-gray-400">Port</div>
                        <div class="font-medium">${service.ports.join(', ')}</div>
                    </div>
                </div>

                ${service.status === 'online' && service.metrics ? `
                    <div class="grid grid-cols-3 gap-2 mb-4">
                        <div class="text-center">
                            <div class="text-xs text-gray-400">CPU</div>
                            <div class="text-sm font-medium">${service.metrics.cpu}%</div>
                        </div>
                        <div class="text-center">
                            <div class="text-xs text-gray-400">Memory</div>
                            <div class="text-sm font-medium">${service.metrics.memory}%</div>
                        </div>
                        <div class="text-center">
                            <div class="text-xs text-gray-400">Network</div>
                            <div class="text-sm font-medium">${service.metrics.network} MB/s</div>
                        </div>
                    </div>
                ` : ''}

                <div class="flex space-x-2">
                    <button onclick="servicesManager.openService('${service.id}')" 
                            class="btn btn-secondary flex-1 text-sm py-2" 
                            ${service.status !== 'online' ? 'disabled' : ''}>
                        Open
                    </button>
                    ${service.restartable ? `
                        <button onclick="servicesManager.restartService('${service.id}')" 
                                class="btn btn-secondary text-sm py-2">
                            Restart
                        </button>
                    ` : ''}
                    ${service.configurable ? `
                        <button onclick="servicesManager.configureService('${service.id}')" 
                                class="btn btn-secondary text-sm py-2">
                            Config
                        </button>
                    ` : ''}
                </div>

                ${service.errors.length > 0 ? `
                    <div class="mt-4 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                        <div class="text-sm font-medium text-red-400 mb-2">Recent Errors</div>
                        <div class="space-y-1">
                            ${service.errors.slice(0, 3).map(error => `
                                <div class="text-xs text-gray-400">
                                    ${this.formatTime(error.timestamp)}: ${error.message}
                                </div>
                            `).join('')}
                        </div>
                    </div>
                ` : ''}
            </div>
        `).join('');

        container.innerHTML = html;
    }

    getStatusColor(status) {
        const colors = {
            'online': 'green',
            'offline': 'pink',
            'warning': 'yellow',
            'unknown': 'gray'
        };
        return colors[status] || 'gray';
    }

    formatUptime(seconds) {
        if (seconds < 60) return `${seconds}s`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
        return `${Math.floor(seconds / 86400)}d`;
    }

    formatTime(date) {
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    async openService(serviceId) {
        const service = this.services.get(serviceId);
        if (service && service.status === 'online') {
            window.open(service.url, '_blank');
        }
    }

    async restartService(serviceId) {
        const service = this.services.get(serviceId);
        if (!service) return;

        try {
            window.dashboard?.showLoading(true);
            
            const response = await fetch(`/api/services/${serviceId}/restart`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                window.dashboard?.showNotification(`${service.name} restart initiated`, 'success');
                
                // Update status to show restarting
                service.status = 'warning';
                this.updateServicesDisplay();
                
                // Check status after a delay
                setTimeout(() => {
                    this.checkServiceHealth(serviceId);
                }, 5000);
            } else {
                throw new Error('Restart failed');
            }
        } catch (error) {
            window.dashboard?.showNotification(`Failed to restart ${service.name}: ${error.message}`, 'error');
        } finally {
            window.dashboard?.showLoading(false);
        }
    }

    async configureService(serviceId) {
        const service = this.services.get(serviceId);
        if (!service) return;

        // Open configuration modal or page
        this.showConfigurationModal(service);
    }

    showConfigurationModal(service) {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center';
        modal.innerHTML = `
            <div class="bg-dark-secondary rounded-xl border border-glass-border w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
                <div class="p-6 border-b border-glass-border">
                    <div class="flex items-center justify-between">
                        <h3 class="text-xl font-semibold">Configure ${service.name}</h3>
                        <button onclick="this.closest('.fixed').remove()" class="p-2 hover:bg-glass rounded-lg">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="p-6">
                    <div class="space-y-4">
                        <div>
                            <label class="block text-sm font-medium mb-2">Service URL</label>
                            <input type="url" value="${service.url}" class="form-input w-full" readonly>
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-2">Health Check Endpoint</label>
                            <input type="text" value="${service.healthEndpoint}" class="form-input w-full">
                        </div>
                        <div>
                            <label class="block text-sm font-medium mb-2">Ports</label>
                            <input type="text" value="${service.ports.join(', ')}" class="form-input w-full">
                        </div>
                        <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                            <div>
                                <div class="font-medium">Auto-restart on failure</div>
                                <div class="text-sm text-gray-400">Automatically restart this service if it goes offline</div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" ${service.autoRestart ? 'checked' : ''}>
                                <span class="slider"></span>
                            </label>
                        </div>
                        <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                            <div>
                                <div class="font-medium">Send notifications</div>
                                <div class="text-sm text-gray-400">Get notified when this service has issues</div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" ${service.notifications !== false ? 'checked' : ''}>
                                <span class="slider"></span>
                            </label>
                        </div>
                    </div>
                    <div class="flex space-x-3 mt-6">
                        <button class="btn btn-primary flex-1">Save Changes</button>
                        <button onclick="this.closest('.fixed').remove()" class="btn btn-secondary">Cancel</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }

    showServiceDetails(serviceId) {
        const service = this.services.get(serviceId);
        if (!service) return;

        // Navigate to services page and highlight the service
        if (window.dashboard) {
            window.dashboard.showPage('services');
            
            // Scroll to service after a brief delay
            setTimeout(() => {
                const serviceElement = document.querySelector(`[data-service-id="${serviceId}"]`);
                if (serviceElement) {
                    serviceElement.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    serviceElement.style.boxShadow = '0 0 20px rgba(58, 134, 255, 0.5)';
                    setTimeout(() => {
                        serviceElement.style.boxShadow = '';
                    }, 2000);
                }
            }, 500);
        }
    }

    async refreshAllServices() {
        window.dashboard?.showLoading(true);
        
        try {
            await this.checkAllServices();
            window.dashboard?.showNotification('All services refreshed successfully', 'success');
        } catch (error) {
            window.dashboard?.showNotification('Failed to refresh services', 'error');
        } finally {
            window.dashboard?.showLoading(false);
        }
    }

    getServiceStats() {
        const stats = {
            total: this.services.size,
            online: 0,
            offline: 0,
            warning: 0,
            unknown: 0
        };

        this.services.forEach(service => {
            stats[service.status]++;
        });

        return stats;
    }

    destroy() {
        if (this.healthCheckInterval) {
            clearInterval(this.healthCheckInterval);
        }
    }
}

// Initialize services manager
const servicesManager = new ServicesManager();

// Export for global access
window.servicesManager = servicesManager;