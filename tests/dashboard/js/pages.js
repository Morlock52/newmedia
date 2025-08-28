// Page Content Management Module

class PageManager {
    constructor() {
        this.pages = new Map();
        this.init();
    }

    init() {
        this.registerPageGenerators();
    }

    registerPageGenerators() {
        this.pages.set('media', this.generateMediaPage.bind(this));
        this.pages.set('services', this.generateServicesPage.bind(this));
        this.pages.set('monitoring', this.generateMonitoringPage.bind(this));
        this.pages.set('logs', this.generateLogsPage.bind(this));
        this.pages.set('users', this.generateUsersPage.bind(this));
        this.pages.set('backup', this.generateBackupPage.bind(this));
        this.pages.set('settings', this.generateSettingsPage.bind(this));
        this.pages.set('api-docs', this.generateAPIDocsPage.bind(this));
        this.pages.set('help', this.generateHelpPage.bind(this));
    }

    getPage(pageId) {
        const generator = this.pages.get(pageId);
        return generator ? generator() : this.generateNotFoundPage();
    }

    generateMediaPage() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">Media Library</h2>
                    <div class="flex space-x-2">
                        <button onclick="pageManager.scanLibrary()" class="btn btn-secondary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            Scan Library
                        </button>
                        <button onclick="pageManager.addMedia()" class="btn btn-primary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
                            </svg>
                            Add Media
                        </button>
                    </div>
                </div>

                <!-- Media Statistics -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Movies</h3>
                            <div class="text-2xl">🎬</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-blue">1,247</div>
                        <div class="text-sm text-gray-400">Total movies</div>
                        <div class="text-xs text-neon-green mt-1">+12 this week</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">TV Shows</h3>
                            <div class="text-2xl">📺</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-purple">342</div>
                        <div class="text-sm text-gray-400">Total series</div>
                        <div class="text-xs text-neon-green mt-1">+3 this week</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Episodes</h3>
                            <div class="text-2xl">📼</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-green">15,432</div>
                        <div class="text-sm text-gray-400">Total episodes</div>
                        <div class="text-xs text-neon-green mt-1">+45 this week</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Music</h3>
                            <div class="text-2xl">🎵</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-yellow">8,451</div>
                        <div class="text-sm text-gray-400">Total tracks</div>
                        <div class="text-xs text-neon-green mt-1">+23 this week</div>
                    </div>
                </div>

                <!-- Search and Filters -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <div class="flex flex-col md:flex-row gap-4 mb-6">
                        <div class="flex-1">
                            <input type="text" placeholder="Search media..." class="form-input w-full">
                        </div>
                        <div class="flex space-x-2">
                            <select class="form-input">
                                <option>All Types</option>
                                <option>Movies</option>
                                <option>TV Shows</option>
                                <option>Music</option>
                            </select>
                            <select class="form-input">
                                <option>All Genres</option>
                                <option>Action</option>
                                <option>Comedy</option>
                                <option>Drama</option>
                                <option>Sci-Fi</option>
                            </select>
                            <select class="form-input">
                                <option>Sort by Date</option>
                                <option>Sort by Name</option>
                                <option>Sort by Rating</option>
                            </select>
                        </div>
                    </div>

                    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4" id="media-grid">
                        ${this.generateMediaItems()}
                    </div>
                </div>

                <!-- Recently Added -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">Recently Added</h3>
                    <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
                        ${this.generateRecentMedia()}
                    </div>
                </div>
            </div>
        `;
    }

    generateServicesPage() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">Services Management</h2>
                    <div class="flex space-x-2">
                        <button onclick="servicesManager.refreshAllServices()" class="btn btn-secondary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            Refresh All
                        </button>
                        <button onclick="pageManager.addService()" class="btn btn-primary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
                            </svg>
                            Add Service
                        </button>
                    </div>
                </div>

                <!-- Service Status Overview -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-2">
                            <h3 class="text-gray-400">Online</h3>
                            <div class="w-3 h-3 bg-neon-green rounded-full animate-pulse"></div>
                        </div>
                        <div class="text-3xl font-bold text-neon-green" id="services-online">8</div>
                        <div class="text-sm text-gray-400">Services running</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-2">
                            <h3 class="text-gray-400">Warning</h3>
                            <div class="w-3 h-3 bg-neon-yellow rounded-full"></div>
                        </div>
                        <div class="text-3xl font-bold text-neon-yellow" id="services-warning">1</div>
                        <div class="text-sm text-gray-400">Need attention</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-2">
                            <h3 class="text-gray-400">Offline</h3>
                            <div class="w-3 h-3 bg-neon-pink rounded-full"></div>
                        </div>
                        <div class="text-3xl font-bold text-neon-pink" id="services-offline">1</div>
                        <div class="text-sm text-gray-400">Not responding</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-2">
                            <h3 class="text-gray-400">Avg Response</h3>
                            <div class="text-2xl">⚡</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-blue">145ms</div>
                        <div class="text-sm text-gray-400">Response time</div>
                    </div>
                </div>

                <!-- Detailed Services Grid -->
                <div id="services-detailed" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <!-- Services will be populated by ServicesManager -->
                </div>

                <!-- Service Configuration -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">Global Settings</h3>
                    <div class="space-y-4">
                        <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                            <div>
                                <div class="font-medium">Auto-restart failed services</div>
                                <div class="text-sm text-gray-400">Automatically restart services that go offline</div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" checked>
                                <span class="slider"></span>
                            </label>
                        </div>
                        
                        <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                            <div>
                                <div class="font-medium">Health check notifications</div>
                                <div class="text-sm text-gray-400">Get notified when services go offline</div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" checked>
                                <span class="slider"></span>
                            </label>
                        </div>
                        
                        <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                            <div>
                                <div class="font-medium">Performance monitoring</div>
                                <div class="text-sm text-gray-400">Monitor service performance metrics</div>
                            </div>
                            <label class="switch">
                                <input type="checkbox" checked>
                                <span class="slider"></span>
                            </label>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    generateMonitoringPage() {
        return `
            <div class="space-y-6">
                <h2 class="text-2xl font-bold">System Monitoring</h2>

                <!-- System Overview -->
                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">CPU Usage</h3>
                            <div class="text-2xl">🖥️</div>
                        </div>
                        <div class="text-2xl font-bold text-neon-blue">23%</div>
                        <div class="progress-bar mt-2">
                            <div class="progress-fill" style="width: 23%"></div>
                        </div>
                        <div class="text-xs text-gray-400 mt-1">4 cores • 3.2 GHz</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Memory</h3>
                            <div class="text-2xl">💾</div>
                        </div>
                        <div class="text-2xl font-bold text-neon-purple">67%</div>
                        <div class="progress-bar mt-2">
                            <div class="progress-fill" style="width: 67%"></div>
                        </div>
                        <div class="text-xs text-gray-400 mt-1">10.7 GB / 16 GB</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Disk Usage</h3>
                            <div class="text-2xl">💿</div>
                        </div>
                        <div class="text-2xl font-bold text-neon-green">62%</div>
                        <div class="progress-bar mt-2">
                            <div class="progress-fill" style="width: 62%"></div>
                        </div>
                        <div class="text-xs text-gray-400 mt-1">12.4 TB / 20 TB</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Network</h3>
                            <div class="text-2xl">🌐</div>
                        </div>
                        <div class="text-2xl font-bold text-neon-yellow">145</div>
                        <div class="text-sm text-gray-400">MB/s</div>
                        <div class="text-xs text-gray-400 mt-1">↓ 120 MB/s ↑ 25 MB/s</div>
                    </div>
                </div>

                <!-- Real-time Charts -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-lg font-semibold">System Performance</h3>
                            <div class="flex space-x-2">
                                <button class="px-3 py-1 bg-neon-blue/20 rounded-lg text-sm">Live</button>
                                <button class="px-3 py-1 bg-glass rounded-lg text-sm">1H</button>
                                <button class="px-3 py-1 bg-glass rounded-lg text-sm">1D</button>
                            </div>
                        </div>
                        <div class="chart-container">
                            <canvas id="systemChart" height="300"></canvas>
                        </div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-lg font-semibold">Network Traffic</h3>
                            <div class="flex space-x-2">
                                <button class="px-3 py-1 bg-neon-green/20 rounded-lg text-sm">Live</button>
                                <button class="px-3 py-1 bg-glass rounded-lg text-sm">1H</button>
                                <button class="px-3 py-1 bg-glass rounded-lg text-sm">1D</button>
                            </div>
                        </div>
                        <div class="chart-container">
                            <canvas id="networkChart" height="300"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Process List -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-lg font-semibold">Top Processes</h3>
                        <button onclick="pageManager.refreshProcesses()" class="btn btn-secondary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            Refresh
                        </button>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Process</th>
                                    <th>PID</th>
                                    <th>CPU %</th>
                                    <th>Memory</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody id="processes-table">
                                ${this.generateProcessTableRows()}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Alerts and Thresholds -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">Alert Thresholds</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div class="space-y-3">
                            <div class="flex items-center justify-between">
                                <label class="text-sm font-medium">CPU Alert Threshold</label>
                                <input type="number" value="80" min="0" max="100" class="form-input w-20 text-center">
                            </div>
                            <div class="flex items-center justify-between">
                                <label class="text-sm font-medium">Memory Alert Threshold</label>
                                <input type="number" value="85" min="0" max="100" class="form-input w-20 text-center">
                            </div>
                            <div class="flex items-center justify-between">
                                <label class="text-sm font-medium">Disk Alert Threshold</label>
                                <input type="number" value="90" min="0" max="100" class="form-input w-20 text-center">
                            </div>
                        </div>
                        <div class="space-y-3">
                            <div class="flex items-center justify-between">
                                <label class="text-sm font-medium">Temperature Alert</label>
                                <input type="number" value="75" min="0" max="100" class="form-input w-20 text-center">
                            </div>
                            <div class="flex items-center justify-between">
                                <label class="text-sm font-medium">Network Alert (MB/s)</label>
                                <input type="number" value="500" min="0" class="form-input w-20 text-center">
                            </div>
                            <div class="flex items-center justify-between p-3 bg-glass rounded-lg">
                                <span class="text-sm">Enable Alerts</span>
                                <label class="switch">
                                    <input type="checkbox" checked>
                                    <span class="slider"></span>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    generateLogsPage() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">System Logs</h2>
                    <div class="flex space-x-2">
                        <button onclick="pageManager.clearLogs()" class="btn btn-secondary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"></path>
                            </svg>
                            Clear Logs
                        </button>
                        <button onclick="pageManager.downloadLogs()" class="btn btn-secondary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                            Download
                        </button>
                        <button onclick="pageManager.refreshLogs()" class="btn btn-primary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path>
                            </svg>
                            Refresh
                        </button>
                    </div>
                </div>

                <!-- Log Filters -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <div class="flex flex-col md:flex-row gap-4">
                        <div class="flex-1">
                            <input type="text" placeholder="Search logs..." class="form-input w-full" id="log-search">
                        </div>
                        <div class="flex space-x-2">
                            <select class="form-input" id="log-level-filter">
                                <option value="">All Levels</option>
                                <option value="error">Error</option>
                                <option value="warning">Warning</option>
                                <option value="info">Info</option>
                                <option value="debug">Debug</option>
                            </select>
                            <select class="form-input" id="log-service-filter">
                                <option value="">All Services</option>
                                <option value="plex">Plex</option>
                                <option value="sonarr">Sonarr</option>
                                <option value="radarr">Radarr</option>
                                <option value="system">System</option>
                            </select>
                            <select class="form-input" id="log-time-filter">
                                <option value="">All Time</option>
                                <option value="1h">Last Hour</option>
                                <option value="24h">Last 24 Hours</option>
                                <option value="7d">Last 7 Days</option>
                            </select>
                        </div>
                    </div>
                </div>

                <!-- Live Log Stream -->
                <div class="bg-dark-secondary rounded-xl border border-glass-border">
                    <div class="p-4 border-b border-glass-border">
                        <div class="flex items-center justify-between">
                            <h3 class="text-lg font-semibold">Live Log Stream</h3>
                            <div class="flex items-center space-x-4">
                                <div class="flex items-center space-x-2">
                                    <div class="w-2 h-2 bg-neon-green rounded-full animate-pulse"></div>
                                    <span class="text-sm text-gray-400">Live</span>
                                </div>
                                <button onclick="pageManager.toggleLogStream()" class="btn btn-secondary">
                                    <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M10 9v6m4-6v6m7-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                                    </svg>
                                    Pause
                                </button>
                            </div>
                        </div>
                    </div>
                    <div class="p-0">
                        <div id="log-stream" class="h-96 overflow-y-auto font-mono text-sm bg-black/20 p-4 space-y-1">
                            ${this.generateLogEntries()}
                        </div>
                    </div>
                </div>

                <!-- Log Analytics -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">Log Level Distribution</h3>
                        <div class="space-y-3">
                            <div class="flex items-center justify-between">
                                <div class="flex items-center space-x-2">
                                    <div class="w-3 h-3 bg-red-500 rounded-full"></div>
                                    <span class="text-sm">Errors</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <div class="text-sm font-medium">23</div>
                                    <div class="w-20 h-2 bg-glass rounded-full overflow-hidden">
                                        <div class="w-1/4 h-full bg-red-500"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="flex items-center justify-between">
                                <div class="flex items-center space-x-2">
                                    <div class="w-3 h-3 bg-yellow-500 rounded-full"></div>
                                    <span class="text-sm">Warnings</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <div class="text-sm font-medium">45</div>
                                    <div class="w-20 h-2 bg-glass rounded-full overflow-hidden">
                                        <div class="w-1/2 h-full bg-yellow-500"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="flex items-center justify-between">
                                <div class="flex items-center space-x-2">
                                    <div class="w-3 h-3 bg-blue-500 rounded-full"></div>
                                    <span class="text-sm">Info</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <div class="text-sm font-medium">156</div>
                                    <div class="w-20 h-2 bg-glass rounded-full overflow-hidden">
                                        <div class="w-full h-full bg-blue-500"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="flex items-center justify-between">
                                <div class="flex items-center space-x-2">
                                    <div class="w-3 h-3 bg-gray-500 rounded-full"></div>
                                    <span class="text-sm">Debug</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <div class="text-sm font-medium">89</div>
                                    <div class="w-20 h-2 bg-glass rounded-full overflow-hidden">
                                        <div class="w-3/4 h-full bg-gray-500"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">Recent Errors</h3>
                        <div class="space-y-3">
                            ${this.generateRecentErrors()}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    generateUsersPage() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">User Management</h2>
                    <div class="flex space-x-2">
                        <button onclick="pageManager.exportUsers()" class="btn btn-secondary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"></path>
                            </svg>
                            Export
                        </button>
                        <button onclick="pageManager.addUser()" class="btn btn-primary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
                            </svg>
                            Add User
                        </button>
                    </div>
                </div>

                <!-- User Statistics -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Total Users</h3>
                            <div class="text-2xl">👥</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-blue">24</div>
                        <div class="text-sm text-gray-400">Active accounts</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Online Now</h3>
                            <div class="text-2xl">🟢</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-green">7</div>
                        <div class="text-sm text-gray-400">Currently active</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Admin Users</h3>
                            <div class="text-2xl">👑</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-purple">3</div>
                        <div class="text-sm text-gray-400">Administrator accounts</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">New This Week</h3>
                            <div class="text-2xl">✨</div>
                        </div>
                        <div class="text-3xl font-bold text-neon-yellow">2</div>
                        <div class="text-sm text-gray-400">Recent signups</div>
                    </div>
                </div>

                <!-- User Management Table -->
                <div class="bg-dark-secondary rounded-xl border border-glass-border">
                    <div class="p-6 border-b border-glass-border">
                        <div class="flex flex-col md:flex-row gap-4">
                            <div class="flex-1">
                                <input type="text" placeholder="Search users..." class="form-input w-full">
                            </div>
                            <div class="flex space-x-2">
                                <select class="form-input">
                                    <option>All Roles</option>
                                    <option>Administrators</option>
                                    <option>Users</option>
                                    <option>Viewers</option>
                                </select>
                                <select class="form-input">
                                    <option>All Status</option>
                                    <option>Active</option>
                                    <option>Inactive</option>
                                    <option>Banned</option>
                                </select>
                            </div>
                        </div>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>User</th>
                                    <th>Email</th>
                                    <th>Role</th>
                                    <th>Status</th>
                                    <th>Last Login</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${this.generateUserTableRows()}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Role Management -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">Role Permissions</h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div class="space-y-3">
                            <h4 class="font-medium text-neon-purple">Administrator</h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-green" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>Full system access</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-green" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>User management</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-green" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>Service configuration</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="space-y-3">
                            <h4 class="font-medium text-neon-blue">User</h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-green" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>Media streaming</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-green" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>Request content</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-pink" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>System configuration</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="space-y-3">
                            <h4 class="font-medium text-neon-green">Viewer</h4>
                            <div class="space-y-2 text-sm">
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-green" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>View-only access</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-pink" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>Content requests</span>
                                </div>
                                <div class="flex items-center space-x-2">
                                    <svg class="w-4 h-4 text-neon-pink" fill="currentColor" viewBox="0 0 20 20">
                                        <path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path>
                                    </svg>
                                    <span>System modifications</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    generateBackupPage() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">Backup & Restore</h2>
                    <div class="flex space-x-2">
                        <button onclick="pageManager.scheduleBackup()" class="btn btn-secondary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path>
                            </svg>
                            Schedule
                        </button>
                        <button onclick="pageManager.startBackup()" class="btn btn-primary">
                            <svg class="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M8 7H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V9a2 2 0 00-2-2h-3m-1 4l-3 3m0 0l-3-3m3 3V4"></path>
                            </svg>
                            Start Backup
                        </button>
                    </div>
                </div>

                <!-- Backup Status -->
                <div class="grid grid-cols-1 md:grid-cols-4 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Last Backup</h3>
                            <div class="text-2xl">💾</div>
                        </div>
                        <div class="text-xl font-bold text-neon-green">2 hours ago</div>
                        <div class="text-sm text-gray-400">Completed successfully</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Backup Size</h3>
                            <div class="text-2xl">📦</div>
                        </div>
                        <div class="text-xl font-bold text-neon-blue">2.3 GB</div>
                        <div class="text-sm text-gray-400">Last backup</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Storage Used</h3>
                            <div class="text-2xl">🗃️</div>
                        </div>
                        <div class="text-xl font-bold text-neon-purple">45.2 GB</div>
                        <div class="text-sm text-gray-400">Total backup storage</div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <div class="flex items-center justify-between mb-4">
                            <h3 class="text-gray-400">Retention</h3>
                            <div class="text-2xl">🕒</div>
                        </div>
                        <div class="text-xl font-bold text-neon-yellow">30 days</div>
                        <div class="text-sm text-gray-400">Auto-cleanup period</div>
                    </div>
                </div>

                <!-- Current Backup Progress -->
                <div id="backup-progress" class="bg-dark-secondary rounded-xl p-6 border border-glass-border hidden">
                    <div class="flex items-center justify-between mb-4">
                        <h3 class="text-lg font-semibold">Backup in Progress</h3>
                        <button onclick="pageManager.cancelBackup()" class="btn btn-danger">Cancel</button>
                    </div>
                    <div class="space-y-4">
                        <div class="flex items-center justify-between">
                            <span class="text-sm">Configuration Files</span>
                            <span class="text-sm text-neon-green">✓ Complete</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-sm">Database</span>
                            <span class="text-sm text-neon-blue">In Progress...</span>
                        </div>
                        <div class="flex items-center justify-between">
                            <span class="text-sm">Media Metadata</span>
                            <span class="text-sm text-gray-400">Pending</span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: 45%"></div>
                        </div>
                        <div class="text-sm text-gray-400">Progress: 45% (1.2 GB / 2.7 GB)</div>
                    </div>
                </div>

                <!-- Backup History -->
                <div class="bg-dark-secondary rounded-xl border border-glass-border">
                    <div class="p-6 border-b border-glass-border">
                        <h3 class="text-lg font-semibold">Backup History</h3>
                    </div>
                    <div class="overflow-x-auto">
                        <table class="data-table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Type</th>
                                    <th>Size</th>
                                    <th>Duration</th>
                                    <th>Status</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${this.generateBackupHistoryRows()}
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Backup Configuration -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">Backup Settings</h3>
                        <div class="space-y-4">
                            <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                <div>
                                    <div class="font-medium">Auto Backup</div>
                                    <div class="text-sm text-gray-400">Enable automatic scheduled backups</div>
                                </div>
                                <label class="switch">
                                    <input type="checkbox" checked>
                                    <span class="slider"></span>
                                </label>
                            </div>
                            
                            <div class="space-y-2">
                                <label class="block text-sm font-medium">Backup Schedule</label>
                                <select class="form-input w-full">
                                    <option>Daily at 2:00 AM</option>
                                    <option>Weekly on Sunday</option>
                                    <option>Monthly on 1st</option>
                                    <option>Custom</option>
                                </select>
                            </div>
                            
                            <div class="space-y-2">
                                <label class="block text-sm font-medium">Retention Period</label>
                                <select class="form-input w-full">
                                    <option>7 days</option>
                                    <option>30 days</option>
                                    <option>90 days</option>
                                    <option>1 year</option>
                                </select>
                            </div>
                            
                            <div class="space-y-2">
                                <label class="block text-sm font-medium">Backup Location</label>
                                <input type="text" value="/backups" class="form-input w-full">
                            </div>
                        </div>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">What's Included</h3>
                        <div class="space-y-3">
                            <div class="flex items-center justify-between p-3 bg-glass rounded-lg">
                                <div class="flex items-center space-x-3">
                                    <div class="text-2xl">⚙️</div>
                                    <div>
                                        <div class="font-medium">Configuration Files</div>
                                        <div class="text-sm text-gray-400">Service configs, settings</div>
                                    </div>
                                </div>
                                <label class="switch">
                                    <input type="checkbox" checked>
                                    <span class="slider"></span>
                                </label>
                            </div>
                            
                            <div class="flex items-center justify-between p-3 bg-glass rounded-lg">
                                <div class="flex items-center space-x-3">
                                    <div class="text-2xl">🗄️</div>
                                    <div>
                                        <div class="font-medium">Databases</div>
                                        <div class="text-sm text-gray-400">SQLite databases, metadata</div>
                                    </div>
                                </div>
                                <label class="switch">
                                    <input type="checkbox" checked>
                                    <span class="slider"></span>
                                </label>
                            </div>
                            
                            <div class="flex items-center justify-between p-3 bg-glass rounded-lg">
                                <div class="flex items-center space-x-3">
                                    <div class="text-2xl">🎬</div>
                                    <div>
                                        <div class="font-medium">Media Metadata</div>
                                        <div class="text-sm text-gray-400">Posters, descriptions, ratings</div>
                                    </div>
                                </div>
                                <label class="switch">
                                    <input type="checkbox" checked>
                                    <span class="slider"></span>
                                </label>
                            </div>
                            
                            <div class="flex items-center justify-between p-3 bg-glass rounded-lg">
                                <div class="flex items-center space-x-3">
                                    <div class="text-2xl">📁</div>
                                    <div>
                                        <div class="font-medium">Media Files</div>
                                        <div class="text-sm text-gray-400">Actual video/audio files</div>
                                    </div>
                                </div>
                                <label class="switch">
                                    <input type="checkbox">
                                    <span class="slider"></span>
                                </label>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Storage Usage Chart -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">Storage Usage</h3>
                    <div class="chart-container">
                        <canvas id="storageChart" height="300"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    generateSettingsPage() {
        return `
            <div class="space-y-6">
                <h2 class="text-2xl font-bold">Settings</h2>

                <!-- Settings Navigation -->
                <div class="bg-dark-secondary rounded-xl border border-glass-border">
                    <div class="flex flex-wrap border-b border-glass-border">
                        <button onclick="pageManager.showSettingsTab('general')" class="settings-tab active px-6 py-4 font-medium border-b-2 border-neon-blue">General</button>
                        <button onclick="pageManager.showSettingsTab('security')" class="settings-tab px-6 py-4 font-medium hover:bg-glass">Security</button>
                        <button onclick="pageManager.showSettingsTab('notifications')" class="settings-tab px-6 py-4 font-medium hover:bg-glass">Notifications</button>
                        <button onclick="pageManager.showSettingsTab('appearance')" class="settings-tab px-6 py-4 font-medium hover:bg-glass">Appearance</button>
                        <button onclick="pageManager.showSettingsTab('advanced')" class="settings-tab px-6 py-4 font-medium hover:bg-glass">Advanced</button>
                    </div>

                    <!-- General Settings -->
                    <div id="settings-general" class="settings-content p-6 space-y-6">
                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">System Information</h3>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div class="space-y-2">
                                    <label class="block text-sm font-medium">Server Name</label>
                                    <input type="text" value="MediaFlow Server" class="form-input w-full">
                                </div>
                                <div class="space-y-2">
                                    <label class="block text-sm font-medium">Time Zone</label>
                                    <select class="form-input w-full">
                                        <option>UTC-5 (Eastern Time)</option>
                                        <option>UTC-8 (Pacific Time)</option>
                                        <option>UTC+0 (Greenwich Time)</option>
                                    </select>
                                </div>
                                <div class="space-y-2">
                                    <label class="block text-sm font-medium">Language</label>
                                    <select class="form-input w-full">
                                        <option>English</option>
                                        <option>Spanish</option>
                                        <option>French</option>
                                        <option>German</option>
                                    </select>
                                </div>
                                <div class="space-y-2">
                                    <label class="block text-sm font-medium">Date Format</label>
                                    <select class="form-input w-full">
                                        <option>MM/DD/YYYY</option>
                                        <option>DD/MM/YYYY</option>
                                        <option>YYYY-MM-DD</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">Performance</h3>
                            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Hardware Acceleration</div>
                                        <div class="text-sm text-gray-400">Use GPU for transcoding</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox" checked>
                                        <span class="slider"></span>
                                    </label>
                                </div>
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Auto Updates</div>
                                        <div class="text-sm text-gray-400">Automatically update services</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox">
                                        <span class="slider"></span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Security Settings -->
                    <div id="settings-security" class="settings-content p-6 space-y-6 hidden">
                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">Authentication</h3>
                            <div class="space-y-4">
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Two-Factor Authentication</div>
                                        <div class="text-sm text-gray-400">Add extra security to your account</div>
                                    </div>
                                    <button class="btn btn-primary">Enable</button>
                                </div>
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Session Timeout</div>
                                        <div class="text-sm text-gray-400">Auto-logout after inactivity</div>
                                    </div>
                                    <select class="form-input w-32">
                                        <option>1 hour</option>
                                        <option>4 hours</option>
                                        <option>24 hours</option>
                                        <option>Never</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">Access Control</h3>
                            <div class="space-y-4">
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Secure Connections Only</div>
                                        <div class="text-sm text-gray-400">Force HTTPS for all connections</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox" checked>
                                        <span class="slider"></span>
                                    </label>
                                </div>
                                <div class="space-y-2">
                                    <label class="block text-sm font-medium">Allowed Networks</label>
                                    <textarea class="form-input w-full h-24" placeholder="192.168.1.0/24&#10;10.0.0.0/8"></textarea>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Notification Settings -->
                    <div id="settings-notifications" class="settings-content p-6 space-y-6 hidden">
                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">Notification Preferences</h3>
                            <div class="space-y-4">
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Desktop Notifications</div>
                                        <div class="text-sm text-gray-400">Show notifications on desktop</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox" checked>
                                        <span class="slider"></span>
                                    </label>
                                </div>
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Email Notifications</div>
                                        <div class="text-sm text-gray-400">Send important alerts via email</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox">
                                        <span class="slider"></span>
                                    </label>
                                </div>
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Sound Alerts</div>
                                        <div class="text-sm text-gray-400">Play sounds for notifications</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox" checked>
                                        <span class="slider"></span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Appearance Settings -->
                    <div id="settings-appearance" class="settings-content p-6 space-y-6 hidden">
                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">Theme</h3>
                            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                                <div class="p-4 bg-glass rounded-lg border-2 border-neon-blue">
                                    <div class="w-full h-20 bg-gradient-to-br from-dark-primary to-dark-secondary rounded mb-3"></div>
                                    <div class="text-center font-medium">Dark (Current)</div>
                                </div>
                                <div class="p-4 bg-glass rounded-lg border-2 border-transparent hover:border-glass-border cursor-pointer">
                                    <div class="w-full h-20 bg-gradient-to-br from-white to-gray-100 rounded mb-3"></div>
                                    <div class="text-center font-medium">Light</div>
                                </div>
                                <div class="p-4 bg-glass rounded-lg border-2 border-transparent hover:border-glass-border cursor-pointer">
                                    <div class="w-full h-20 bg-gradient-to-br from-gray-800 to-blue-900 rounded mb-3"></div>
                                    <div class="text-center font-medium">Auto</div>
                                </div>
                            </div>
                        </div>

                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">Layout</h3>
                            <div class="space-y-4">
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Compact Mode</div>
                                        <div class="text-sm text-gray-400">Reduce spacing and padding</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox">
                                        <span class="slider"></span>
                                    </label>
                                </div>
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Sidebar Auto-hide</div>
                                        <div class="text-sm text-gray-400">Hide sidebar on mobile</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox" checked>
                                        <span class="slider"></span>
                                    </label>
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Advanced Settings -->
                    <div id="settings-advanced" class="settings-content p-6 space-y-6 hidden">
                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">API Configuration</h3>
                            <div class="space-y-4">
                                <div class="space-y-2">
                                    <label class="block text-sm font-medium">API Base URL</label>
                                    <input type="url" value="http://localhost:3001/api" class="form-input w-full">
                                </div>
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">API Rate Limiting</div>
                                        <div class="text-sm text-gray-400">Limit API requests per minute</div>
                                    </div>
                                    <input type="number" value="100" class="form-input w-20 text-center">
                                </div>
                            </div>
                        </div>

                        <div class="space-y-4">
                            <h3 class="text-lg font-semibold">Debug & Logging</h3>
                            <div class="space-y-4">
                                <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                                    <div>
                                        <div class="font-medium">Debug Mode</div>
                                        <div class="text-sm text-gray-400">Enable detailed logging</div>
                                    </div>
                                    <label class="switch">
                                        <input type="checkbox">
                                        <span class="slider"></span>
                                    </label>
                                </div>
                                <div class="space-y-2">
                                    <label class="block text-sm font-medium">Log Level</label>
                                    <select class="form-input w-full">
                                        <option>Error</option>
                                        <option>Warning</option>
                                        <option>Info</option>
                                        <option>Debug</option>
                                    </select>
                                </div>
                            </div>
                        </div>

                        <div class="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
                            <h4 class="text-red-400 font-medium mb-2">Danger Zone</h4>
                            <div class="space-y-3">
                                <button onclick="pageManager.resetSettings()" class="btn btn-danger">Reset All Settings</button>
                                <button onclick="pageManager.factoryReset()" class="btn btn-danger">Factory Reset</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Save Button -->
                <div class="flex justify-end space-x-3">
                    <button onclick="pageManager.resetCurrentSettings()" class="btn btn-secondary">Reset</button>
                    <button onclick="pageManager.saveSettings()" class="btn btn-primary">Save Changes</button>
                </div>
            </div>
        `;
    }

    generateAPIDocsPage() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">API Documentation</h2>
                    <div class="flex space-x-2">
                        <button onclick="pageManager.generateAPIKey()" class="btn btn-secondary">Generate API Key</button>
                        <button onclick="pageManager.testAPI()" class="btn btn-primary">Test API</button>
                    </div>
                </div>

                <!-- API Overview -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">API Overview</h3>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div class="text-center">
                            <div class="text-2xl font-bold text-neon-blue">REST API</div>
                            <div class="text-sm text-gray-400">RESTful web services</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl font-bold text-neon-green">WebSocket</div>
                            <div class="text-sm text-gray-400">Real-time updates</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl font-bold text-neon-purple">GraphQL</div>
                            <div class="text-sm text-gray-400">Flexible queries</div>
                        </div>
                    </div>
                </div>

                <!-- API Endpoints -->
                <div class="space-y-4">
                    ${this.generateAPIEndpoints()}
                </div>

                <!-- API Playground -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">API Playground</h3>
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div class="space-y-4">
                            <div class="space-y-2">
                                <label class="block text-sm font-medium">Endpoint</label>
                                <select class="form-input w-full">
                                    <option>GET /api/services</option>
                                    <option>GET /api/system/status</option>
                                    <option>POST /api/services/{id}/restart</option>
                                </select>
                            </div>
                            <div class="space-y-2">
                                <label class="block text-sm font-medium">Request Body</label>
                                <textarea class="form-input w-full h-32 font-mono text-sm" placeholder='{\n  "example": "value"\n}'></textarea>
                            </div>
                            <button class="btn btn-primary w-full">Send Request</button>
                        </div>
                        <div class="space-y-2">
                            <label class="block text-sm font-medium">Response</label>
                            <div class="bg-black/20 rounded-lg p-4 h-64 overflow-y-auto font-mono text-sm">
                                <pre class="text-green-400">{
  "status": "success",
  "data": {
    "services": [
      {
        "id": "plex",
        "name": "Plex Media Server",
        "status": "online",
        "uptime": 3600
      }
    ]
  }
}</pre>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    generateHelpPage() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">Help & Support</h2>
                    <div class="flex space-x-2">
                        <button onclick="pageManager.contactSupport()" class="btn btn-secondary">Contact Support</button>
                        <button onclick="pageManager.openTutorial()" class="btn btn-primary">Start Tutorial</button>
                    </div>
                </div>

                <!-- Quick Help -->
                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border text-center">
                        <div class="text-4xl mb-4">🚀</div>
                        <h3 class="text-lg font-semibold mb-2">Getting Started</h3>
                        <p class="text-sm text-gray-400 mb-4">Learn the basics of MediaFlow</p>
                        <button class="btn btn-primary w-full">View Guide</button>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border text-center">
                        <div class="text-4xl mb-4">❓</div>
                        <h3 class="text-lg font-semibold mb-2">FAQ</h3>
                        <p class="text-sm text-gray-400 mb-4">Common questions and answers</p>
                        <button class="btn btn-primary w-full">Browse FAQ</button>
                    </div>
                    
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border text-center">
                        <div class="text-4xl mb-4">🛠️</div>
                        <h3 class="text-lg font-semibold mb-2">Troubleshooting</h3>
                        <p class="text-sm text-gray-400 mb-4">Fix common issues</p>
                        <button class="btn btn-primary w-full">Get Help</button>
                    </div>
                </div>

                <!-- Documentation Sections -->
                <div class="bg-dark-secondary rounded-xl border border-glass-border">
                    <div class="p-6 border-b border-glass-border">
                        <h3 class="text-lg font-semibold">Documentation</h3>
                    </div>
                    <div class="p-6">
                        <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                            ${this.generateHelpSections()}
                        </div>
                    </div>
                </div>

                <!-- System Information -->
                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">System Information</h3>
                    <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div class="space-y-3">
                            <div class="flex justify-between">
                                <span class="text-gray-400">Version:</span>
                                <span class="font-medium">2.1.0</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-400">Build:</span>
                                <span class="font-medium">20250103</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-400">Platform:</span>
                                <span class="font-medium">Docker</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-400">Architecture:</span>
                                <span class="font-medium">x86_64</span>
                            </div>
                        </div>
                        <div class="space-y-3">
                            <div class="flex justify-between">
                                <span class="text-gray-400">Uptime:</span>
                                <span class="font-medium">2d 14h 32m</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-400">Memory Usage:</span>
                                <span class="font-medium">2.1 GB</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-400">Disk Usage:</span>
                                <span class="font-medium">45.2 GB</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-400">Active Users:</span>
                                <span class="font-medium">7</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    // Helper methods for generating content
    generateMediaItems() {
        const items = [
            { title: "The Matrix", type: "movie", year: "1999", poster: "🎬" },
            { title: "Breaking Bad", type: "tv", year: "2008", poster: "📺" },
            { title: "Inception", type: "movie", year: "2010", poster: "🎬" },
            { title: "Stranger Things", type: "tv", year: "2016", poster: "📺" },
            { title: "Interstellar", type: "movie", year: "2014", poster: "🎬" },
            { title: "The Office", type: "tv", year: "2005", poster: "📺" }
        ];

        return items.map(item => `
            <div class="bg-glass rounded-lg p-4 hover:bg-white/20 transition-colors cursor-pointer">
                <div class="text-4xl mb-2 text-center">${item.poster}</div>
                <div class="text-sm font-medium truncate">${item.title}</div>
                <div class="text-xs text-gray-400">${item.year}</div>
            </div>
        `).join('');
    }

    generateRecentMedia() {
        const items = Array.from({ length: 8 }, (_, i) => ({
            title: `Media Item ${i + 1}`,
            poster: ['🎬', '📺', '🎵'][i % 3]
        }));

        return items.map(item => `
            <div class="bg-glass rounded-lg p-3 hover:bg-white/20 transition-colors cursor-pointer">
                <div class="text-2xl mb-1 text-center">${item.poster}</div>
                <div class="text-xs font-medium truncate">${item.title}</div>
            </div>
        `).join('');
    }

    generateProcessTableRows() {
        const processes = [
            { name: "plex", pid: "1234", cpu: "15.2", memory: "1.2 GB", status: "running" },
            { name: "sonarr", pid: "1235", cpu: "3.1", memory: "256 MB", status: "running" },
            { name: "radarr", pid: "1236", cpu: "2.8", memory: "198 MB", status: "running" },
            { name: "lidarr", pid: "1237", cpu: "1.5", memory: "145 MB", status: "warning" },
            { name: "prowlarr", pid: "1238", cpu: "0.8", memory: "89 MB", status: "running" }
        ];

        return processes.map(proc => `
            <tr>
                <td class="font-medium">${proc.name}</td>
                <td>${proc.pid}</td>
                <td>${proc.cpu}%</td>
                <td>${proc.memory}</td>
                <td><span class="status-badge status-${proc.status === 'running' ? 'online' : 'warning'}">${proc.status}</span></td>
                <td>
                    <div class="flex space-x-2">
                        <button class="btn btn-secondary text-xs py-1 px-2">View</button>
                        <button class="btn btn-danger text-xs py-1 px-2">Kill</button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    generateLogEntries() {
        const logs = [
            { time: "18:34:22", level: "info", service: "plex", message: "New media library scan completed" },
            { time: "18:34:15", level: "warning", service: "lidarr", message: "Connection timeout to indexer" },
            { time: "18:34:10", level: "error", service: "sonarr", message: "Failed to download episode metadata" },
            { time: "18:34:05", level: "info", service: "system", message: "User admin logged in from 192.168.1.100" },
            { time: "18:33:58", level: "debug", service: "radarr", message: "Processing movie search request" }
        ];

        return logs.map(log => `
            <div class="flex items-start space-x-3 text-xs">
                <span class="text-gray-500 font-mono">${log.time}</span>
                <span class="px-2 py-1 rounded text-xs font-medium ${this.getLogLevelClass(log.level)}">${log.level.toUpperCase()}</span>
                <span class="text-neon-blue">[${log.service}]</span>
                <span class="flex-1">${log.message}</span>
            </div>
        `).join('');
    }

    generateRecentErrors() {
        const errors = [
            { service: "sonarr", message: "Failed to download episode", time: "2 min ago" },
            { service: "lidarr", message: "Indexer connection timeout", time: "5 min ago" },
            { service: "system", message: "Disk space warning", time: "1 hour ago" }
        ];

        return errors.map(error => `
            <div class="flex items-start space-x-3 p-3 bg-red-500/10 rounded-lg">
                <div class="w-2 h-2 bg-red-500 rounded-full mt-2 flex-shrink-0"></div>
                <div class="flex-1">
                    <div class="font-medium text-red-400">${error.service}</div>
                    <div class="text-sm text-gray-400">${error.message}</div>
                    <div class="text-xs text-gray-500">${error.time}</div>
                </div>
            </div>
        `).join('');
    }

    generateUserTableRows() {
        const users = [
            { name: "Admin", email: "admin@mediaflow.local", role: "Administrator", status: "active", lastLogin: "2 min ago" },
            { name: "John Doe", email: "john@example.com", role: "User", status: "active", lastLogin: "1 hour ago" },
            { name: "Jane Smith", email: "jane@example.com", role: "User", status: "active", lastLogin: "Yesterday" },
            { name: "Bob Wilson", email: "bob@example.com", role: "Viewer", status: "inactive", lastLogin: "1 week ago" }
        ];

        return users.map(user => `
            <tr>
                <td>
                    <div class="flex items-center space-x-3">
                        <div class="w-8 h-8 bg-gradient-to-br from-neon-blue to-neon-purple rounded-full flex items-center justify-center text-sm font-bold">
                            ${user.name.split(' ').map(n => n[0]).join('')}
                        </div>
                        <div class="font-medium">${user.name}</div>
                    </div>
                </td>
                <td>${user.email}</td>
                <td><span class="px-2 py-1 bg-glass rounded-full text-xs">${user.role}</span></td>
                <td><span class="status-badge status-${user.status === 'active' ? 'online' : 'offline'}">${user.status}</span></td>
                <td>${user.lastLogin}</td>
                <td>
                    <div class="flex space-x-2">
                        <button class="btn btn-secondary text-xs py-1 px-2">Edit</button>
                        <button class="btn btn-danger text-xs py-1 px-2">Delete</button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    generateBackupHistoryRows() {
        const backups = [
            { date: "2025-01-03 02:00", type: "Scheduled", size: "2.3 GB", duration: "12 min", status: "success" },
            { date: "2025-01-02 02:00", type: "Scheduled", size: "2.1 GB", duration: "11 min", status: "success" },
            { date: "2025-01-01 14:30", type: "Manual", size: "2.0 GB", duration: "10 min", status: "success" },
            { date: "2025-01-01 02:00", type: "Scheduled", size: "1.9 GB", duration: "15 min", status: "failed" }
        ];

        return backups.map(backup => `
            <tr>
                <td>${backup.date}</td>
                <td><span class="px-2 py-1 bg-glass rounded-full text-xs">${backup.type}</span></td>
                <td>${backup.size}</td>
                <td>${backup.duration}</td>
                <td><span class="status-badge status-${backup.status === 'success' ? 'online' : 'offline'}">${backup.status}</span></td>
                <td>
                    <div class="flex space-x-2">
                        <button class="btn btn-secondary text-xs py-1 px-2">Download</button>
                        <button class="btn btn-primary text-xs py-1 px-2">Restore</button>
                    </div>
                </td>
            </tr>
        `).join('');
    }

    generateAPIEndpoints() {
        const endpoints = [
            { method: "GET", path: "/api/services", description: "Get all service statuses" },
            { method: "GET", path: "/api/services/{id}", description: "Get specific service status" },
            { method: "POST", path: "/api/services/{id}/restart", description: "Restart a service" },
            { method: "GET", path: "/api/system/status", description: "Get system health information" },
            { method: "GET", path: "/api/media", description: "Get media library information" },
            { method: "GET", path: "/api/users", description: "Get user list (admin only)" },
            { method: "POST", path: "/api/backup", description: "Trigger backup process" },
            { method: "GET", path: "/api/logs", description: "Get system logs" }
        ];

        return endpoints.map(endpoint => `
            <div class="bg-dark-secondary rounded-xl p-4 border border-glass-border">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-3">
                        <span class="px-2 py-1 rounded text-xs font-medium ${this.getMethodClass(endpoint.method)}">${endpoint.method}</span>
                        <code class="text-sm text-neon-blue">${endpoint.path}</code>
                    </div>
                    <button class="btn btn-secondary text-xs">Try it</button>
                </div>
                <p class="text-sm text-gray-400">${endpoint.description}</p>
            </div>
        `).join('');
    }

    generateHelpSections() {
        const sections = [
            { title: "Installation Guide", description: "Set up MediaFlow from scratch", icon: "🛠️" },
            { title: "Service Configuration", description: "Configure Plex, Sonarr, Radarr and more", icon: "⚙️" },
            { title: "User Management", description: "Add and manage user accounts", icon: "👥" },
            { title: "Backup & Restore", description: "Protect your data with regular backups", icon: "💾" },
            { title: "API Reference", description: "Complete API documentation", icon: "🔌" },
            { title: "Troubleshooting", description: "Common issues and solutions", icon: "🔧" },
            { title: "Security Guide", description: "Secure your MediaFlow installation", icon: "🔒" },
            { title: "Performance Tips", description: "Optimize your setup for better performance", icon: "⚡" }
        ];

        return sections.map(section => `
            <div class="flex items-center space-x-3 p-3 bg-glass rounded-lg hover:bg-white/20 transition-colors cursor-pointer">
                <div class="text-2xl">${section.icon}</div>
                <div class="flex-1">
                    <div class="font-medium">${section.title}</div>
                    <div class="text-sm text-gray-400">${section.description}</div>
                </div>
                <svg class="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"></path>
                </svg>
            </div>
        `).join('');
    }

    getLogLevelClass(level) {
        const classes = {
            'error': 'bg-red-500/20 text-red-400',
            'warning': 'bg-yellow-500/20 text-yellow-400',
            'info': 'bg-blue-500/20 text-blue-400',
            'debug': 'bg-gray-500/20 text-gray-400'
        };
        return classes[level] || classes.info;
    }

    getMethodClass(method) {
        const classes = {
            'GET': 'bg-green-500/20 text-green-400',
            'POST': 'bg-blue-500/20 text-blue-400',
            'PUT': 'bg-yellow-500/20 text-yellow-400',
            'DELETE': 'bg-red-500/20 text-red-400'
        };
        return classes[method] || classes.GET;
    }

    generateNotFoundPage() {
        return `
            <div class="text-center py-12">
                <div class="text-6xl mb-4">🔍</div>
                <h2 class="text-2xl font-bold mb-2">Page Not Found</h2>
                <p class="text-gray-400 mb-6">The page you're looking for doesn't exist.</p>
                <button onclick="dashboard.showPage('dashboard')" class="btn btn-primary">Go to Dashboard</button>
            </div>
        `;
    }

    // Page-specific action methods (can be implemented as needed)
    async scanLibrary() {
        window.dashboard?.showNotification('Library scan started', 'info');
    }

    async addMedia() {
        // Implementation for adding media
    }

    async addService() {
        // Implementation for adding service
    }

    async refreshProcesses() {
        window.dashboard?.showNotification('Process list refreshed', 'success');
    }

    async clearLogs() {
        if (confirm('Are you sure you want to clear all logs?')) {
            window.dashboard?.showNotification('Logs cleared', 'success');
        }
    }

    async downloadLogs() {
        window.dashboard?.showNotification('Downloading logs...', 'info');
    }

    async refreshLogs() {
        window.dashboard?.showNotification('Logs refreshed', 'success');
    }

    toggleLogStream() {
        // Implementation for toggling log stream
    }

    async exportUsers() {
        window.dashboard?.showNotification('Exporting users...', 'info');
    }

    async addUser() {
        // Implementation for adding user
    }

    async scheduleBackup() {
        // Implementation for scheduling backup
    }

    async startBackup() {
        document.getElementById('backup-progress').classList.remove('hidden');
        window.dashboard?.showNotification('Backup started', 'info');
    }

    async cancelBackup() {
        document.getElementById('backup-progress').classList.add('hidden');
        window.dashboard?.showNotification('Backup cancelled', 'warning');
    }

    showSettingsTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.settings-content').forEach(content => {
            content.classList.add('hidden');
        });
        document.querySelectorAll('.settings-tab').forEach(tab => {
            tab.classList.remove('active', 'border-neon-blue');
        });

        // Show selected tab
        document.getElementById(`settings-${tabName}`).classList.remove('hidden');
        event.target.classList.add('active', 'border-neon-blue');
    }

    async saveSettings() {
        window.dashboard?.showNotification('Settings saved successfully', 'success');
    }

    async resetCurrentSettings() {
        if (confirm('Reset current settings to defaults?')) {
            window.dashboard?.showNotification('Settings reset', 'info');
        }
    }

    async resetSettings() {
        if (confirm('This will reset ALL settings to defaults. Continue?')) {
            window.dashboard?.showNotification('All settings reset', 'warning');
        }
    }

    async factoryReset() {
        if (confirm('This will completely reset MediaFlow to factory defaults. ALL DATA WILL BE LOST. Continue?')) {
            window.dashboard?.showNotification('Factory reset initiated', 'error');
        }
    }

    async generateAPIKey() {
        window.dashboard?.showNotification('New API key generated', 'success');
    }

    async testAPI() {
        window.dashboard?.showNotification('API test completed', 'success');
    }

    async contactSupport() {
        window.open('mailto:support@mediaflow.com?subject=MediaFlow Support Request');
    }

    async openTutorial() {
        window.dashboard?.showNotification('Tutorial started', 'info');
    }
}

// Initialize page manager
const pageManager = new PageManager();

// Export for global access
window.pageManager = pageManager;