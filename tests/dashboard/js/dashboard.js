// MediaFlow Pro Dashboard JavaScript

class MediaFlowDashboard {
    constructor() {
        this.currentPage = 'dashboard';
        this.services = new Map();
        this.notifications = [];
        this.theme = localStorage.getItem('theme') || 'dark';
        this.init();
    }

    init() {
        this.initTheme();
        this.initEventListeners();
        this.initServices();
        this.initNotifications();
        this.initSearch();
        this.startRealTimeUpdates();
        this.loadDashboardData();
    }

    initTheme() {
        document.documentElement.className = this.theme;
        this.updateThemeIcon();
    }

    initEventListeners() {
        // Global search
        const searchInput = document.getElementById('global-search');
        if (searchInput) {
            searchInput.addEventListener('input', this.handleSearch.bind(this));
            searchInput.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    this.hideSearchResults();
                }
            });
        }

        // Close dropdowns when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.dropdown')) {
                this.closeAllDropdowns();
            }
        });

        // Keyboard shortcuts
        document.addEventListener('keydown', this.handleKeyboardShortcuts.bind(this));
    }

    initServices() {
        this.services.set('plex', { name: 'Plex', status: 'online', url: 'http://localhost:32400', icon: '🎬' });
        this.services.set('sonarr', { name: 'Sonarr', status: 'online', url: 'http://localhost:8989', icon: '📺' });
        this.services.set('radarr', { name: 'Radarr', status: 'online', url: 'http://localhost:7878', icon: '🎥' });
        this.services.set('lidarr', { name: 'Lidarr', status: 'warning', url: 'http://localhost:8686', icon: '🎵' });
        this.services.set('prowlarr', { name: 'Prowlarr', status: 'online', url: 'http://localhost:9696', icon: '🔍' });
        this.services.set('bazarr', { name: 'Bazarr', status: 'online', url: 'http://localhost:6767', icon: '📝' });
        this.services.set('overseerr', { name: 'Overseerr', status: 'online', url: 'http://localhost:5055', icon: '📋' });
        this.services.set('tautulli', { name: 'Tautulli', status: 'online', url: 'http://localhost:8181', icon: '📊' });
        
        this.renderServices();
    }

    initNotifications() {
        this.notifications = [
            { id: 1, title: 'New Movie Added', message: 'The Matrix Resurrections has been added to your library', time: '5 min ago', read: false, type: 'success' },
            { id: 2, title: 'Service Warning', message: 'Lidarr is experiencing connection issues', time: '12 min ago', read: false, type: 'warning' },
            { id: 3, title: 'Download Complete', message: 'Breaking Bad S05E16 download completed', time: '1 hour ago', read: true, type: 'info' }
        ];
        
        this.updateNotificationCount();
        this.renderNotifications();
    }

    initSearch() {
        this.searchResults = [];
        this.searchTimeout = null;
    }

    handleSearch(e) {
        const query = e.target.value.trim();
        
        if (this.searchTimeout) {
            clearTimeout(this.searchTimeout);
        }

        if (query.length < 2) {
            this.hideSearchResults();
            return;
        }

        this.searchTimeout = setTimeout(() => {
            this.performSearch(query);
        }, 300);
    }

    async performSearch(query) {
        try {
            // Simulate API search
            const results = await this.searchAPI(query);
            this.displaySearchResults(results);
        } catch (error) {
            console.error('Search error:', error);
        }
    }

    async searchAPI(query) {
        // Simulate search across services, media, logs, etc.
        const mockResults = [
            { type: 'service', title: 'Plex Media Server', description: 'Streaming service', url: '/services#plex' },
            { type: 'media', title: 'The Matrix', description: '1999 • Action, Sci-Fi', url: '/media#matrix' },
            { type: 'user', title: 'Admin User', description: 'System administrator', url: '/users#admin' },
            { type: 'log', title: 'Recent Errors', description: 'View system logs', url: '/logs#errors' }
        ].filter(item => item.title.toLowerCase().includes(query.toLowerCase()));

        return new Promise(resolve => {
            setTimeout(() => resolve(mockResults), 200);
        });
    }

    displaySearchResults(results) {
        const searchInput = document.getElementById('global-search');
        let resultsContainer = document.getElementById('search-results');
        
        if (!resultsContainer) {
            resultsContainer = document.createElement('div');
            resultsContainer.id = 'search-results';
            resultsContainer.className = 'search-results hidden';
            searchInput.parentElement.appendChild(resultsContainer);
        }

        if (results.length === 0) {
            resultsContainer.innerHTML = '<div class="search-result-item">No results found</div>';
        } else {
            resultsContainer.innerHTML = results.map(result => `
                <div class="search-result-item" onclick="navigateToResult('${result.url}')">
                    <div class="font-medium">${result.title}</div>
                    <div class="text-sm text-gray-400">${result.description}</div>
                    <div class="text-xs text-neon-blue">${result.type}</div>
                </div>
            `).join('');
        }

        resultsContainer.classList.remove('hidden');
    }

    hideSearchResults() {
        const resultsContainer = document.getElementById('search-results');
        if (resultsContainer) {
            resultsContainer.classList.add('hidden');
        }
    }

    handleKeyboardShortcuts(e) {
        // Cmd/Ctrl + K for search
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            const searchInput = document.getElementById('global-search');
            if (searchInput) {
                searchInput.focus();
            }
        }
        
        // Escape to close modals/dropdowns
        if (e.key === 'Escape') {
            this.closeAllDropdowns();
            this.hideSearchResults();
        }
    }

    showPage(pageId) {
        // Update navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.remove('active');
        });
        
        event.target.closest('.nav-item').classList.add('active');

        // Update page title
        const titles = {
            'dashboard': 'Dashboard',
            'media': 'Media Library',
            'services': 'Services',
            'monitoring': 'System Monitoring',
            'logs': 'System Logs',
            'users': 'User Management',
            'backup': 'Backup & Restore',
            'settings': 'Settings',
            'api-docs': 'API Documentation',
            'help': 'Help & Support'
        };
        
        document.getElementById('page-title').textContent = titles[pageId] || pageId;

        // Load page content
        this.loadPageContent(pageId);
        this.currentPage = pageId;

        // Close mobile sidebar
        if (window.innerWidth < 768) {
            this.toggleSidebar();
        }
    }

    async loadPageContent(pageId) {
        const container = document.getElementById('page-content');
        
        // Show loading
        this.showLoading(true);

        try {
            const content = await this.getPageContent(pageId);
            container.innerHTML = content;
            
            // Initialize page-specific functionality
            this.initPageFeatures(pageId);
        } catch (error) {
            console.error('Error loading page:', error);
            container.innerHTML = `
                <div class="text-center py-12">
                    <div class="text-red-400 mb-4">Error loading page</div>
                    <button onclick="dashboard.loadPageContent('${pageId}')" class="btn btn-primary">Retry</button>
                </div>
            `;
        } finally {
            this.showLoading(false);
        }
    }

    async getPageContent(pageId) {
        const pages = {
            'dashboard': this.getDashboardContent(),
            'media': this.getMediaContent(),
            'services': this.getServicesContent(),
            'monitoring': this.getMonitoringContent(),
            'logs': this.getLogsContent(),
            'users': this.getUsersContent(),
            'backup': this.getBackupContent(),
            'settings': this.getSettingsContent(),
            'api-docs': this.getAPIDocsContent(),
            'help': this.getHelpContent()
        };

        return pages[pageId] || '<div>Page not found</div>';
    }

    getDashboardContent() {
        return document.getElementById('dashboard-page').innerHTML;
    }

    getMediaContent() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">Media Library</h2>
                    <div class="flex space-x-2">
                        <button class="btn btn-secondary" onclick="dashboard.scanLibrary()">Scan Library</button>
                        <button class="btn btn-primary" onclick="dashboard.addMedia()">Add Media</button>
                    </div>
                </div>

                <div class="grid grid-cols-1 md:grid-cols-3 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">Movies</h3>
                        <div class="text-3xl font-bold text-neon-blue">1,247</div>
                        <div class="text-sm text-gray-400">Total movies</div>
                    </div>
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">TV Shows</h3>
                        <div class="text-3xl font-bold text-neon-purple">342</div>
                        <div class="text-sm text-gray-400">Total series</div>
                    </div>
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">Music</h3>
                        <div class="text-3xl font-bold text-neon-green">8,451</div>
                        <div class="text-sm text-gray-400">Total tracks</div>
                    </div>
                </div>

                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">Recently Added</h3>
                    <div id="recent-media" class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
                        <!-- Media items will be loaded here -->
                    </div>
                </div>
            </div>
        `;
    }

    getServicesContent() {
        return `
            <div class="space-y-6">
                <div class="flex items-center justify-between">
                    <h2 class="text-2xl font-bold">Services Management</h2>
                    <button class="btn btn-primary" onclick="dashboard.refreshAllServices()">Refresh All</button>
                </div>

                <div id="services-detailed" class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    <!-- Services will be loaded here -->
                </div>

                <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                    <h3 class="text-lg font-semibold mb-4">Service Configuration</h3>
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
                    </div>
                </div>
            </div>
        `;
    }

    getMonitoringContent() {
        return `
            <div class="space-y-6">
                <h2 class="text-2xl font-bold">System Monitoring</h2>

                <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-gray-400 mb-2">CPU Usage</h3>
                        <div class="text-2xl font-bold text-neon-blue">23%</div>
                        <div class="progress-bar mt-2">
                            <div class="progress-fill" style="width: 23%"></div>
                        </div>
                    </div>
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-gray-400 mb-2">Memory</h3>
                        <div class="text-2xl font-bold text-neon-purple">67%</div>
                        <div class="progress-bar mt-2">
                            <div class="progress-fill" style="width: 67%"></div>
                        </div>
                    </div>
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-gray-400 mb-2">Disk I/O</h3>
                        <div class="text-2xl font-bold text-neon-green">12%</div>
                        <div class="progress-bar mt-2">
                            <div class="progress-fill" style="width: 12%"></div>
                        </div>
                    </div>
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-gray-400 mb-2">Network</h3>
                        <div class="text-2xl font-bold text-neon-yellow">145 MB/s</div>
                        <div class="text-sm text-gray-400">Avg throughput</div>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">CPU & Memory Usage</h3>
                        <canvas id="systemChart" height="200"></canvas>
                    </div>
                    <div class="bg-dark-secondary rounded-xl p-6 border border-glass-border">
                        <h3 class="text-lg font-semibold mb-4">Network Traffic</h3>
                        <canvas id="networkChart" height="200"></canvas>
                    </div>
                </div>
            </div>
        `;
    }

    initPageFeatures(pageId) {
        switch (pageId) {
            case 'services':
                this.renderDetailedServices();
                break;
            case 'monitoring':
                this.initMonitoringCharts();
                break;
            case 'logs':
                this.initLogViewer();
                break;
            case 'users':
                this.loadUsers();
                break;
            case 'backup':
                this.initBackupFeatures();
                break;
            case 'settings':
                this.loadSettings();
                break;
        }
    }

    renderServices() {
        const container = document.getElementById('services-grid');
        if (!container) return;

        container.innerHTML = Array.from(this.services.values()).map(service => `
            <div class="service-card p-4 bg-glass rounded-xl border border-glass-border hover:border-neon-${this.getServiceColor(service.status)}/50 transition-all">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center space-x-2">
                        <span class="text-lg">${service.icon}</span>
                        <span class="font-medium">${service.name}</span>
                    </div>
                    <div class="w-2 h-2 bg-neon-${this.getServiceColor(service.status)} rounded-full ${service.status === 'online' ? 'animate-pulse' : ''}"></div>
                </div>
                <p class="text-sm text-gray-400 capitalize">${service.status}</p>
            </div>
        `).join('');
    }

    getServiceColor(status) {
        const colors = {
            'online': 'green',
            'offline': 'pink',
            'warning': 'yellow'
        };
        return colors[status] || 'gray';
    }

    async refreshServices() {
        const refreshIcon = document.getElementById('refresh-icon');
        if (refreshIcon) {
            refreshIcon.classList.add('animate-spin');
        }

        try {
            // Simulate API call
            await new Promise(resolve => setTimeout(resolve, 1000));
            
            // Update service statuses (simulate some changes)
            this.services.forEach((service, key) => {
                if (Math.random() > 0.8) {
                    const statuses = ['online', 'warning', 'offline'];
                    service.status = statuses[Math.floor(Math.random() * statuses.length)];
                }
            });

            this.renderServices();
            this.showNotification('Services refreshed successfully', 'success');
        } catch (error) {
            this.showNotification('Failed to refresh services', 'error');
        } finally {
            if (refreshIcon) {
                refreshIcon.classList.remove('animate-spin');
            }
        }
    }

    updateNotificationCount() {
        const unreadCount = this.notifications.filter(n => !n.read).length;
        const countElement = document.getElementById('notification-count');
        if (countElement) {
            countElement.textContent = unreadCount;
            countElement.style.display = unreadCount > 0 ? 'flex' : 'none';
        }
    }

    renderNotifications() {
        const container = document.getElementById('notifications-list');
        if (!container) return;

        container.innerHTML = this.notifications.map(notification => `
            <div class="notification-item p-3 rounded-lg ${notification.read ? '' : 'unread'}" onclick="dashboard.markAsRead(${notification.id})">
                <div class="flex items-start space-x-3">
                    <div class="w-2 h-2 bg-neon-${this.getNotificationColor(notification.type)} rounded-full mt-2 flex-shrink-0"></div>
                    <div class="flex-1">
                        <div class="font-medium">${notification.title}</div>
                        <div class="text-sm text-gray-400">${notification.message}</div>
                        <div class="text-xs text-gray-500 mt-1">${notification.time}</div>
                    </div>
                </div>
            </div>
        `).join('');
    }

    getNotificationColor(type) {
        const colors = {
            'success': 'green',
            'warning': 'yellow',
            'error': 'pink',
            'info': 'blue'
        };
        return colors[type] || 'blue';
    }

    markAsRead(notificationId) {
        const notification = this.notifications.find(n => n.id === notificationId);
        if (notification) {
            notification.read = true;
            this.updateNotificationCount();
            this.renderNotifications();
        }
    }

    toggleNotifications() {
        const panel = document.getElementById('notification-panel');
        if (panel) {
            panel.classList.toggle('translate-x-full');
        }
    }

    toggleProfile() {
        const menu = document.getElementById('profile-menu');
        if (menu) {
            menu.classList.toggle('translate-x-full');
        }
    }

    toggleTheme() {
        this.theme = this.theme === 'dark' ? 'light' : 'dark';
        document.documentElement.className = this.theme;
        localStorage.setItem('theme', this.theme);
        this.updateThemeIcon();
    }

    updateThemeIcon() {
        const icon = document.getElementById('theme-icon');
        if (icon) {
            if (this.theme === 'dark') {
                icon.innerHTML = `<path fill-rule="evenodd" d="M10 2a1 1 0 011 1v1a1 1 0 11-2 0V3a1 1 0 011-1zm4 8a4 4 0 11-8 0 4 4 0 018 0zm-.464 4.95l.707.707a1 1 0 001.414-1.414l-.707-.707a1 1 0 00-1.414 1.414zm2.12-10.607a1 1 0 010 1.414l-.706.707a1 1 0 11-1.414-1.414l.707-.707a1 1 0 011.414 0zM17 11a1 1 0 100-2h-1a1 1 0 100 2h1zm-7 4a1 1 0 011 1v1a1 1 0 11-2 0v-1a1 1 0 011-1zM5.05 6.464A1 1 0 106.465 5.05l-.708-.707a1 1 0 00-1.414 1.414l.707.707zm1.414 8.486l-.707.707a1 1 0 01-1.414-1.414l.707-.707a1 1 0 011.414 1.414zM4 11a1 1 0 100-2H3a1 1 0 000 2h1z" clip-rule="evenodd"/>`;
            } else {
                icon.innerHTML = `<path d="M17.293 13.293A8 8 0 016.707 2.707a8.001 8.001 0 1010.586 10.586z"/>`;
            }
        }
    }

    toggleSidebar() {
        const sidebar = document.getElementById('sidebar');
        if (sidebar) {
            sidebar.classList.toggle('-translate-x-full');
        }
    }

    showLoading(show) {
        const overlay = document.getElementById('loading-overlay');
        if (overlay) {
            overlay.classList.toggle('hidden', !show);
        }
    }

    showNotification(message, type = 'info') {
        const notification = {
            id: Date.now(),
            title: type.charAt(0).toUpperCase() + type.slice(1),
            message: message,
            time: 'Just now',
            read: false,
            type: type
        };

        this.notifications.unshift(notification);
        this.updateNotificationCount();
        this.renderNotifications();

        // Auto-remove after 5 seconds
        setTimeout(() => {
            const index = this.notifications.findIndex(n => n.id === notification.id);
            if (index > -1) {
                this.notifications.splice(index, 1);
                this.updateNotificationCount();
                this.renderNotifications();
            }
        }, 5000);
    }

    closeAllDropdowns() {
        document.querySelectorAll('.dropdown').forEach(dropdown => {
            dropdown.classList.remove('active');
        });
    }

    startRealTimeUpdates() {
        // Update stats every 30 seconds
        setInterval(() => {
            this.updateRealTimeStats();
        }, 30000);

        // Check service health every 60 seconds
        setInterval(() => {
            this.checkServiceHealth();
        }, 60000);
    }

    async updateRealTimeStats() {
        try {
            // Simulate API call for real-time stats
            const stats = await this.fetchStats();
            this.updateDashboardStats(stats);
        } catch (error) {
            console.error('Error updating stats:', error);
        }
    }

    async fetchStats() {
        // Simulate API response
        return new Promise(resolve => {
            setTimeout(() => {
                resolve({
                    activeStreams: Math.floor(Math.random() * 20) + 1,
                    totalMedia: 2847 + Math.floor(Math.random() * 10),
                    storageUsed: (12.4 + Math.random() * 0.5).toFixed(1),
                    systemHealth: Math.floor(Math.random() * 5) + 95
                });
            }, 500);
        });
    }

    updateDashboardStats(stats) {
        const elements = {
            'active-streams': stats.activeStreams,
            'total-media': stats.totalMedia.toLocaleString(),
            'storage-used': stats.storageUsed,
            'system-health': stats.systemHealth
        };

        Object.entries(elements).forEach(([id, value]) => {
            const element = document.getElementById(id);
            if (element) {
                element.textContent = value + (id === 'system-health' ? '%' : '');
            }
        });
    }

    async checkServiceHealth() {
        // Simulate health checks
        this.services.forEach(async (service, key) => {
            try {
                const health = await this.pingService(service.url);
                service.status = health ? 'online' : 'offline';
            } catch (error) {
                service.status = 'offline';
            }
        });

        this.renderServices();
    }

    async pingService(url) {
        // Simulate service ping
        return new Promise(resolve => {
            setTimeout(() => {
                resolve(Math.random() > 0.1); // 90% uptime simulation
            }, Math.random() * 1000);
        });
    }

    async loadDashboardData() {
        try {
            // Load recent activity
            const activity = await this.fetchRecentActivity();
            this.renderRecentActivity(activity);
        } catch (error) {
            console.error('Error loading dashboard data:', error);
        }
    }

    async fetchRecentActivity() {
        return new Promise(resolve => {
            setTimeout(() => {
                resolve([
                    { type: 'download', title: 'The Batman (2022)', time: '2 min ago', icon: '⬇️' },
                    { type: 'stream', title: 'Breaking Bad S01E01', time: '5 min ago', icon: '▶️' },
                    { type: 'add', title: 'House of the Dragon added', time: '15 min ago', icon: '➕' },
                    { type: 'user', title: 'New user registered', time: '1 hour ago', icon: '👤' }
                ]);
            }, 300);
        });
    }

    renderRecentActivity(activities) {
        const container = document.getElementById('recent-activity');
        if (!container) return;

        container.innerHTML = activities.map(activity => `
            <div class="flex items-center space-x-3 p-3 bg-glass rounded-lg hover:bg-white/20 transition-colors">
                <span class="text-lg">${activity.icon}</span>
                <div class="flex-1">
                    <div class="font-medium">${activity.title}</div>
                    <div class="text-sm text-gray-400">${activity.time}</div>
                </div>
                <div class="text-xs text-gray-500 capitalize">${activity.type}</div>
            </div>
        `).join('');
    }
}

// Voice control functionality
function toggleVoice() {
    const statusElement = document.getElementById('voice-status');
    
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        const recognition = new SpeechRecognition();
        
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';
        
        recognition.onstart = () => {
            statusElement.className = 'w-2 h-2 bg-neon-pink rounded-full animate-pulse';
            dashboard.showNotification('Voice recognition started. Say "Hey MediaFlow"', 'info');
        };
        
        recognition.onresult = (event) => {
            const command = event.results[0][0].transcript.toLowerCase();
            console.log('Voice command:', command);
            
            if (command.includes('hey mediaflow') || command.includes('media flow')) {
                dashboard.processVoiceCommand(command);
            }
        };
        
        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            statusElement.className = 'w-2 h-2 bg-neon-pink rounded-full';
            dashboard.showNotification('Voice recognition error: ' + event.error, 'error');
        };
        
        recognition.onend = () => {
            statusElement.className = 'w-2 h-2 bg-neon-green rounded-full animate-pulse';
        };
        
        recognition.start();
    } else {
        dashboard.showNotification('Speech recognition not supported in this browser', 'warning');
    }
}

// Global functions
function showPage(pageId) {
    if (window.dashboard) {
        window.dashboard.showPage(pageId);
    }
}

function refreshServices() {
    if (window.dashboard) {
        window.dashboard.refreshServices();
    }
}

function toggleSidebar() {
    if (window.dashboard) {
        window.dashboard.toggleSidebar();
    }
}

function toggleTheme() {
    if (window.dashboard) {
        window.dashboard.toggleTheme();
    }
}

function toggleNotifications() {
    if (window.dashboard) {
        window.dashboard.toggleNotifications();
    }
}

function toggleProfile() {
    if (window.dashboard) {
        window.dashboard.toggleProfile();
    }
}

function navigateToResult(url) {
    console.log('Navigating to:', url);
    // Implement navigation logic
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new MediaFlowDashboard();
});