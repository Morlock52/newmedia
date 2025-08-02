// HoloMedia Hub - App Integration Script
class HoloMediaHub {
    constructor() {
        this.currentPage = 'dashboard';
        this.notifications = [];
        this.settings = this.loadSettings();
        this.activityLog = [];
        this.router = null;
        this.init();
    }

    init() {
        this.setupRouter();
        this.setupNavigation();
        this.setupSearch();
        this.setupNotifications();
        this.setupTheme();
        this.setupSettings();
        this.setupQuickActions();
        this.setupActivityFeed();
        this.setupKeyboardShortcuts();
        this.setupServiceWorkerUpdates();
        this.startPeriodicUpdates();
    }

    // Setup Modern Router
    setupRouter() {
        this.router = new Router({
            root: '/holographic-dashboard',
            transitionDuration: 300,
            beforeRoute: async (to, from) => {
                // Show loading
                const navHeader = document.querySelector('navigation-header');
                if (navHeader) navHeader.showLoading();
                return true;
            },
            afterRoute: async (context) => {
                // Hide loading
                const navHeader = document.querySelector('navigation-header');
                if (navHeader) navHeader.hideLoading();
                
                // Update active links
                this.router.updateActiveLinks();
                
                // Log activity
                this.logActivity(`Navigated to ${context.path}`, '📍');
            },
            notFound: (path) => {
                console.error('Route not found:', path);
                this.addNotification('Page Not Found', `The page ${path} could not be found.`, 'error');
                this.router.navigate('/dashboard');
            }
        });

        // Define routes
        this.router
            .route('/dashboard', (ctx) => this.showPage('dashboard'))
            .route('/media-library', (ctx) => this.showPage('media-library'))
            .route('/config-manager', (ctx) => this.showPage('config-manager'))
            .route('/env-editor', (ctx) => this.showPage('env-editor'))
            .route('/ai-assistant', (ctx) => this.showPage('ai-assistant'))
            .route('/workflow-builder', (ctx) => this.showPage('workflow-builder'))
            .route('/health-monitor', (ctx) => this.showPage('health-monitor'))
            .route('/documentation', (ctx) => this.showPage('documentation'))
            .route('/', (ctx) => this.router.navigate('/dashboard', {}, true));

        // Handle initial route
        const path = window.location.pathname.replace('/holographic-dashboard/main-app.html', '');
        if (path === '' || path === '/') {
            this.router.navigate('/dashboard', {}, true);
        }
    }

    // Navigation System
    setupNavigation() {
        const sidebarToggle = document.getElementById('sidebarToggle');
        const sidebar = document.getElementById('sidebar');

        if (sidebarToggle && sidebar) {
            sidebarToggle.addEventListener('click', () => {
                sidebar.classList.toggle('collapsed');
                this.settings.compactSidebar = sidebar.classList.contains('collapsed');
                this.saveSettings();
            });
        }

        // Handle hash-based navigation for backward compatibility
        if (window.location.hash) {
            const page = window.location.hash.substring(1);
            this.router.navigate(`/${page}`);
        }
    }

    showPage(page) {
        // Update active nav item
        document.querySelectorAll('.nav-item').forEach(item => {
            item.classList.toggle('active', item.dataset.page === page);
        });

        // Update active page content with smooth transition
        const allPages = document.querySelectorAll('.page-content');
        const targetPage = document.getElementById(`${page}-page`);
        
        // Fade out current page
        allPages.forEach(content => {
            if (content.classList.contains('active')) {
                content.style.opacity = '0';
                setTimeout(() => {
                    content.classList.remove('active');
                    content.style.display = 'none';
                }, 300);
            }
        });

        // Fade in new page
        if (targetPage) {
            setTimeout(() => {
                targetPage.style.display = 'block';
                targetPage.classList.add('active');
                setTimeout(() => {
                    targetPage.style.opacity = '1';
                }, 50);
            }, 300);
        }

        this.currentPage = page;

        // Page-specific initialization
        this.initializePage(page);
    }

    navigateToPage(page) {
        // Use the router for navigation
        this.router.navigate(`/${page}`);
    }

    initializePage(page) {
        switch(page) {
            case 'dashboard':
                this.updateDashboardStats();
                break;
            case 'media-library':
                this.sendMessageToIframe('media-library', { action: 'refresh' });
                break;
            case 'ai-assistant':
                this.sendMessageToIframe('ai-assistant', { action: 'focus' });
                break;
            // Add other page initializations as needed
        }
    }

    // Global Search
    setupSearch() {
        const searchInput = document.getElementById('globalSearch');
        let searchTimeout;

        searchInput.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this.performSearch(e.target.value);
            }, 300);
        });

        searchInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') {
                this.performSearch(e.target.value);
            }
        });
    }

    performSearch(query) {
        if (!query.trim()) return;

        // Search across all features
        const results = this.searchAllFeatures(query);
        this.displaySearchResults(results);
        this.logActivity(`Searched for: "${query}"`, '🔍');
    }

    searchAllFeatures(query) {
        // Implement search logic across all features
        const results = [];
        const lowerQuery = query.toLowerCase();

        // Search navigation items
        document.querySelectorAll('.nav-item').forEach(item => {
            const label = item.querySelector('.label').textContent;
            if (label.toLowerCase().includes(lowerQuery)) {
                results.push({
                    type: 'navigation',
                    title: label,
                    action: () => this.navigateToPage(item.dataset.page)
                });
            }
        });

        // Add more search contexts here
        return results;
    }

    // Notification System
    setupNotifications() {
        const notificationBtn = document.getElementById('notificationBtn');
        const notificationPanel = document.getElementById('notificationPanel');
        const closeBtn = document.getElementById('closeNotifications');

        notificationBtn.addEventListener('click', () => {
            notificationPanel.classList.toggle('active');
            this.markNotificationsAsRead();
        });

        closeBtn.addEventListener('click', () => {
            notificationPanel.classList.remove('active');
        });

        // Request notification permission
        if ('Notification' in window && this.settings.desktopNotifications) {
            Notification.requestPermission();
        }

        // Load initial notifications
        this.loadNotifications();
    }

    addNotification(title, message, type = 'info', actions = []) {
        const notification = {
            id: Date.now(),
            title,
            message,
            type,
            actions,
            timestamp: new Date(),
            read: false
        };

        this.notifications.unshift(notification);
        this.updateNotificationBadge();
        this.renderNotifications();

        // Show desktop notification if enabled
        if (this.settings.desktopNotifications && Notification.permission === 'granted') {
            new Notification(title, {
                body: message,
                icon: '/icons/icon-192x192.png'
            });
        }

        // Play sound if enabled
        if (this.settings.soundAlerts) {
            this.playNotificationSound();
        }

        return notification;
    }

    // Theme Management
    setupTheme() {
        const themeToggle = document.getElementById('themeToggle');
        const darkModeToggle = document.getElementById('darkModeToggle');

        // Apply saved theme
        document.body.classList.toggle('light-theme', !this.settings.darkMode);

        themeToggle.addEventListener('click', () => {
            this.settings.darkMode = !this.settings.darkMode;
            document.body.classList.toggle('light-theme', !this.settings.darkMode);
            themeToggle.textContent = this.settings.darkMode ? '🌙' : '☀️';
            this.saveSettings();
        });

        if (darkModeToggle) {
            darkModeToggle.checked = this.settings.darkMode;
            darkModeToggle.addEventListener('change', (e) => {
                this.settings.darkMode = e.target.checked;
                document.body.classList.toggle('light-theme', !this.settings.darkMode);
                themeToggle.textContent = this.settings.darkMode ? '🌙' : '☀️';
                this.saveSettings();
            });
        }
    }

    // Settings Management
    setupSettings() {
        const settingsBtn = document.getElementById('settingsBtn');
        const settingsModal = document.getElementById('settingsModal');
        const closeSettings = document.getElementById('closeSettings');
        const saveSettings = document.getElementById('saveSettings');
        const cancelSettings = document.getElementById('cancelSettings');

        settingsBtn.addEventListener('click', () => {
            this.openSettings();
        });

        closeSettings.addEventListener('click', () => {
            settingsModal.classList.remove('active');
        });

        cancelSettings.addEventListener('click', () => {
            settingsModal.classList.remove('active');
        });

        saveSettings.addEventListener('click', () => {
            this.saveSettingsFromModal();
            settingsModal.classList.remove('active');
        });

        // Apply settings
        this.applySettings();
    }

    openSettings() {
        const modal = document.getElementById('settingsModal');
        modal.classList.add('active');

        // Load current settings into form
        document.getElementById('darkModeToggle').checked = this.settings.darkMode;
        document.getElementById('compactSidebarToggle').checked = this.settings.compactSidebar;
        document.getElementById('desktopNotificationsToggle').checked = this.settings.desktopNotifications;
        document.getElementById('soundAlertsToggle').checked = this.settings.soundAlerts;
        document.getElementById('animationsToggle').checked = this.settings.animations;
        document.getElementById('autoRefreshToggle').checked = this.settings.autoRefresh;
    }

    saveSettingsFromModal() {
        this.settings = {
            darkMode: document.getElementById('darkModeToggle').checked,
            compactSidebar: document.getElementById('compactSidebarToggle').checked,
            desktopNotifications: document.getElementById('desktopNotificationsToggle').checked,
            soundAlerts: document.getElementById('soundAlertsToggle').checked,
            animations: document.getElementById('animationsToggle').checked,
            autoRefresh: document.getElementById('autoRefreshToggle').checked
        };

        this.saveSettings();
        this.applySettings();
        this.addNotification('Settings Updated', 'Your preferences have been saved', 'success');
    }

    // Quick Actions
    setupQuickActions() {
        const quickActionBtn = document.getElementById('quickAction');
        const quickActionMenu = document.getElementById('quickActionMenu');

        quickActionBtn.addEventListener('click', () => {
            quickActionMenu.classList.toggle('active');
        });

        // Setup quick action buttons
        document.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const action = e.currentTarget.dataset.action;
                this.handleQuickAction(action);
            });
        });

        // Close menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!quickActionMenu.contains(e.target) && e.target !== quickActionBtn) {
                quickActionMenu.classList.remove('active');
            }
        });
    }

    handleQuickAction(action) {
        switch(action) {
            case 'scan-media':
                this.scanMediaLibraries();
                break;
            case 'add-media':
                this.openAddMediaDialog();
                break;
            case 'run-workflow':
                this.navigateToPage('workflow-builder');
                break;
            case 'view-logs':
                this.viewSystemLogs();
                break;
            case 'scan-all':
                this.scanAllLibraries();
                break;
            case 'restart-services':
                this.restartServices();
                break;
            case 'clear-cache':
                this.clearCache();
                break;
            case 'export-config':
                this.exportConfiguration();
                break;
        }

        // Close quick action menu
        document.getElementById('quickActionMenu').classList.remove('active');
    }

    // Activity Feed
    setupActivityFeed() {
        this.updateActivityFeed();
    }

    logActivity(message, icon = '📌') {
        const activity = {
            message,
            icon,
            timestamp: new Date()
        };

        this.activityLog.unshift(activity);
        if (this.activityLog.length > 100) {
            this.activityLog = this.activityLog.slice(0, 100);
        }

        this.updateActivityFeed();
        this.saveActivityLog();
    }

    updateActivityFeed() {
        const feed = document.getElementById('activityFeed');
        if (!feed) return;

        const recentActivities = this.activityLog.slice(0, 10);
        
        feed.innerHTML = recentActivities.map(activity => `
            <div class="activity-item">
                <span class="activity-icon">${activity.icon}</span>
                <div class="activity-content">
                    <p>${activity.message}</p>
                    <span class="activity-time">${this.getRelativeTime(activity.timestamp)}</span>
                </div>
            </div>
        `).join('');
    }

    // Keyboard Shortcuts
    setupKeyboardShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl/Cmd + K for search
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                document.getElementById('globalSearch').focus();
            }

            // Ctrl/Cmd + / for shortcuts help
            if ((e.ctrlKey || e.metaKey) && e.key === '/') {
                e.preventDefault();
                this.showKeyboardShortcuts();
            }

            // Number keys for quick navigation
            if (e.altKey && e.key >= '1' && e.key <= '9') {
                e.preventDefault();
                const navItems = document.querySelectorAll('.nav-item');
                const index = parseInt(e.key) - 1;
                if (navItems[index]) {
                    const page = navItems[index].dataset.page;
                    this.router.navigate(`/${page}`);
                }
            }
        });
    }

    // Service Worker Updates
    setupServiceWorkerUpdates() {
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.addEventListener('controllerchange', () => {
                this.addNotification(
                    'App Updated',
                    'HoloMedia Hub has been updated. Refresh to see changes.',
                    'info',
                    [{ label: 'Refresh', action: () => window.location.reload() }]
                );
            });
        }
    }

    // Periodic Updates
    startPeriodicUpdates() {
        if (this.settings.autoRefresh) {
            // Update dashboard stats every 30 seconds
            setInterval(() => {
                if (this.currentPage === 'dashboard') {
                    this.updateDashboardStats();
                }
            }, 30000);

            // Check for notifications every minute
            setInterval(() => {
                this.checkForNewNotifications();
            }, 60000);
        }
    }

    // Helper Methods
    updateDashboardStats() {
        // Implement dashboard stats update
        console.log('Updating dashboard stats...');
    }

    sendMessageToIframe(page, message) {
        const iframe = document.querySelector(`#${page}-page iframe`);
        if (iframe && iframe.contentWindow) {
            iframe.contentWindow.postMessage(message, '*');
        }
    }

    getPageTitle(page) {
        const titles = {
            'dashboard': 'Dashboard',
            'media-library': 'Media Library',
            'config-manager': 'Configuration Manager',
            'env-editor': '.env Editor',
            'ai-assistant': 'AI Assistant',
            'workflow-builder': 'Workflow Builder',
            'health-monitor': 'Health Monitor',
            'documentation': 'Documentation'
        };
        return titles[page] || page;
    }

    getRelativeTime(date) {
        const now = new Date();
        const diff = now - date;
        const seconds = Math.floor(diff / 1000);
        const minutes = Math.floor(seconds / 60);
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
        if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
        if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
        return 'Just now';
    }

    // Storage Methods
    loadSettings() {
        const defaults = {
            darkMode: true,
            compactSidebar: false,
            desktopNotifications: false,
            soundAlerts: true,
            animations: true,
            autoRefresh: true
        };
        
        try {
            const saved = localStorage.getItem('holomedia-settings');
            return saved ? { ...defaults, ...JSON.parse(saved) } : defaults;
        } catch {
            return defaults;
        }
    }

    saveSettings() {
        try {
            localStorage.setItem('holomedia-settings', JSON.stringify(this.settings));
        } catch (e) {
            console.error('Failed to save settings:', e);
        }
    }

    loadNotifications() {
        try {
            const saved = localStorage.getItem('holomedia-notifications');
            this.notifications = saved ? JSON.parse(saved) : [];
            this.updateNotificationBadge();
            this.renderNotifications();
        } catch {
            this.notifications = [];
        }
    }

    saveNotifications() {
        try {
            localStorage.setItem('holomedia-notifications', JSON.stringify(this.notifications));
        } catch (e) {
            console.error('Failed to save notifications:', e);
        }
    }

    loadActivityLog() {
        try {
            const saved = localStorage.getItem('holomedia-activity');
            this.activityLog = saved ? JSON.parse(saved) : [];
        } catch {
            this.activityLog = [];
        }
    }

    saveActivityLog() {
        try {
            localStorage.setItem('holomedia-activity', JSON.stringify(this.activityLog));
        } catch (e) {
            console.error('Failed to save activity log:', e);
        }
    }

    // Notification Methods
    updateNotificationBadge() {
        const badge = document.getElementById('notificationBadge');
        const unreadCount = this.notifications.filter(n => !n.read).length;
        
        if (badge) {
            badge.textContent = unreadCount;
            badge.style.display = unreadCount > 0 ? 'block' : 'none';
        }
    }

    renderNotifications() {
        const list = document.getElementById('notificationList');
        if (!list) return;

        list.innerHTML = this.notifications.map(notification => `
            <div class="notification-item ${notification.type} ${notification.read ? 'read' : ''}" data-id="${notification.id}">
                <div class="notification-content">
                    <h4>${notification.title}</h4>
                    <p>${notification.message}</p>
                    <span class="notification-time">${this.getRelativeTime(notification.timestamp)}</span>
                </div>
                ${notification.actions && notification.actions.length > 0 ? `
                    <div class="notification-actions">
                        ${notification.actions.map(action => `
                            <button class="notification-action" data-action="${action.label}">${action.label}</button>
                        `).join('')}
                    </div>
                ` : ''}
            </div>
        `).join('');

        // Add action listeners
        list.querySelectorAll('.notification-action').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const notificationId = parseInt(e.target.closest('.notification-item').dataset.id);
                const actionLabel = e.target.dataset.action;
                const notification = this.notifications.find(n => n.id === notificationId);
                const action = notification.actions.find(a => a.label === actionLabel);
                if (action && action.action) {
                    action.action();
                }
            });
        });
    }

    markNotificationsAsRead() {
        let updated = false;
        this.notifications.forEach(n => {
            if (!n.read) {
                n.read = true;
                updated = true;
            }
        });
        
        if (updated) {
            this.updateNotificationBadge();
            this.saveNotifications();
        }
    }

    // Action Methods
    scanMediaLibraries() {
        this.addNotification('Media Scan Started', 'Scanning all media libraries...', 'info');
        this.logActivity('Started media library scan', '📡');
        
        // Simulate scan completion
        setTimeout(() => {
            this.addNotification('Media Scan Complete', 'Found 23 new items', 'success');
            this.logActivity('Media scan completed: 23 new items', '✅');
        }, 3000);
    }

    openAddMediaDialog() {
        // Implement add media dialog
        this.logActivity('Opened add media dialog', '➕');
    }

    viewSystemLogs() {
        // Implement system logs viewer
        this.logActivity('Viewing system logs', '📋');
    }

    scanAllLibraries() {
        this.addNotification('Full Scan Started', 'Scanning all libraries and services...', 'info');
        this.logActivity('Started full system scan', '🔍');
    }

    restartServices() {
        this.addNotification('Restarting Services', 'All services are being restarted...', 'warning');
        this.logActivity('Restarting all services', '🔄');
    }

    clearCache() {
        if ('caches' in window) {
            caches.keys().then(names => {
                names.forEach(name => caches.delete(name));
            });
            this.addNotification('Cache Cleared', 'All cached data has been removed', 'success');
            this.logActivity('Cleared application cache', '🗑️');
        }
    }

    exportConfiguration() {
        // Implement configuration export
        this.addNotification('Configuration Exported', 'Configuration saved to Downloads', 'success');
        this.logActivity('Exported system configuration', '💾');
    }

    checkForNewNotifications() {
        // Implement notification check
        console.log('Checking for new notifications...');
    }

    playNotificationSound() {
        // Create and play notification sound
        const audio = new Audio('data:audio/wav;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBDGAzvLZhjsGGGS07OScTgwOUqzz8rZeFwFGnN7tu2gmBTGS0fXZiiMFLITO+NaE');
        audio.volume = 0.3;
        audio.play();
    }

    showKeyboardShortcuts() {
        const shortcuts = [
            { keys: 'Ctrl/Cmd + K', description: 'Focus search' },
            { keys: 'Alt + 1-9', description: 'Quick navigation' },
            { keys: 'Ctrl/Cmd + /', description: 'Show shortcuts' },
            { keys: 'Escape', description: 'Close dialogs' }
        ];

        const content = shortcuts.map(s => `${s.keys}: ${s.description}`).join('\n');
        alert('Keyboard Shortcuts:\n\n' + content);
    }

    applySettings() {
        const sidebar = document.getElementById('sidebar');
        if (this.settings.compactSidebar) {
            sidebar.classList.add('collapsed');
        }

        if (!this.settings.animations) {
            document.body.classList.add('no-animations');
        } else {
            document.body.classList.remove('no-animations');
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Add smooth transitions CSS
    const style = document.createElement('style');
    style.textContent = `
        .page-content {
            transition: opacity 0.3s ease;
            opacity: 0;
            display: none;
        }
        .page-content.active {
            display: block;
            opacity: 1;
        }
        [data-router-view] {
            min-height: calc(100vh - 80px);
        }
    `;
    document.head.appendChild(style);
    
    window.holoMediaHub = new HoloMediaHub();
});

// Handle messages from iframes
window.addEventListener('message', (event) => {
    // Verify origin if needed
    if (event.data && event.data.action) {
        switch(event.data.action) {
            case 'navigate':
                window.holoMediaHub.navigateToPage(event.data.page);
                break;
            case 'notification':
                window.holoMediaHub.addNotification(event.data.title, event.data.message, event.data.type);
                break;
            case 'activity':
                window.holoMediaHub.logActivity(event.data.message, event.data.icon);
                break;
        }
    }
});