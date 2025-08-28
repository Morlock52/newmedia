// Notifications System Module

class NotificationSystem {
    constructor() {
        this.notifications = [];
        this.settings = {
            maxNotifications: 50,
            autoRemoveDelay: 5000,
            soundEnabled: true,
            desktopEnabled: true,
            categories: {
                system: { enabled: true, sound: true },
                services: { enabled: true, sound: true },
                media: { enabled: true, sound: false },
                security: { enabled: true, sound: true },
                updates: { enabled: true, sound: false }
            }
        };
        this.init();
    }

    init() {
        this.loadSettings();
        this.requestPermissions();
        this.initWebSocket();
        this.loadNotifications();
    }

    loadSettings() {
        const saved = localStorage.getItem('notification-settings');
        if (saved) {
            this.settings = { ...this.settings, ...JSON.parse(saved) };
        }
    }

    saveSettings() {
        localStorage.setItem('notification-settings', JSON.stringify(this.settings));
    }

    async requestPermissions() {
        if ('Notification' in window && Notification.permission === 'default') {
            try {
                await Notification.requestPermission();
            } catch (error) {
                console.error('Error requesting notification permission:', error);
            }
        }
    }

    initWebSocket() {
        if (typeof io !== 'undefined') {
            try {
                this.socket = io();
                this.socket.on('notification', (data) => {
                    this.addNotification(data);
                });
                this.socket.on('system-alert', (data) => {
                    this.addNotification({ ...data, type: 'system', priority: 'high' });
                });
            } catch (error) {
                console.log('WebSocket not available, using polling fallback');
                this.startPolling();
            }
        } else {
            this.startPolling();
        }
    }

    startPolling() {
        setInterval(async () => {
            try {
                const response = await fetch('/api/notifications');
                if (response.ok) {
                    const data = await response.json();
                    data.notifications?.forEach(notification => {
                        if (!this.notifications.find(n => n.id === notification.id)) {
                            this.addNotification(notification);
                        }
                    });
                }
            } catch (error) {
                console.error('Error polling notifications:', error);
            }
        }, 30000);
    }

    loadNotifications() {
        // Load initial notifications
        const initialNotifications = [
            {
                id: 1,
                title: 'System Startup',
                message: 'MediaFlow Dashboard started successfully',
                type: 'system',
                category: 'system',
                priority: 'normal',
                timestamp: new Date(Date.now() - 5 * 60 * 1000),
                read: false,
                actions: []
            },
            {
                id: 2,
                title: 'Service Warning',
                message: 'Lidarr is experiencing connection issues',
                type: 'warning',
                category: 'services',
                priority: 'high',
                timestamp: new Date(Date.now() - 12 * 60 * 1000),
                read: false,
                actions: [
                    { label: 'Restart Service', action: 'restart-lidarr' },
                    { label: 'View Logs', action: 'view-logs-lidarr' }
                ]
            },
            {
                id: 3,
                title: 'New Content Added',
                message: 'Breaking Bad S05E16 has been added to your library',
                type: 'success',
                category: 'media',
                priority: 'normal',
                timestamp: new Date(Date.now() - 1 * 60 * 60 * 1000),
                read: true,
                actions: [
                    { label: 'View in Plex', action: 'open-plex' }
                ]
            },
            {
                id: 4,
                title: 'Download Complete',
                message: 'The Matrix Resurrections (2021) download completed',
                type: 'info',
                category: 'media',
                priority: 'normal',
                timestamp: new Date(Date.now() - 2 * 60 * 60 * 1000),
                read: true,
                actions: []
            },
            {
                id: 5,
                title: 'Backup Scheduled',
                message: 'Weekly backup will start in 2 hours',
                type: 'info',
                category: 'system',
                priority: 'normal',
                timestamp: new Date(Date.now() - 10 * 60 * 1000),
                read: false,
                actions: [
                    { label: 'View Schedule', action: 'view-backup-schedule' },
                    { label: 'Start Now', action: 'start-backup-now' }
                ]
            }
        ];

        this.notifications = initialNotifications;
        this.updateDisplay();
    }

    addNotification(notificationData) {
        const notification = {
            id: notificationData.id || Date.now() + Math.random(),
            title: notificationData.title || 'Notification',
            message: notificationData.message || '',
            type: notificationData.type || 'info',
            category: notificationData.category || 'system',
            priority: notificationData.priority || 'normal',
            timestamp: new Date(notificationData.timestamp || Date.now()),
            read: false,
            actions: notificationData.actions || [],
            metadata: notificationData.metadata || {}
        };

        // Check if category is enabled
        if (!this.settings.categories[notification.category]?.enabled) {
            return;
        }

        // Add to notifications array
        this.notifications.unshift(notification);

        // Limit notifications
        if (this.notifications.length > this.settings.maxNotifications) {
            this.notifications = this.notifications.slice(0, this.settings.maxNotifications);
        }

        // Show desktop notification
        this.showDesktopNotification(notification);

        // Play sound
        this.playNotificationSound(notification);

        // Show toast notification
        this.showToastNotification(notification);

        // Update display
        this.updateDisplay();

        // Auto-remove non-critical notifications
        if (notification.priority !== 'critical') {
            setTimeout(() => {
                this.removeNotification(notification.id);
            }, this.settings.autoRemoveDelay);
        }
    }

    showDesktopNotification(notification) {
        if (!this.settings.desktopEnabled || Notification.permission !== 'granted') {
            return;
        }

        const options = {
            body: notification.message,
            icon: this.getNotificationIcon(notification.type),
            tag: notification.id,
            requireInteraction: notification.priority === 'critical',
            actions: notification.actions.slice(0, 2).map(action => ({
                action: action.action,
                title: action.label
            }))
        };

        const desktopNotification = new Notification(notification.title, options);

        desktopNotification.onclick = () => {
            window.focus();
            this.markAsRead(notification.id);
            desktopNotification.close();
        };

        // Auto-close after delay
        if (notification.priority !== 'critical') {
            setTimeout(() => {
                desktopNotification.close();
            }, 5000);
        }
    }

    playNotificationSound(notification) {
        if (!this.settings.soundEnabled || !this.settings.categories[notification.category]?.sound) {
            return;
        }

        const audio = new Audio();
        const soundMap = {
            'error': '/assets/sounds/error.mp3',
            'warning': '/assets/sounds/warning.mp3',
            'success': '/assets/sounds/success.mp3',
            'info': '/assets/sounds/info.mp3'
        };

        audio.src = soundMap[notification.type] || soundMap.info;
        audio.volume = 0.3;
        audio.play().catch(error => {
            console.log('Could not play notification sound:', error);
        });
    }

    showToastNotification(notification) {
        const toast = document.createElement('div');
        toast.className = `fixed top-4 right-4 z-50 max-w-sm bg-dark-secondary border border-glass-border rounded-xl p-4 shadow-2xl transform translate-x-full transition-transform duration-300 toast-${notification.type}`;
        
        toast.innerHTML = `
            <div class="flex items-start space-x-3">
                <div class="flex-shrink-0">
                    <div class="w-8 h-8 rounded-full bg-neon-${this.getTypeColor(notification.type)}/20 flex items-center justify-center">
                        ${this.getTypeIcon(notification.type)}
                    </div>
                </div>
                <div class="flex-1 min-w-0">
                    <div class="font-medium text-white">${notification.title}</div>
                    <div class="text-sm text-gray-400 mt-1">${notification.message}</div>
                    ${notification.actions.length > 0 ? `
                        <div class="flex space-x-2 mt-3">
                            ${notification.actions.slice(0, 2).map(action => `
                                <button onclick="notificationSystem.executeAction('${action.action}', ${notification.id})" 
                                        class="text-xs px-3 py-1 bg-neon-${this.getTypeColor(notification.type)}/20 rounded-full hover:bg-neon-${this.getTypeColor(notification.type)}/30 transition-colors">
                                    ${action.label}
                                </button>
                            `).join('')}
                        </div>
                    ` : ''}
                </div>
                <button onclick="this.parentElement.parentElement.remove()" class="flex-shrink-0 text-gray-400 hover:text-white">
                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        `;

        document.body.appendChild(toast);

        // Show toast
        setTimeout(() => {
            toast.classList.remove('translate-x-full');
        }, 100);

        // Auto-remove toast
        setTimeout(() => {
            toast.classList.add('translate-x-full');
            setTimeout(() => {
                if (toast.parentElement) {
                    toast.remove();
                }
            }, 300);
        }, notification.priority === 'critical' ? 10000 : 5000);
    }

    updateDisplay() {
        this.updateNotificationCount();
        this.renderNotificationsList();
    }

    updateNotificationCount() {
        const unreadCount = this.notifications.filter(n => !n.read).length;
        const countElement = document.getElementById('notification-count');
        
        if (countElement) {
            countElement.textContent = unreadCount;
            countElement.style.display = unreadCount > 0 ? 'flex' : 'none';
            
            // Add pulsing animation for new notifications
            if (unreadCount > 0) {
                countElement.classList.add('animate-pulse');
                setTimeout(() => {
                    countElement.classList.remove('animate-pulse');
                }, 2000);
            }
        }
    }

    renderNotificationsList() {
        const container = document.getElementById('notifications-list');
        if (!container) return;

        if (this.notifications.length === 0) {
            container.innerHTML = `
                <div class="text-center py-8 text-gray-400">
                    <svg class="w-12 h-12 mx-auto mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 17h5l-5 5v-5zM9 7h6v2H9V7zm0 4h6v2H9v-2zm0 4h4v2H9v-2z"></path>
                    </svg>
                    <p>No notifications</p>
                </div>
            `;
            return;
        }

        const html = this.notifications.map(notification => `
            <div class="notification-item p-4 rounded-lg ${notification.read ? '' : 'unread'} border-l-4 border-neon-${this.getTypeColor(notification.type)} hover:bg-white/5 transition-colors"
                 data-notification-id="${notification.id}">
                <div class="flex items-start space-x-3">
                    <div class="flex-shrink-0 mt-1">
                        <div class="w-6 h-6 rounded-full bg-neon-${this.getTypeColor(notification.type)}/20 flex items-center justify-center text-neon-${this.getTypeColor(notification.type)}">
                            ${this.getTypeIcon(notification.type)}
                        </div>
                    </div>
                    <div class="flex-1 min-w-0">
                        <div class="flex items-start justify-between">
                            <div class="flex-1">
                                <div class="font-medium ${notification.read ? 'text-gray-300' : 'text-white'}">${notification.title}</div>
                                <div class="text-sm text-gray-400 mt-1">${notification.message}</div>
                                <div class="flex items-center space-x-4 mt-2">
                                    <span class="text-xs text-gray-500">${this.formatTimestamp(notification.timestamp)}</span>
                                    <span class="text-xs px-2 py-1 bg-glass rounded-full">${notification.category}</span>
                                    ${notification.priority === 'high' || notification.priority === 'critical' ? 
                                        `<span class="text-xs px-2 py-1 bg-red-500/20 text-red-400 rounded-full">${notification.priority}</span>` : ''}
                                </div>
                            </div>
                            <div class="flex items-center space-x-2 ml-4">
                                ${!notification.read ? `
                                    <button onclick="notificationSystem.markAsRead(${notification.id})" 
                                            class="p-1 text-gray-400 hover:text-neon-blue transition-colors" 
                                            title="Mark as read">
                                        <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path>
                                        </svg>
                                    </button>
                                ` : ''}
                                <button onclick="notificationSystem.removeNotification(${notification.id})" 
                                        class="p-1 text-gray-400 hover:text-red-400 transition-colors" 
                                        title="Remove">
                                    <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                                    </svg>
                                </button>
                            </div>
                        </div>
                        ${notification.actions.length > 0 ? `
                            <div class="flex flex-wrap gap-2 mt-3">
                                ${notification.actions.map(action => `
                                    <button onclick="notificationSystem.executeAction('${action.action}', ${notification.id})" 
                                            class="text-xs px-3 py-1 bg-neon-${this.getTypeColor(notification.type)}/20 text-neon-${this.getTypeColor(notification.type)} rounded-full hover:bg-neon-${this.getTypeColor(notification.type)}/30 transition-colors">
                                        ${action.label}
                                    </button>
                                `).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            </div>
        `).join('');

        container.innerHTML = html;
    }

    markAsRead(notificationId) {
        const notification = this.notifications.find(n => n.id === notificationId);
        if (notification) {
            notification.read = true;
            this.updateDisplay();
        }
    }

    markAllAsRead() {
        this.notifications.forEach(n => n.read = true);
        this.updateDisplay();
    }

    removeNotification(notificationId) {
        this.notifications = this.notifications.filter(n => n.id !== notificationId);
        this.updateDisplay();
    }

    clearAll() {
        this.notifications = [];
        this.updateDisplay();
    }

    executeAction(actionId, notificationId) {
        const notification = this.notifications.find(n => n.id === notificationId);
        if (!notification) return;

        switch (actionId) {
            case 'restart-lidarr':
                window.servicesManager?.restartService('lidarr');
                break;
            case 'view-logs-lidarr':
                window.dashboard?.showPage('logs');
                break;
            case 'open-plex':
                window.open('http://localhost:32400', '_blank');
                break;
            case 'view-backup-schedule':
                window.dashboard?.showPage('backup');
                break;
            case 'start-backup-now':
                // Implement backup start logic
                this.addNotification({
                    title: 'Backup Started',
                    message: 'Manual backup has been initiated',
                    type: 'info',
                    category: 'system'
                });
                break;
            default:
                console.log('Unknown action:', actionId);
        }

        // Mark notification as read after action
        this.markAsRead(notificationId);
    }

    getTypeColor(type) {
        const colors = {
            'success': 'green',
            'warning': 'yellow',
            'error': 'pink',
            'info': 'blue',
            'system': 'purple'
        };
        return colors[type] || 'blue';
    }

    getTypeIcon(type) {
        const icons = {
            'success': '<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clip-rule="evenodd"></path></svg>',
            'warning': '<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clip-rule="evenodd"></path></svg>',
            'error': '<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clip-rule="evenodd"></path></svg>',
            'info': '<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clip-rule="evenodd"></path></svg>',
            'system': '<svg class="w-4 h-4" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M11.49 3.17c-.38-1.56-2.6-1.56-2.98 0a1.532 1.532 0 01-2.286.948c-1.372-.836-2.942.734-2.106 2.106.54.886.061 2.042-.947 2.287-1.561.379-1.561 2.6 0 2.978a1.532 1.532 0 01.947 2.287c-.836 1.372.734 2.942 2.106 2.106a1.532 1.532 0 012.287.947c.379 1.561 2.6 1.561 2.978 0a1.533 1.533 0 012.287-.947c1.372.836 2.942-.734 2.106-2.106a1.533 1.533 0 01.947-2.287c1.561-.379 1.561-2.6 0-2.978a1.532 1.532 0 01-.947-2.287c.836-1.372-.734-2.942-2.106-2.106a1.532 1.532 0 01-2.287-.947zM10 13a3 3 0 100-6 3 3 0 000 6z" clip-rule="evenodd"></path></svg>'
        };
        return icons[type] || icons.info;
    }

    getNotificationIcon(type) {
        // Return icon URL for desktop notifications
        const baseUrl = window.location.origin;
        return `${baseUrl}/assets/icons/notification-${type}.png`;
    }

    formatTimestamp(timestamp) {
        const now = new Date();
        const diff = now - timestamp;
        const minutes = Math.floor(diff / (1000 * 60));
        const hours = Math.floor(minutes / 60);
        const days = Math.floor(hours / 24);

        if (minutes < 1) return 'Just now';
        if (minutes < 60) return `${minutes}m ago`;
        if (hours < 24) return `${hours}h ago`;
        if (days < 7) return `${days}d ago`;
        
        return timestamp.toLocaleDateString();
    }

    showSettingsModal() {
        const modal = document.createElement('div');
        modal.className = 'fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center';
        modal.innerHTML = `
            <div class="bg-dark-secondary rounded-xl border border-glass-border w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
                <div class="p-6 border-b border-glass-border">
                    <div class="flex items-center justify-between">
                        <h3 class="text-xl font-semibold">Notification Settings</h3>
                        <button onclick="this.closest('.fixed').remove()" class="p-2 hover:bg-glass rounded-lg">
                            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path>
                            </svg>
                        </button>
                    </div>
                </div>
                <div class="p-6 space-y-6">
                    <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                        <div>
                            <div class="font-medium">Desktop Notifications</div>
                            <div class="text-sm text-gray-400">Show notifications on your desktop</div>
                        </div>
                        <label class="switch">
                            <input type="checkbox" ${this.settings.desktopEnabled ? 'checked' : ''} 
                                   onchange="notificationSystem.settings.desktopEnabled = this.checked; notificationSystem.saveSettings();">
                            <span class="slider"></span>
                        </label>
                    </div>
                    
                    <div class="flex items-center justify-between p-4 bg-glass rounded-lg">
                        <div>
                            <div class="font-medium">Sound Notifications</div>
                            <div class="text-sm text-gray-400">Play sounds for notifications</div>
                        </div>
                        <label class="switch">
                            <input type="checkbox" ${this.settings.soundEnabled ? 'checked' : ''} 
                                   onchange="notificationSystem.settings.soundEnabled = this.checked; notificationSystem.saveSettings();">
                            <span class="slider"></span>
                        </label>
                    </div>

                    <div class="space-y-3">
                        <h4 class="font-medium">Category Settings</h4>
                        ${Object.entries(this.settings.categories).map(([category, settings]) => `
                            <div class="flex items-center justify-between p-3 bg-glass rounded-lg">
                                <div class="flex-1">
                                    <div class="font-medium capitalize">${category}</div>
                                </div>
                                <div class="flex items-center space-x-4">
                                    <label class="flex items-center space-x-2">
                                        <span class="text-sm">Enabled</span>
                                        <input type="checkbox" ${settings.enabled ? 'checked' : ''} 
                                               onchange="notificationSystem.settings.categories.${category}.enabled = this.checked; notificationSystem.saveSettings();"
                                               class="rounded">
                                    </label>
                                    <label class="flex items-center space-x-2">
                                        <span class="text-sm">Sound</span>
                                        <input type="checkbox" ${settings.sound ? 'checked' : ''} 
                                               onchange="notificationSystem.settings.categories.${category}.sound = this.checked; notificationSystem.saveSettings();"
                                               class="rounded">
                                    </label>
                                </div>
                            </div>
                        `).join('')}
                    </div>

                    <div class="flex space-x-3">
                        <button onclick="notificationSystem.markAllAsRead()" class="btn btn-secondary flex-1">Mark All Read</button>
                        <button onclick="notificationSystem.clearAll()" class="btn btn-danger flex-1">Clear All</button>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
    }
}

// Initialize notification system
const notificationSystem = new NotificationSystem();

// Export for global access
window.notificationSystem = notificationSystem;