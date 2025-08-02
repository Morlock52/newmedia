// Fixed Main Application Entry Point

class HolographicMediaDashboard {
    constructor() {
        this.scene = null;
        this.mediaCardsManager = null;
        this.audioVisualizer = null;
        this.uiController = null;
        this.wsClient = null;
        
        this.init();
    }

    async init() {
        try {
            console.log('Initializing Holographic Media Dashboard...');
            
            // Show loading progress
            this.updateLoadingProgress(10, 'Initializing 3D scene...');
            
            // Initialize 3D scene
            const container = document.getElementById('webgl-container');
            if (!container) {
                throw new Error('WebGL container not found');
            }
            
            console.log('Creating HolographicScene...');
            this.scene = new HolographicScene(container);
            
            this.updateLoadingProgress(30, 'Loading media system...');
            
            // Initialize media cards manager
            console.log('Creating MediaCardsManager...');
            this.mediaCardsManager = new MediaCardsManager(
                this.scene.scene,
                this.scene.camera
            );
            
            this.updateLoadingProgress(50, 'Setting up audio visualizer...');
            
            // Initialize audio visualizer
            console.log('Creating AudioVisualizer...');
            this.audioVisualizer = new AudioVisualizer(this.scene.scene);
            
            this.updateLoadingProgress(70, 'Initializing UI controls...');
            
            // Initialize UI controller
            console.log('Creating UIController...');
            this.uiController = new UIController(
                this.scene,
                this.mediaCardsManager,
                this.audioVisualizer
            );
            
            this.updateLoadingProgress(90, 'Connecting to server...');
            
            // Initialize WebSocket client
            console.log('Initializing WebSocket...');
            this.initializeWebSocket();
            
            // Initialize navigation
            this.initializeNavigation();
            
            // Setup performance monitoring
            this.setupPerformanceMonitoring();
            
            // Start render loop
            this.animate();
            
            this.updateLoadingProgress(100, 'Ready!');
            
            // Hide loading screen with animation
            setTimeout(() => {
                const loadingScreen = document.getElementById('loading-screen');
                if (loadingScreen) {
                    loadingScreen.style.opacity = '0';
                    setTimeout(() => {
                        loadingScreen.classList.add('hidden');
                    }, 500);
                }
            }, 500);
            
            console.log('Dashboard initialized successfully');
            
            // Auto-detect quality settings
            CONFIG.autoDetectQuality();
            
            // Apply saved preferences
            this.loadUserPreferences();
        } catch (error) {
            console.error('Error during initialization:', error);
            this.handleInitError(error);
        }
    }

    initializeNavigation() {
        // Enable all navigation buttons
        const navButtons = document.querySelectorAll('.nav-btn');
        navButtons.forEach(btn => {
            btn.disabled = false;
            console.log(`Enabled navigation: ${btn.dataset.section}`);
        });

        // Initialize page manager if available
        if (window.PageManager && !window.pageManager) {
            window.pageManager = new PageManager();
            console.log('PageManager initialized');
        }

        // Initialize navigation manager if available
        if (window.NavigationManager && !window.navigationManager) {
            window.navigationManager = new NavigationManager();
            console.log('NavigationManager initialized');
            
            // Refresh navigation after a short delay
            setTimeout(() => {
                window.navigationManager.refresh();
            }, 100);
        }
    }

    handleInitError(error) {
        console.error('Initialization error:', error);
        
        // Hide loading screen
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            loadingScreen.style.opacity = '0';
            setTimeout(() => {
                loadingScreen.classList.add('hidden');
            }, 500);
        }
        
        // Enable navigation anyway
        this.initializeNavigation();
        
        // Show error notification if UI controller exists
        if (this.uiController) {
            this.uiController.showNotification('3D initialization failed, running in reduced mode', 'warning');
        }
    }

    initializeWebSocket() {
        // Try to connect to WebSocket server
        this.wsClient = new WebSocketClient(CONFIG.websocket.url);
        
        // Setup event handlers
        this.wsClient.on('connected', () => {
            console.log('Connected to media server');
            if (this.uiController) {
                this.uiController.showNotification('Connected to media server', 'success');
            }
            
            // Request initial data
            this.wsClient.requestStats();
            this.wsClient.requestMediaList();
        });
        
        this.wsClient.on('disconnected', () => {
            console.log('Disconnected from media server');
            if (this.uiController) {
                this.uiController.showNotification('Connection lost', 'error');
            }
        });
        
        this.wsClient.on('stats-update', (stats) => {
            this.handleStatsUpdate(stats);
        });
        
        this.wsClient.on('media-update', (data) => {
            this.handleMediaUpdate(data);
        });
        
        this.wsClient.on('activity', (activity) => {
            if (this.uiController) {
                this.uiController.addActivity(activity.icon, activity.title);
            }
        });
        
        this.wsClient.on('system-alert', (alert) => {
            if (this.uiController) {
                this.uiController.showNotification(alert.message, alert.level);
            }
        });
        
        this.wsClient.on('performance-metrics', (metrics) => {
            this.updatePerformanceDisplay(metrics);
        });
        
        // If connection fails, start demo mode
        this.wsClient.on('reconnect-failed', () => {
            console.log('Starting demo mode');
            this.wsClient.startDemoMode();
            if (this.uiController) {
                this.uiController.showNotification('Running in demo mode', 'warning');
            }
        });
        
        // Start in demo mode immediately for this demo
        setTimeout(() => {
            if (!this.wsClient.isConnected) {
                this.wsClient.startDemoMode();
            }
        }, 2000);
    }

    setupPerformanceMonitoring() {
        if (!CONFIG.debug.showStats) return;
        
        // Create stats display
        const stats = new Stats();
        stats.showPanel(0); // FPS
        document.body.appendChild(stats.dom);
        stats.dom.style.position = 'absolute';
        stats.dom.style.left = '10px';
        stats.dom.style.top = '10px';
        
        this.stats = stats;
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        
        if (this.stats) this.stats.begin();
        
        const deltaTime = this.scene.clock.getDelta();
        
        // Update media cards
        if (this.mediaCardsManager) {
            this.mediaCardsManager.update(deltaTime);
        }
        
        // Update audio visualizer
        if (this.audioVisualizer) {
            this.audioVisualizer.update(deltaTime);
        }
        
        if (this.stats) this.stats.end();
    }

    handleStatsUpdate(stats) {
        // Update UI with new stats
        document.getElementById('total-media').textContent = Utils.formatNumber(stats.totalMedia);
        document.getElementById('storage-used').textContent = stats.storageUsed.toFixed(1) + 'TB';
        document.getElementById('active-users').textContent = stats.activeUsers;
        document.getElementById('bandwidth').textContent = stats.bandwidth + 'Mbps';
        document.getElementById('active-streams').textContent = stats.activeStreams;
        document.getElementById('gpu-usage').textContent = stats.gpuUsage + '%';
    }

    handleMediaUpdate(data) {
        if (!this.uiController) return;
        
        switch (data.action) {
            case 'added':
                this.uiController.addActivity('🎬', `New media added: ${data.media.title}`);
                break;
            case 'removed':
                this.uiController.addActivity('🗑️', `Media removed: ${data.media.title}`);
                break;
            case 'updated':
                this.uiController.addActivity('🔄', `Media updated: ${data.media.title}`);
                break;
        }
    }

    updatePerformanceDisplay(metrics) {
        // Update performance indicators
        if (metrics.fps < 30 && this.uiController) {
            this.uiController.showNotification('Performance degraded', 'warning');
        }
        
        // Adjust quality if needed
        if (CONFIG.performance.adaptiveQuality && this.scene) {
            if (metrics.fps < 30 && CONFIG.particles.count > 500) {
                CONFIG.setQuality('low');
                this.scene.particleSystem.dispose();
                this.scene.setupParticleSystems();
            } else if (metrics.fps > 50 && CONFIG.particles.count < 2000) {
                CONFIG.setQuality('high');
                this.scene.particleSystem.dispose();
                this.scene.setupParticleSystems();
            }
        }
    }

    updateLoadingProgress(percent, message) {
        const progressBar = document.getElementById('loading-progress');
        const loadingText = document.querySelector('.loading-text');
        
        if (progressBar) {
            progressBar.style.width = percent + '%';
        }
        
        if (loadingText) {
            loadingText.textContent = message;
        }
    }

    loadUserPreferences() {
        const preferences = Utils.storage.get('userPreferences', {
            effectsEnabled: true,
            particlesEnabled: true,
            audioVisualizerEnabled: false,
            quality: 'medium'
        });
        
        // Apply preferences
        if (this.scene) {
            this.scene.setEffectsEnabled(preferences.effectsEnabled);
            this.scene.setParticlesEnabled(preferences.particlesEnabled);
        }
        
        if (this.audioVisualizer) {
            this.audioVisualizer.setEnabled(preferences.audioVisualizerEnabled);
        }
        
        CONFIG.setQuality(preferences.quality);
        
        // Update UI
        if (preferences.effectsEnabled) {
            document.getElementById('toggle-effects').classList.add('active');
        }
        if (preferences.particlesEnabled) {
            document.getElementById('toggle-particles').classList.add('active');
        }
        if (preferences.audioVisualizerEnabled) {
            document.getElementById('toggle-audio').classList.add('active');
        }
    }

    saveUserPreferences() {
        const preferences = {
            effectsEnabled: document.getElementById('toggle-effects').classList.contains('active'),
            particlesEnabled: document.getElementById('toggle-particles').classList.contains('active'),
            audioVisualizerEnabled: document.getElementById('toggle-audio').classList.contains('active'),
            quality: 'medium'
        };
        
        Utils.storage.set('userPreferences', preferences);
    }
}

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    
    // Show user-friendly error message
    if (window.dashboard && window.dashboard.uiController) {
        window.dashboard.uiController.showNotification(
            'An error occurred. Please refresh the page.',
            'error'
        );
    }
});

// Handle visibility change
document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
        // Reduce render quality when tab is hidden
        if (window.dashboard && window.dashboard.scene) {
            window.dashboard.scene.clock.stop();
        }
    } else {
        // Resume normal rendering
        if (window.dashboard && window.dashboard.scene) {
            window.dashboard.scene.clock.start();
        }
    }
});

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Force hide loading screen after 10 seconds as ultimate fallback
    setTimeout(() => {
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen && !loadingScreen.classList.contains('hidden')) {
            console.warn('Force hiding loading screen due to timeout');
            loadingScreen.style.opacity = '0';
            setTimeout(() => {
                loadingScreen.classList.add('hidden');
            }, 500);
        }
    }, 10000);
    
    // Check WebGL support
    if (!window.WebGLRenderingContext) {
        document.body.innerHTML = `
            <div style="display: flex; align-items: center; justify-content: center; height: 100vh; text-align: center;">
                <div>
                    <h1 style="color: #00FFFF;">WebGL Not Supported</h1>
                    <p>This dashboard requires WebGL support. Please use a modern browser.</p>
                </div>
            </div>
        `;
        return;
    }
    
    // Wait a bit for all scripts to load properly
    setTimeout(() => {
        try {
            // Create dashboard instance
            window.dashboard = new HolographicMediaDashboard();
        } catch (error) {
            console.error('Failed to initialize dashboard:', error);
            // Hide loading screen on error
            const loadingScreen = document.getElementById('loading-screen');
            if (loadingScreen) {
                loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    loadingScreen.classList.add('hidden');
                }, 500);
            }
        }
    }, 1000);
    
    // Save preferences on page unload
    window.addEventListener('beforeunload', () => {
        if (window.dashboard) {
            window.dashboard.saveUserPreferences();
        }
    });
});

// Performance optimization for mobile
if ('ontouchstart' in window) {
    CONFIG.setQuality('low');
    CONFIG.particles.count = 500;
    CONFIG.mediaCards.rows = 2;
    CONFIG.mediaCards.columns = 2;
}

// Stats.js helper (minimal implementation if not loaded)
if (typeof Stats === 'undefined') {
    window.Stats = function() {
        return {
            showPanel: () => {},
            begin: () => {},
            end: () => {},
            dom: document.createElement('div')
        };
    };
}