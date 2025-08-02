// Enhanced Main Application with Comprehensive Error Handling and Logging

class HolographicMediaDashboard {
    constructor() {
        this.debugLogger = window.debugLogger;
        this.scene = null;
        this.mediaCardsManager = null;
        this.audioVisualizer = null;
        this.uiController = null;
        this.wsClient = null;
        this.initComplete = false;
        
        // Start initialization with error recovery
        this.safeInit();
    }

    async safeInit() {
        try {
            this.debugLogger.info('Starting HolographicMediaDashboard initialization');
            
            // Check dependencies first
            await this.waitForDependencies();
            
            // Initialize with comprehensive error handling
            await this.init();
            
        } catch (error) {
            this.debugLogger.error('Critical initialization error', {
                error: error.message,
                stack: error.stack
            });
            
            // Attempt recovery
            this.handleCriticalError(error);
        }
    }

    async waitForDependencies() {
        this.debugLogger.startStep('dependency-check', 10000);
        
        const maxAttempts = 100;
        let attempts = 0;
        
        while (attempts < maxAttempts) {
            const deps = this.debugLogger.checkDependencies();
            
            if (deps.allLoaded) {
                this.debugLogger.completeStep('dependency-check', deps);
                return true;
            }
            
            if (attempts % 10 === 0) {
                this.debugLogger.warn(`Waiting for dependencies (attempt ${attempts})`, deps.missing);
            }
            
            await new Promise(resolve => setTimeout(resolve, 100));
            attempts++;
        }
        
        throw new Error(`Dependencies not loaded after ${maxAttempts} attempts`);
    }

    async init() {
        this.debugLogger.info('Initializing Holographic Media Dashboard...');
        
        try {
            // Step 1: WebGL Check
            this.debugLogger.startStep('webgl-check');
            const webglInfo = this.debugLogger.checkWebGL();
            if (!webglInfo) {
                throw new Error('WebGL not supported');
            }
            this.debugLogger.completeStep('webgl-check', webglInfo);
            
            // Step 2: DOM Check
            this.debugLogger.startStep('dom-check');
            const container = document.getElementById('webgl-container');
            if (!container) {
                throw new Error('WebGL container not found');
            }
            this.debugLogger.completeStep('dom-check', { containerId: 'webgl-container' });
            
            // Show loading progress
            this.updateLoadingProgress(10, 'Initializing 3D scene...');
            
            // Step 3: Initialize 3D Scene
            this.debugLogger.startStep('3d-scene-init', 10000);
            try {
                this.scene = new HolographicScene(container);
                this.debugLogger.completeStep('3d-scene-init', { 
                    renderer: this.scene.renderer ? 'initialized' : 'failed',
                    camera: this.scene.camera ? 'initialized' : 'failed'
                });
            } catch (sceneError) {
                this.debugLogger.failStep('3d-scene-init', sceneError);
                throw new Error(`3D Scene initialization failed: ${sceneError.message}`);
            }
            
            this.updateLoadingProgress(30, 'Loading media system...');
            
            // Step 4: Initialize Media Cards Manager
            this.debugLogger.startStep('media-cards-init');
            try {
                this.mediaCardsManager = new MediaCardsManager(
                    this.scene.scene,
                    this.scene.camera
                );
                this.debugLogger.completeStep('media-cards-init');
            } catch (mediaError) {
                this.debugLogger.failStep('media-cards-init', mediaError);
                this.debugLogger.warn('Media cards disabled', mediaError);
                // Continue without media cards
            }
            
            this.updateLoadingProgress(50, 'Setting up audio visualizer...');
            
            // Step 5: Initialize Audio Visualizer
            this.debugLogger.startStep('audio-visualizer-init');
            try {
                this.audioVisualizer = new AudioVisualizer(this.scene.scene);
                this.debugLogger.completeStep('audio-visualizer-init');
            } catch (audioError) {
                this.debugLogger.failStep('audio-visualizer-init', audioError);
                this.debugLogger.warn('Audio visualizer disabled', audioError);
                // Continue without audio visualizer
            }
            
            this.updateLoadingProgress(70, 'Initializing UI controls...');
            
            // Step 6: Initialize UI Controller
            this.debugLogger.startStep('ui-controller-init');
            try {
                this.uiController = new UIController(
                    this.scene,
                    this.mediaCardsManager,
                    this.audioVisualizer
                );
                this.debugLogger.completeStep('ui-controller-init');
            } catch (uiError) {
                this.debugLogger.failStep('ui-controller-init', uiError);
                this.debugLogger.error('UI Controller failed', uiError);
                // This is critical, but try to continue
            }
            
            this.updateLoadingProgress(85, 'Setting up navigation...');
            
            // Step 7: Initialize Navigation
            this.debugLogger.startStep('navigation-init');
            this.initializeNavigation();
            this.debugLogger.completeStep('navigation-init');
            
            this.updateLoadingProgress(90, 'Connecting to server...');
            
            // Step 8: Initialize WebSocket
            this.debugLogger.startStep('websocket-init');
            try {
                this.initializeWebSocket();
                this.debugLogger.completeStep('websocket-init');
            } catch (wsError) {
                this.debugLogger.failStep('websocket-init', wsError);
                this.debugLogger.warn('WebSocket disabled, using demo mode');
            }
            
            // Step 9: Setup Performance Monitoring
            this.debugLogger.startStep('performance-init');
            this.setupPerformanceMonitoring();
            this.debugLogger.completeStep('performance-init');
            
            // Step 10: Start Render Loop
            this.debugLogger.startStep('render-loop-init');
            this.animate();
            this.debugLogger.completeStep('render-loop-init');
            
            this.updateLoadingProgress(100, 'Ready!');
            
            // Hide loading screen
            this.hideLoadingScreen();
            
            // Load preferences
            this.loadUserPreferences();
            
            // Auto-detect quality
            if (CONFIG && CONFIG.autoDetectQuality) {
                CONFIG.autoDetectQuality();
            }
            
            this.initComplete = true;
            this.debugLogger.success('Dashboard initialization complete!');
            
            // Generate initialization report
            const report = this.debugLogger.generateReport();
            
            // Show success notification
            if (this.uiController && this.uiController.showNotification) {
                this.uiController.showNotification('Dashboard ready!', 'success');
            }
            
        } catch (error) {
            this.debugLogger.error('Initialization failed', {
                error: error.message,
                stack: error.stack,
                phase: 'init'
            });
            throw error;
        }
    }

    initializeNavigation() {
        try {
            // Enable all navigation buttons
            const navButtons = document.querySelectorAll('.nav-btn');
            navButtons.forEach(btn => {
                btn.disabled = false;
                this.debugLogger.debug(`Enabled navigation: ${btn.dataset.section}`);
            });

            // Initialize page manager
            if (window.PageManager && !window.pageManager) {
                window.pageManager = new PageManager();
                this.debugLogger.info('PageManager initialized');
            }

            // Initialize navigation manager
            if (window.NavigationManager && !window.navigationManager) {
                window.navigationManager = new NavigationManager();
                this.debugLogger.info('NavigationManager initialized');
                
                // Refresh navigation
                setTimeout(() => {
                    if (window.navigationManager && window.navigationManager.refresh) {
                        window.navigationManager.refresh();
                        this.debugLogger.debug('Navigation refreshed');
                    }
                }, 100);
            }
        } catch (navError) {
            this.debugLogger.error('Navigation initialization error', navError);
            // Enable buttons anyway
            document.querySelectorAll('.nav-btn').forEach(btn => btn.disabled = false);
        }
    }

    initializeWebSocket() {
        try {
            this.wsClient = new WebSocketClient(CONFIG.websocket.url);
            
            // Setup event handlers
            this.wsClient.on('connected', () => {
                this.debugLogger.info('Connected to media server');
                if (this.uiController) {
                    this.uiController.showNotification('Connected to media server', 'success');
                }
                this.wsClient.requestStats();
                this.wsClient.requestMediaList();
            });
            
            this.wsClient.on('disconnected', () => {
                this.debugLogger.warn('Disconnected from media server');
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
            
            this.wsClient.on('reconnect-failed', () => {
                this.debugLogger.info('Starting demo mode');
                this.wsClient.startDemoMode();
                if (this.uiController) {
                    this.uiController.showNotification('Running in demo mode', 'warning');
                }
            });
            
            // Start in demo mode after timeout
            setTimeout(() => {
                if (!this.wsClient.isConnected) {
                    this.wsClient.startDemoMode();
                }
            }, 2000);
            
        } catch (wsError) {
            this.debugLogger.error('WebSocket initialization failed', wsError);
            // Continue without websocket
        }
    }

    setupPerformanceMonitoring() {
        try {
            // Show performance monitor
            const perfMonitor = document.querySelector('.performance-monitor');
            if (perfMonitor) {
                perfMonitor.style.display = 'block';
            }
            
            // Update performance stats
            setInterval(() => {
                if (this.scene && this.scene.renderer) {
                    const info = this.scene.renderer.info;
                    document.getElementById('fps-counter').textContent = `FPS: ${Math.round(1000 / this.scene.clock.getDelta())}`;
                    
                    if (performance.memory) {
                        const memUsed = (performance.memory.usedJSHeapSize / 1048576).toFixed(1);
                        const memTotal = (performance.memory.totalJSHeapSize / 1048576).toFixed(1);
                        document.getElementById('memory-usage').textContent = `Memory: ${memUsed}/${memTotal}MB`;
                    }
                    
                    document.getElementById('webgl-status').textContent = `WebGL: Active`;
                }
            }, 1000);
            
        } catch (perfError) {
            this.debugLogger.warn('Performance monitoring setup failed', perfError);
        }
    }

    animate() {
        try {
            requestAnimationFrame(this.animate.bind(this));
            
            if (!this.scene || !this.scene.clock) return;
            
            const deltaTime = this.scene.clock.getDelta();
            
            // Update components with error handling
            if (this.mediaCardsManager) {
                try {
                    this.mediaCardsManager.update(deltaTime);
                } catch (e) {
                    this.debugLogger.debug('Media cards update error', e);
                }
            }
            
            if (this.audioVisualizer) {
                try {
                    this.audioVisualizer.update(deltaTime);
                } catch (e) {
                    this.debugLogger.debug('Audio visualizer update error', e);
                }
            }
            
        } catch (animError) {
            this.debugLogger.error('Animation loop error', animError);
        }
    }

    handleStatsUpdate(stats) {
        try {
            document.getElementById('total-media').textContent = Utils.formatNumber(stats.totalMedia);
            document.getElementById('storage-used').textContent = stats.storageUsed.toFixed(1) + 'TB';
            document.getElementById('active-users').textContent = stats.activeUsers;
            document.getElementById('bandwidth').textContent = stats.bandwidth + 'Mbps';
            document.getElementById('active-streams').textContent = stats.activeStreams;
            document.getElementById('gpu-usage').textContent = stats.gpuUsage + '%';
        } catch (error) {
            this.debugLogger.debug('Stats update error', error);
        }
    }

    handleMediaUpdate(data) {
        try {
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
        } catch (error) {
            this.debugLogger.debug('Media update error', error);
        }
    }

    updateLoadingProgress(percent, message) {
        try {
            const progressBar = document.getElementById('loading-progress');
            const loadingText = document.querySelector('.loading-text');
            
            if (progressBar) {
                progressBar.style.width = percent + '%';
            }
            
            if (loadingText) {
                loadingText.textContent = message;
            }
            
            this.debugLogger.info(`Loading: ${percent}% - ${message}`);
        } catch (error) {
            this.debugLogger.debug('Progress update error', error);
        }
    }

    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            setTimeout(() => {
                loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    loadingScreen.classList.add('hidden');
                    this.debugLogger.info('Loading screen hidden');
                }, 500);
            }, 500);
        }
    }

    loadUserPreferences() {
        try {
            const preferences = Utils.storage.get('userPreferences', {
                effectsEnabled: true,
                particlesEnabled: true,
                audioVisualizerEnabled: false,
                quality: 'medium'
            });
            
            if (this.scene) {
                this.scene.setEffectsEnabled(preferences.effectsEnabled);
                this.scene.setParticlesEnabled(preferences.particlesEnabled);
            }
            
            if (this.audioVisualizer) {
                this.audioVisualizer.setEnabled(preferences.audioVisualizerEnabled);
            }
            
            CONFIG.setQuality(preferences.quality);
            
            this.debugLogger.info('User preferences loaded', preferences);
        } catch (error) {
            this.debugLogger.warn('Failed to load preferences', error);
        }
    }

    handleCriticalError(error) {
        this.debugLogger.error('Attempting recovery from critical error', error);
        
        // Hide loading screen
        this.hideLoadingScreen();
        
        // Enable navigation anyway
        this.initializeNavigation();
        
        // Show error message
        const container = document.getElementById('webgl-container');
        if (container) {
            container.innerHTML = `
                <div style="
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100vh;
                    text-align: center;
                    background: linear-gradient(135deg, rgba(0,255,255,0.1), rgba(255,0,255,0.1));
                ">
                    <div>
                        <h1 style="color: #00FFFF; text-shadow: 0 0 20px rgba(0,255,255,0.8);">
                            Holographic System Error
                        </h1>
                        <p style="color: #FFF; margin: 20px 0;">
                            The 3D interface encountered an error. Navigation is still available.
                        </p>
                        <button onclick="location.reload()" style="
                            background: #00FFFF;
                            color: #000;
                            border: none;
                            padding: 10px 20px;
                            font-size: 16px;
                            cursor: pointer;
                            margin: 10px;
                        ">Reload Dashboard</button>
                        <button onclick="window.debugLogger.showDebugPanel()" style="
                            background: #FF00FF;
                            color: #FFF;
                            border: none;
                            padding: 10px 20px;
                            font-size: 16px;
                            cursor: pointer;
                            margin: 10px;
                        ">Show Debug Info</button>
                    </div>
                </div>
            `;
        }
        
        // Show debug panel
        this.debugLogger.showDebugPanel();
    }
}

// Initialize when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.debugLogger.info('DOM Content Loaded');
    
    // Set a maximum timeout for initialization
    const initTimeout = setTimeout(() => {
        window.debugLogger.error('Initialization timeout - forcing loading screen hide');
        const loadingScreen = document.getElementById('loading-screen');
        if (loadingScreen) {
            loadingScreen.style.display = 'none';
        }
        
        // Enable navigation
        document.querySelectorAll('.nav-btn').forEach(btn => btn.disabled = false);
    }, 15000);
    
    // Wait a moment for scripts to fully load
    setTimeout(() => {
        try {
            window.debugLogger.info('Creating dashboard instance');
            window.dashboard = new HolographicMediaDashboard();
            clearTimeout(initTimeout);
        } catch (error) {
            window.debugLogger.error('Failed to create dashboard', error);
            clearTimeout(initTimeout);
            
            // Show error state
            const loadingScreen = document.getElementById('loading-screen');
            if (loadingScreen) {
                loadingScreen.innerHTML = `
                    <div style="text-align: center;">
                        <h2 style="color: #FF0066;">Initialization Failed</h2>
                        <p>${error.message}</p>
                        <button onclick="location.reload()" style="
                            background: #00FFFF;
                            color: #000;
                            border: none;
                            padding: 10px 20px;
                            cursor: pointer;
                        ">Retry</button>
                    </div>
                `;
            }
        }
    }, 1000);
});

// Handle page unload
window.addEventListener('beforeunload', () => {
    if (window.dashboard && window.dashboard.saveUserPreferences) {
        window.dashboard.saveUserPreferences();
    }
});

// Performance optimization for mobile
if ('ontouchstart' in window) {
    CONFIG.setQuality('low');
    CONFIG.particles.count = 500;
    CONFIG.mediaCards.rows = 2;
    CONFIG.mediaCards.columns = 2;
    window.debugLogger.info('Mobile mode activated');
}

// Minimal Stats implementation if not loaded
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