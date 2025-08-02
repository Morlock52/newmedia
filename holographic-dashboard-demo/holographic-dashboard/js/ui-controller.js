// UI Controller for Holographic Dashboard

class UIController {
    constructor(scene, mediaCardsManager, audioVisualizer) {
        this.scene = scene;
        this.mediaCardsManager = mediaCardsManager;
        this.audioVisualizer = audioVisualizer;
        
        this.stats = {
            totalMedia: 0,
            storageUsed: 0,
            activeUsers: 0,
            bandwidth: 0,
            activeStreams: 0,
            gpuUsage: 0
        };
        
        this.activityFeed = [];
        this.currentSection = 'dashboard';
        
        this.init();
    }

    init() {
        // Initialize UI elements
        this.initializeNavigation();
        this.initializeControlPanel();
        this.initializeMediaPreview();
        this.initializeStats();
        
        // Start update loops
        this.startUpdateLoops();
        
        // Listen for media card clicks
        window.addEventListener('mediaCardClick', this.onMediaCardClick.bind(this));
        
        // Initialize with demo data
        this.loadDemoData();
    }

    initializeNavigation() {
        const navButtons = document.querySelectorAll('.nav-btn');
        
        navButtons.forEach(button => {
            button.addEventListener('click', (e) => {
                e.preventDefault();
                
                // Prevent navigation if currently navigating
                if (window.navigationManager && window.navigationManager.isNavigating) {
                    return;
                }
                
                // Update active state
                navButtons.forEach(btn => btn.classList.remove('active'));
                button.classList.add('active');
                
                // Get section
                const section = button.dataset.section;
                const route = this.getSectionRoute(section);
                
                // Navigate using navigation manager if available
                if (window.navigationManager) {
                    window.navigationManager.navigate(route);
                } else {
                    // Fallback to direct section switching
                    this.switchSection(section);
                }
                
                // Add ripple effect
                this.createRippleEffect(e, button);
            });
        });
        
        // Initialize keyboard navigation
        this.initializeKeyboardNavigation();
    }

    initializeControlPanel() {
        // Toggle effects
        const toggleEffects = document.getElementById('toggle-effects');
        toggleEffects.addEventListener('click', () => {
            toggleEffects.classList.toggle('active');
            const enabled = toggleEffects.classList.contains('active');
            this.scene.setEffectsEnabled(enabled);
            this.showNotification(enabled ? 'Effects enabled' : 'Effects disabled', 'info');
        });
        
        // Toggle particles
        const toggleParticles = document.getElementById('toggle-particles');
        toggleParticles.addEventListener('click', () => {
            toggleParticles.classList.toggle('active');
            const enabled = toggleParticles.classList.contains('active');
            this.scene.setParticlesEnabled(enabled);
            this.showNotification(enabled ? 'Particles enabled' : 'Particles disabled', 'info');
        });
        
        // Toggle audio visualizer
        const toggleAudio = document.getElementById('toggle-audio');
        toggleAudio.addEventListener('click', () => {
            toggleAudio.classList.toggle('active');
            const enabled = toggleAudio.classList.contains('active');
            this.audioVisualizer.setEnabled(enabled);
            this.showNotification(enabled ? 'Audio visualizer enabled' : 'Audio visualizer disabled', 'info');
        });
        
        // Fullscreen
        const fullscreenBtn = document.getElementById('fullscreen');
        fullscreenBtn.addEventListener('click', () => {
            if (!document.fullscreenElement) {
                document.documentElement.requestFullscreen();
                fullscreenBtn.innerHTML = '<span>⛶</span>';
            } else {
                document.exitFullscreen();
                fullscreenBtn.innerHTML = '<span>⛶</span>';
            }
        });
        
        // Set initial states
        toggleEffects.classList.add('active');
        toggleParticles.classList.add('active');
    }

    initializeMediaPreview() {
        const previewPanel = document.getElementById('media-preview');
        const closeBtn = previewPanel.querySelector('.preview-close');
        
        closeBtn.addEventListener('click', () => {
            this.hideMediaPreview();
        });
        
        // Action buttons
        const playBtn = previewPanel.querySelector('.play-btn');
        playBtn.addEventListener('click', () => {
            this.showNotification('Playback started', 'success');
            this.hideMediaPreview();
        });
        
        const infoBtn = previewPanel.querySelector('.info-btn');
        infoBtn.addEventListener('click', () => {
            this.showNotification('Loading media information...', 'info');
        });
        
        const downloadBtn = previewPanel.querySelector('.download-btn');
        downloadBtn.addEventListener('click', () => {
            this.showNotification('Download started', 'success');
        });
    }

    initializeStats() {
        // Animate stat values on load
        setTimeout(() => {
            this.animateStatTo('total-media', 2847);
            this.animateStatTo('storage-used', 47.3, 'TB');
            this.animateStatTo('active-users', 12);
            this.animateStatTo('bandwidth', 450, 'Mbps');
            this.animateStatTo('active-streams', 12);
            this.animateStatTo('gpu-usage', 65, '%');
        }, 1000);
    }

    startUpdateLoops() {
        // Update stats
        setInterval(() => {
            this.updateStats();
        }, CONFIG.ui.statsUpdateInterval);
        
        // Update activity feed
        setInterval(() => {
            this.addRandomActivity();
        }, 5000);
        
        // Hide loading screen
        setTimeout(() => {
            this.hideLoadingScreen();
        }, 2000);
    }

    updateStats() {
        // Simulate stat changes
        const changes = {
            activeStreams: Math.random() > 0.7 ? Utils.randomInt(-2, 3) : 0,
            bandwidth: Utils.randomInt(-50, 50),
            gpuUsage: Utils.randomInt(-5, 5)
        };
        
        // Update active streams
        const streamsElement = document.getElementById('active-streams');
        const currentStreams = parseInt(streamsElement.textContent);
        const newStreams = Math.max(0, currentStreams + changes.activeStreams);
        if (newStreams !== currentStreams) {
            this.animateStatTo('active-streams', newStreams);
        }
        
        // Update bandwidth
        const bandwidthElement = document.getElementById('bandwidth');
        const currentBandwidth = parseInt(bandwidthElement.textContent);
        const newBandwidth = Utils.clamp(currentBandwidth + changes.bandwidth, 100, 1000);
        this.animateStatTo('bandwidth', newBandwidth, 'Mbps');
        
        // Update GPU usage
        const gpuElement = document.getElementById('gpu-usage');
        const currentGPU = parseInt(gpuElement.textContent);
        const newGPU = Utils.clamp(currentGPU + changes.gpuUsage, 0, 100);
        this.animateStatTo('gpu-usage', newGPU, '%');
    }

    animateStatTo(elementId, value, suffix = '') {
        const element = document.getElementById(elementId);
        const startValue = parseFloat(element.textContent) || 0;
        const isFloat = value % 1 !== 0;
        
        gsap.to({ val: startValue }, {
            val: value,
            duration: 1,
            ease: "power2.out",
            onUpdate: function() {
                const currentVal = isFloat ? this.targets()[0].val.toFixed(1) : Math.floor(this.targets()[0].val);
                element.textContent = Utils.formatNumber(currentVal) + suffix;
            }
        });
    }

    addRandomActivity() {
        const activities = [
            { icon: '🎬', title: 'New movie added', details: ['Dune Part Two', 'Avatar 3', 'Blade Runner 2099'] },
            { icon: '👤', title: 'User started watching', details: ['The Mandalorian', 'Westworld', 'Foundation'] },
            { icon: '⬇️', title: 'Download completed', details: ['The Matrix Resurrections', 'Interstellar', 'Inception'] },
            { icon: '🔄', title: 'Transcoding finished', details: ['12 episodes', '5 movies', '8 documentaries'] },
            { icon: '📡', title: 'Live stream started', details: ['Channel 1', 'Sports HD', 'News 24'] },
            { icon: '✅', title: 'System scan completed', details: ['All systems operational', 'Database optimized', 'Cache cleared'] }
        ];
        
        const activity = activities[Math.floor(Math.random() * activities.length)];
        const detail = activity.details[Math.floor(Math.random() * activity.details.length)];
        
        this.addActivity(activity.icon, `${activity.title}: ${detail}`);
    }

    addActivity(icon, text) {
        const feedElement = document.getElementById('activity-feed');
        
        // Create activity item
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        activityItem.innerHTML = `
            <div class="activity-icon">${icon}</div>
            <div class="activity-content">
                <div class="activity-title">${text}</div>
                <div class="activity-time">Just now</div>
            </div>
        `;
        
        // Add to feed
        feedElement.insertBefore(activityItem, feedElement.firstChild);
        
        // Limit feed size
        while (feedElement.children.length > CONFIG.ui.activityFeedLimit) {
            feedElement.removeChild(feedElement.lastChild);
        }
        
        // Update timestamps
        this.updateActivityTimestamps();
    }

    updateActivityTimestamps() {
        const activities = document.querySelectorAll('.activity-item');
        activities.forEach((item, index) => {
            const timeElement = item.querySelector('.activity-time');
            if (index === 0) {
                timeElement.textContent = 'Just now';
            } else {
                timeElement.textContent = Utils.getRelativeTime(Date.now() - index * 60000);
            }
        });
    }

    getSectionRoute(section) {
        const routes = {
            'dashboard': '/dashboard',
            'movies': '/movies',
            'series': '/series',
            'music': '/music',
            'live': '/live',
            'analytics': '/analytics'
        };
        return routes[section] || '/dashboard';
    }
    
    switchSection(section) {
        this.currentSection = section;
        
        // Show loading state
        this.showSectionLoading(section);
        
        // Simulate async loading with proper error handling
        setTimeout(() => {
            try {
                // Load appropriate content
                switch (section) {
                    case 'dashboard':
                        this.loadDashboard();
                        break;
                    case 'movies':
                        this.loadMovies();
                        break;
                    case 'series':
                        this.loadSeries();
                        break;
                    case 'music':
                        this.loadMusic();
                        break;
                    case 'live':
                        this.loadLiveTV();
                        break;
                    case 'analytics':
                        this.loadAnalytics();
                        break;
                    default:
                        this.loadDashboard();
                }
                
                this.hideSectionLoading(section);
                this.showNotification(`Loaded ${section}`, 'success');
                
                // Track section change
                this.trackSectionChange(section);
                
            } catch (error) {
                console.error(`Error loading ${section}:`, error);
                this.hideSectionLoading(section);
                this.showNotification(`Failed to load ${section}`, 'error');
                // Fallback to dashboard
                this.loadDashboard();
            }
        }, 300); // Realistic loading delay
    }

    loadDashboard() {
        const demoData = this.mediaCardsManager.generateDemoData();
        this.mediaCardsManager.loadMediaData(demoData);
    }

    loadMovies() {
        const movieData = this.mediaCardsManager.generateDemoData()
            .map(item => ({ ...item, type: 'movie' }));
        this.mediaCardsManager.loadMediaData(movieData);
    }

    loadSeries() {
        const seriesData = this.mediaCardsManager.generateDemoData()
            .map(item => ({ ...item, type: 'series' }));
        this.mediaCardsManager.loadMediaData(seriesData);
    }

    loadMusic() {
        const musicData = this.mediaCardsManager.generateDemoData()
            .map(item => ({ ...item, type: 'music' }));
        this.mediaCardsManager.loadMediaData(musicData);
        
        // Enable audio visualizer for music section
        this.audioVisualizer.setEnabled(true);
        document.getElementById('toggle-audio').classList.add('active');
    }

    loadLiveTV() {
        const liveData = this.mediaCardsManager.generateDemoData()
            .map(item => ({ ...item, type: 'live' }));
        this.mediaCardsManager.loadMediaData(liveData);
    }

    loadAnalytics() {
        // Clear media cards and show analytics visualization
        this.mediaCardsManager.clearCards();
        
        // Create analytics visualization
        this.createAnalyticsVisualization();
        
        this.showNotification('Analytics loaded', 'success');
    }
    
    loadLiveTV() {
        const liveData = this.generateLiveData();
        this.mediaCardsManager.loadMediaData(liveData);
        
        // Enable real-time updates for live content
        this.startLiveUpdates();
    }
    
    generateLiveData() {
        const channels = [
            { title: 'Sports HD', type: 'live', status: 'live', viewers: 2847 },
            { title: 'News 24/7', type: 'live', status: 'live', viewers: 1923 },
            { title: 'Movie Channel', type: 'live', status: 'live', viewers: 5674 },
            { title: 'Documentary', type: 'live', status: 'live', viewers: 892 },
            { title: 'Music TV', type: 'live', status: 'live', viewers: 3456 },
            { title: 'Tech Talk', type: 'live', status: 'offline', viewers: 0 }
        ];
        
        return channels.map((channel, index) => ({
            id: `live_${index}`,
            title: channel.title,
            type: channel.type,
            status: channel.status,
            metadata: {
                viewers: channel.viewers,
                quality: channel.status === 'live' ? '1080p' : 'Offline',
                duration: channel.status === 'live' ? 'Live' : 'Offline',
                year: new Date().getFullYear()
            }
        }));
    }
    
    createAnalyticsVisualization() {
        // Create analytics data cards in 3D space
        const analyticsData = [
            { title: 'Total Views', value: '2.4M', trend: '+12%' },
            { title: 'Active Users', value: '14.2K', trend: '+8%' },
            { title: 'Storage Used', value: '47.3TB', trend: '+3%' },
            { title: 'Bandwidth', value: '450Mbps', trend: '+15%' },
            { title: 'Transcoding', value: '12 jobs', trend: '-2%' },
            { title: 'Error Rate', value: '0.03%', trend: '-45%' }
        ];
        
        const cardData = analyticsData.map((data, index) => ({
            id: `analytics_${index}`,
            title: data.title,
            type: 'analytics',
            metadata: {
                value: data.value,
                trend: data.trend,
                quality: 'Real-time',
                duration: 'Live Data',
                year: new Date().getFullYear()
            }
        }));
        
        this.mediaCardsManager.loadMediaData(cardData);
    }

    loadDemoData() {
        this.loadDashboard();
        
        // Add initial activities
        this.addActivity('🚀', 'System initialized');
        this.addActivity('✅', 'All services online');
        this.addActivity('📊', 'Dashboard loaded');
    }

    onMediaCardClick(event) {
        const { data } = event.detail;
        this.showMediaPreview(data);
    }

    showMediaPreview(mediaData) {
        const previewPanel = document.getElementById('media-preview');
        const titleElement = previewPanel.querySelector('.preview-title');
        const metaElement = previewPanel.querySelector('.preview-meta');
        
        titleElement.textContent = mediaData.title;
        metaElement.innerHTML = `
            <div>Type: ${mediaData.type}</div>
            <div>Year: ${mediaData.metadata.year}</div>
            <div>Quality: ${mediaData.metadata.quality}</div>
            <div>Duration: ${mediaData.metadata.duration}</div>
        `;
        
        previewPanel.style.display = 'block';
        
        // Animate in
        gsap.from(previewPanel, {
            scale: 0.8,
            opacity: 0,
            duration: 0.3,
            ease: "back.out(1.7)"
        });
    }

    hideMediaPreview() {
        const previewPanel = document.getElementById('media-preview');
        
        gsap.to(previewPanel, {
            scale: 0.8,
            opacity: 0,
            duration: 0.2,
            ease: "power2.in",
            onComplete: () => {
                previewPanel.style.display = 'none';
            }
        });
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${this.getNotificationIcon(type)}</span>
                <span class="notification-message">${message}</span>
            </div>
        `;
        
        // Add to body
        document.body.appendChild(notification);
        
        // Position
        notification.style.cssText = `
            position: fixed;
            top: 2rem;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            border: 1px solid ${type === 'success' ? '#0FF1CE' : type === 'error' ? '#FF10F0' : '#00FFFF'};
            border-radius: 8px;
            padding: 1rem 2rem;
            color: white;
            font-weight: 500;
            z-index: 1000;
            box-shadow: 0 4px 20px rgba(0, 255, 255, 0.3);
        `;
        
        // Animate in
        gsap.from(notification, {
            y: -50,
            opacity: 0,
            duration: 0.3,
            ease: "power2.out"
        });
        
        // Remove after duration
        setTimeout(() => {
            gsap.to(notification, {
                y: -50,
                opacity: 0,
                duration: 0.3,
                ease: "power2.in",
                onComplete: () => {
                    notification.remove();
                }
            });
        }, CONFIG.ui.notificationDuration);
    }

    getNotificationIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }

    createRippleEffect(event, element) {
        const ripple = document.createElement('div');
        ripple.className = 'ripple';
        
        const rect = element.getBoundingClientRect();
        const size = Math.max(rect.width, rect.height);
        const x = event.clientX - rect.left - size / 2;
        const y = event.clientY - rect.top - size / 2;
        
        ripple.style.cssText = `
            position: absolute;
            width: ${size}px;
            height: ${size}px;
            border-radius: 50%;
            background: rgba(0, 255, 255, 0.5);
            transform: translate(${x}px, ${y}px) scale(0);
            pointer-events: none;
        `;
        
        element.appendChild(ripple);
        
        // Animate
        gsap.to(ripple, {
            scale: 2,
            opacity: 0,
            duration: 0.6,
            ease: "power2.out",
            onComplete: () => {
                ripple.remove();
            }
        });
    }

    hideLoadingScreen() {
        const loadingScreen = document.getElementById('loading-screen');
        const progressBar = document.getElementById('loading-progress');
        
        if (!loadingScreen || !progressBar) return;
        
        // Check if gsap is available
        if (typeof gsap !== 'undefined') {
            // Animate progress to 100%
            gsap.to(progressBar, {
                width: '100%',
                duration: 0.5,
                ease: "power2.out",
                onComplete: () => {
                    // Fade out loading screen
                    gsap.to(loadingScreen, {
                        opacity: 0,
                        duration: 0.5,
                        ease: "power2.out",
                        onComplete: () => {
                            loadingScreen.classList.add('hidden');
                            loadingScreen.style.display = 'none';
                            // Enable navigation after loading
                            this.enableNavigation();
                        }
                    });
                }
            });
        } else {
            // Fallback without animation
            progressBar.style.width = '100%';
            setTimeout(() => {
                loadingScreen.style.opacity = '0';
                setTimeout(() => {
                    loadingScreen.classList.add('hidden');
                    loadingScreen.style.display = 'none';
                    this.enableNavigation();
                }, 500);
            }, 500);
        }
    }
    
    enableNavigation() {
        // Enable all navigation elements
        const navButtons = document.querySelectorAll('.nav-btn');
        navButtons.forEach(btn => {
            btn.disabled = false;
            btn.style.pointerEvents = 'auto';
        });
        
        // Initialize back button if available
        this.initializeBackButton();
    }
    
    initializeBackButton() {
        // Create back button if not exists
        let backButton = document.querySelector('.back-button');
        if (!backButton) {
            backButton = document.createElement('button');
            backButton.className = 'back-button control-btn';
            backButton.innerHTML = '<span>←</span>';
            backButton.title = 'Go Back';
            backButton.style.cssText = `
                position: fixed;
                top: 100px;
                left: 20px;
                z-index: 1000;
                opacity: 0;
                pointer-events: none;
                transition: opacity 0.3s ease;
            `;
            document.body.appendChild(backButton);
        }
        
        backButton.addEventListener('click', () => {
            if (window.navigationManager) {
                window.navigationManager.navigateBack();
            } else {
                window.history.back();
            }
        });
        
        // Show back button when not on dashboard
        this.updateBackButtonVisibility();
    }
    
    updateBackButtonVisibility() {
        const backButton = document.querySelector('.back-button');
        if (backButton) {
            const shouldShow = this.currentSection !== 'dashboard' && 
                              window.location.pathname !== '/dashboard';
            backButton.style.opacity = shouldShow ? '1' : '0';
            backButton.style.pointerEvents = shouldShow ? 'auto' : 'none';
        }
    }
    
    initializeKeyboardNavigation() {
        document.addEventListener('keydown', (e) => {
            // Alt + Arrow keys for navigation
            if (e.altKey) {
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    this.navigateBack();
                } else if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    this.navigateForward();
                }
            }
            
            // Ctrl/Cmd + K for quick search/navigation
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.showQuickNavigation();
            }
            
            // Number keys for section navigation
            if (e.key >= '1' && e.key <= '6') {
                const sections = ['dashboard', 'movies', 'series', 'music', 'live', 'analytics'];
                const sectionIndex = parseInt(e.key) - 1;
                if (sections[sectionIndex]) {
                    e.preventDefault();
                    this.navigateToSection(sections[sectionIndex]);
                }
            }
            
            // Escape to close any overlays
            if (e.key === 'Escape') {
                this.closeOverlays();
            }
        });
    }
    
    navigateBack() {
        if (window.navigationManager) {
            window.navigationManager.navigateBack();
        } else {
            window.history.back();
        }
    }
    
    navigateForward() {
        if (window.navigationManager) {
            window.navigationManager.navigateForward();
        } else {
            window.history.forward();
        }
    }
    
    navigateToSection(section) {
        const navButton = document.querySelector(`[data-section="${section}"]`);
        if (navButton) {
            navButton.click();
        }
    }
    
    showQuickNavigation() {
        // Create quick navigation overlay
        const overlay = document.createElement('div');
        overlay.className = 'quick-nav-overlay';
        overlay.innerHTML = `
            <div class="quick-nav-content">
                <h3>Quick Navigation</h3>
                <div class="quick-nav-items">
                    <div class="quick-nav-item" data-section="dashboard">1. Dashboard</div>
                    <div class="quick-nav-item" data-section="movies">2. Movies</div>
                    <div class="quick-nav-item" data-section="series">3. Series</div>
                    <div class="quick-nav-item" data-section="music">4. Music</div>
                    <div class="quick-nav-item" data-section="live">5. Live TV</div>
                    <div class="quick-nav-item" data-section="analytics">6. Analytics</div>
                </div>
                <div class="quick-nav-help">Press 1-6 or click to navigate</div>
            </div>
        `;
        
        overlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(10, 10, 15, 0.9);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        `;
        
        document.body.appendChild(overlay);
        
        // Add click handlers
        overlay.querySelectorAll('.quick-nav-item').forEach(item => {
            item.addEventListener('click', () => {
                const section = item.dataset.section;
                this.navigateToSection(section);
                overlay.remove();
            });
        });
        
        // Close on overlay click or escape
        overlay.addEventListener('click', (e) => {
            if (e.target === overlay) {
                overlay.remove();
            }
        });
        
        // Auto-close after 5 seconds
        setTimeout(() => {
            if (overlay.parentNode) {
                overlay.remove();
            }
        }, 5000);
    }
    
    closeOverlays() {
        // Close quick navigation
        const quickNav = document.querySelector('.quick-nav-overlay');
        if (quickNav) {
            quickNav.remove();
        }
        
        // Close media preview
        this.hideMediaPreview();
    }
    
    showSectionLoading(section) {
        const sectionElement = document.querySelector('.hud-content');
        if (sectionElement) {
            sectionElement.classList.add('loading');
        }
    }
    
    hideSectionLoading(section) {
        const sectionElement = document.querySelector('.hud-content');
        if (sectionElement) {
            sectionElement.classList.remove('loading');
        }
    }
    
    trackSectionChange(section) {
        // Analytics tracking
        if (window.gtag) {
            window.gtag('event', 'navigation', {
                'section': section,
                'timestamp': Date.now()
            });
        }
        
        // Update document title
        document.title = `${section.charAt(0).toUpperCase() + section.slice(1)} - Holographic Media Dashboard`;
        
        // Update back button visibility
        this.updateBackButtonVisibility();
    }
    
    startLiveUpdates() {
        // Start real-time updates for live content
        if (this.liveUpdateInterval) {
            clearInterval(this.liveUpdateInterval);
        }
        
        this.liveUpdateInterval = setInterval(() => {
            if (this.currentSection === 'live') {
                // Update viewer counts and status
                this.updateLiveData();
            }
        }, 10000); // Update every 10 seconds
    }
    
    updateLiveData() {
        // Simulate live data updates
        const viewerChanges = ['+12', '-5', '+23', '+7', '-3', '+15'];
        viewerChanges.forEach((change, index) => {
            const activity = `Live Channel ${index + 1}: ${change} viewers`;
            if (Math.random() > 0.7) { // 30% chance to show update
                this.addActivity('📡', activity);
            }
        });
    }
}

// Export for use in other modules
window.UIController = UIController;