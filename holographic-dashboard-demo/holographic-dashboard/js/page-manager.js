// Page Manager - Handles dynamic page content and back navigation
// Manages all page content, transitions, and back button functionality

class PageManager {
    constructor(options = {}) {
        this.options = {
            containerSelector: options.containerSelector || '.hud-content',
            transitionDuration: options.transitionDuration || 300,
            enableBackButton: options.enableBackButton !== false,
            ...options
        };

        this.currentPage = 'dashboard';
        this.pageHistory = [];
        this.pageContent = new Map();
        this.isTransitioning = false;

        this.init();
    }

    init() {
        this.setupContainer();
        this.setupBackButton();
        this.loadPageContent();
        this.createPages();
    }

    setupContainer() {
        this.container = document.querySelector(this.options.containerSelector);
        if (!this.container) {
            throw new Error(`Container not found: ${this.options.containerSelector}`);
        }

        // Add page container structure
        this.container.innerHTML = `
            <div class="page-container">
                <div class="page-header">
                    <button class="back-button" style="display: none;">
                        <span class="back-icon">←</span>
                        <span class="back-text">Back</span>
                    </button>
                    <div class="page-breadcrumbs"></div>
                </div>
                <div class="page-content-wrapper">
                    <div id="dashboard-page" class="page-content active">
                        ${this.getDashboardContent()}
                    </div>
                </div>
            </div>
        `;
    }

    setupBackButton() {
        if (!this.options.enableBackButton) return;

        this.backButton = this.container.querySelector('.back-button');
        if (this.backButton) {
            this.backButton.addEventListener('click', () => {
                this.goBack();
            });
        }
    }

    createPages() {
        const pages = [
            'movies', 'series', 'music', 'live', 'analytics'
        ];

        pages.forEach(pageId => {
            this.createPage(pageId);
        });
    }

    createPage(pageId) {
        const wrapper = this.container.querySelector('.page-content-wrapper');
        
        const pageElement = document.createElement('div');
        pageElement.id = `${pageId}-page`;
        pageElement.className = 'page-content';
        pageElement.style.display = 'none';
        pageElement.innerHTML = this.getPageContent(pageId);
        
        wrapper.appendChild(pageElement);
    }

    async showPage(pageId) {
        if (this.isTransitioning || pageId === this.currentPage) {
            return false;
        }

        this.isTransitioning = true;

        try {
            // Add to history if it's a new page
            if (pageId !== this.currentPage) {
                this.pageHistory.push(this.currentPage);
                
                // Limit history size
                if (this.pageHistory.length > 10) {
                    this.pageHistory.shift();
                }
            }

            // Get page elements
            const currentPageElement = document.getElementById(`${this.currentPage}-page`);
            const targetPageElement = document.getElementById(`${pageId}-page`);

            if (!targetPageElement) {
                console.error(`Page not found: ${pageId}`);
                this.isTransitioning = false;
                return false;
            }

            // Transition out current page
            if (currentPageElement && currentPageElement !== targetPageElement) {
                await this.transitionOut(currentPageElement);
            }

            // Transition in new page
            await this.transitionIn(targetPageElement);

            // Update current page
            this.currentPage = pageId;

            // Update back button visibility
            this.updateBackButton();

            // Update breadcrumbs
            this.updateBreadcrumbs(pageId);

            // Initialize page-specific features
            this.initializePageFeatures(pageId);

            this.isTransitioning = false;
            return true;

        } catch (error) {
            console.error('Page transition error:', error);
            this.isTransitioning = false;
            return false;
        }
    }

    async transitionOut(element) {
        return new Promise(resolve => {
            element.style.transition = `opacity ${this.options.transitionDuration}ms ease-out, transform ${this.options.transitionDuration}ms ease-out`;
            element.style.opacity = '0';
            element.style.transform = 'translateX(-20px)';
            
            setTimeout(() => {
                element.classList.remove('active');
                element.style.display = 'none';
                resolve();
            }, this.options.transitionDuration);
        });
    }

    async transitionIn(element) {
        return new Promise(resolve => {
            element.style.display = 'block';
            element.style.opacity = '0';
            element.style.transform = 'translateX(20px)';
            element.style.transition = `opacity ${this.options.transitionDuration}ms ease-in, transform ${this.options.transitionDuration}ms ease-in`;
            
            // Force reflow
            element.offsetHeight;
            
            element.classList.add('active');
            element.style.opacity = '1';
            element.style.transform = 'translateX(0)';
            
            setTimeout(() => {
                element.style.transition = '';
                resolve();
            }, this.options.transitionDuration);
        });
    }

    goBack() {
        if (this.pageHistory.length > 0) {
            const previousPage = this.pageHistory.pop();
            this.showPage(previousPage);
        } else {
            // Fallback to dashboard
            this.showPage('dashboard');
        }
    }

    updateBackButton() {
        if (!this.backButton) return;

        const showBackButton = this.currentPage !== 'dashboard' && this.pageHistory.length > 0;
        this.backButton.style.display = showBackButton ? 'flex' : 'none';
    }

    updateBreadcrumbs(pageId) {
        const breadcrumbsContainer = this.container.querySelector('.page-breadcrumbs');
        if (!breadcrumbsContainer) return;

        let breadcrumbs = 'Dashboard';
        if (pageId !== 'dashboard') {
            const pageName = this.getPageTitle(pageId);
            breadcrumbs += ` > ${pageName}`;
        }

        breadcrumbsContainer.textContent = breadcrumbs;
    }

    getPageTitle(pageId) {
        const titles = {
            'dashboard': 'Dashboard',
            'movies': 'Movies',
            'series': 'TV Series',
            'music': 'Music Library',
            'live': 'Live TV',
            'analytics': 'Analytics'
        };
        return titles[pageId] || pageId;
    }

    initializePageFeatures(pageId) {
        // Initialize page-specific features
        switch (pageId) {
            case 'movies':
                this.initializeMoviesPage();
                break;
            case 'series':
                this.initializeSeriesPage();
                break;
            case 'music':
                this.initializeMusicPage();
                break;
            case 'live':
                this.initializeLivePage();
                break;
            case 'analytics':
                this.initializeAnalyticsPage();
                break;
        }

        // Dispatch page ready event
        window.dispatchEvent(new CustomEvent('pageready', {
            detail: { pageId, timestamp: Date.now() }
        }));
    }

    // Page Content Generators
    getDashboardContent() {
        return `
            <!-- Stats Panel -->
            <div class="stats-panel glass-panel">
                <div class="panel-title">System Overview</div>
                <div class="stats-grid">
                    <div class="stat-item">
                        <div class="stat-value" id="total-media">0</div>
                        <div class="stat-label">Total Media</div>
                        <div class="stat-graph" data-stat="media"></div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="storage-used">0TB</div>
                        <div class="stat-label">Storage</div>
                        <div class="stat-graph" data-stat="storage"></div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="active-users">0</div>
                        <div class="stat-label">Active Users</div>
                        <div class="stat-graph" data-stat="users"></div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-value" id="bandwidth">0Mbps</div>
                        <div class="stat-label">Bandwidth</div>
                        <div class="stat-graph" data-stat="bandwidth"></div>
                    </div>
                </div>
            </div>
            
            <!-- Activity Feed -->
            <div class="activity-panel glass-panel">
                <h3 class="panel-title">Recent Activity</h3>
                <div id="activity-feed" class="activity-feed">
                    <div class="activity-item">
                        <span class="activity-icon">🎬</span>
                        <span class="activity-text">System initialized</span>
                        <span class="activity-time">now</span>
                    </div>
                </div>
            </div>
        `;
    }

    getPageContent(pageId) {
        const content = {
            movies: `
                <div class="page-header-content">
                    <h2 class="page-title">🎬 Movies</h2>
                    <div class="page-actions">
                        <button class="action-btn" onclick="this.searchMovies()">
                            <span>🔍</span> Search
                        </button>
                        <button class="action-btn" onclick="this.addMovie()">
                            <span>➕</span> Add Movie
                        </button>
                        <button class="action-btn" onclick="this.refreshMovies()">
                            <span>🔄</span> Refresh
                        </button>
                    </div>
                </div>
                <div class="content-area">
                    <div class="media-grid" id="movies-grid">
                        <div class="media-card">
                            <div class="media-poster placeholder"></div>
                            <div class="media-info">
                                <h3>Movie Library</h3>
                                <p>Your movie collection will appear here</p>
                            </div>
                        </div>
                        <div class="media-card">
                            <div class="media-poster placeholder"></div>
                            <div class="media-info">
                                <h3>Recently Added</h3>
                                <p>Recently added movies</p>
                            </div>
                        </div>
                        <div class="media-card">
                            <div class="media-poster placeholder"></div>
                            <div class="media-info">
                                <h3>Trending</h3>
                                <p>Popular movies</p>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            series: `
                <div class="page-header-content">
                    <h2 class="page-title">📺 TV Series</h2>
                    <div class="page-actions">
                        <button class="action-btn" onclick="this.searchSeries()">
                            <span>🔍</span> Search
                        </button>
                        <button class="action-btn" onclick="this.addSeries()">
                            <span>➕</span> Add Series
                        </button>
                        <button class="action-btn" onclick="this.refreshSeries()">
                            <span>🔄</span> Refresh
                        </button>
                    </div>
                </div>
                <div class="content-area">
                    <div class="media-grid" id="series-grid">
                        <div class="media-card">
                            <div class="media-poster placeholder"></div>
                            <div class="media-info">
                                <h3>TV Series Library</h3>
                                <p>Your TV show collection</p>
                            </div>
                        </div>
                        <div class="media-card">
                            <div class="media-poster placeholder"></div>
                            <div class="media-info">
                                <h3>Currently Watching</h3>
                                <p>Continue watching</p>
                            </div>
                        </div>
                        <div class="media-card">
                            <div class="media-poster placeholder"></div>
                            <div class="media-info">
                                <h3>New Episodes</h3>
                                <p>Latest episodes available</p>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            music: `
                <div class="page-header-content">
                    <h2 class="page-title">🎵 Music Library</h2>
                    <div class="page-actions">
                        <button class="action-btn" onclick="this.searchMusic()">
                            <span>🔍</span> Search
                        </button>
                        <button class="action-btn" onclick="this.createPlaylist()">
                            <span>➕</span> Playlist
                        </button>
                        <button class="action-btn audio-viz-toggle" onclick="this.toggleAudioViz()">
                            <span>🎵</span> Visualizer
                        </button>
                    </div>
                </div>
                <div class="content-area">
                    <div class="music-layout">
                        <div class="music-sidebar">
                            <div class="music-section">
                                <h4>Quick Access</h4>
                                <div class="music-menu">
                                    <div class="menu-item active">📀 All Music</div>
                                    <div class="menu-item">⭐ Favorites</div>
                                    <div class="menu-item">🎵 Playlists</div>
                                    <div class="menu-item">🎤 Artists</div>
                                    <div class="menu-item">💿 Albums</div>
                                </div>
                            </div>
                        </div>
                        <div class="music-main">
                            <div class="music-grid" id="music-grid">
                                <div class="album-card">
                                    <div class="album-cover placeholder"></div>
                                    <div class="album-info">
                                        <h3>Music Library</h3>
                                        <p>Your music collection</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            live: `
                <div class="page-header-content">
                    <h2 class="page-title">📡 Live TV</h2>
                    <div class="page-actions">
                        <button class="action-btn" onclick="this.refreshChannels()">
                            <span>🔄</span> Refresh
                        </button>
                        <button class="action-btn" onclick="this.recordShow()">
                            <span>⏺️</span> Record
                        </button>
                        <button class="action-btn" onclick="this.showGuide()">
                            <span>📋</span> TV Guide
                        </button>
                    </div>
                </div>
                <div class="content-area">
                    <div class="live-tv-layout">
                        <div class="channel-list">
                            <h4>Channels</h4>
                            <div class="channels">
                                <div class="channel-item active">
                                    <span class="channel-number">1</span>
                                    <span class="channel-name">Demo Channel</span>
                                </div>
                                <div class="channel-item">
                                    <span class="channel-number">2</span>
                                    <span class="channel-name">Movies HD</span>
                                </div>
                                <div class="channel-item">
                                    <span class="channel-number">3</span>
                                    <span class="channel-name">Sports TV</span>
                                </div>
                            </div>
                        </div>
                        <div class="live-player">
                            <div class="player-container">
                                <div class="player-placeholder">
                                    <div class="play-icon">▶</div>
                                    <p>Select a channel to start streaming</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `,
            analytics: `
                <div class="page-header-content">
                    <h2 class="page-title">📊 Analytics</h2>
                    <div class="page-actions">
                        <button class="action-btn" onclick="this.refreshAnalytics()">
                            <span>🔄</span> Refresh
                        </button>
                        <button class="action-btn" onclick="this.exportData()">
                            <span>⬇️</span> Export
                        </button>
                        <button class="action-btn" onclick="this.customReport()">
                            <span>📋</span> Report
                        </button>
                    </div>
                </div>
                <div class="content-area">
                    <div class="analytics-dashboard">
                        <div class="metrics-row">
                            <div class="metric-card">
                                <h4>Usage Statistics</h4>
                                <div class="metric-chart" id="usage-chart">
                                    <div class="chart-placeholder">Usage data will appear here</div>
                                </div>
                            </div>
                            <div class="metric-card">
                                <h4>Popular Content</h4>
                                <div class="metric-list">
                                    <div class="metric-item">
                                        <span>🎬 Movie Stream</span>
                                        <span class="metric-value">85%</span>
                                    </div>
                                    <div class="metric-item">
                                        <span>📺 TV Shows</span>
                                        <span class="metric-value">72%</span>
                                    </div>
                                    <div class="metric-item">
                                        <span>🎵 Music</span>
                                        <span class="metric-value">43%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="metrics-row">
                            <div class="metric-card full-width">
                                <h4>System Performance</h4>
                                <div class="performance-grid">
                                    <div class="perf-item">
                                        <div class="perf-label">CPU Usage</div>
                                        <div class="perf-bar">
                                            <div class="perf-fill" style="width: 45%"></div>
                                        </div>
                                        <div class="perf-value">45%</div>
                                    </div>
                                    <div class="perf-item">
                                        <div class="perf-label">Memory</div>
                                        <div class="perf-bar">
                                            <div class="perf-fill" style="width: 68%"></div>
                                        </div>
                                        <div class="perf-value">68%</div>
                                    </div>
                                    <div class="perf-item">
                                        <div class="perf-label">Storage</div>
                                        <div class="perf-bar">
                                            <div class="perf-fill" style="width: 32%"></div>
                                        </div>
                                        <div class="perf-value">32%</div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            `
        };

        return content[pageId] || '<div>Page content not found</div>';
    }

    loadPageContent() {
        // This method can be used to load content dynamically from API
        // For now, content is generated statically
    }

    // Page-specific initialization methods
    initializeMoviesPage() {
        console.log('Movies page initialized');
        // Enable specific features for movies
        if (window.dashboard && window.dashboard.mediaCardsManager) {
            window.dashboard.mediaCardsManager.setFilter('movies');
        }
    }

    initializeSeriesPage() {
        console.log('Series page initialized');
        if (window.dashboard && window.dashboard.mediaCardsManager) {
            window.dashboard.mediaCardsManager.setFilter('series');
        }
    }

    initializeMusicPage() {
        console.log('Music page initialized');
        // Enable audio visualizer for music page
        if (window.dashboard && window.dashboard.audioVisualizer) {
            window.dashboard.audioVisualizer.setEnabled(true);
        }
        
        // Setup music page specific features
        const audioVizToggle = document.querySelector('.audio-viz-toggle');
        if (audioVizToggle) {
            audioVizToggle.addEventListener('click', () => {
                const isEnabled = window.dashboard.audioVisualizer.isEnabled();
                window.dashboard.audioVisualizer.setEnabled(!isEnabled);
                audioVizToggle.classList.toggle('active', !isEnabled);
            });
        }
    }

    initializeLivePage() {
        console.log('Live TV page initialized');
        // Setup channel selection
        const channelItems = document.querySelectorAll('.channel-item');
        channelItems.forEach(item => {
            item.addEventListener('click', () => {
                channelItems.forEach(c => c.classList.remove('active'));
                item.classList.add('active');
                // Simulate channel change
                console.log('Switched to channel:', item.querySelector('.channel-name').textContent);
            });
        });
    }

    initializeAnalyticsPage() {
        console.log('Analytics page initialized');
        // Initialize charts and metrics
        this.updateAnalyticsData();
    }

    updateAnalyticsData() {
        // Simulate real-time analytics updates
        const performanceBars = document.querySelectorAll('.perf-fill');
        performanceBars.forEach(bar => {
            const randomWidth = Math.floor(Math.random() * 100);
            bar.style.width = randomWidth + '%';
            const valueElement = bar.parentElement.nextElementSibling;
            if (valueElement) {
                valueElement.textContent = randomWidth + '%';
            }
        });
    }

    // Public API methods
    getCurrentPage() {
        return this.currentPage;
    }

    getPageHistory() {
        return [...this.pageHistory];
    }

    canGoBack() {
        return this.pageHistory.length > 0;
    }

    // Navigation button handlers (can be called from buttons)
    searchMovies() {
        console.log('Search movies');
        // Implement search functionality
    }

    addMovie() {
        console.log('Add movie');
        // Implement add movie functionality
    }

    refreshMovies() {
        console.log('Refresh movies');
        // Implement refresh functionality
    }

    // Add similar methods for other pages...
}

// Initialize page manager when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Wait for other components to initialize first
    setTimeout(() => {
        if (!window.pageManager) {
            window.pageManager = new PageManager();
            console.log('Page Manager initialized');
        }
    }, 100);
});

// Export for use in other modules
window.PageManager = PageManager;