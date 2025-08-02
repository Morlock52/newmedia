// Navigation Manager - Comprehensive navigation solution for HoloMedia Hub
// Handles routing, error boundaries, loading states, and smooth transitions

class NavigationManager {
    constructor(options = {}) {
        this.options = {
            root: options.root || '/holographic-dashboard',
            defaultRoute: options.defaultRoute || '/dashboard',
            transitionDuration: options.transitionDuration || 300,
            enableBreadcrumbs: options.enableBreadcrumbs !== false,
            enableBackToTop: options.enableBackToTop !== false,
            enableKeyboardShortcuts: options.enableKeyboardShortcuts !== false,
            ...options
        };

        this.router = null;
        this.history = [];
        this.maxHistorySize = 50;
        this.isNavigating = false;
        this.errorCount = 0;
        this.maxErrors = 3;

        this.init();
    }

    init() {
        this.setupRouter();
        this.setupErrorBoundary();
        this.setupLoadingStates();
        this.setupBackToTop();
        this.setupKeyboardNavigation();
        this.setupAccessibility();
        this.setupPerformanceMonitoring();
    }

    setupRouter() {
        this.router = new Router({
            root: this.options.root,
            transitionDuration: this.options.transitionDuration,
            beforeRoute: this.beforeRoute.bind(this),
            afterRoute: this.afterRoute.bind(this),
            notFound: this.handleNotFound.bind(this)
        });

        // Define application routes
        this.defineRoutes();
    }

    defineRoutes() {
        const routes = [
            { path: '/dashboard', handler: 'dashboard', title: 'Dashboard' },
            { path: '/movies', handler: 'movies', title: 'Movies' },
            { path: '/series', handler: 'series', title: 'TV Series' },
            { path: '/music', handler: 'music', title: 'Music Library' },
            { path: '/live', handler: 'live', title: 'Live TV' },
            { path: '/analytics', handler: 'analytics', title: 'Analytics' },
            { path: '/', handler: () => this.router.navigate(this.options.defaultRoute, {}, true) }
        ];

        routes.forEach(route => {
            if (typeof route.handler === 'string') {
                this.router.route(route.path, (ctx) => this.loadPage(route.handler, route.title, ctx));
            } else {
                this.router.route(route.path, route.handler);
            }
        });
    }

    async beforeRoute(to, from) {
        if (this.isNavigating) {
            console.warn('Navigation already in progress');
            return false;
        }

        this.isNavigating = true;
        this.showProgress();

        // Check if user needs to save changes
        if (from && this.hasUnsavedChanges()) {
            const shouldContinue = await this.confirmNavigation();
            if (!shouldContinue) {
                this.isNavigating = false;
                this.hideProgress();
                return false;
            }
        }

        // Add to history
        this.addToHistory(to, from);

        // Dispatch navigation start event
        this.dispatchEvent('navigationstart', { to, from });

        return true;
    }

    async afterRoute(context) {
        this.isNavigating = false;
        this.hideProgress();
        this.updateBreadcrumbs(context);
        this.updateDocumentTitle(context);
        this.updateActiveStates();
        
        // Reset error count on successful navigation
        this.errorCount = 0;

        // Dispatch navigation end event
        this.dispatchEvent('navigationend', { context });

        // Analytics tracking
        this.trackPageView(context);
    }

    async loadPage(pageId, title, context) {
        try {
            const startTime = performance.now();

            // Show loading state
            this.showPageLoading(pageId);

            // Use page manager if available, otherwise fallback to direct loading
            if (window.pageManager) {
                await window.pageManager.showPage(pageId);
            } else {
                // Fallback to simple page loading
                await this.loadPageContent(pageId, context);
            }

            // Hide loading state
            this.hidePageLoading(pageId);

            // Performance tracking
            const loadTime = performance.now() - startTime;
            this.trackPerformance(pageId, loadTime);

            // Update page title
            document.title = `${title} - Holographic Media Dashboard`;
            
            // Update UI controller if available
            if (window.dashboard && window.dashboard.uiController) {
                window.dashboard.uiController.switchSection(pageId);
            }

        } catch (error) {
            this.handlePageLoadError(pageId, error);
        }
    }

    async loadPageContent(pageId, context) {
        // Check if page manager is available
        if (window.pageManager) {
            // Let page manager handle the content loading
            return;
        }
        
        // Fallback implementation for direct page loading
        const pages = document.querySelectorAll('.page-content');
        const targetPage = document.getElementById(`${pageId}-page`);

        if (!targetPage) {
            // Create page container if it doesn't exist
            const hudContent = document.querySelector('.hud-content');
            if (hudContent) {
                const newPage = document.createElement('div');
                newPage.id = `${pageId}-page`;
                newPage.className = 'page-content';
                newPage.style.cssText = `
                    display: none;
                    width: 100%;
                    height: 100%;
                    position: absolute;
                    top: 0;
                    left: 0;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                `;
                hudContent.appendChild(newPage);
            } else {
                throw new Error(`Page container not found and cannot create: ${pageId}`);
            }
        }

        // Transition out current page
        const currentPage = document.querySelector('.page-content.active');
        if (currentPage && currentPage !== document.getElementById(`${pageId}-page`)) {
            await this.transitionOut(currentPage);
        }

        // Transition in new page
        await this.transitionIn(document.getElementById(`${pageId}-page`));

        // Initialize page-specific features
        this.initializePage(pageId, context);
    }

    async transitionOut(element) {
        return new Promise(resolve => {
            element.classList.add('page-transition-exit');
            setTimeout(() => {
                element.classList.remove('active', 'page-transition-exit');
                element.style.display = 'none';
                resolve();
            }, this.options.transitionDuration);
        });
    }

    async transitionIn(element) {
        return new Promise(resolve => {
            element.style.display = 'block';
            element.classList.add('page-transition-enter');
            
            // Force reflow
            element.offsetHeight;
            
            element.classList.add('active');
            setTimeout(() => {
                element.classList.remove('page-transition-enter');
                resolve();
            }, this.options.transitionDuration);
        });
    }

    initializePage(pageId, context) {
        // Page-specific initialization
        switch (pageId) {
            case 'dashboard':
                this.initDashboard(context);
                break;
            case 'movies':
                this.initMovies(context);
                break;
            case 'series':
                this.initSeries(context);
                break;
            case 'music':
                this.initMusic(context);
                break;
            case 'live':
                this.initLive(context);
                break;
            case 'analytics':
                this.initAnalytics(context);
                break;
        }

        // Dispatch page ready event
        this.dispatchEvent('pageready', { pageId, context });
    }

    // Error Handling
    setupErrorBoundary() {
        window.addEventListener('error', (event) => {
            if (this.isNavigating) {
                event.preventDefault();
                this.handleNavigationError(event.error);
            }
        });

        window.addEventListener('unhandledrejection', (event) => {
            if (this.isNavigating) {
                event.preventDefault();
                this.handleNavigationError(event.reason);
            }
        });
    }

    handleNavigationError(error) {
        console.error('Navigation error:', error);
        this.errorCount++;

        if (this.errorCount >= this.maxErrors) {
            this.showCriticalError();
            return;
        }

        this.showError('Navigation failed. Please try again.');
        this.isNavigating = false;
        this.hideProgress();
    }

    handlePageLoadError(pageId, error) {
        console.error(`Failed to load page ${pageId}:`, error);
        this.showError(`Failed to load ${pageId}. Please try again.`);
        
        // Fallback to dashboard
        if (pageId !== 'dashboard') {
            setTimeout(() => {
                this.router.navigate('/dashboard');
            }, 2000);
        }
    }

    handleNotFound(path) {
        console.warn('Route not found:', path);
        this.show404Page(path);
    }

    show404Page(path) {
        const content = document.querySelector('[data-router-view]');
        if (content) {
            content.innerHTML = `
                <div class="error-page">
                    <h1>404 - Page Not Found</h1>
                    <p>The page "${path}" could not be found.</p>
                    <button onclick="navigationManager.navigateHome()">Go to Dashboard</button>
                </div>
            `;
        }
    }

    // Loading States
    setupLoadingStates() {
        // Create loading overlay
        const loadingOverlay = document.createElement('div');
        loadingOverlay.className = 'nav-loading-overlay';
        loadingOverlay.innerHTML = '<div class="nav-loading-spinner"></div>';
        document.body.appendChild(loadingOverlay);

        // Create progress bar
        const progressBar = document.createElement('div');
        progressBar.className = 'nav-progress';
        document.body.appendChild(progressBar);
    }

    showProgress() {
        const progress = document.querySelector('.nav-progress');
        if (progress) {
            progress.classList.add('active');
        }
    }

    hideProgress() {
        const progress = document.querySelector('.nav-progress');
        if (progress) {
            progress.classList.remove('active');
            // Reset after animation
            setTimeout(() => {
                progress.style.transform = 'scaleX(0)';
            }, 300);
        }
    }

    showPageLoading(pageId) {
        const page = document.getElementById(`${pageId}-page`);
        if (page) {
            page.classList.add('loading');
        }
    }

    hidePageLoading(pageId) {
        const page = document.getElementById(`${pageId}-page`);
        if (page) {
            page.classList.remove('loading');
        }
    }

    // Back to Top
    setupBackToTop() {
        if (!this.options.enableBackToTop) return;

        const button = document.createElement('div');
        button.className = 'back-to-top';
        button.setAttribute('role', 'button');
        button.setAttribute('aria-label', 'Back to top');
        button.setAttribute('tabindex', '0');
        document.body.appendChild(button);

        // Show/hide based on scroll
        let scrollTimeout;
        window.addEventListener('scroll', () => {
            clearTimeout(scrollTimeout);
            scrollTimeout = setTimeout(() => {
                const scrolled = window.pageYOffset > 300;
                button.classList.toggle('visible', scrolled);
            }, 100);
        });

        // Click handler
        button.addEventListener('click', () => {
            window.scrollTo({ top: 0, behavior: 'smooth' });
        });

        // Keyboard support
        button.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                button.click();
            }
        });
    }

    // Keyboard Navigation
    setupKeyboardNavigation() {
        if (!this.options.enableKeyboardShortcuts) return;

        document.addEventListener('keydown', (e) => {
            // Alt + Arrow keys for history navigation
            if (e.altKey) {
                if (e.key === 'ArrowLeft') {
                    e.preventDefault();
                    this.navigateBack();
                } else if (e.key === 'ArrowRight') {
                    e.preventDefault();
                    this.navigateForward();
                }
            }

            // Ctrl/Cmd + K for search focus
            if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
                e.preventDefault();
                this.focusSearch();
            }

            // Escape to close modals
            if (e.key === 'Escape') {
                this.closeModals();
            }
        });
    }

    // Accessibility
    setupAccessibility() {
        // Skip to content link
        const skipLink = document.createElement('a');
        skipLink.className = 'skip-to-content';
        skipLink.href = '#main-content';
        skipLink.textContent = 'Skip to main content';
        document.body.insertBefore(skipLink, document.body.firstChild);

        // Announce navigation to screen readers
        this.announcer = document.createElement('div');
        this.announcer.setAttribute('role', 'status');
        this.announcer.setAttribute('aria-live', 'polite');
        this.announcer.setAttribute('aria-atomic', 'true');
        this.announcer.style.position = 'absolute';
        this.announcer.style.left = '-9999px';
        document.body.appendChild(this.announcer);
    }

    announceNavigation(message) {
        if (this.announcer) {
            this.announcer.textContent = message;
            // Clear after announcement
            setTimeout(() => {
                this.announcer.textContent = '';
            }, 1000);
        }
    }

    // Breadcrumbs
    updateBreadcrumbs(context) {
        if (!this.options.enableBreadcrumbs) return;

        const breadcrumbsContainer = document.querySelector('.breadcrumbs');
        if (!breadcrumbsContainer) return;

        const parts = context.path.split('/').filter(p => p);
        let html = '<a href="/" class="breadcrumb-link">Home</a>';

        parts.forEach((part, index) => {
            const isLast = index === parts.length - 1;
            const path = '/' + parts.slice(0, index + 1).join('/');
            const name = this.formatBreadcrumbName(part);

            html += ' <span class="breadcrumb-separator">›</span> ';
            
            if (isLast) {
                html += `<span class="breadcrumb-current">${name}</span>`;
            } else {
                html += `<a href="${path}" class="breadcrumb-link">${name}</a>`;
            }
        });

        breadcrumbsContainer.innerHTML = html;
    }

    formatBreadcrumbName(name) {
        return name.split('-').map(word => 
            word.charAt(0).toUpperCase() + word.slice(1)
        ).join(' ');
    }

    // History Management
    addToHistory(to, from) {
        this.history.push({
            to,
            from,
            timestamp: Date.now()
        });

        // Limit history size
        if (this.history.length > this.maxHistorySize) {
            this.history.shift();
        }
    }

    navigateBack() {
        this.router.back();
    }

    navigateForward() {
        this.router.forward();
    }

    navigateHome() {
        this.router.navigate(this.options.defaultRoute);
    }

    // Utility Methods
    hasUnsavedChanges() {
        // Check for unsaved changes in forms or editors
        return false; // Implement based on your needs
    }

    async confirmNavigation() {
        return confirm('You have unsaved changes. Do you want to leave this page?');
    }

    updateDocumentTitle(context) {
        const pageName = this.formatBreadcrumbName(context.path.split('/').pop() || 'dashboard');
        document.title = `${pageName} - HoloMedia Hub`;
    }

    updateActiveStates() {
        // Update navigation active states
        const currentPath = window.location.pathname;
        document.querySelectorAll('[data-route]').forEach(link => {
            const route = link.getAttribute('data-route');
            const isActive = currentPath.includes(route);
            link.classList.toggle('active', isActive);
        });
    }

    focusSearch() {
        const searchInput = document.querySelector('.search-input');
        if (searchInput) {
            searchInput.focus();
            searchInput.select();
        }
    }

    closeModals() {
        // Close any open modals
        document.querySelectorAll('.modal.active').forEach(modal => {
            modal.classList.remove('active');
        });
    }

    // Performance Monitoring
    setupPerformanceMonitoring() {
        this.performanceData = new Map();
        
        // Monitor navigation timing
        if ('PerformanceObserver' in window) {
            const observer = new PerformanceObserver((list) => {
                for (const entry of list.getEntries()) {
                    if (entry.entryType === 'navigation') {
                        this.trackNavigationPerformance(entry);
                    }
                }
            });
            observer.observe({ entryTypes: ['navigation'] });
        }
    }

    trackPerformance(pageId, loadTime) {
        if (!this.performanceData.has(pageId)) {
            this.performanceData.set(pageId, []);
        }
        
        this.performanceData.get(pageId).push({
            loadTime,
            timestamp: Date.now()
        });

        // Log slow page loads
        if (loadTime > 1000) {
            console.warn(`Slow page load detected: ${pageId} took ${loadTime.toFixed(2)}ms`);
        }
    }

    trackPageView(context) {
        // Analytics tracking
        if (window.gtag) {
            window.gtag('config', 'GA_MEASUREMENT_ID', {
                page_path: context.path
            });
        }
    }

    trackNavigationPerformance(entry) {
        const metrics = {
            domContentLoaded: entry.domContentLoadedEventEnd - entry.domContentLoadedEventStart,
            loadComplete: entry.loadEventEnd - entry.loadEventStart,
            domInteractive: entry.domInteractive - entry.fetchStart,
            timeToFirstByte: entry.responseStart - entry.requestStart
        };

        console.log('Navigation performance:', metrics);
    }

    // Error UI
    showError(message) {
        const error = document.createElement('div');
        error.className = 'nav-error';
        error.textContent = message;
        document.body.appendChild(error);

        setTimeout(() => {
            error.remove();
        }, 5000);
    }

    showCriticalError() {
        const overlay = document.querySelector('.nav-loading-overlay');
        if (overlay) {
            overlay.innerHTML = `
                <div class="critical-error">
                    <h2>Navigation Error</h2>
                    <p>We're having trouble loading the page. Please refresh and try again.</p>
                    <button onclick="location.reload()">Refresh Page</button>
                </div>
            `;
            overlay.classList.add('active');
        }
    }

    // Event System
    dispatchEvent(eventName, detail) {
        window.dispatchEvent(new CustomEvent(`navigation:${eventName}`, {
            detail,
            bubbles: true
        }));
    }

    // Page-specific initializations
    initDashboard(context) {
        console.log('Dashboard initialized', context);
        // Dashboard is handled by page manager or UI controller
    }

    initMovies(context) {
        console.log('Movies page initialized', context);
    }
    
    initSeries(context) {
        console.log('Series page initialized', context);
    }
    
    initMusic(context) {
        console.log('Music page initialized', context);
        // Enable audio visualizer for music
        if (window.dashboard && window.dashboard.audioVisualizer) {
            window.dashboard.audioVisualizer.setEnabled(true);
        }
    }
    
    initLive(context) {
        console.log('Live TV page initialized', context);
    }
    
    initAnalytics(context) {
        console.log('Analytics page initialized', context);
    }

    // Public API
    navigate(path, state = {}) {
        return this.router.navigate(path, state);
    }

    getCurrentRoute() {
        return this.router.getCurrentRoute();
    }

    refresh() {
        const current = this.getCurrentRoute();
        if (current) {
            this.router.navigate(current.path, current.state, true);
        }
    }
}

// Initialize navigation manager
window.navigationManager = new NavigationManager({
    root: '/holographic-dashboard',
    defaultRoute: '/dashboard',
    transitionDuration: 300,
    enableBreadcrumbs: true,
    enableBackToTop: true,
    enableKeyboardShortcuts: true
});

// Export for use in other modules
window.NavigationManager = NavigationManager;