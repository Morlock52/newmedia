// Modern SPA Router for HoloMedia Hub
// Implements client-side routing with History API for smooth navigation

class Router {
    constructor(options = {}) {
        this.routes = new Map();
        this.currentRoute = null;
        this.root = options.root || '/';
        this.notFound = options.notFound || (() => console.error('Route not found'));
        this.beforeRoute = options.beforeRoute || null;
        this.afterRoute = options.afterRoute || null;
        this.transitionDuration = options.transitionDuration || 300;
        this.isNavigating = false;
        this.errorCount = 0;
        this.maxErrors = 3;
        
        // Initialize
        this.init();
    }

    init() {
        // Handle browser back/forward buttons
        window.addEventListener('popstate', (e) => {
            this.handleRoute(window.location.pathname, e.state);
        });

        // Intercept all link clicks
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href]');
            if (link && this.shouldHandleLink(link)) {
                e.preventDefault();
                this.navigate(link.getAttribute('href'));
            }
        });

        // Handle initial route
        this.handleRoute(window.location.pathname);
    }

    shouldHandleLink(link) {
        // Only handle internal links
        const href = link.getAttribute('href');
        return (
            !link.hasAttribute('download') &&
            !link.hasAttribute('target') &&
            href &&
            !href.startsWith('http') &&
            !href.startsWith('//') &&
            !href.startsWith('#') &&
            !href.startsWith('mailto:') &&
            !href.startsWith('tel:')
        );
    }

    route(path, handler, options = {}) {
        // Support wildcards and parameters
        const regex = this.pathToRegex(path);
        this.routes.set(regex, {
            handler,
            path,
            options,
            regex
        });
        return this;
    }

    pathToRegex(path) {
        // Convert path to regex, supporting :param and * wildcards
        const pattern = path
            .replace(/\//g, '\\/')
            .replace(/:(\w+)/g, '(?<$1>[^/]+)')
            .replace(/\*/g, '.*');
        return new RegExp(`^${pattern}$`);
    }

    async navigate(path, state = {}, replace = false) {
        // Prevent navigation if currently navigating
        if (this.isNavigating) {
            console.warn('Navigation already in progress');
            return false;
        }
        
        // Normalize path
        const fullPath = this.normalizePath(path);
        
        // Don't navigate if we're already on this path
        if (fullPath === window.location.pathname && !replace) {
            return false;
        }
        
        this.isNavigating = true;
        
        try {
            // Execute before route hook
            if (this.beforeRoute) {
                const shouldContinue = await this.beforeRoute(fullPath, this.currentRoute);
                if (!shouldContinue) {
                    this.isNavigating = false;
                    return false;
                }
            }

            // Update browser history
            if (replace) {
                window.history.replaceState(state, '', fullPath);
            } else {
                window.history.pushState(state, '', fullPath);
            }

            // Handle the route
            await this.handleRoute(fullPath, state);
            return true;
            
        } catch (error) {
            console.error('Navigation error:', error);
            this.isNavigating = false;
            return false;
        }
    }

    normalizePath(path) {
        // Remove trailing slashes and ensure proper format
        if (!path.startsWith('/')) {
            path = '/' + path;
        }
        if (path.length > 1 && path.endsWith('/')) {
            path = path.slice(0, -1);
        }
        return path;
    }

    async handleRoute(path, state = {}) {
        const normalizedPath = this.normalizePath(path);
        
        try {
            // Find matching route
            for (const [regex, route] of this.routes) {
                const match = normalizedPath.match(regex);
                if (match) {
                    // Extract parameters
                    const params = match.groups || {};
                    
                    // Create route context
                    const context = {
                        path: normalizedPath,
                        params,
                        state,
                        query: this.parseQuery(window.location.search),
                        route: route.path,
                        timestamp: Date.now()
                    };

                    // Store current route
                    this.currentRoute = context;

                    // Execute route handler with transition and error handling
                    await this.executeWithTransition(async () => {
                        try {
                            await route.handler(context);
                            this.errorCount = 0; // Reset error count on success
                        } catch (error) {
                            console.error('Route handler error:', error);
                            this.handleRouteError(error, context);
                        }
                    });

                    // Execute after route hook
                    if (this.afterRoute) {
                        try {
                            await this.afterRoute(context);
                        } catch (error) {
                            console.error('After route hook error:', error);
                        }
                    }
                    
                    this.isNavigating = false;
                    return;
                }
            }

            // No matching route found
            this.notFound(normalizedPath);
            this.isNavigating = false;
            
        } catch (error) {
            console.error('Route handling error:', error);
            this.handleRouteError(error, { path: normalizedPath });
            this.isNavigating = false;
        }
    }
    
    handleRouteError(error, context) {
        this.errorCount++;
        
        if (this.errorCount >= this.maxErrors) {
            // Too many errors, show critical error
            this.showCriticalError(error);
            return;
        }
        
        // Show error notification
        if (window.dashboard && window.dashboard.uiController) {
            window.dashboard.uiController.showNotification(
                'Navigation error occurred. Please try again.',
                'error'
            );
        }
        
        // Try to navigate to dashboard as fallback
        setTimeout(() => {
            if (context.path !== '/dashboard') {
                this.navigate('/dashboard', {}, true);
            }
        }, 1000);
    }
    
    showCriticalError(error) {
        const errorOverlay = document.createElement('div');
        errorOverlay.className = 'critical-error-overlay';
        errorOverlay.innerHTML = `
            <div class="critical-error-content">
                <h2>Navigation System Error</h2>
                <p>The navigation system encountered a critical error.</p>
                <pre>${error.message}</pre>
                <button onclick="location.reload()">Reload Application</button>
                <button onclick="this.parentElement.parentElement.remove()">Continue</button>
            </div>
        `;
        
        errorOverlay.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(255, 0, 0, 0.1);
            backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10001;
        `;
        
        document.body.appendChild(errorOverlay);
    }

    async executeWithTransition(handler) {
        const content = document.querySelector('[data-router-view]') || document.querySelector('.hud-content');
        if (!content) {
            await handler();
            return;
        }

        try {
            // Add loading state
            content.classList.add('route-loading');
            
            // Fade out with error handling
            content.style.transition = `opacity ${this.transitionDuration}ms ease-out`;
            content.style.opacity = '0.5';

            await new Promise(resolve => setTimeout(resolve, this.transitionDuration / 2));

            // Execute handler with timeout
            const timeoutPromise = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Route handler timeout')), 5000)
            );
            
            await Promise.race([handler(), timeoutPromise]);

            // Fade in
            content.style.opacity = '1';
            content.classList.remove('route-loading');
            
        } catch (error) {
            // Restore content on error
            content.style.opacity = '1';
            content.classList.remove('route-loading');
            throw error;
        }
    }

    parseQuery(queryString) {
        const params = new URLSearchParams(queryString);
        const query = {};
        for (const [key, value] of params) {
            query[key] = value;
        }
        return query;
    }

    back() {
        window.history.back();
    }

    forward() {
        window.history.forward();
    }

    go(delta) {
        window.history.go(delta);
    }

    // Helper method to update active navigation links
    updateActiveLinks() {
        const currentPath = window.location.pathname;
        
        // Update navigation buttons
        document.querySelectorAll('[data-route]').forEach(link => {
            const route = link.getAttribute('data-route');
            const isActive = currentPath === route || currentPath.startsWith(route + '/');
            link.classList.toggle('active', isActive);
        });
        
        // Update section-based navigation
        document.querySelectorAll('[data-section]').forEach(button => {
            const section = button.getAttribute('data-section');
            const sectionRoute = this.getSectionRoute(section);
            const isActive = currentPath === sectionRoute;
            button.classList.toggle('active', isActive);
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

    // Get current route information
    getCurrentRoute() {
        return this.currentRoute;
    }

    // Programmatic navigation with animation
    async navigateWithAnimation(path, animation = 'slide') {
        const content = document.querySelector('[data-router-view]') || document.querySelector('.hud-content');
        if (!content) {
            return this.navigate(path);
        }

        try {
            // Apply exit animation
            content.classList.add(`route-exit-${animation}`);
            await new Promise(resolve => setTimeout(resolve, this.transitionDuration));

            // Navigate
            const success = await this.navigate(path);
            if (!success) {
                // Revert animation if navigation failed
                content.classList.remove(`route-exit-${animation}`);
                return false;
            }

            // Apply enter animation
            content.classList.remove(`route-exit-${animation}`);
            content.classList.add(`route-enter-${animation}`);
            
            setTimeout(() => {
                content.classList.remove(`route-enter-${animation}`);
            }, this.transitionDuration);
            
            return true;
            
        } catch (error) {
            console.error('Animated navigation error:', error);
            content.classList.remove(`route-exit-${animation}`, `route-enter-${animation}`);
            return false;
        }
    }

    // Middleware support
    use(middleware) {
        const originalBeforeRoute = this.beforeRoute;
        this.beforeRoute = async (path, from) => {
            const next = async () => {
                if (originalBeforeRoute) {
                    return await originalBeforeRoute(path, from);
                }
                return true;
            };
            return await middleware(path, from, next);
        };
        return this;
    }
}

// Enhanced router instance with error recovery
class EnhancedRouter extends Router {
    constructor(options = {}) {
        super(options);
        this.setupErrorRecovery();
        this.setupPerformanceMonitoring();
    }
    
    setupErrorRecovery() {
        // Global error handler for navigation issues
        window.addEventListener('error', (event) => {
            if (this.isNavigating) {
                console.error('Navigation-related error:', event.error);
                this.handleRouteError(event.error, this.currentRoute);
            }
        });
        
        // Handle unhandled promise rejections during navigation
        window.addEventListener('unhandledrejection', (event) => {
            if (this.isNavigating) {
                console.error('Navigation promise rejection:', event.reason);
                this.handleRouteError(event.reason, this.currentRoute);
            }
        });
    }
    
    setupPerformanceMonitoring() {
        // Monitor navigation performance
        this.navigationTimes = new Map();
        
        const originalNavigate = this.navigate;
        this.navigate = async function(path, state, replace) {
            const startTime = performance.now();
            const result = await originalNavigate.call(this, path, state, replace);
            const endTime = performance.now();
            
            this.navigationTimes.set(path, endTime - startTime);
            
            // Log slow navigations
            if (endTime - startTime > 1000) {
                console.warn(`Slow navigation to ${path}: ${(endTime - startTime).toFixed(2)}ms`);
            }
            
            return result;
        }.bind(this);
    }
    
    getNavigationStats() {
        const times = Array.from(this.navigationTimes.values());
        return {
            totalNavigations: times.length,
            averageTime: times.length > 0 ? times.reduce((a, b) => a + b, 0) / times.length : 0,
            slowestNavigation: Math.max(...times, 0),
            fastestNavigation: Math.min(...times, 0)
        };
    }
    
    // Enhanced back/forward with error handling
    back() {
        try {
            if (window.history.length > 1) {
                window.history.back();
            } else {
                // Fallback to dashboard if no history
                this.navigate('/dashboard');
            }
        } catch (error) {
            console.error('Back navigation error:', error);
            this.navigate('/dashboard');
        }
    }
    
    forward() {
        try {
            window.history.forward();
        } catch (error) {
            console.error('Forward navigation error:', error);
        }
    }
}

// Export for use in other modules
window.Router = Router;
window.EnhancedRouter = EnhancedRouter;