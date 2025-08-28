/**
 * Optimized Feature Rendering Engine
 * Extracted and modularized for reusability
 */

class FeatureRenderer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.performanceMonitor = new PerformanceMonitor();
        this.observer = null;
    }

    /**
     * Render features with optimized DOM manipulation
     * @param {Array} featureData - Array of feature objects
     */
    render(featureData) {
        if (!this.container) {
            console.error('Container not found');
            return;
        }

        const fragment = document.createDocumentFragment();
        
        featureData.forEach(feature => {
            const card = this.createFeatureCard(feature);
            fragment.appendChild(card);
        });
        
        // Single DOM append for better performance
        this.container.appendChild(fragment);
        this.initializeAnimations();
    }

    /**
     * Create optimized feature card
     * @param {Object} feature - Feature data object
     * @returns {HTMLElement} Feature card element
     */
    createFeatureCard(feature) {
        const card = document.createElement('div');
        card.className = 'feature-card glass-card focus-visible';
        card.tabIndex = 0;
        
        // Use template literals for efficient HTML generation
        card.innerHTML = `
            <div class="feature-header">
                <div class="feature-icon">${feature.icon}</div>
                <h3 class="feature-title">${feature.title}</h3>
                <span class="feature-badge">${feature.badge}</span>
            </div>
            <p class="feature-description">${feature.desc}</p>
            <div class="feature-tech">
                ${feature.tech.map(tech => `<span class="tech-tag">${tech}</span>`).join('')}
            </div>
            <div class="feature-stats">
                ${this.renderStats(feature.stats)}
            </div>
        `;

        return card;
    }

    /**
     * Render statistics with optimized parsing
     * @param {Array} stats - Array of stat strings
     * @returns {string} HTML string for stats
     */
    renderStats(stats) {
        return stats.map(stat => {
            const parts = stat.split(' ');
            const value = parts[0];
            const label = parts.slice(1).join(' ');
            
            return `
                <div class="feature-stat">
                    <span class="feature-stat-value">${value}</span>
                    <span class="feature-stat-label">${label}</span>
                </div>
            `;
        }).join('');
    }

    /**
     * Initialize intersection observer for animations
     */
    initializeAnimations() {
        if (this.observer) {
            this.observer.disconnect();
        }

        this.observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                    // Unobserve after animation to improve performance
                    this.observer.unobserve(entry.target);
                }
            });
        }, { 
            threshold: 0.1, 
            rootMargin: '0px 0px -50px 0px' 
        });

        // Apply initial styles and observe
        document.querySelectorAll('.feature-card').forEach(card => {
            card.style.opacity = '0';
            card.style.transform = 'translateY(20px)';
            card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
            this.observer.observe(card);
        });
    }

    /**
     * Clean up resources
     */
    destroy() {
        if (this.observer) {
            this.observer.disconnect();
        }
    }
}

/**
 * Optimized Performance Monitor
 */
class PerformanceMonitor {
    constructor() {
        this.startTime = performance.now();
        this.metrics = {};
        this.init();
    }

    init() {
        this.trackPageLoad();
        this.trackInteractions();
        this.trackWebVitals();
    }

    trackPageLoad() {
        if (document.readyState === 'complete') {
            this.recordLoadTime();
        } else {
            window.addEventListener('load', () => this.recordLoadTime());
        }
    }

    recordLoadTime() {
        const loadTime = performance.now() - this.startTime;
        this.metrics.pageLoad = loadTime;
        console.log(`Page loaded in ${loadTime.toFixed(2)}ms`);
    }

    trackInteractions() {
        // Use event delegation for better performance
        document.addEventListener('click', (e) => {
            const card = e.target.closest('.feature-card');
            if (card) {
                const title = card.querySelector('.feature-title')?.textContent;
                this.trackEvent('feature_card_click', title);
            }
        }, { passive: true });
    }

    trackWebVitals() {
        // Track Core Web Vitals
        const observer = new PerformanceObserver((entryList) => {
            for (const entry of entryList.getEntries()) {
                switch (entry.entryType) {
                    case 'largest-contentful-paint':
                        this.metrics.lcp = entry.startTime;
                        break;
                    case 'first-input':
                        this.metrics.fid = entry.processingStart - entry.startTime;
                        break;
                    case 'layout-shift':
                        if (!entry.hadRecentInput) {
                            this.metrics.cls = (this.metrics.cls || 0) + entry.value;
                        }
                        break;
                }
            }
        });

        observer.observe({ entryTypes: ['largest-contentful-paint', 'first-input', 'layout-shift'] });
    }

    trackEvent(event, data) {
        console.log(`Event: ${event}`, data);
        // Send to analytics service if available
        if (window.gtag) {
            window.gtag('event', event, { custom_parameter: data });
        }
    }

    getMetrics() {
        return this.metrics;
    }
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { FeatureRenderer, PerformanceMonitor };
}