// Advanced Performance Monitoring System
// Tracks Core Web Vitals, custom metrics, and provides optimization insights

class AdvancedPerformanceMonitor {
    constructor(options = {}) {
        this.options = {
            enableWebVitals: true,
            enableResourceTiming: true,
            enableNetworkTiming: true,
            enableUserTiming: true,
            samplingRate: 1.0,
            reportingUrl: '/api/performance',
            reportingInterval: 30000,
            ...options
        };
        
        this.metrics = new Map();
        this.observers = new Map();
        this.reportQueue = [];
        this.isReporting = false;
        
        this.init();
    }

    init() {
        console.log('🚀 Advanced Performance Monitor initializing...');
        
        if (this.options.enableWebVitals) {
            this.initWebVitals();
        }
        
        if (this.options.enableResourceTiming) {
            this.initResourceTiming();
        }
        
        if (this.options.enableNetworkTiming) {
            this.initNetworkTiming();
        }
        
        if (this.options.enableUserTiming) {
            this.initUserTiming();
        }
        
        this.initCustomMetrics();
        this.startReporting();
        this.initBudgetMonitoring();
        
        console.log('✅ Performance Monitor active');
    }

    // Core Web Vitals Measurement
    initWebVitals() {
        // Largest Contentful Paint (LCP)
        this.observeMetric('largest-contentful-paint', (entries) => {
            const lastEntry = entries[entries.length - 1];
            this.recordMetric('lcp', {
                value: lastEntry.startTime,
                element: lastEntry.element?.tagName || 'unknown',
                url: lastEntry.url || '',
                timestamp: Date.now()
            });
        });

        // First Input Delay (FID)
        this.observeMetric('first-input', (entries) => {
            const firstInput = entries[0];
            const fid = firstInput.processingStart - firstInput.startTime;
            this.recordMetric('fid', {
                value: fid,
                name: firstInput.name,
                target: firstInput.target?.tagName || 'unknown',
                timestamp: Date.now()
            });
        });

        // Cumulative Layout Shift (CLS)
        let clsValue = 0;
        let sessionValue = 0;
        let maxSessionValue = 0;
        
        this.observeMetric('layout-shift', (entries) => {
            for (const entry of entries) {
                if (!entry.hadRecentInput) {
                    sessionValue += entry.value;
                    maxSessionValue = Math.max(maxSessionValue, sessionValue);
                } else {
                    sessionValue = 0;
                }
                clsValue += entry.value;
            }
            
            this.recordMetric('cls', {
                value: maxSessionValue,
                totalShifts: clsValue,
                timestamp: Date.now()
            });
        });

        // Time to First Byte (TTFB)
        const navEntry = performance.getEntriesByType('navigation')[0];
        if (navEntry) {
            this.recordMetric('ttfb', {
                value: navEntry.responseStart - navEntry.requestStart,
                timestamp: Date.now()
            });
        }

        // First Contentful Paint (FCP)
        this.observeMetric('paint', (entries) => {
            for (const entry of entries) {
                if (entry.name === 'first-contentful-paint') {
                    this.recordMetric('fcp', {
                        value: entry.startTime,
                        timestamp: Date.now()
                    });
                }
            }
        });
    }

    // Resource Performance Monitoring
    initResourceTiming() {
        const processResources = (entries) => {
            for (const entry of entries) {
                const resourceMetrics = {
                    name: entry.name,
                    duration: entry.duration,
                    size: entry.transferSize || 0,
                    compressed: entry.encodedBodySize || 0,
                    uncompressed: entry.decodedBodySize || 0,
                    type: this.getResourceType(entry.name),
                    cached: entry.transferSize === 0 && entry.decodedBodySize > 0,
                    timing: {
                        dns: entry.domainLookupEnd - entry.domainLookupStart,
                        tcp: entry.connectEnd - entry.connectStart,
                        ssl: entry.secureConnectionStart > 0 ? 
                             entry.connectEnd - entry.secureConnectionStart : 0,
                        request: entry.responseStart - entry.requestStart,
                        response: entry.responseEnd - entry.responseStart,
                        total: entry.responseEnd - entry.startTime
                    },
                    timestamp: Date.now()
                };
                
                this.recordMetric('resource', resourceMetrics);
            }
        };

        // Process existing resources
        processResources(performance.getEntriesByType('resource'));

        // Observe new resources
        this.observeMetric('resource', processResources);
    }

    // Network Performance Monitoring
    initNetworkTiming() {
        // Monitor network connection
        if ('connection' in navigator) {
            const connection = navigator.connection;
            this.recordMetric('network', {
                effectiveType: connection.effectiveType,
                downlink: connection.downlink,
                rtt: connection.rtt,
                saveData: connection.saveData,
                timestamp: Date.now()
            });

            connection.addEventListener('change', () => {
                this.recordMetric('network-change', {
                    effectiveType: connection.effectiveType,
                    downlink: connection.downlink,
                    rtt: connection.rtt,
                    timestamp: Date.now()
                });
            });
        }

        // Monitor service worker performance
        if ('serviceWorker' in navigator && navigator.serviceWorker.controller) {
            this.monitorServiceWorkerPerformance();
        }
    }

    // User Timing API Integration
    initUserTiming() {
        this.observeMetric('measure', (entries) => {
            for (const entry of entries) {
                this.recordMetric('user-timing', {
                    name: entry.name,
                    duration: entry.duration,
                    timestamp: Date.now()
                });
            }
        });
    }

    // Custom Application Metrics
    initCustomMetrics() {
        // Memory usage monitoring
        if ('memory' in performance) {
            const measureMemory = () => {
                this.recordMetric('memory', {
                    used: performance.memory.usedJSHeapSize,
                    total: performance.memory.totalJSHeapSize,
                    limit: performance.memory.jsHeapSizeLimit,
                    utilization: performance.memory.usedJSHeapSize / performance.memory.totalJSHeapSize,
                    timestamp: Date.now()
                });
            };

            measureMemory();
            setInterval(measureMemory, 10000); // Every 10 seconds
        }

        // Bundle size analysis
        this.analyzeBundleSize();

        // Service availability monitoring
        this.monitorServiceAvailability();

        // Long task monitoring
        this.observeMetric('longtask', (entries) => {
            for (const entry of entries) {
                this.recordMetric('long-task', {
                    duration: entry.duration,
                    startTime: entry.startTime,
                    attribution: entry.attribution || [],
                    timestamp: Date.now()
                });
            }
        });
    }

    // Performance Budget Monitoring
    initBudgetMonitoring() {
        const budgets = {
            lcp: 2500,      // 2.5 seconds
            fid: 100,       // 100 milliseconds
            cls: 0.1,       // 0.1 units
            fcp: 1800,      // 1.8 seconds
            ttfb: 800,      // 800 milliseconds
            totalJS: 200,   // 200KB
            totalCSS: 100,  // 100KB
            totalImages: 500 // 500KB
        };

        this.budgets = budgets;
        this.budgetViolations = [];
    }

    // Helper Methods
    observeMetric(type, callback) {
        try {
            const observer = new PerformanceObserver((list) => {
                if (Math.random() <= this.options.samplingRate) {
                    callback(list.getEntries());
                }
            });
            
            observer.observe({ entryTypes: [type] });
            this.observers.set(type, observer);
        } catch (error) {
            console.warn(`Performance Observer for ${type} not supported:`, error);
        }
    }

    recordMetric(type, data) {
        if (!this.metrics.has(type)) {
            this.metrics.set(type, []);
        }
        
        const metrics = this.metrics.get(type);
        metrics.push(data);
        
        // Keep only recent metrics to prevent memory bloat
        if (metrics.length > 100) {
            metrics.shift();
        }
        
        // Check against performance budgets
        this.checkBudget(type, data);
        
        // Queue for reporting
        this.queueForReporting(type, data);
    }

    checkBudget(type, data) {
        const budget = this.budgets[type];
        if (budget && data.value > budget) {
            const violation = {
                type,
                budget,
                actual: data.value,
                excess: data.value - budget,
                timestamp: Date.now()
            };
            
            this.budgetViolations.push(violation);
            console.warn(`🚨 Performance budget violation: ${type}`, violation);
            
            // Trigger immediate reporting for critical violations
            if (type === 'lcp' || type === 'cls') {
                this.reportViolation(violation);
            }
        }
    }

    queueForReporting(type, data) {
        this.reportQueue.push({ type, data, timestamp: Date.now() });
        
        // Prevent queue from growing too large
        if (this.reportQueue.length > 1000) {
            this.reportQueue = this.reportQueue.slice(-500);
        }
    }

    // Reporting System
    startReporting() {
        setInterval(() => {
            if (!this.isReporting && this.reportQueue.length > 0) {
                this.sendReport();
            }
        }, this.options.reportingInterval);
    }

    async sendReport() {
        if (this.reportQueue.length === 0) return;
        
        this.isReporting = true;
        
        const report = {
            timestamp: Date.now(),
            url: window.location.href,
            userAgent: navigator.userAgent,
            metrics: this.reportQueue.splice(0, 100), // Send in batches
            budgetViolations: this.budgetViolations.splice(0),
            summary: this.generateSummary()
        };
        
        try {
            if (navigator.sendBeacon) {
                // Use sendBeacon for reliability
                navigator.sendBeacon(
                    this.options.reportingUrl,
                    JSON.stringify(report)
                );
            } else {
                // Fallback to fetch
                await fetch(this.options.reportingUrl, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(report)
                });
            }
        } catch (error) {
            console.error('Failed to send performance report:', error);
        } finally {
            this.isReporting = false;
        }
    }

    async reportViolation(violation) {
        try {
            await fetch('/api/performance/violation', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(violation)
            });
        } catch (error) {
            console.error('Failed to report performance violation:', error);
        }
    }

    // Analysis Methods
    generateSummary() {
        const summary = {};
        
        for (const [type, metrics] of this.metrics) {
            if (metrics.length === 0) continue;
            
            const values = metrics.map(m => m.value).filter(v => v != null);
            if (values.length === 0) continue;
            
            summary[type] = {
                count: values.length,
                min: Math.min(...values),
                max: Math.max(...values),
                avg: values.reduce((a, b) => a + b, 0) / values.length,
                median: this.calculateMedian(values),
                p95: this.calculatePercentile(values, 95),
                p99: this.calculatePercentile(values, 99)
            };
        }
        
        return summary;
    }

    calculatePerformanceScore() {
        const lcp = this.getLatestMetric('lcp');
        const fid = this.getLatestMetric('fid');
        const cls = this.getLatestMetric('cls');
        const fcp = this.getLatestMetric('fcp');
        const ttfb = this.getLatestMetric('ttfb');
        
        if (!lcp || !fid || !cls) return null;
        
        // Lighthouse scoring algorithm (simplified)
        const lcpScore = this.calculateLCPScore(lcp.value);
        const fidScore = this.calculateFIDScore(fid.value);
        const clsScore = this.calculateCLSScore(cls.value);
        const fcpScore = fcp ? this.calculateFCPScore(fcp.value) : 100;
        const ttfbScore = ttfb ? this.calculateTTFBScore(ttfb.value) : 100;
        
        // Weighted average
        const score = Math.round(
            lcpScore * 0.25 +
            fidScore * 0.25 +
            clsScore * 0.25 +
            fcpScore * 0.15 +
            ttfbScore * 0.10
        );
        
        return {
            overall: score,
            lcp: lcpScore,
            fid: fidScore,
            cls: clsScore,
            fcp: fcpScore,
            ttfb: ttfbScore
        };
    }

    // Utility Methods
    getLatestMetric(type) {
        const metrics = this.metrics.get(type);
        return metrics && metrics.length > 0 ? metrics[metrics.length - 1] : null;
    }

    calculateMedian(values) {
        const sorted = [...values].sort((a, b) => a - b);
        const mid = Math.floor(sorted.length / 2);
        return sorted.length % 2 === 0 
            ? (sorted[mid - 1] + sorted[mid]) / 2 
            : sorted[mid];
    }

    calculatePercentile(values, percentile) {
        const sorted = [...values].sort((a, b) => a - b);
        const index = Math.ceil((percentile / 100) * sorted.length) - 1;
        return sorted[Math.max(0, index)];
    }

    getResourceType(url) {
        if (url.match(/\.(js|mjs)$/)) return 'script';
        if (url.match(/\.css$/)) return 'stylesheet';
        if (url.match(/\.(png|jpg|jpeg|gif|svg|webp)$/)) return 'image';
        if (url.match(/\.(woff|woff2|ttf|otf)$/)) return 'font';
        return 'other';
    }

    // Scoring functions (based on Lighthouse)
    calculateLCPScore(lcp) {
        if (lcp <= 2500) return 100;
        if (lcp <= 4000) return Math.round(100 - ((lcp - 2500) / 1500) * 50);
        return Math.max(0, Math.round(50 - ((lcp - 4000) / 1000) * 10));
    }

    calculateFIDScore(fid) {
        if (fid <= 100) return 100;
        if (fid <= 300) return Math.round(100 - ((fid - 100) / 200) * 40);
        return Math.max(0, Math.round(60 - ((fid - 300) / 100) * 10));
    }

    calculateCLSScore(cls) {
        if (cls <= 0.1) return 100;
        if (cls <= 0.25) return Math.round(100 - ((cls - 0.1) / 0.15) * 25);
        return Math.max(0, Math.round(75 - ((cls - 0.25) / 0.1) * 15));
    }

    calculateFCPScore(fcp) {
        if (fcp <= 1800) return 100;
        if (fcp <= 3000) return Math.round(100 - ((fcp - 1800) / 1200) * 40);
        return Math.max(0, Math.round(60 - ((fcp - 3000) / 1000) * 10));
    }

    calculateTTFBScore(ttfb) {
        if (ttfb <= 800) return 100;
        if (ttfb <= 1800) return Math.round(100 - ((ttfb - 800) / 1000) * 30);
        return Math.max(0, Math.round(70 - ((ttfb - 1800) / 1000) * 10));
    }

    // Advanced Analysis
    analyzeBundleSize() {
        const scriptResources = this.metrics.has('resource') 
            ? this.metrics.get('resource').filter(r => r.type === 'script')
            : [];
        
        const totalJS = scriptResources.reduce((sum, r) => sum + (r.size || 0), 0);
        const totalCompressed = scriptResources.reduce((sum, r) => sum + (r.compressed || 0), 0);
        
        this.recordMetric('bundle-size', {
            totalJS,
            totalCompressed,
            compressionRatio: totalJS > 0 ? totalCompressed / totalJS : 0,
            scriptCount: scriptResources.length,
            timestamp: Date.now()
        });
    }

    monitorServiceAvailability() {
        const services = [
            { name: 'jellyfin', port: 8096 },
            { name: 'sonarr', port: 8989 },
            { name: 'radarr', port: 7878 },
            { name: 'prowlarr', port: 9696 }
        ];

        const checkServices = async () => {
            for (const service of services) {
                const startTime = performance.now();
                
                try {
                    await fetch(`http://localhost:${service.port}`, {
                        mode: 'no-cors',
                        signal: AbortSignal.timeout(5000)
                    });
                    
                    this.recordMetric('service-availability', {
                        service: service.name,
                        available: true,
                        responseTime: performance.now() - startTime,
                        timestamp: Date.now()
                    });
                } catch (error) {
                    this.recordMetric('service-availability', {
                        service: service.name,
                        available: false,
                        error: error.message,
                        responseTime: performance.now() - startTime,
                        timestamp: Date.now()
                    });
                }
            }
        };

        // Initial check
        checkServices();
        
        // Periodic checks
        setInterval(checkServices, 30000);
    }

    monitorServiceWorkerPerformance() {
        const channel = new MessageChannel();
        
        channel.port1.onmessage = (event) => {
            if (event.data.type === 'SW_PERFORMANCE') {
                this.recordMetric('service-worker', event.data.metrics);
            }
        };
        
        navigator.serviceWorker.controller.postMessage({
            type: 'PERFORMANCE_REQUEST'
        }, [channel.port2]);
    }

    // Public API
    mark(name) {
        performance.mark(name);
    }

    measure(name, startMark, endMark) {
        performance.measure(name, startMark, endMark);
    }

    getMetrics(type) {
        return this.metrics.get(type) || [];
    }

    getPerformanceScore() {
        return this.calculatePerformanceScore();
    }

    getBudgetStatus() {
        return {
            budgets: this.budgets,
            violations: this.budgetViolations,
            status: this.budgetViolations.length === 0 ? 'good' : 'needs-improvement'
        };
    }

    // Cleanup
    destroy() {
        for (const observer of this.observers.values()) {
            observer.disconnect();
        }
        this.observers.clear();
        this.metrics.clear();
        console.log('Performance Monitor destroyed');
    }
}

// Global instance
window.PerformanceMonitor = AdvancedPerformanceMonitor;

// Auto-initialize if running in browser
if (typeof window !== 'undefined') {
    window.performanceMonitor = new AdvancedPerformanceMonitor();
}

export default AdvancedPerformanceMonitor;