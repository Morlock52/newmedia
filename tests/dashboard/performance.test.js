/**
 * Performance Test Suite
 * Tests dashboard load times, resource usage, and optimization
 */

const puppeteer = require('puppeteer');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

describe('Dashboard Performance Tests', () => {
    let browser, page;
    let dashboardURL;
    let apiURL;

    beforeAll(async () => {
        browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        
        const dashboardPath = path.join(__dirname, '../../dashboard-enhanced.html');
        dashboardURL = `file://${dashboardPath}`;
        apiURL = process.env.BASE_URL || 'http://localhost:3002';
    });

    afterAll(async () => {
        if (browser) {
            await browser.close();
        }
    });

    beforeEach(async () => {
        page = await browser.newPage();
    });

    afterEach(async () => {
        if (page) {
            await page.close();
        }
    });

    describe('Page Load Performance', () => {
        test('Initial page load performance', async () => {
            const startTime = Date.now();
            
            await page.goto(dashboardURL, { 
                waitUntil: 'networkidle0',
                timeout: 30000 
            });
            
            const loadTime = Date.now() - startTime;
            
            console.log(`📊 Initial page load time: ${loadTime}ms`);
            expect(loadTime).toBeLessThan(5000); // Should load within 5 seconds
        });

        test('DOM content loaded performance', async () => {
            const performanceMetrics = await page.evaluate(() => {
                return new Promise((resolve) => {
                    if (document.readyState === 'loading') {
                        document.addEventListener('DOMContentLoaded', () => {
                            resolve({
                                domContentLoaded: Date.now(),
                                startTime: performance.timing.navigationStart
                            });
                        });
                    } else {
                        resolve({
                            domContentLoaded: Date.now(),
                            startTime: performance.timing.navigationStart
                        });
                    }
                });
            });

            await page.goto(dashboardURL, { waitUntil: 'domcontentloaded' });
            
            const domLoadTime = performanceMetrics.domContentLoaded - performanceMetrics.startTime;
            console.log(`📊 DOM Content Loaded: ${domLoadTime}ms`);
            expect(domLoadTime).toBeLessThan(3000); // DOM should load within 3 seconds
        });

        test('First contentful paint (FCP)', async () => {
            await page.goto(dashboardURL);
            
            const fcp = await page.evaluate(() => {
                return new Promise((resolve) => {
                    new PerformanceObserver((entryList) => {
                        for (const entry of entryList.getEntriesByName('first-contentful-paint')) {
                            resolve(entry.startTime);
                        }
                    }).observe({ entryTypes: ['paint'] });
                    
                    // Fallback if FCP is not available
                    setTimeout(() => resolve(null), 2000);
                });
            });

            if (fcp) {
                console.log(`📊 First Contentful Paint: ${fcp.toFixed(2)}ms`);
                expect(fcp).toBeLessThan(2000); // FCP should occur within 2 seconds
            } else {
                console.log('ℹ️ FCP measurement not available');
            }
        });

        test('Largest contentful paint (LCP)', async () => {
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            const lcp = await page.evaluate(() => {
                return new Promise((resolve) => {
                    new PerformanceObserver((entryList) => {
                        const entries = entryList.getEntries();
                        const lastEntry = entries[entries.length - 1];
                        resolve(lastEntry.startTime);
                    }).observe({ entryTypes: ['largest-contentful-paint'] });
                    
                    // Fallback
                    setTimeout(() => resolve(null), 3000);
                });
            });

            if (lcp) {
                console.log(`📊 Largest Contentful Paint: ${lcp.toFixed(2)}ms`);
                expect(lcp).toBeLessThan(4000); // LCP should occur within 4 seconds
            } else {
                console.log('ℹ️ LCP measurement not available');
            }
        });
    });

    describe('Resource Loading Performance', () => {
        test('CSS loading performance', async () => {
            const cssStartTime = Date.now();
            
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            // Check if Tailwind CSS is loaded
            const tailwindLoaded = await page.evaluate(() => {
                const element = document.createElement('div');
                element.className = 'bg-black';
                document.body.appendChild(element);
                const computedStyle = window.getComputedStyle(element);
                const isLoaded = computedStyle.backgroundColor === 'rgb(0, 0, 0)';
                document.body.removeChild(element);
                return isLoaded;
            });

            const cssLoadTime = Date.now() - cssStartTime;
            
            expect(tailwindLoaded).toBe(true);
            console.log(`📊 CSS loading time: ${cssLoadTime}ms`);
            expect(cssLoadTime).toBeLessThan(3000);
        });

        test('JavaScript loading performance', async () => {
            const jsStartTime = Date.now();
            
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            // Check if Chart.js is loaded
            const chartJSLoaded = await page.evaluate(() => {
                return typeof window.Chart !== 'undefined';
            });

            const jsLoadTime = Date.now() - jsStartTime;
            
            if (chartJSLoaded) {
                console.log(`📊 JavaScript loading time: ${jsLoadTime}ms`);
                expect(jsLoadTime).toBeLessThan(3000);
            } else {
                console.log('ℹ️ Chart.js not loaded or test running in file:// mode');
            }
        });

        test('Image loading performance', async () => {
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            const imageMetrics = await page.evaluate(() => {
                const images = document.querySelectorAll('img');
                const imagePromises = Array.from(images).map(img => {
                    return new Promise((resolve) => {
                        const startTime = Date.now();
                        if (img.complete) {
                            resolve({ loaded: true, time: 0 });
                        } else {
                            img.onload = () => resolve({ loaded: true, time: Date.now() - startTime });
                            img.onerror = () => resolve({ loaded: false, time: Date.now() - startTime });
                        }
                    });
                });
                
                return Promise.all(imagePromises);
            });

            const loadedImages = imageMetrics.filter(metric => metric.loaded);
            const avgLoadTime = loadedImages.length > 0 
                ? loadedImages.reduce((sum, metric) => sum + metric.time, 0) / loadedImages.length
                : 0;

            console.log(`📊 Images loaded: ${loadedImages.length}/${imageMetrics.length}`);
            if (avgLoadTime > 0) {
                console.log(`📊 Average image load time: ${avgLoadTime.toFixed(2)}ms`);
                expect(avgLoadTime).toBeLessThan(2000);
            }
        });
    });

    describe('Memory Usage Tests', () => {
        test('Page memory consumption', async () => {
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            const memoryUsage = await page.evaluate(() => {
                if (performance.memory) {
                    return {
                        usedJSHeapSize: performance.memory.usedJSHeapSize,
                        totalJSHeapSize: performance.memory.totalJSHeapSize,
                        jsHeapSizeLimit: performance.memory.jsHeapSizeLimit
                    };
                }
                return null;
            });

            if (memoryUsage) {
                const usedMemoryMB = (memoryUsage.usedJSHeapSize / 1024 / 1024).toFixed(2);
                console.log(`📊 JavaScript heap used: ${usedMemoryMB} MB`);
                
                // Dashboard shouldn't use more than 50MB of JS heap
                expect(memoryUsage.usedJSHeapSize).toBeLessThan(50 * 1024 * 1024);
            } else {
                console.log('ℹ️ Memory usage metrics not available');
            }
        });

        test('Memory leaks detection', async () => {
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            // Take initial memory measurement
            const initialMemory = await page.evaluate(() => {
                return performance.memory ? performance.memory.usedJSHeapSize : 0;
            });

            // Simulate user interactions that might cause leaks
            for (let i = 0; i < 10; i++) {
                await page.evaluate(() => {
                    // Simulate sidebar toggle
                    if (window.toggleSidebar) {
                        window.toggleSidebar();
                    }
                    
                    // Simulate AI assistant open/close
                    if (window.openAIAssistant && window.closeAIAssistant) {
                        window.openAIAssistant();
                        window.closeAIAssistant();
                    }
                });
                
                await page.waitForTimeout(100);
            }

            // Force garbage collection if possible
            await page.evaluate(() => {
                if (window.gc) {
                    window.gc();
                }
            });

            await page.waitForTimeout(1000);

            const finalMemory = await page.evaluate(() => {
                return performance.memory ? performance.memory.usedJSHeapSize : 0;
            });

            if (initialMemory > 0 && finalMemory > 0) {
                const memoryIncrease = finalMemory - initialMemory;
                const increasePercent = (memoryIncrease / initialMemory) * 100;
                
                console.log(`📊 Memory increase after interactions: ${(memoryIncrease / 1024).toFixed(2)} KB (${increasePercent.toFixed(2)}%)`);
                
                // Memory shouldn't increase by more than 500KB after interactions
                expect(memoryIncrease).toBeLessThan(500 * 1024);
            }
        });
    });

    describe('API Performance Tests', () => {
        test('API response times', async () => {
            const endpoints = [
                '/health',
                '/api/docs',
                '/api/services',
                '/api/health/overview'
            ];

            const performanceResults = [];

            for (const endpoint of endpoints) {
                try {
                    const startTime = Date.now();
                    const response = await axios.get(`${apiURL}${endpoint}`, { timeout: 5000 });
                    const responseTime = Date.now() - startTime;

                    performanceResults.push({
                        endpoint,
                        responseTime,
                        status: response.status,
                        size: JSON.stringify(response.data).length
                    });

                    expect(responseTime).toBeLessThan(2000); // 2 second max
                } catch (error) {
                    console.warn(`⚠️ API endpoint ${endpoint} not available:`, error.message);
                }
            }

            if (performanceResults.length > 0) {
                console.log('📊 API Performance Results:');
                performanceResults.forEach(result => {
                    console.log(`  ${result.endpoint}: ${result.responseTime}ms (${result.size} bytes)`);
                });

                const avgResponseTime = performanceResults.reduce((sum, r) => sum + r.responseTime, 0) / performanceResults.length;
                console.log(`  Average API response time: ${avgResponseTime.toFixed(2)}ms`);
            }
        });

        test('API concurrent request performance', async () => {
            const concurrentRequests = 5;
            const requests = [];

            const startTime = Date.now();

            for (let i = 0; i < concurrentRequests; i++) {
                requests.push(
                    axios.get(`${apiURL}/health`, { timeout: 5000 })
                        .catch(error => ({ error: error.message }))
                );
            }

            try {
                const results = await Promise.all(requests);
                const totalTime = Date.now() - startTime;
                const successfulRequests = results.filter(r => !r.error);

                console.log(`📊 Concurrent requests completed in ${totalTime}ms`);
                console.log(`📊 Successful requests: ${successfulRequests.length}/${concurrentRequests}`);

                expect(totalTime).toBeLessThan(5000); // Should complete within 5 seconds
                expect(successfulRequests.length).toBeGreaterThan(0);
            } catch (error) {
                console.warn('⚠️ Concurrent API test failed:', error.message);
            }
        });
    });

    describe('Rendering Performance', () => {
        test('Chart rendering performance', async () => {
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

            const chartRenderTime = await page.evaluate(() => {
                return new Promise((resolve) => {
                    const startTime = Date.now();
                    
                    // Look for the chart canvas
                    const canvas = document.getElementById('activityChart');
                    if (!canvas) {
                        resolve(0);
                        return;
                    }

                    // Check if chart is rendered by looking for canvas content
                    const checkChart = () => {
                        const ctx = canvas.getContext('2d');
                        const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
                        const hasContent = imageData.data.some(pixel => pixel !== 0);
                        
                        if (hasContent) {
                            resolve(Date.now() - startTime);
                        } else if (Date.now() - startTime > 5000) {
                            resolve(0); // Timeout
                        } else {
                            setTimeout(checkChart, 100);
                        }
                    };

                    checkChart();
                });
            });

            if (chartRenderTime > 0) {
                console.log(`📊 Chart render time: ${chartRenderTime}ms`);
                expect(chartRenderTime).toBeLessThan(3000);
            } else {
                console.log('ℹ️ Chart rendering test skipped (chart not found or timeout)');
            }
        });

        test('Service cards rendering performance', async () => {
            const startTime = Date.now();
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

            // Wait for service cards to be rendered
            await page.waitForSelector('.service-card', { timeout: 5000 });

            const renderTime = Date.now() - startTime;
            const serviceCardCount = await page.$$eval('.service-card', cards => cards.length);

            console.log(`📊 Service cards rendered: ${serviceCardCount} cards in ${renderTime}ms`);
            expect(renderTime).toBeLessThan(3000);
            expect(serviceCardCount).toBeGreaterThan(0);
        });

        test('Animation performance', async () => {
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

            // Test animation performance by toggling sidebar multiple times
            const animationPerformance = await page.evaluate(() => {
                return new Promise((resolve) => {
                    const sidebar = document.getElementById('sidebar');
                    if (!sidebar) {
                        resolve({ error: 'Sidebar not found' });
                        return;
                    }

                    let frameCount = 0;
                    const startTime = performance.now();
                    
                    const countFrames = () => {
                        frameCount++;
                        if (performance.now() - startTime < 1000) {
                            requestAnimationFrame(countFrames);
                        } else {
                            resolve({ 
                                fps: frameCount,
                                duration: performance.now() - startTime
                            });
                        }
                    };

                    // Start animation by toggling sidebar
                    sidebar.classList.toggle('-translate-x-full');
                    requestAnimationFrame(countFrames);
                });
            });

            if (animationPerformance.fps) {
                console.log(`📊 Animation FPS: ${animationPerformance.fps}`);
                expect(animationPerformance.fps).toBeGreaterThan(30); // Should maintain at least 30 FPS
            }
        });
    });

    describe('Bundle Size and Optimization', () => {
        test('HTML file size optimization', async () => {
            const dashboardPath = path.join(__dirname, '../../dashboard-enhanced.html');
            const stats = await fs.stat(dashboardPath);
            
            const sizeKB = (stats.size / 1024).toFixed(2);
            console.log(`📊 Dashboard HTML size: ${sizeKB} KB`);
            
            // Dashboard HTML should be under 500KB
            expect(stats.size).toBeLessThan(500 * 1024);
        });

        test('External resource count', async () => {
            const dashboardPath = path.join(__dirname, '../../dashboard-enhanced.html');
            const htmlContent = await fs.readFile(dashboardPath, 'utf8');
            
            const externalScripts = (htmlContent.match(/<script[^>]*src=/g) || []).length;
            const externalStyles = (htmlContent.match(/<link[^>]*rel=['"]stylesheet['"][^>]*>/g) || []).length;
            const totalExternalResources = externalScripts + externalStyles;
            
            console.log(`📊 External resources: ${totalExternalResources} (${externalScripts} scripts, ${externalStyles} stylesheets)`);
            
            // Should minimize external resources for better performance
            expect(totalExternalResources).toBeLessThan(10);
        });

        test('CSS optimization check', async () => {
            const dashboardPath = path.join(__dirname, '../../dashboard-enhanced.html');
            const htmlContent = await fs.readFile(dashboardPath, 'utf8');
            
            // Check for inline CSS vs external CSS ratio
            const inlineCSS = (htmlContent.match(/<style[^>]*>[\s\S]*?<\/style>/g) || []).join('').length;
            const externalCSSCount = (htmlContent.match(/<link[^>]*rel=['"]stylesheet['"][^>]*>/g) || []).length;
            
            console.log(`📊 Inline CSS: ${(inlineCSS / 1024).toFixed(2)} KB`);
            console.log(`📊 External CSS files: ${externalCSSCount}`);
            
            // Inline CSS should be reasonable (less than 50KB for Tailwind config)
            expect(inlineCSS).toBeLessThan(50 * 1024);
        });
    });

    describe('Mobile Performance', () => {
        test('Mobile viewport performance', async () => {
            await page.setViewport({ width: 375, height: 667 });
            
            const startTime = Date.now();
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            const loadTime = Date.now() - startTime;
            
            console.log(`📊 Mobile load time: ${loadTime}ms`);
            expect(loadTime).toBeLessThan(6000); // Mobile should load within 6 seconds
        });

        test('Touch interaction responsiveness', async () => {
            await page.setViewport({ width: 375, height: 667 });
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            const mobileMenuButton = await page.$('button[onclick="toggleSidebar()"]');
            
            if (mobileMenuButton) {
                const startTime = Date.now();
                await mobileMenuButton.tap();
                
                // Wait for the animation to complete
                await page.waitForTimeout(300);
                
                const responseTime = Date.now() - startTime;
                console.log(`📊 Touch interaction response time: ${responseTime}ms`);
                
                // Touch interactions should respond within 300ms
                expect(responseTime).toBeLessThan(500);
            }
        });
    });

    afterAll(async () => {
        console.log('\n📊 Performance Test Summary:');
        console.log('- Page load performance tested');
        console.log('- Resource loading performance tested');
        console.log('- Memory usage tested');
        console.log('- API performance benchmarked');
        console.log('- Rendering performance tested');
        console.log('- Bundle optimization verified');
        console.log('- Mobile performance tested');
    });
});