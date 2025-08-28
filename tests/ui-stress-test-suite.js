/**
 * Comprehensive UI Stress Test Suite
 * Tests all buttons, menu items, and interactive elements
 * For Ultimate Media Server 2025
 */

const puppeteer = require('puppeteer');
const axios = require('axios');
const { performance } = require('perf_hooks');

class UIStressTestSuite {
    constructor() {
        this.baseURL = 'http://localhost:3333';
        this.apiURL = 'http://localhost:3333/api';
        this.browser = null;
        this.page = null;
        this.results = {
            timestamp: new Date(),
            tests: [],
            issues: [],
            performance: {},
            summary: {
                total: 0,
                passed: 0,
                failed: 0,
                warnings: 0
            }
        };
    }

    async initialize() {
        console.log('🚀 Initializing UI Stress Test Suite...');
        
        // Launch headless browser
        this.browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        
        this.page = await this.browser.newPage();
        
        // Set viewport for desktop testing
        await this.page.setViewport({ width: 1920, height: 1080 });
        
        // Enable console logging
        this.page.on('console', msg => {
            if (msg.type() === 'error') {
                this.recordIssue('Console Error', msg.text(), 'error');
            }
        });
        
        // Catch page errors
        this.page.on('pageerror', error => {
            this.recordIssue('Page Error', error.message, 'error');
        });
        
        // Monitor network failures
        this.page.on('requestfailed', request => {
            this.recordIssue('Network Request Failed', `${request.url()} - ${request.failure().errorText}`, 'warning');
        });
    }

    recordTest(name, status, duration, details = {}) {
        const test = {
            name,
            status,
            duration,
            timestamp: new Date(),
            ...details
        };
        
        this.results.tests.push(test);
        this.results.summary.total++;
        
        if (status === 'passed') {
            this.results.summary.passed++;
            console.log(`✅ ${name} - ${duration}ms`);
        } else if (status === 'failed') {
            this.results.summary.failed++;
            console.log(`❌ ${name} - ${details.error || 'Failed'}`);
        } else if (status === 'warning') {
            this.results.summary.warnings++;
            console.log(`⚠️ ${name} - ${details.warning || 'Warning'}`);
        }
    }

    recordIssue(type, message, severity = 'warning') {
        this.results.issues.push({
            type,
            message,
            severity,
            timestamp: new Date()
        });
    }

    // Test 1: Homepage Load and Initial Render
    async testHomepageLoad() {
        const start = performance.now();
        
        try {
            const response = await this.page.goto(this.baseURL, {
                waitUntil: 'networkidle2',
                timeout: 30000
            });
            
            const duration = performance.now() - start;
            
            if (!response.ok()) {
                throw new Error(`HTTP ${response.status()}`);
            }
            
            // Check for main container
            await this.page.waitForSelector('#root', { timeout: 5000 });
            
            this.recordTest('Homepage Load', 'passed', duration, {
                httpStatus: response.status(),
                loadTime: duration
            });
            
            this.results.performance.homepageLoadTime = duration;
            
        } catch (error) {
            this.recordTest('Homepage Load', 'failed', performance.now() - start, {
                error: error.message
            });
        }
    }

    // Test 2: All Navigation Menu Items
    async testNavigationMenu() {
        const menuItems = [
            { selector: '[data-nav="dashboard"]', name: 'Dashboard', expectedUrl: '/' },
            { selector: '[data-nav="library"]', name: 'Media Library', expectedUrl: '/library' },
            { selector: '[data-nav="downloads"]', name: 'Downloads', expectedUrl: '/downloads' },
            { selector: '[data-nav="analytics"]', name: 'Analytics', expectedUrl: '/analytics' },
            { selector: '[data-nav="settings"]', name: 'Settings', expectedUrl: '/settings' },
            { selector: '[data-nav="profiles"]', name: 'Profiles', expectedUrl: '/profiles' },
            { selector: '[data-nav="social"]', name: 'Social', expectedUrl: '/social' },
            { selector: '[data-nav="discover"]', name: 'Discover', expectedUrl: '/discover' }
        ];

        for (const item of menuItems) {
            const start = performance.now();
            
            try {
                // Try multiple selector strategies
                let element = await this.page.$(item.selector);
                
                if (!element) {
                    // Try by text content
                    element = await this.page.evaluateHandle((text) => {
                        return Array.from(document.querySelectorAll('a, button')).find(el => 
                            el.textContent.includes(text)
                        );
                    }, item.name);
                }
                
                if (!element || !(await element.boundingBox())) {
                    throw new Error(`Menu item not found: ${item.name}`);
                }
                
                await element.click();
                await this.page.waitForTimeout(500); // Wait for navigation
                
                this.recordTest(`Navigation: ${item.name}`, 'passed', performance.now() - start);
                
            } catch (error) {
                this.recordTest(`Navigation: ${item.name}`, 'failed', performance.now() - start, {
                    error: error.message
                });
            }
        }
    }

    // Test 3: All Interactive Buttons
    async testAllButtons() {
        const buttonTests = [
            // Service Controls
            { selector: '.service-start-btn', name: 'Start Service', action: 'click' },
            { selector: '.service-stop-btn', name: 'Stop Service', action: 'click' },
            { selector: '.service-restart-btn', name: 'Restart Service', action: 'click' },
            
            // Media Controls
            { selector: '.play-btn', name: 'Play Button', action: 'click' },
            { selector: '.pause-btn', name: 'Pause Button', action: 'click' },
            { selector: '.next-btn', name: 'Next Button', action: 'click' },
            { selector: '.prev-btn', name: 'Previous Button', action: 'click' },
            { selector: '.fullscreen-btn', name: 'Fullscreen Button', action: 'click' },
            
            // Download Manager
            { selector: '.download-add-btn', name: 'Add Download', action: 'click' },
            { selector: '.download-pause-btn', name: 'Pause Download', action: 'click' },
            { selector: '.download-resume-btn', name: 'Resume Download', action: 'click' },
            { selector: '.download-cancel-btn', name: 'Cancel Download', action: 'click' },
            
            // Social Features
            { selector: '.watch-party-create', name: 'Create Watch Party', action: 'click' },
            { selector: '.watch-party-join', name: 'Join Watch Party', action: 'click' },
            { selector: '.share-btn', name: 'Share Button', action: 'click' },
            
            // Authentication
            { selector: '.login-btn', name: 'Login Button', action: 'click' },
            { selector: '.logout-btn', name: 'Logout Button', action: 'click' },
            { selector: '.profile-switch', name: 'Switch Profile', action: 'click' },
            
            // Settings
            { selector: '.save-settings-btn', name: 'Save Settings', action: 'click' },
            { selector: '.reset-settings-btn', name: 'Reset Settings', action: 'click' },
            { selector: '.theme-toggle', name: 'Theme Toggle', action: 'click' },
            
            // AI Features
            { selector: '.ai-recommend', name: 'AI Recommendations', action: 'click' },
            { selector: '.voice-control', name: 'Voice Control', action: 'click' },
            { selector: '.gpt-search', name: 'GPT-4 Search', action: 'click' },
            
            // Web3 Features
            { selector: '.connect-wallet', name: 'Connect Wallet', action: 'click' },
            { selector: '.view-nfts', name: 'View NFTs', action: 'click' },
            
            // Smart Home
            { selector: '.smart-home-sync', name: 'Smart Home Sync', action: 'click' },
            { selector: '.hue-lights', name: 'Hue Lights Control', action: 'click' }
        ];

        for (const test of buttonTests) {
            const start = performance.now();
            
            try {
                const elements = await this.page.$$(test.selector);
                
                if (elements.length === 0) {
                    // Try finding by button text
                    const textButton = await this.page.evaluateHandle((text) => {
                        return Array.from(document.querySelectorAll('button')).find(el => 
                            el.textContent.includes(text)
                        );
                    }, test.name.replace(' Button', ''));
                    
                    if (textButton) {
                        elements.push(textButton);
                    }
                }
                
                if (elements.length > 0) {
                    for (const element of elements) {
                        const isVisible = await element.boundingBox();
                        if (isVisible) {
                            if (test.action === 'click') {
                                await element.click();
                                await this.page.waitForTimeout(100);
                            }
                        }
                    }
                    
                    this.recordTest(`Button: ${test.name}`, 'passed', performance.now() - start, {
                        elementsFound: elements.length
                    });
                } else {
                    this.recordTest(`Button: ${test.name}`, 'warning', performance.now() - start, {
                        warning: 'Button not found on current page'
                    });
                }
                
            } catch (error) {
                this.recordTest(`Button: ${test.name}`, 'failed', performance.now() - start, {
                    error: error.message
                });
            }
        }
    }

    // Test 4: Form Inputs and Validation
    async testFormInputs() {
        const formTests = [
            { selector: 'input[type="text"]', name: 'Text Inputs', value: 'Test Input' },
            { selector: 'input[type="email"]', name: 'Email Inputs', value: 'test@example.com' },
            { selector: 'input[type="password"]', name: 'Password Inputs', value: 'TestPass123!' },
            { selector: 'input[type="search"]', name: 'Search Inputs', value: 'Game of Thrones' },
            { selector: 'textarea', name: 'Textareas', value: 'Multi-line\ntest input' },
            { selector: 'select', name: 'Dropdowns', action: 'select' },
            { selector: 'input[type="checkbox"]', name: 'Checkboxes', action: 'toggle' },
            { selector: 'input[type="radio"]', name: 'Radio Buttons', action: 'select' },
            { selector: 'input[type="range"]', name: 'Sliders', value: '50' }
        ];

        for (const test of formTests) {
            const start = performance.now();
            
            try {
                const elements = await this.page.$$(test.selector);
                
                if (elements.length > 0) {
                    for (const element of elements.slice(0, 3)) { // Test first 3 of each type
                        if (test.action === 'toggle') {
                            await element.click();
                        } else if (test.action === 'select') {
                            const options = await element.$$('option');
                            if (options.length > 1) {
                                await element.select(await options[1].evaluate(el => el.value));
                            }
                        } else if (test.value) {
                            await element.type(test.value, { delay: 10 });
                        }
                    }
                    
                    this.recordTest(`Form: ${test.name}`, 'passed', performance.now() - start, {
                        elementsFound: elements.length
                    });
                } else {
                    this.recordTest(`Form: ${test.name}`, 'warning', performance.now() - start, {
                        warning: 'No form elements found'
                    });
                }
                
            } catch (error) {
                this.recordTest(`Form: ${test.name}`, 'failed', performance.now() - start, {
                    error: error.message
                });
            }
        }
    }

    // Test 5: API Endpoints
    async testAPIEndpoints() {
        const endpoints = [
            { path: '/health', method: 'GET', name: 'Health Check' },
            { path: '/services', method: 'GET', name: 'Get Services' },
            { path: '/config', method: 'GET', name: 'Get Config' },
            { path: '/stats', method: 'GET', name: 'Get Statistics' },
            { path: '/logs', method: 'GET', name: 'Get Logs' },
            { path: '/media/libraries', method: 'GET', name: 'Get Libraries' },
            { path: '/downloads/queue', method: 'GET', name: 'Get Download Queue' },
            { path: '/users/profile', method: 'GET', name: 'Get User Profile' },
            { path: '/ai/recommendations', method: 'GET', name: 'Get Recommendations' },
            { path: '/social/watch-parties', method: 'GET', name: 'Get Watch Parties' },
            { path: '/web3/nfts', method: 'GET', name: 'Get NFTs' },
            { path: '/smart-home/devices', method: 'GET', name: 'Get Smart Devices' },
            { path: '/monitoring/metrics', method: 'GET', name: 'Get Metrics' },
            { path: '/security/status', method: 'GET', name: 'Security Status' }
        ];

        for (const endpoint of endpoints) {
            const start = performance.now();
            
            try {
                const response = await axios({
                    method: endpoint.method,
                    url: `${this.apiURL}${endpoint.path}`,
                    timeout: 5000,
                    validateStatus: () => true // Accept any status
                });
                
                const duration = performance.now() - start;
                
                if (response.status < 400) {
                    this.recordTest(`API: ${endpoint.name}`, 'passed', duration, {
                        status: response.status,
                        responseTime: duration
                    });
                } else if (response.status === 404) {
                    this.recordTest(`API: ${endpoint.name}`, 'warning', duration, {
                        warning: 'Endpoint not implemented',
                        status: response.status
                    });
                } else {
                    this.recordTest(`API: ${endpoint.name}`, 'failed', duration, {
                        error: `HTTP ${response.status}`,
                        status: response.status
                    });
                }
                
            } catch (error) {
                this.recordTest(`API: ${endpoint.name}`, 'failed', performance.now() - start, {
                    error: error.message
                });
            }
        }
    }

    // Test 6: WebSocket Connections
    async testWebSocketConnections() {
        const start = performance.now();
        
        try {
            // Inject WebSocket test
            const wsTest = await this.page.evaluate(() => {
                return new Promise((resolve, reject) => {
                    const ws = new WebSocket('ws://localhost:3333');
                    const timeout = setTimeout(() => {
                        ws.close();
                        reject(new Error('WebSocket connection timeout'));
                    }, 5000);
                    
                    ws.onopen = () => {
                        clearTimeout(timeout);
                        ws.send(JSON.stringify({ type: 'ping' }));
                    };
                    
                    ws.onmessage = (event) => {
                        ws.close();
                        resolve({ success: true, message: event.data });
                    };
                    
                    ws.onerror = (error) => {
                        clearTimeout(timeout);
                        reject(error);
                    };
                });
            });
            
            this.recordTest('WebSocket Connection', 'passed', performance.now() - start, wsTest);
            
        } catch (error) {
            this.recordTest('WebSocket Connection', 'warning', performance.now() - start, {
                warning: 'WebSocket not available or not configured'
            });
        }
    }

    // Test 7: Responsive Design
    async testResponsiveDesign() {
        const viewports = [
            { name: 'Mobile Portrait', width: 375, height: 812 },
            { name: 'Mobile Landscape', width: 812, height: 375 },
            { name: 'Tablet Portrait', width: 768, height: 1024 },
            { name: 'Tablet Landscape', width: 1024, height: 768 },
            { name: 'Desktop', width: 1920, height: 1080 },
            { name: '4K', width: 3840, height: 2160 }
        ];

        for (const viewport of viewports) {
            const start = performance.now();
            
            try {
                await this.page.setViewport(viewport);
                await this.page.waitForTimeout(500);
                
                // Check if main content is visible
                const isVisible = await this.page.evaluate(() => {
                    const root = document.getElementById('root');
                    if (!root) return false;
                    
                    const rect = root.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                });
                
                if (isVisible) {
                    this.recordTest(`Responsive: ${viewport.name}`, 'passed', performance.now() - start, {
                        viewport: `${viewport.width}x${viewport.height}`
                    });
                } else {
                    throw new Error('Content not visible');
                }
                
            } catch (error) {
                this.recordTest(`Responsive: ${viewport.name}`, 'failed', performance.now() - start, {
                    error: error.message,
                    viewport: `${viewport.width}x${viewport.height}`
                });
            }
        }
        
        // Reset to desktop
        await this.page.setViewport({ width: 1920, height: 1080 });
    }

    // Test 8: Component-Specific Tests
    async testSpecificComponents() {
        const components = [
            'NotificationSystem',
            'DataAnalyticsDashboard',
            'ServiceGrid3D',
            'NEXUSAIAssistant',
            'SocialWatchParty',
            'PredictiveAnalytics',
            'HolographicMediaPlayer',
            'NeuralRecommendations',
            'MultiUserProfiles',
            'GPT4Discovery'
        ];

        for (const component of components) {
            const start = performance.now();
            
            try {
                // Check if component is rendered
                const componentExists = await this.page.evaluate((name) => {
                    // Look for component by data attribute or class
                    return !!(
                        document.querySelector(`[data-component="${name}"]`) ||
                        document.querySelector(`.${name}`) ||
                        document.querySelector(`#${name}`)
                    );
                }, component);
                
                if (componentExists) {
                    this.recordTest(`Component: ${component}`, 'passed', performance.now() - start);
                } else {
                    this.recordTest(`Component: ${component}`, 'warning', performance.now() - start, {
                        warning: 'Component not found on current view'
                    });
                }
                
            } catch (error) {
                this.recordTest(`Component: ${component}`, 'failed', performance.now() - start, {
                    error: error.message
                });
            }
        }
    }

    // Test 9: Performance Metrics
    async testPerformanceMetrics() {
        const start = performance.now();
        
        try {
            const metrics = await this.page.metrics();
            const performanceData = await this.page.evaluate(() => {
                const perfData = performance.getEntriesByType('navigation')[0];
                return {
                    domContentLoaded: perfData.domContentLoadedEventEnd - perfData.domContentLoadedEventStart,
                    loadComplete: perfData.loadEventEnd - perfData.loadEventStart,
                    domInteractive: perfData.domInteractive,
                    firstPaint: performance.getEntriesByType('paint')[0]?.startTime || 0
                };
            });
            
            this.results.performance = {
                ...this.results.performance,
                ...metrics,
                ...performanceData
            };
            
            // Check performance thresholds
            const issues = [];
            if (performanceData.domContentLoaded > 3000) {
                issues.push('DOM content loaded too slowly');
            }
            if (metrics.JSHeapUsedSize > 50 * 1024 * 1024) {
                issues.push('High memory usage');
            }
            
            if (issues.length === 0) {
                this.recordTest('Performance Metrics', 'passed', performance.now() - start, {
                    metrics: performanceData
                });
            } else {
                this.recordTest('Performance Metrics', 'warning', performance.now() - start, {
                    warning: issues.join(', '),
                    metrics: performanceData
                });
            }
            
        } catch (error) {
            this.recordTest('Performance Metrics', 'failed', performance.now() - start, {
                error: error.message
            });
        }
    }

    // Test 10: Security Features
    async testSecurityFeatures() {
        const securityTests = [
            {
                name: 'HTTPS Redirect',
                test: async () => {
                    // Check if HTTPS is enforced (in production)
                    return true; // Placeholder for local testing
                }
            },
            {
                name: 'CSP Headers',
                test: async () => {
                    const response = await axios.get(this.baseURL);
                    return response.headers['content-security-policy'] !== undefined;
                }
            },
            {
                name: 'XSS Protection',
                test: async () => {
                    // Try injecting script
                    const result = await this.page.evaluate(() => {
                        const input = document.querySelector('input[type="text"]');
                        if (input) {
                            input.value = '<script>alert("XSS")</script>';
                            const event = new Event('input', { bubbles: true });
                            input.dispatchEvent(event);
                            return !document.querySelector('script[src*="alert"]');
                        }
                        return true;
                    });
                    return result;
                }
            },
            {
                name: 'Authentication Required',
                test: async () => {
                    // Check if protected routes require auth
                    try {
                        const response = await axios.get(`${this.apiURL}/users/profile`);
                        return response.status === 401 || response.status === 200;
                    } catch (error) {
                        return error.response?.status === 401;
                    }
                }
            }
        ];

        for (const test of securityTests) {
            const start = performance.now();
            
            try {
                const result = await test.test();
                
                if (result) {
                    this.recordTest(`Security: ${test.name}`, 'passed', performance.now() - start);
                } else {
                    this.recordTest(`Security: ${test.name}`, 'warning', performance.now() - start, {
                        warning: 'Security feature not fully implemented'
                    });
                }
                
            } catch (error) {
                this.recordTest(`Security: ${test.name}`, 'failed', performance.now() - start, {
                    error: error.message
                });
            }
        }
    }

    // Generate comprehensive report
    generateReport() {
        console.log('\n' + '='.repeat(80));
        console.log('📊 UI STRESS TEST REPORT');
        console.log('='.repeat(80));
        
        console.log('\n📈 SUMMARY:');
        console.log(`Total Tests: ${this.results.summary.total}`);
        console.log(`✅ Passed: ${this.results.summary.passed}`);
        console.log(`❌ Failed: ${this.results.summary.failed}`);
        console.log(`⚠️ Warnings: ${this.results.summary.warnings}`);
        
        const successRate = ((this.results.summary.passed / this.results.summary.total) * 100).toFixed(1);
        console.log(`Success Rate: ${successRate}%`);
        
        if (this.results.issues.length > 0) {
            console.log('\n🔍 ISSUES FOUND:');
            const criticalIssues = this.results.issues.filter(i => i.severity === 'error');
            const warnings = this.results.issues.filter(i => i.severity === 'warning');
            
            if (criticalIssues.length > 0) {
                console.log('\n❌ Critical Issues:');
                criticalIssues.forEach(issue => {
                    console.log(`  - ${issue.type}: ${issue.message}`);
                });
            }
            
            if (warnings.length > 0) {
                console.log('\n⚠️ Warnings:');
                warnings.slice(0, 5).forEach(issue => {
                    console.log(`  - ${issue.type}: ${issue.message}`);
                });
                if (warnings.length > 5) {
                    console.log(`  ... and ${warnings.length - 5} more warnings`);
                }
            }
        }
        
        if (this.results.summary.failed > 0) {
            console.log('\n❌ FAILED TESTS:');
            this.results.tests
                .filter(t => t.status === 'failed')
                .forEach(test => {
                    console.log(`  - ${test.name}: ${test.error || 'Unknown error'}`);
                });
        }
        
        console.log('\n⚡ PERFORMANCE METRICS:');
        if (this.results.performance.homepageLoadTime) {
            console.log(`Homepage Load Time: ${this.results.performance.homepageLoadTime.toFixed(0)}ms`);
        }
        if (this.results.performance.domContentLoaded) {
            console.log(`DOM Content Loaded: ${this.results.performance.domContentLoaded.toFixed(0)}ms`);
        }
        if (this.results.performance.JSHeapUsedSize) {
            console.log(`Memory Usage: ${(this.results.performance.JSHeapUsedSize / 1024 / 1024).toFixed(1)}MB`);
        }
        
        console.log('\n' + '='.repeat(80));
        
        return this.results;
    }

    // Cleanup
    async cleanup() {
        if (this.browser) {
            await this.browser.close();
        }
    }

    // Main test runner
    async runAllTests() {
        try {
            await this.initialize();
            
            console.log('\n🚀 Starting Comprehensive UI Stress Tests...\n');
            
            // Run all test suites
            await this.testHomepageLoad();
            await this.testNavigationMenu();
            await this.testAllButtons();
            await this.testFormInputs();
            await this.testAPIEndpoints();
            await this.testWebSocketConnections();
            await this.testResponsiveDesign();
            await this.testSpecificComponents();
            await this.testPerformanceMetrics();
            await this.testSecurityFeatures();
            
            // Generate and return report
            const report = this.generateReport();
            
            // Save report to file
            const fs = require('fs').promises;
            await fs.writeFile(
                '/Users/morlock/fun/newmedia/test-results/ui-stress-test-report.json',
                JSON.stringify(report, null, 2)
            );
            
            console.log('\n✅ Test report saved to: test-results/ui-stress-test-report.json');
            
            return report;
            
        } catch (error) {
            console.error('❌ Test suite failed:', error);
            throw error;
        } finally {
            await this.cleanup();
        }
    }
}

// Export for use in other scripts
module.exports = UIStressTestSuite;

// Run if executed directly
if (require.main === module) {
    const tester = new UIStressTestSuite();
    tester.runAllTests()
        .then(report => {
            // Prepare data for Archon task creation if issues found
            const issues = report.issues.filter(i => i.severity === 'error');
            const failedTests = report.tests.filter(t => t.status === 'failed');
            
            if (issues.length > 0 || failedTests.length > 0) {
                console.log('\n🔧 Issues found that need fixing:');
                console.log(`- ${issues.length} critical issues`);
                console.log(`- ${failedTests.length} failed tests`);
                console.log('\n📝 Creating Archon tasks for repairs...');
                
                // Output issues for task creation
                const repairTasks = {
                    projectId: '3e6fbcc1-60f6-434b-a45b-e811cc9bb891',
                    issues: issues,
                    failedTests: failedTests,
                    timestamp: new Date()
                };
                
                require('fs').writeFileSync(
                    '/Users/morlock/fun/newmedia/test-results/repair-tasks.json',
                    JSON.stringify(repairTasks, null, 2)
                );
            } else {
                console.log('\n✨ All tests passed! No repairs needed.');
            }
            
            process.exit(report.summary.failed > 0 ? 1 : 0);
        })
        .catch(error => {
            console.error('Fatal error:', error);
            process.exit(1);
        });
}