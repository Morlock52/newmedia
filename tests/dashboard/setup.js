/**
 * Test Setup Configuration
 * Global test setup for dashboard test suite
 */

const { TextEncoder, TextDecoder } = require('util');

// Polyfills for Node.js environment
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Set longer timeout for slow systems and network requests
jest.setTimeout(30000);

// Console filtering to reduce noise
const originalConsoleError = console.error;
const originalConsoleWarn = console.warn;
const originalConsoleLog = console.log;

// Filter out common warnings and debug messages
console.error = (...args) => {
    const message = args[0];
    
    // Skip React warnings and other common warnings
    if (typeof message === 'string') {
        if (message.includes('Warning:') ||
            message.includes('deprecated') ||
            message.includes('JSDOM') ||
            message.includes('ResourceLoader')) {
            return;
        }
    }
    
    originalConsoleError.call(console, ...args);
};

console.warn = (...args) => {
    const message = args[0];
    
    if (typeof message === 'string') {
        if (message.includes('deprecated') ||
            message.includes('experimental') ||
            message.includes('puppeteer')) {
            return;
        }
    }
    
    originalConsoleWarn.call(console, ...args);
};

// Add test result formatting
const testResults = {
    passed: 0,
    failed: 0,
    skipped: 0,
    suites: []
};

// Global test hooks
beforeAll(() => {
    console.log('🧪 Dashboard Test Suite Starting...\n');
});

afterAll(() => {
    console.log('\n📊 Test Execution Complete');
});

// Per-suite tracking
beforeEach(() => {
    // Individual test setup if needed
});

afterEach(() => {
    // Individual test cleanup if needed
});

// Error handling for unhandled promises
process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Global test utilities
global.testUtils = {
    // Utility to wait for element
    waitForElement: async (page, selector, timeout = 5000) => {
        try {
            await page.waitForSelector(selector, { timeout });
            return true;
        } catch (error) {
            return false;
        }
    },
    
    // Utility to check if URL is accessible
    isUrlAccessible: async (url, timeout = 5000) => {
        try {
            const axios = require('axios');
            const response = await axios.get(url, { 
                timeout,
                validateStatus: status => status < 500
            });
            return response.status < 400;
        } catch (error) {
            return false;
        }
    },
    
    // Utility to create screenshot filename
    getScreenshotPath: (testName, viewport = '') => {
        const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
        const viewportSuffix = viewport ? `-${viewport}` : '';
        return `./reports/screenshot-${testName}${viewportSuffix}-${timestamp}.png`;
    },
    
    // Utility to log test progress
    logProgress: (message, type = 'info') => {
        const timestamp = new Date().toISOString();
        const prefix = type === 'error' ? '❌' : type === 'warn' ? '⚠️' : '✅';
        console.log(`${prefix} [${timestamp}] ${message}`);
    },
    
    // Utility to measure performance
    measureTime: async (fn, label = 'Operation') => {
        const startTime = Date.now();
        const result = await fn();
        const endTime = Date.now();
        const duration = endTime - startTime;
        
        console.log(`📊 ${label} completed in ${duration}ms`);
        return { result, duration };
    }
};

// Environment detection
global.testEnvironment = {
    isCI: process.env.CI === 'true',
    platform: process.platform,
    nodeVersion: process.version,
    hasDisplay: process.env.DISPLAY !== undefined || process.platform === 'darwin',
    
    // Feature detection
    hasDocker: (() => {
        try {
            require('child_process').execSync('docker --version', { stdio: 'ignore' });
            return true;
        } catch {
            return false;
        }
    })(),
    
    // Service availability
    services: {
        api: false,
        dashboard: false,
        docker: false
    }
};

// Test data generators
global.testData = {
    // Generate test viewport configurations
    getViewports: () => [
        { name: 'mobile', width: 375, height: 667 },
        { name: 'tablet', width: 768, height: 1024 },
        { name: 'desktop', width: 1280, height: 720 },
        { name: 'large', width: 1920, height: 1080 }
    ],
    
    // Generate test API endpoints
    getApiEndpoints: (baseUrl = 'http://localhost:3002') => [
        `${baseUrl}/health`,
        `${baseUrl}/api/docs`,
        `${baseUrl}/api/services`,
        `${baseUrl}/api/config`,
        `${baseUrl}/api/health/overview`
    ],
    
    // Generate test service configurations
    getTestServices: () => [
        { name: 'jellyfin', port: 8096, path: '/web' },
        { name: 'plex', port: 32400, path: '/web' },
        { name: 'sonarr', port: 8989, path: '/' },
        { name: 'radarr', port: 7878, path: '/' },
        { name: 'prowlarr', port: 9696, path: '/' }
    ]
};

// Mock implementations for testing
global.mocks = {
    // Mock WebSocket for testing
    createMockWebSocket: () => {
        const EventEmitter = require('events');
        class MockWebSocket extends EventEmitter {
            constructor(url) {
                super();
                this.url = url;
                this.readyState = 1; // OPEN
                setTimeout(() => this.emit('open'), 10);
            }
            
            send(data) {
                // Echo back for testing
                setTimeout(() => {
                    try {
                        const message = JSON.parse(data);
                        if (message.action === 'ping') {
                            this.emit('message', JSON.stringify({
                                type: 'pong',
                                timestamp: new Date().toISOString()
                            }));
                        }
                    } catch (error) {
                        this.emit('message', JSON.stringify({
                            type: 'error',
                            message: 'Invalid message format'
                        }));
                    }
                }, 50);
            }
            
            close() {
                this.readyState = 3; // CLOSED
                this.emit('close');
            }
        }
        
        return MockWebSocket;
    },
    
    // Mock Puppeteer page for testing
    createMockPage: () => ({
        goto: jest.fn().mockResolvedValue({}),
        setViewport: jest.fn().mockResolvedValue({}),
        screenshot: jest.fn().mockResolvedValue(Buffer.from('mock-image')),
        title: jest.fn().mockResolvedValue('Mock Title'),
        $: jest.fn().mockResolvedValue({}),
        $$: jest.fn().mockResolvedValue([]),
        evaluate: jest.fn().mockResolvedValue({}),
        waitForSelector: jest.fn().mockResolvedValue({}),
        close: jest.fn().mockResolvedValue({})
    })
};

// Test configuration based on environment
if (global.testEnvironment.isCI) {
    // CI-specific configuration
    jest.setTimeout(60000); // Longer timeout for CI
    console.log('🤖 Running in CI environment');
} else {
    // Local development configuration
    console.log('💻 Running in local development environment');
}

// Export configuration for other test files
module.exports = {
    testResults,
    testEnvironment: global.testEnvironment,
    testUtils: global.testUtils,
    testData: global.testData,
    mocks: global.mocks
};