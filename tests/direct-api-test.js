/**
 * Direct API and Component Test Suite
 * Tests all endpoints and components directly without browser automation
 */

const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

class DirectTestSuite {
    constructor() {
        this.baseURL = 'http://localhost:3333';
        this.apiURL = 'http://localhost:3333/api';
        this.results = {
            timestamp: new Date(),
            tests: [],
            issues: [],
            summary: {
                total: 0,
                passed: 0,
                failed: 0,
                warnings: 0
            }
        };
    }

    recordTest(category, name, status, details = {}) {
        const test = {
            category,
            name,
            status,
            timestamp: new Date(),
            ...details
        };
        
        this.results.tests.push(test);
        this.results.summary.total++;
        
        if (status === 'passed') {
            this.results.summary.passed++;
            console.log(`✅ ${category}: ${name}`);
        } else if (status === 'failed') {
            this.results.summary.failed++;
            console.log(`❌ ${category}: ${name} - ${details.error || 'Failed'}`);
            this.results.issues.push({
                category,
                name,
                error: details.error || 'Unknown error',
                severity: 'error'
            });
        } else if (status === 'warning') {
            this.results.summary.warnings++;
            console.log(`⚠️  ${category}: ${name} - ${details.warning || 'Warning'}`);
        }
    }

    // Test all API endpoints
    async testAPIEndpoints() {
        console.log('\n📡 Testing API Endpoints...\n');
        
        const endpoints = [
            // Core endpoints
            { path: '/services', method: 'GET', name: 'List Services' },
            { path: '/services/jellyfin', method: 'GET', name: 'Jellyfin Service' },
            { path: '/services/plex', method: 'GET', name: 'Plex Service' },
            { path: '/services/sonarr', method: 'GET', name: 'Sonarr Service' },
            { path: '/services/radarr', method: 'GET', name: 'Radarr Service' },
            
            // Component endpoints
            { path: '/components/notification-system', method: 'GET', name: 'Notification System' },
            { path: '/components/data-analytics', method: 'GET', name: 'Data Analytics' },
            { path: '/components/mobile-pwa', method: 'GET', name: 'Mobile PWA' },
            { path: '/components/smart-download', method: 'GET', name: 'Smart Download' },
            { path: '/components/voice-control', method: 'GET', name: 'Voice Control' },
            { path: '/components/ar-vr-experience', method: 'GET', name: 'AR/VR Experience' },
            { path: '/components/real-time-monitoring', method: 'GET', name: 'Real-time Monitoring' },
            { path: '/components/unified-api', method: 'GET', name: 'Unified API' },
            { path: '/components/service-grid-3d', method: 'GET', name: 'Service Grid 3D' },
            { path: '/components/nexus-ai', method: 'GET', name: 'NEXUS AI' },
            { path: '/components/service-grid', method: 'GET', name: 'Service Grid' },
            { path: '/components/cyberpunk-theme', method: 'GET', name: 'Cyberpunk Theme' },
            { path: '/components/social-watch-party', method: 'GET', name: 'Social Watch Party' },
            { path: '/components/predictive-analytics', method: 'GET', name: 'Predictive Analytics' },
            { path: '/components/holographic-player', method: 'GET', name: 'Holographic Player' },
            { path: '/components/neural-recommendations', method: 'GET', name: 'Neural Recommendations' },
            { path: '/components/multi-user-profiles', method: 'GET', name: 'Multi-User Profiles' },
            { path: '/components/gpt4-discovery', method: 'GET', name: 'GPT-4 Discovery' },
            
            // Advanced features
            { path: '/components/zero-trust-security', method: 'GET', name: 'Zero Trust Security' },
            { path: '/components/web3-integration', method: 'GET', name: 'Web3 Integration' },
            { path: '/components/smart-home-hub', method: 'GET', name: 'Smart Home Hub' },
            { path: '/components/performance-optimizer', method: 'GET', name: 'Performance Optimizer' },
            { path: '/components/mobile-sync', method: 'GET', name: 'Mobile Sync' },
            { path: '/components/infrastructure-monitor', method: 'GET', name: 'Infrastructure Monitor' },
            
            // Integration endpoints
            { path: '/integrations/jellyfin/status', method: 'GET', name: 'Jellyfin Integration' },
            { path: '/integrations/plex/status', method: 'GET', name: 'Plex Integration' },
            { path: '/integrations/sonarr/status', method: 'GET', name: 'Sonarr Integration' },
            { path: '/integrations/radarr/status', method: 'GET', name: 'Radarr Integration' },
            { path: '/integrations/prowlarr/status', method: 'GET', name: 'Prowlarr Integration' },
            { path: '/integrations/jellyseerr/status', method: 'GET', name: 'Jellyseerr Integration' },
            { path: '/integrations/tautulli/status', method: 'GET', name: 'Tautulli Integration' },
            { path: '/integrations/netflow/status', method: 'GET', name: 'NetFlow Integration' }
        ];

        for (const endpoint of endpoints) {
            try {
                const response = await axios({
                    method: endpoint.method,
                    url: `${this.apiURL}${endpoint.path}`,
                    timeout: 5000,
                    validateStatus: () => true
                });
                
                if (response.status === 200) {
                    this.recordTest('API', endpoint.name, 'passed', {
                        status: response.status,
                        hasData: !!response.data
                    });
                } else if (response.status === 404) {
                    this.recordTest('API', endpoint.name, 'warning', {
                        warning: 'Endpoint not implemented',
                        status: response.status
                    });
                } else {
                    this.recordTest('API', endpoint.name, 'failed', {
                        error: `HTTP ${response.status}`,
                        status: response.status
                    });
                }
            } catch (error) {
                this.recordTest('API', endpoint.name, 'failed', {
                    error: error.message
                });
            }
        }
    }

    // Test component files exist
    async testComponentFiles() {
        console.log('\n📁 Testing Component Files...\n');
        
        const components = [
            'NotificationSystem.tsx',
            'DataAnalyticsDashboard.tsx',
            'MobilePWAInterface.tsx',
            'SmartDownloadManager.tsx',
            'VoiceControlSystem.tsx',
            'ARVRMediaExperience.tsx',
            'RealTimeMonitoringSystem.tsx',
            'UnifiedMediaAPI.tsx',
            'ServiceGrid3D.tsx',
            'NEXUSAIAssistant.tsx',
            'ServiceGrid.tsx',
            'CyberpunkTheme.tsx',
            'SocialWatchParty.tsx',
            'PredictiveAnalytics.tsx',
            'HolographicMediaPlayer.tsx',
            'NeuralRecommendations.tsx',
            'MultiUserProfiles.tsx',
            'GPT4Discovery.tsx',
            'ZeroTrustSecurity.tsx',
            'Web3Integration.tsx',
            'SmartHomeHub.tsx',
            'PerformanceOptimizer.tsx',
            'MobileSync.tsx',
            'InfrastructureMonitor.tsx'
        ];

        const componentPath = '/Users/morlock/fun/newmedia/dashboard/src/components';
        
        for (const component of components) {
            try {
                await fs.access(path.join(componentPath, component));
                this.recordTest('Files', component, 'passed');
            } catch (error) {
                this.recordTest('Files', component, 'failed', {
                    error: 'File not found'
                });
            }
        }
    }

    // Test service files
    async testServiceFiles() {
        console.log('\n📂 Testing Service Files...\n');
        
        const services = [
            'DockerManager.js',
            'ConfigManager.js',
            'HealthMonitor.js',
            'SeedboxManager.js',
            'LogManager.js',
            'Web3Service.js',
            'SmartHomeService.js',
            'SecurityService.js',
            'VPNService.js',
            'MonitoringService.js',
            'JellyfinAuthService.js'
        ];

        const servicePath = '/Users/morlock/fun/newmedia/api/services';
        
        for (const service of services) {
            try {
                await fs.access(path.join(servicePath, service));
                this.recordTest('Services', service, 'passed');
            } catch (error) {
                this.recordTest('Services', service, 'failed', {
                    error: 'File not found'
                });
            }
        }
    }

    // Test integration files
    async testIntegrationFiles() {
        console.log('\n📄 Testing Integration Files...\n');
        
        const integrations = [
            'jellyfin.js',
            'plex.js',
            'sonarr.js',
            'radarr.js',
            'prowlarr.js',
            'jellyseerr.js',
            'tautulli.js',
            'netflow.js'
        ];

        const integrationPath = '/Users/morlock/fun/newmedia/api/integrations';
        
        for (const integration of integrations) {
            try {
                await fs.access(path.join(integrationPath, integration));
                this.recordTest('Integrations', integration, 'passed');
            } catch (error) {
                this.recordTest('Integrations', integration, 'failed', {
                    error: 'File not found'
                });
            }
        }
    }

    // Test WebSocket connection
    async testWebSocket() {
        console.log('\n🔌 Testing WebSocket...\n');
        
        try {
            // Simple HTTP check for WebSocket upgrade capability
            const response = await axios.get(this.baseURL, {
                headers: {
                    'Upgrade': 'websocket',
                    'Connection': 'Upgrade'
                },
                validateStatus: () => true
            });
            
            // If server supports WebSocket, it should return 101 or have upgrade headers
            if (response.status === 101 || response.headers['upgrade']) {
                this.recordTest('WebSocket', 'Connection Support', 'passed');
            } else {
                this.recordTest('WebSocket', 'Connection Support', 'warning', {
                    warning: 'WebSocket not configured'
                });
            }
        } catch (error) {
            this.recordTest('WebSocket', 'Connection Support', 'warning', {
                warning: 'WebSocket not available'
            });
        }
    }

    // Test CORS configuration
    async testCORS() {
        console.log('\n🔒 Testing CORS Configuration...\n');
        
        try {
            const response = await axios.options(this.baseURL, {
                headers: {
                    'Origin': 'file://',
                    'Access-Control-Request-Method': 'GET'
                }
            });
            
            const corsHeaders = response.headers['access-control-allow-origin'];
            
            if (corsHeaders === '*' || corsHeaders === 'file://') {
                this.recordTest('CORS', 'File Protocol Support', 'passed', {
                    allowedOrigin: corsHeaders
                });
            } else {
                this.recordTest('CORS', 'File Protocol Support', 'failed', {
                    error: 'CORS not configured for file://'
                });
            }
        } catch (error) {
            this.recordTest('CORS', 'File Protocol Support', 'failed', {
                error: error.message
            });
        }
    }

    // Test dashboard HTML structure
    async testDashboardHTML() {
        console.log('\n📄 Testing Dashboard HTML...\n');
        
        try {
            const response = await axios.get(this.baseURL);
            const html = response.data;
            
            // Check for essential elements
            const checks = [
                { pattern: /<div id="root"/, name: 'Root Element' },
                { pattern: /<script.*src=.*bundle\.js/, name: 'JavaScript Bundle' },
                { pattern: /<link.*href=.*\.css/, name: 'CSS Styles' },
                { pattern: /React|react/, name: 'React Framework' }
            ];
            
            for (const check of checks) {
                if (check.pattern.test(html)) {
                    this.recordTest('HTML', check.name, 'passed');
                } else {
                    this.recordTest('HTML', check.name, 'warning', {
                        warning: 'Element not found in HTML'
                    });
                }
            }
        } catch (error) {
            this.recordTest('HTML', 'Dashboard Structure', 'failed', {
                error: error.message
            });
        }
    }

    // Generate report
    generateReport() {
        console.log('\n' + '='.repeat(80));
        console.log('📊 TEST RESULTS SUMMARY');
        console.log('='.repeat(80));
        
        console.log('\n📈 Statistics:');
        console.log(`Total Tests: ${this.results.summary.total}`);
        console.log(`✅ Passed: ${this.results.summary.passed}`);
        console.log(`❌ Failed: ${this.results.summary.failed}`);
        console.log(`⚠️  Warnings: ${this.results.summary.warnings}`);
        
        const successRate = this.results.summary.total > 0
            ? ((this.results.summary.passed / this.results.summary.total) * 100).toFixed(1)
            : 0;
        console.log(`Success Rate: ${successRate}%`);
        
        if (this.results.issues.length > 0) {
            console.log('\n🔍 Issues Found:');
            this.results.issues.forEach(issue => {
                console.log(`  - ${issue.category}: ${issue.name} - ${issue.error}`);
            });
        }
        
        console.log('\n' + '='.repeat(80));
        
        return this.results;
    }

    // Save results
    async saveResults() {
        const resultsPath = '/Users/morlock/fun/newmedia/test-results/direct-test-results.json';
        await fs.writeFile(resultsPath, JSON.stringify(this.results, null, 2));
        console.log(`\n💾 Results saved to: ${resultsPath}`);
    }

    // Main test runner
    async runAllTests() {
        console.log('🚀 Starting Direct Test Suite...');
        console.log('Testing: http://localhost:3333\n');
        
        await this.testCORS();
        await this.testDashboardHTML();
        await this.testAPIEndpoints();
        await this.testComponentFiles();
        await this.testServiceFiles();
        await this.testIntegrationFiles();
        await this.testWebSocket();
        
        const report = this.generateReport();
        await this.saveResults();
        
        return report;
    }
}

// Run tests
async function main() {
    const tester = new DirectTestSuite();
    
    try {
        const report = await tester.runAllTests();
        
        // Check if repairs are needed
        if (report.summary.failed > 0) {
            console.log('\n🔧 Repairs needed! Creating tasks for swarm coordination...');
            
            const repairData = {
                projectId: '3e6fbcc1-60f6-434b-a45b-e811cc9bb891',
                issues: report.issues,
                timestamp: new Date(),
                needsSwarmCoordination: true
            };
            
            await fs.writeFile(
                '/Users/morlock/fun/newmedia/test-results/repair-tasks.json',
                JSON.stringify(repairData, null, 2)
            );
            
            console.log('📝 Repair tasks created for swarm coordination.');
        } else {
            console.log('\n✨ All tests passed or have warnings only!');
        }
        
        process.exit(report.summary.failed > 0 ? 1 : 0);
        
    } catch (error) {
        console.error('Fatal error:', error);
        process.exit(1);
    }
}

if (require.main === module) {
    main().catch(console.error);
}

module.exports = DirectTestSuite;