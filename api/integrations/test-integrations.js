/**
 * Integration Test Suite
 * Tests all service integrations to ensure they work correctly
 */

const {
    JellyfinIntegration,
    PlexIntegration,
    SonarrIntegration,
    RadarrIntegration,
    ProwlarrIntegration,
    JellyseerrIntegration,
    TautulliIntegration,
    NetflowIntegration,
    IntegrationsManager
} = require('./index');

class IntegrationTester {
    constructor() {
        this.results = {
            timestamp: new Date(),
            tests: [],
            summary: {
                total: 0,
                passed: 0,
                failed: 0,
                skipped: 0
            }
        };
    }

    /**
     * Run a test and record results
     */
    async runTest(name, testFn, skipReason = null) {
        const test = {
            name,
            status: 'running',
            startTime: new Date(),
            error: null,
            skipped: false,
            skipReason
        };

        this.results.tests.push(test);
        this.results.summary.total++;

        if (skipReason) {
            test.status = 'skipped';
            test.skipped = true;
            this.results.summary.skipped++;
            console.log(`⏸️  SKIP: ${name} - ${skipReason}`);
            return;
        }

        try {
            console.log(`🧪 TEST: ${name}`);
            await testFn();
            test.status = 'passed';
            this.results.summary.passed++;
            console.log(`✅ PASS: ${name}`);
        } catch (error) {
            test.status = 'failed';
            test.error = error.message;
            this.results.summary.failed++;
            console.error(`❌ FAIL: ${name} - ${error.message}`);
        } finally {
            test.endTime = new Date();
            test.duration = test.endTime - test.startTime;
        }
    }

    /**
     * Test Jellyfin Integration
     */
    async testJellyfinIntegration() {
        const hasConfig = process.env.JELLYFIN_URL && process.env.JELLYFIN_API_KEY;
        
        await this.runTest('Jellyfin - Initialize', async () => {
            const jellyfin = new JellyfinIntegration({
                baseURL: 'http://localhost:8096',
                apiKey: 'test-key'
            });
            if (!jellyfin) throw new Error('Failed to initialize Jellyfin integration');
        });

        await this.runTest('Jellyfin - Connection Test', async () => {
            const jellyfin = new JellyfinIntegration({
                baseURL: process.env.JELLYFIN_URL || 'http://localhost:8096',
                apiKey: process.env.JELLYFIN_API_KEY || 'test-key'
            });
            
            if (hasConfig) {
                const result = await jellyfin.testConnection();
                if (!result.success && !result.error.includes('authentication')) {
                    throw new Error(`Connection test failed: ${result.error}`);
                }
            } else {
                // Just test that methods exist
                if (typeof jellyfin.getServerInfo !== 'function') {
                    throw new Error('getServerInfo method missing');
                }
            }
        }, hasConfig ? null : 'No Jellyfin configuration provided');

        await this.runTest('Jellyfin - Method Validation', async () => {
            const jellyfin = new JellyfinIntegration({
                baseURL: 'http://localhost:8096',
                apiKey: 'test-key'
            });

            const requiredMethods = [
                'authenticate', 'getServerInfo', 'getUsers', 'getLibraries',
                'searchItems', 'getItem', 'markPlayed', 'getActivity'
            ];

            for (const method of requiredMethods) {
                if (typeof jellyfin[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }
        });
    }

    /**
     * Test Plex Integration
     */
    async testPlexIntegration() {
        const hasConfig = process.env.PLEX_URL && process.env.PLEX_TOKEN;

        await this.runTest('Plex - Initialize', async () => {
            const plex = new PlexIntegration({
                baseURL: 'http://localhost:32400',
                token: 'test-token'
            });
            if (!plex) throw new Error('Failed to initialize Plex integration');
        });

        await this.runTest('Plex - Connection Test', async () => {
            const plex = new PlexIntegration({
                baseURL: process.env.PLEX_URL || 'http://localhost:32400',
                token: process.env.PLEX_TOKEN || 'test-token'
            });
            
            if (hasConfig) {
                const result = await plex.testConnection();
                if (!result.success && !result.error.includes('authentication')) {
                    throw new Error(`Connection test failed: ${result.error}`);
                }
            } else {
                // Just test that methods exist
                if (typeof plex.getServerInfo !== 'function') {
                    throw new Error('getServerInfo method missing');
                }
            }
        }, hasConfig ? null : 'No Plex configuration provided');

        await this.runTest('Plex - Method Validation', async () => {
            const plex = new PlexIntegration({
                baseURL: 'http://localhost:32400',
                token: 'test-token'
            });

            const requiredMethods = [
                'authenticate', 'getServerInfo', 'getLibraries', 'search',
                'getMetadata', 'getSessions', 'markWatched', 'getPlaylists'
            ];

            for (const method of requiredMethods) {
                if (typeof plex[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }
        });
    }

    /**
     * Test Sonarr Integration
     */
    async testSonarrIntegration() {
        const hasConfig = process.env.SONARR_URL && process.env.SONARR_API_KEY;

        await this.runTest('Sonarr - Initialize', async () => {
            const sonarr = new SonarrIntegration({
                baseURL: 'http://localhost:8989',
                apiKey: 'test-key'
            });
            if (!sonarr) throw new Error('Failed to initialize Sonarr integration');
        });

        await this.runTest('Sonarr - Method Validation', async () => {
            const sonarr = new SonarrIntegration({
                baseURL: 'http://localhost:8989',
                apiKey: 'test-key'
            });

            const requiredMethods = [
                'getSystemStatus', 'getSeries', 'addSeries', 'getEpisodes',
                'getCalendar', 'getQueue', 'getHistory', 'searchSeries'
            ];

            for (const method of requiredMethods) {
                if (typeof sonarr[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }
        });
    }

    /**
     * Test Radarr Integration
     */
    async testRadarrIntegration() {
        const hasConfig = process.env.RADARR_URL && process.env.RADARR_API_KEY;

        await this.runTest('Radarr - Initialize', async () => {
            const radarr = new RadarrIntegration({
                baseURL: 'http://localhost:7878',
                apiKey: 'test-key'
            });
            if (!radarr) throw new Error('Failed to initialize Radarr integration');
        });

        await this.runTest('Radarr - Method Validation', async () => {
            const radarr = new RadarrIntegration({
                baseURL: 'http://localhost:7878',
                apiKey: 'test-key'
            });

            const requiredMethods = [
                'getSystemStatus', 'getMovies', 'addMovie', 'getMovieFiles',
                'getCalendar', 'getQueue', 'getHistory', 'searchMovies'
            ];

            for (const method of requiredMethods) {
                if (typeof radarr[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }
        });
    }

    /**
     * Test Prowlarr Integration
     */
    async testProwlarrIntegration() {
        const hasConfig = process.env.PROWLARR_URL && process.env.PROWLARR_API_KEY;

        await this.runTest('Prowlarr - Initialize', async () => {
            const prowlarr = new ProwlarrIntegration({
                baseURL: 'http://localhost:9696',
                apiKey: 'test-key'
            });
            if (!prowlarr) throw new Error('Failed to initialize Prowlarr integration');
        });

        await this.runTest('Prowlarr - Method Validation', async () => {
            const prowlarr = new ProwlarrIntegration({
                baseURL: 'http://localhost:9696',
                apiKey: 'test-key'
            });

            const requiredMethods = [
                'getSystemStatus', 'getIndexers', 'search', 'getApplications',
                'getDownloadClients', 'getStatistics', 'testIndexer'
            ];

            for (const method of requiredMethods) {
                if (typeof prowlarr[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }
        });
    }

    /**
     * Test Jellyseerr Integration
     */
    async testJellyseerrIntegration() {
        const hasConfig = process.env.JELLYSEERR_URL && process.env.JELLYSEERR_API_KEY;

        await this.runTest('Jellyseerr - Initialize', async () => {
            const jellyseerr = new JellyseerrIntegration({
                baseURL: 'http://localhost:5055',
                apiKey: 'test-key'
            });
            if (!jellyseerr) throw new Error('Failed to initialize Jellyseerr integration');
        });

        await this.runTest('Jellyseerr - Method Validation', async () => {
            const jellyseerr = new JellyseerrIntegration({
                baseURL: 'http://localhost:5055',
                apiKey: 'test-key'
            });

            const requiredMethods = [
                'getStatus', 'getRequests', 'createRequest', 'approveRequest',
                'searchMedia', 'getUsers', 'getIssues', 'getStatistics'
            ];

            for (const method of requiredMethods) {
                if (typeof jellyseerr[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }
        });
    }

    /**
     * Test Tautulli Integration
     */
    async testTautulliIntegration() {
        const hasConfig = process.env.TAUTULLI_URL && process.env.TAUTULLI_API_KEY;

        await this.runTest('Tautulli - Initialize', async () => {
            const tautulli = new TautulliIntegration({
                baseURL: 'http://localhost:8181',
                apiKey: 'test-key'
            });
            if (!tautulli) throw new Error('Failed to initialize Tautulli integration');
        });

        await this.runTest('Tautulli - Method Validation', async () => {
            const tautulli = new TautulliIntegration({
                baseURL: 'http://localhost:8181',
                apiKey: 'test-key'
            });

            const requiredMethods = [
                'getServerInfo', 'getActivity', 'getHistory', 'getLibraries',
                'getUsers', 'getPlaysByDate', 'getStatistics'
            ];

            for (const method of requiredMethods) {
                if (typeof tautulli[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }
        });
    }

    /**
     * Test NetFlow Integration
     */
    async testNetflowIntegration() {
        await this.runTest('NetFlow - Initialize', async () => {
            const netflow = new NetflowIntegration({
                analysisEnabled: false // Don't start UDP server in tests
            });
            if (!netflow) throw new Error('Failed to initialize NetFlow integration');
        });

        await this.runTest('NetFlow - Method Validation', async () => {
            const netflow = new NetflowIntegration({
                analysisEnabled: false
            });

            const requiredMethods = [
                'getStatistics', 'getFlowHistory', 'searchFlows',
                'analyzeFlows', 'exportFlows', 'testConnection'
            ];

            for (const method of requiredMethods) {
                if (typeof netflow[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }

            // Test basic functionality
            const stats = netflow.getStatistics();
            if (!stats || typeof stats !== 'object') {
                throw new Error('getStatistics should return an object');
            }

            // Cleanup
            netflow.cleanup();
        });

        await this.runTest('NetFlow - Flow Processing', async () => {
            const netflow = new NetflowIntegration({
                analysisEnabled: false
            });

            // Test flow processing
            const testFlow = {
                srcAddr: '192.168.1.100',
                dstAddr: '192.168.1.200',
                srcPort: 32400,
                dstPort: 8080,
                protocol: 6,
                octets: 1024000,
                packets: 100,
                timestamp: new Date()
            };

            netflow.processFlow(testFlow, { address: 'test' });
            
            const stats = netflow.getStatistics();
            if (stats.totalFlows === 0) {
                throw new Error('Flow was not processed correctly');
            }

            netflow.cleanup();
        });
    }

    /**
     * Test Integrations Manager
     */
    async testIntegrationsManager() {
        await this.runTest('IntegrationsManager - Initialize', async () => {
            const manager = new IntegrationsManager();
            if (!manager) throw new Error('Failed to initialize IntegrationsManager');
        });

        await this.runTest('IntegrationsManager - Method Validation', async () => {
            const manager = new IntegrationsManager();

            const requiredMethods = [
                'initializeAll', 'getIntegration', 'getAllIntegrations',
                'getStatus', 'setupWebhooks', 'getComprehensiveStats', 'cleanup'
            ];

            for (const method of requiredMethods) {
                if (typeof manager[method] !== 'function') {
                    throw new Error(`Required method ${method} not found`);
                }
            }
        });

        await this.runTest('IntegrationsManager - Initialization', async () => {
            const manager = new IntegrationsManager({
                // Mock configurations to test initialization
                jellyfin: { baseURL: 'http://localhost:8096', apiKey: 'test' },
                netflow: { analysisEnabled: false }
            });

            const results = await manager.initializeAll();
            
            if (!results || typeof results !== 'object') {
                throw new Error('initializeAll should return an object');
            }

            // Should have attempted to initialize configured services
            if (!results.jellyfin && !results.netflow) {
                throw new Error('No services were initialized');
            }

            manager.cleanup();
        });
    }

    /**
     * Test Webhook Setup
     */
    async testWebhookSetup() {
        await this.runTest('Webhooks - Setup Validation', async () => {
            const express = require('express');
            const app = express();
            app.use(express.json());

            const integrations = [
                new JellyfinIntegration({ baseURL: 'http://localhost:8096', apiKey: 'test' }),
                new PlexIntegration({ baseURL: 'http://localhost:32400', token: 'test' }),
                new SonarrIntegration({ baseURL: 'http://localhost:8989', apiKey: 'test' }),
                new RadarrIntegration({ baseURL: 'http://localhost:7878', apiKey: 'test' }),
                new ProwlarrIntegration({ baseURL: 'http://localhost:9696', apiKey: 'test' }),
                new JellyseerrIntegration({ baseURL: 'http://localhost:5055', apiKey: 'test' }),
                new TautulliIntegration({ baseURL: 'http://localhost:8181', apiKey: 'test' }),
                new NetflowIntegration({ analysisEnabled: false })
            ];

            for (const integration of integrations) {
                if (typeof integration.setupWebhook === 'function') {
                    try {
                        integration.setupWebhook(app);
                    } catch (error) {
                        throw new Error(`Failed to setup webhook for ${integration.constructor.name}: ${error.message}`);
                    }
                }
            }

            // Cleanup NetFlow
            const netflow = integrations.find(i => i.constructor.name === 'NetflowIntegration');
            if (netflow) netflow.cleanup();
        });
    }

    /**
     * Run all tests
     */
    async runAllTests() {
        console.log('🚀 Starting Integration Tests...\n');

        await this.testJellyfinIntegration();
        await this.testPlexIntegration();
        await this.testSonarrIntegration();
        await this.testRadarrIntegration();
        await this.testProwlarrIntegration();
        await this.testJellyseerrIntegration();
        await this.testTautulliIntegration();
        await this.testNetflowIntegration();
        await this.testIntegrationsManager();
        await this.testWebhookSetup();

        this.generateReport();
        return this.results;
    }

    /**
     * Generate test report
     */
    generateReport() {
        console.log('\n📊 TEST RESULTS SUMMARY');
        console.log('========================');
        console.log(`Total Tests: ${this.results.summary.total}`);
        console.log(`✅ Passed: ${this.results.summary.passed}`);
        console.log(`❌ Failed: ${this.results.summary.failed}`);
        console.log(`⏸️  Skipped: ${this.results.summary.skipped}`);
        
        const successRate = ((this.results.summary.passed / (this.results.summary.total - this.results.summary.skipped)) * 100).toFixed(1);
        console.log(`📈 Success Rate: ${successRate}%`);

        if (this.results.summary.failed > 0) {
            console.log('\n❌ FAILED TESTS:');
            this.results.tests
                .filter(t => t.status === 'failed')
                .forEach(test => {
                    console.log(`   - ${test.name}: ${test.error}`);
                });
        }

        if (this.results.summary.skipped > 0) {
            console.log('\n⏸️  SKIPPED TESTS:');
            this.results.tests
                .filter(t => t.skipped)
                .forEach(test => {
                    console.log(`   - ${test.name}: ${test.skipReason}`);
                });
        }

        console.log('\n✨ Test run completed!');
    }
}

// Run tests if this file is executed directly
if (require.main === module) {
    const tester = new IntegrationTester();
    tester.runAllTests().catch(console.error);
}

module.exports = IntegrationTester;