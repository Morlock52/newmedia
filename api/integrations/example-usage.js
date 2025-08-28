/**
 * Example Usage of Media Server Integrations
 * Demonstrates how to integrate all services into an Express application
 */

const express = require('express');
const { IntegrationsManager } = require('./index');

class MediaServerAPI {
    constructor() {
        this.app = express();
        this.integrationsManager = new IntegrationsManager();
        this.setupMiddleware();
        this.setupRoutes();
    }

    setupMiddleware() {
        this.app.use(express.json());
        this.app.use(express.urlencoded({ extended: true }));
        
        // CORS
        this.app.use((req, res, next) => {
            res.header('Access-Control-Allow-Origin', '*');
            res.header('Access-Control-Allow-Headers', 'Origin, X-Requested-With, Content-Type, Accept, Authorization');
            res.header('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
            next();
        });
    }

    async initialize() {
        console.log('🚀 Initializing Media Server API...');
        
        // Initialize all configured integrations
        const results = await this.integrationsManager.initializeAll();
        console.log('📡 Integration Status:', results);
        
        // Setup webhooks for all integrations
        this.integrationsManager.setupWebhooks(this.app);
        console.log('🎣 Webhooks configured');
        
        return results;
    }

    setupRoutes() {
        // Health check
        this.app.get('/health', async (req, res) => {
            try {
                const status = await this.integrationsManager.getStatus();
                res.json({
                    status: 'healthy',
                    timestamp: new Date(),
                    integrations: status
                });
            } catch (error) {
                res.status(500).json({
                    status: 'error',
                    error: error.message
                });
            }
        });

        // Get all statistics
        this.app.get('/api/stats', async (req, res) => {
            try {
                const stats = await this.integrationsManager.getComprehensiveStats();
                res.json(stats);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Search across all services
        this.app.get('/api/search', async (req, res) => {
            try {
                const { q: query } = req.query;
                if (!query) {
                    return res.status(400).json({ error: 'Query parameter is required' });
                }
                
                const results = await this.integrationsManager.searchAll(query);
                res.json({
                    query,
                    results,
                    timestamp: new Date()
                });
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        // Jellyfin routes
        this.setupJellyfinRoutes();
        
        // Plex routes
        this.setupPlexRoutes();
        
        // Sonarr routes
        this.setupSonarrRoutes();
        
        // Radarr routes
        this.setupRadarrRoutes();
        
        // Prowlarr routes
        this.setupProwlarrRoutes();
        
        // Jellyseerr routes
        this.setupJellyseerrRoutes();
        
        // Tautulli routes
        this.setupTautulliRoutes();
        
        // NetFlow routes
        this.setupNetflowRoutes();
    }

    setupJellyfinRoutes() {
        const jellyfin = this.integrationsManager.getIntegration('jellyfin');
        if (!jellyfin) return;

        this.app.get('/api/jellyfin/info', async (req, res) => {
            try {
                const info = await jellyfin.getServerInfo();
                res.json(info);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/jellyfin/libraries', async (req, res) => {
            try {
                const libraries = await jellyfin.getLibraries();
                res.json(libraries);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/jellyfin/activity', async (req, res) => {
            try {
                const activity = await jellyfin.getActivity();
                res.json(activity);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/jellyfin/stats', async (req, res) => {
            try {
                const stats = await jellyfin.getStatistics();
                res.json(stats);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    setupPlexRoutes() {
        const plex = this.integrationsManager.getIntegration('plex');
        if (!plex) return;

        this.app.get('/api/plex/info', async (req, res) => {
            try {
                const info = await plex.getServerInfo();
                res.json(info);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/plex/libraries', async (req, res) => {
            try {
                const libraries = await plex.getLibraries();
                res.json(libraries);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/plex/sessions', async (req, res) => {
            try {
                const sessions = await plex.getSessions();
                res.json(sessions);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    setupSonarrRoutes() {
        const sonarr = this.integrationsManager.getIntegration('sonarr');
        if (!sonarr) return;

        this.app.get('/api/sonarr/series', async (req, res) => {
            try {
                const series = await sonarr.getSeries();
                res.json(series);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/sonarr/calendar', async (req, res) => {
            try {
                const calendar = await sonarr.getCalendar();
                res.json(calendar);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/sonarr/queue', async (req, res) => {
            try {
                const queue = await sonarr.getQueue();
                res.json(queue);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/sonarr/series', async (req, res) => {
            try {
                const series = await sonarr.addSeries(req.body);
                res.json(series);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    setupRadarrRoutes() {
        const radarr = this.integrationsManager.getIntegration('radarr');
        if (!radarr) return;

        this.app.get('/api/radarr/movies', async (req, res) => {
            try {
                const movies = await radarr.getMovies();
                res.json(movies);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/radarr/calendar', async (req, res) => {
            try {
                const calendar = await radarr.getCalendar();
                res.json(calendar);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/radarr/queue', async (req, res) => {
            try {
                const queue = await radarr.getQueue();
                res.json(queue);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/radarr/movies', async (req, res) => {
            try {
                const movie = await radarr.addMovie(req.body);
                res.json(movie);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    setupProwlarrRoutes() {
        const prowlarr = this.integrationsManager.getIntegration('prowlarr');
        if (!prowlarr) return;

        this.app.get('/api/prowlarr/indexers', async (req, res) => {
            try {
                const indexers = await prowlarr.getIndexers();
                res.json(indexers);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/prowlarr/search', async (req, res) => {
            try {
                const { q: query } = req.query;
                if (!query) {
                    return res.status(400).json({ error: 'Query parameter is required' });
                }
                
                const results = await prowlarr.search(query);
                res.json(results);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/prowlarr/stats', async (req, res) => {
            try {
                const stats = await prowlarr.getStatistics();
                res.json(stats);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    setupJellyseerrRoutes() {
        const jellyseerr = this.integrationsManager.getIntegration('jellyseerr');
        if (!jellyseerr) return;

        this.app.get('/api/jellyseerr/requests', async (req, res) => {
            try {
                const requests = await jellyseerr.getRequests();
                res.json(requests);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/jellyseerr/requests', async (req, res) => {
            try {
                const request = await jellyseerr.createRequest(req.body);
                res.json(request);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/jellyseerr/requests/:id/approve', async (req, res) => {
            try {
                const result = await jellyseerr.approveRequest(req.params.id);
                res.json(result);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/jellyseerr/search', async (req, res) => {
            try {
                const { q: query } = req.query;
                if (!query) {
                    return res.status(400).json({ error: 'Query parameter is required' });
                }
                
                const results = await jellyseerr.searchMedia(query);
                res.json(results);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    setupTautulliRoutes() {
        const tautulli = this.integrationsManager.getIntegration('tautulli');
        if (!tautulli) return;

        this.app.get('/api/tautulli/activity', async (req, res) => {
            try {
                const activity = await tautulli.getActivity();
                res.json(activity);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/tautulli/history', async (req, res) => {
            try {
                const history = await tautulli.getHistory();
                res.json(history);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/tautulli/stats', async (req, res) => {
            try {
                const stats = await tautulli.getStatistics();
                res.json(stats);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/tautulli/libraries', async (req, res) => {
            try {
                const libraries = await tautulli.getLibraries();
                res.json(libraries);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    setupNetflowRoutes() {
        const netflow = this.integrationsManager.getIntegration('netflow');
        if (!netflow) return;

        this.app.get('/api/netflow/stats', async (req, res) => {
            try {
                const stats = netflow.getStatistics();
                res.json(stats);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/netflow/flows', async (req, res) => {
            try {
                const { limit = 100 } = req.query;
                const flows = netflow.getFlowHistory(parseInt(limit));
                res.json(flows);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.post('/api/netflow/search', async (req, res) => {
            try {
                const flows = netflow.searchFlows(req.body);
                res.json(flows);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });

        this.app.get('/api/netflow/export/:format', async (req, res) => {
            try {
                const { format } = req.params;
                const data = netflow.exportFlows(format);
                
                if (format === 'csv') {
                    res.set({
                        'Content-Type': 'text/csv',
                        'Content-Disposition': `attachment; filename="netflow-${Date.now()}.csv"`
                    });
                } else {
                    res.set('Content-Type', 'application/json');
                }
                
                res.send(data);
            } catch (error) {
                res.status(500).json({ error: error.message });
            }
        });
    }

    start(port = 3002) {
        this.app.listen(port, () => {
            console.log(`🌟 Media Server API running on port ${port}`);
            console.log(`📖 Health check: http://localhost:${port}/health`);
            console.log(`📊 Statistics: http://localhost:${port}/api/stats`);
            console.log(`🔍 Search: http://localhost:${port}/api/search?q=query`);
        });
    }

    async cleanup() {
        console.log('🧹 Cleaning up integrations...');
        this.integrationsManager.cleanup();
    }
}

// Example usage
async function main() {
    const api = new MediaServerAPI();
    
    try {
        await api.initialize();
        api.start();
        
        // Graceful shutdown
        process.on('SIGINT', async () => {
            console.log('\n🛑 Shutting down gracefully...');
            await api.cleanup();
            process.exit(0);
        });
        
    } catch (error) {
        console.error('❌ Failed to start API:', error);
        process.exit(1);
    }
}

// Run if this file is executed directly
if (require.main === module) {
    main().catch(console.error);
}

module.exports = MediaServerAPI;