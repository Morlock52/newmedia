const logger = require('../../middleware/logger.js');
/**
 * Media Server Integrations Index
 * Exports all service integration modules
 */

const JellyfinIntegration = require('./JellyfinIntegration');
const PlexIntegration = require('./PlexIntegration');
const SonarrIntegration = require('./SonarrIntegration');
const RadarrIntegration = require('./RadarrIntegration');
const ProwlarrIntegration = require('./ProwlarrIntegration');
const JellyseerrIntegration = require('./JellyseerrIntegration');
const TautulliIntegration = require('./TautulliIntegration');
const NetflowIntegration = require('./NetflowIntegration');

/**
 * Integration factory class for managing all service integrations
 */
class IntegrationsManager {
    constructor(config = {}) {
        this.config = config;
        this.integrations = new Map();
        this.eventHandlers = new Map();
    }

    /**
     * Initialize all integrations based on configuration
     */
    async initializeAll() {
        const results = {};

        // Initialize Jellyfin if configured
        if (this.config.jellyfin || process.env.JELLYFIN_URL) {
            try {
                const jellyfin = new JellyfinIntegration(this.config.jellyfin);
                this.integrations.set('jellyfin', jellyfin);
                this.setupEventHandlers('jellyfin', jellyfin);
                results.jellyfin = await jellyfin.testConnection();
            } catch (error) {
                results.jellyfin = { success: false, error: error.message };
            }
        }

        // Initialize Plex if configured
        if (this.config.plex || process.env.PLEX_URL) {
            try {
                const plex = new PlexIntegration(this.config.plex);
                this.integrations.set('plex', plex);
                this.setupEventHandlers('plex', plex);
                results.plex = await plex.testConnection();
            } catch (error) {
                results.plex = { success: false, error: error.message };
            }
        }

        // Initialize Sonarr if configured
        if (this.config.sonarr || process.env.SONARR_URL) {
            try {
                const sonarr = new SonarrIntegration(this.config.sonarr);
                this.integrations.set('sonarr', sonarr);
                this.setupEventHandlers('sonarr', sonarr);
                results.sonarr = await sonarr.testConnection();
            } catch (error) {
                results.sonarr = { success: false, error: error.message };
            }
        }

        // Initialize Radarr if configured
        if (this.config.radarr || process.env.RADARR_URL) {
            try {
                const radarr = new RadarrIntegration(this.config.radarr);
                this.integrations.set('radarr', radarr);
                this.setupEventHandlers('radarr', radarr);
                results.radarr = await radarr.testConnection();
            } catch (error) {
                results.radarr = { success: false, error: error.message };
            }
        }

        // Initialize Prowlarr if configured
        if (this.config.prowlarr || process.env.PROWLARR_URL) {
            try {
                const prowlarr = new ProwlarrIntegration(this.config.prowlarr);
                this.integrations.set('prowlarr', prowlarr);
                this.setupEventHandlers('prowlarr', prowlarr);
                results.prowlarr = await prowlarr.testConnection();
            } catch (error) {
                results.prowlarr = { success: false, error: error.message };
            }
        }

        // Initialize Jellyseerr if configured
        if (this.config.jellyseerr || process.env.JELLYSEERR_URL) {
            try {
                const jellyseerr = new JellyseerrIntegration(this.config.jellyseerr);
                this.integrations.set('jellyseerr', jellyseerr);
                this.setupEventHandlers('jellyseerr', jellyseerr);
                results.jellyseerr = await jellyseerr.testConnection();
            } catch (error) {
                results.jellyseerr = { success: false, error: error.message };
            }
        }

        // Initialize Tautulli if configured
        if (this.config.tautulli || process.env.TAUTULLI_URL) {
            try {
                const tautulli = new TautulliIntegration(this.config.tautulli);
                this.integrations.set('tautulli', tautulli);
                this.setupEventHandlers('tautulli', tautulli);
                results.tautulli = await tautulli.testConnection();
            } catch (error) {
                results.tautulli = { success: false, error: error.message };
            }
        }

        // Initialize NetFlow if configured
        if (this.config.netflow || process.env.NETFLOW_COLLECTOR_URL) {
            try {
                const netflow = new NetflowIntegration(this.config.netflow);
                this.integrations.set('netflow', netflow);
                this.setupEventHandlers('netflow', netflow);
                results.netflow = await netflow.testConnection();
            } catch (error) {
                results.netflow = { success: false, error: error.message };
            }
        }

        return results;
    }

    /**
     * Setup event handlers for integration
     */
    setupEventHandlers(serviceName, integration) {
        const handlers = {
            error: (error) => {
                logger.error(`${serviceName} error:`, error);
                this.emit('integrationError', { service: serviceName, error });
            },
            webhook: (data) => {
                logger.info(`${serviceName} webhook:`, data);
                this.emit('webhookReceived', { service: serviceName, data });
            }
        };

        this.eventHandlers.set(serviceName, handlers);

        // Attach handlers
        Object.entries(handlers).forEach(([event, handler]) => {
            integration.on(event, handler);
        });
    }

    /**
     * Get integration by name
     */
    getIntegration(name) {
        return this.integrations.get(name.toLowerCase());
    }

    /**
     * Get all integrations
     */
    getAllIntegrations() {
        return Object.fromEntries(this.integrations);
    }

    /**
     * Get integration status
     */
    async getStatus() {
        const status = {};
        
        for (const [name, integration] of this.integrations) {
            try {
                status[name] = await integration.testConnection();
            } catch (error) {
                status[name] = { success: false, error: error.message };
            }
        }
        
        return status;
    }

    /**
     * Setup all webhooks on Express app
     */
    setupWebhooks(app) {
        for (const [name, integration] of this.integrations) {
            if (typeof integration.setupWebhook === 'function') {
                integration.setupWebhook(app);
                logger.info(`Webhook setup for ${name}`);
            }
        }
    }

    /**
     * Get comprehensive statistics from all services
     */
    async getComprehensiveStats() {
        const stats = {
            timestamp: new Date(),
            services: {}
        };

        for (const [name, integration] of this.integrations) {
            try {
                if (typeof integration.getStatistics === 'function') {
                    stats.services[name] = await integration.getStatistics();
                }
            } catch (error) {
                stats.services[name] = { error: error.message };
            }
        }

        return stats;
    }

    /**
     * Search across all compatible services
     */
    async searchAll(query) {
        const results = {};

        for (const [name, integration] of this.integrations) {
            try {
                if (typeof integration.search === 'function') {
                    results[name] = await integration.search(query);
                } else if (typeof integration.searchMovies === 'function' && name === 'radarr') {
                    results[name] = await integration.searchMovies(query);
                } else if (typeof integration.searchSeries === 'function' && name === 'sonarr') {
                    results[name] = await integration.searchSeries(query);
                } else if (typeof integration.searchMedia === 'function') {
                    results[name] = await integration.searchMedia(query);
                }
            } catch (error) {
                results[name] = { error: error.message };
            }
        }

        return results;
    }

    /**
     * Cleanup all integrations
     */
    cleanup() {
        for (const [name, integration] of this.integrations) {
            try {
                // Remove event handlers
                const handlers = this.eventHandlers.get(name);
                if (handlers) {
                    Object.entries(handlers).forEach(([event, handler]) => {
                        integration.removeListener(event, handler);
                    });
                }

                // Cleanup integration if method exists
                if (typeof integration.cleanup === 'function') {
                    integration.cleanup();
                }
            } catch (error) {
                logger.error(`Error cleaning up ${name}:`, error);
            }
        }

        this.integrations.clear();
        this.eventHandlers.clear();
    }

    /**
     * Emit events (EventEmitter pattern)
     */
    emit(event, data) {
        // Simple event emission - could be extended with proper EventEmitter
        logger.info(`IntegrationsManager event: ${event}`, data);
    }
}

module.exports = {
    JellyfinIntegration,
    PlexIntegration,
    SonarrIntegration,
    RadarrIntegration,
    ProwlarrIntegration,
    JellyseerrIntegration,
    TautulliIntegration,
    NetflowIntegration,
    IntegrationsManager
};