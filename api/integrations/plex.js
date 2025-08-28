/**
 * Plex Integration Wrapper
 * Simplified interface for Plex media server integration
 */

const PlexIntegration = require('./PlexIntegration');
const EventEmitter = require('events');

/**
 * Factory function to create a Plex integration instance
 * @param {Object} config - Configuration options
 * @returns {PlexIntegration} Configured Plex integration instance
 */
function createPlexIntegration(config = {}) {
    return new PlexIntegration(config);
}

/**
 * Default configuration for Plex integration
 */
const defaultConfig = {
    baseURL: process.env.PLEX_URL || 'http://localhost:32400',
    token: process.env.PLEX_TOKEN,
    username: process.env.PLEX_USERNAME,
    password: process.env.PLEX_PASSWORD,
    timeout: 30000,
    retries: 3,
    webhookEnabled: true,
    clientIdentifier: 'media-server-api'
};

/**
 * Quick setup function for common use cases
 * @param {Object} options - Setup options
 * @returns {Promise<PlexIntegration>} Configured and authenticated integration
 */
async function quickSetup(options = {}) {
    const config = { ...defaultConfig, ...options };
    const plex = new PlexIntegration(config);
    
    try {
        // Test connection and authenticate if needed
        const connectionResult = await plex.testConnection();
        if (connectionResult.success) {
            console.log('✅ Plex integration setup successfully');
        } else {
            console.warn('⚠️ Plex connection test failed:', connectionResult.error);
        }
        
        return plex;
    } catch (error) {
        console.error('❌ Plex quick setup failed:', error.message);
        throw error;
    }
}

/**
 * Utility functions for common Plex operations
 */
const utils = {
    /**
     * Format media info for display
     * @param {Object} media - Media object from Plex
     * @returns {Object} Formatted media info
     */
    formatMediaInfo(media) {
        return {
            id: media.ratingKey,
            title: media.title,
            type: media.type,
            year: media.year,
            duration: media.duration ? Math.round(media.duration / 60000) : null,
            summary: media.summary,
            genres: media.Genre?.map(g => g.tag) || [],
            rating: media.rating,
            key: media.key,
            thumb: media.thumb,
            art: media.art,
            addedAt: media.addedAt ? new Date(media.addedAt * 1000) : null,
            viewCount: media.viewCount || 0,
            lastViewedAt: media.lastViewedAt ? new Date(media.lastViewedAt * 1000) : null
        };
    },

    /**
     * Build Plex stream URL
     * @param {string} baseURL - Plex server URL
     * @param {string} key - Media key
     * @param {string} token - Plex token
     * @param {Object} options - Streaming options
     * @returns {string} Stream URL
     */
    buildStreamURL(baseURL, key, token, options = {}) {
        const params = new URLSearchParams({
            'X-Plex-Token': token,
            ...(options.videoProfile && { videoProfile: options.videoProfile }),
            ...(options.maxBitrate && { maxBitrate: options.maxBitrate }),
            ...(options.videoQuality && { videoQuality: options.videoQuality })
        });

        return `${baseURL}${key}?${params.toString()}`;
    },

    /**
     * Parse Plex webhook payload
     * @param {Object} payload - Webhook payload
     * @returns {Object} Parsed webhook data
     */
    parseWebhookPayload(payload) {
        const event = payload.event;
        const metadata = payload.Metadata || {};
        const account = payload.Account || {};
        const player = payload.Player || {};
        const server = payload.Server || {};
        
        return {
            event: event,
            timestamp: new Date(),
            user: account.title,
            userId: account.id,
            item: {
                id: metadata.ratingKey,
                title: metadata.title,
                type: metadata.type,
                year: metadata.year,
                thumb: metadata.thumb
            },
            player: {
                title: player.title,
                uuid: player.uuid,
                local: player.local
            },
            server: {
                title: server.title,
                uuid: server.uuid
            }
        };
    },

    /**
     * Convert Plex quality to readable format
     * @param {number} quality - Plex quality value
     * @returns {string} Readable quality
     */
    formatQuality(quality) {
        const qualityMap = {
            0: 'Original',
            1: '20 Mbps',
            2: '12 Mbps',
            3: '10 Mbps',
            4: '8 Mbps',
            5: '4 Mbps',
            6: '2 Mbps',
            7: '1.5 Mbps',
            8: '720 kbps',
            9: '320 kbps'
        };
        return qualityMap[quality] || `${quality} Mbps`;
    }
};

/**
 * Health check function
 * @param {Object} config - Plex configuration
 * @returns {Promise<Object>} Health check result
 */
async function healthCheck(config = {}) {
    try {
        const plex = createPlexIntegration(config);
        const result = await plex.testConnection();
        
        return {
            service: 'plex',
            healthy: result.success,
            timestamp: new Date(),
            response_time: result.responseTime,
            version: result.serverVersion,
            platform: result.platform,
            error: result.success ? null : result.error
        };
    } catch (error) {
        return {
            service: 'plex',
            healthy: false,
            timestamp: new Date(),
            error: error.message
        };
    }
}

/**
 * Library scanning utilities
 */
const scanner = {
    /**
     * Scan all libraries
     * @param {PlexIntegration} plex - Plex integration instance
     * @returns {Promise<Array>} Scan results
     */
    async scanAllLibraries(plex) {
        const libraries = await plex.getLibraries();
        const scanPromises = libraries.map(lib => 
            plex.refreshLibrary(lib.key).catch(err => ({ 
                library: lib.title, 
                error: err.message 
            }))
        );
        
        return Promise.allSettled(scanPromises);
    },

    /**
     * Get library statistics
     * @param {PlexIntegration} plex - Plex integration instance
     * @returns {Promise<Object>} Library statistics
     */
    async getLibraryStats(plex) {
        const libraries = await plex.getLibraries();
        const stats = {};
        
        for (const library of libraries) {
            try {
                const items = await plex.getLibraryItems(library.key, { limit: 0 });
                stats[library.title] = {
                    type: library.type,
                    count: items.totalSize || 0,
                    updatedAt: new Date(library.updatedAt * 1000)
                };
            } catch (error) {
                stats[library.title] = { error: error.message };
            }
        }
        
        return stats;
    }
};

module.exports = {
    PlexIntegration,
    createPlexIntegration,
    quickSetup,
    defaultConfig,
    utils,
    healthCheck,
    scanner,
    
    // Aliases for convenience
    create: createPlexIntegration,
    setup: quickSetup,
    Integration: PlexIntegration
};