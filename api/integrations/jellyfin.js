const logger = require('../../middleware/logger.js');
/**
 * Jellyfin Integration Wrapper
 * Simplified interface for Jellyfin media server integration
 */

const JellyfinIntegration = require('./JellyfinIntegration');
const EventEmitter = require('events');

/**
 * Factory function to create a Jellyfin integration instance
 * @param {Object} config - Configuration options
 * @returns {JellyfinIntegration} Configured Jellyfin integration instance
 */
function createJellyfinIntegration(config = {}) {
    return new JellyfinIntegration(config);
}

/**
 * Default configuration for Jellyfin integration
 */
const defaultConfig = {
    baseURL: process.env.JELLYFIN_URL || 'http://localhost:8096',
    apiKey: process.env.JELLYFIN_API_KEY,
    username: process.env.JELLYFIN_USERNAME,
    password: process.env.JELLYFIN_PASSWORD,
    timeout: 30000,
    retries: 3,
    webhookEnabled: true
};

/**
 * Quick setup function for common use cases
 * @param {Object} options - Setup options
 * @returns {Promise<JellyfinIntegration>} Configured and authenticated integration
 */
async function quickSetup(options = {}) {
    const config = { ...defaultConfig, ...options };
    const jellyfin = new JellyfinIntegration(config);
    
    try {
        // Test connection and authenticate if needed
        const connectionResult = await jellyfin.testConnection();
        if (connectionResult.success) {
            logger.info('✅ Jellyfin integration setup successfully');
        } else {
            logger.warn('⚠️ Jellyfin connection test failed:', connectionResult.error);
        }
        
        return jellyfin;
    } catch (error) {
        logger.error('❌ Jellyfin quick setup failed:', error.message);
        throw error;
    }
}

/**
 * Utility functions for common Jellyfin operations
 */
const utils = {
    /**
     * Format media info for display
     * @param {Object} media - Media object from Jellyfin
     * @returns {Object} Formatted media info
     */
    formatMediaInfo(media) {
        return {
            id: media.Id,
            name: media.Name,
            type: media.Type,
            year: media.ProductionYear,
            runtime: media.RunTimeTicks ? Math.round(media.RunTimeTicks / 10000000 / 60) : null,
            overview: media.Overview,
            genres: media.Genres || [],
            rating: media.CommunityRating,
            path: media.Path,
            dateAdded: media.DateCreated,
            playCount: media.UserData?.PlayCount || 0,
            favorite: media.UserData?.IsFavorite || false
        };
    },

    /**
     * Build Jellyfin stream URL
     * @param {string} baseURL - Jellyfin server URL
     * @param {string} itemId - Media item ID
     * @param {string} accessToken - User access token
     * @param {Object} options - Streaming options
     * @returns {string} Stream URL
     */
    buildStreamURL(baseURL, itemId, accessToken, options = {}) {
        const params = new URLSearchParams({
            api_key: accessToken,
            VideoCodec: options.videoCodec || 'h264',
            AudioCodec: options.audioCodec || 'aac',
            MaxVideoBitrate: options.maxBitrate || '8000000',
            MaxAudioBitrate: options.maxAudioBitrate || '320000'
        });

        return `${baseURL}/Videos/${itemId}/stream?${params.toString()}`;
    },

    /**
     * Parse Jellyfin webhook payload
     * @param {Object} payload - Webhook payload
     * @returns {Object} Parsed webhook data
     */
    parseWebhookPayload(payload) {
        return {
            event: payload.NotificationType,
            timestamp: new Date(payload.Timestamp || Date.now()),
            user: payload.User?.Name,
            userId: payload.UserId,
            item: payload.Item ? {
                id: payload.Item.Id,
                name: payload.Item.Name,
                type: payload.Item.Type,
                year: payload.Item.ProductionYear
            } : null,
            device: payload.DeviceName,
            session: payload.Session ? {
                id: payload.Session.Id,
                client: payload.Session.Client,
                deviceName: payload.Session.DeviceName
            } : null
        };
    }
};

/**
 * Health check function
 * @param {Object} config - Jellyfin configuration
 * @returns {Promise<Object>} Health check result
 */
async function healthCheck(config = {}) {
    try {
        const jellyfin = createJellyfinIntegration(config);
        const result = await jellyfin.testConnection();
        
        return {
            service: 'jellyfin',
            healthy: result.success,
            timestamp: new Date(),
            response_time: result.responseTime,
            version: result.serverVersion,
            error: result.success ? null : result.error
        };
    } catch (error) {
        return {
            service: 'jellyfin',
            healthy: false,
            timestamp: new Date(),
            error: error.message
        };
    }
}

module.exports = {
    JellyfinIntegration,
    createJellyfinIntegration,
    quickSetup,
    defaultConfig,
    utils,
    healthCheck,
    
    // Aliases for convenience
    create: createJellyfinIntegration,
    setup: quickSetup,
    Integration: JellyfinIntegration
};