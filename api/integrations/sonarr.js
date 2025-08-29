const logger = require('../../middleware/logger.js');
/**
 * Sonarr Integration Wrapper
 * Simplified interface for Sonarr TV series management
 */

const SonarrIntegration = require('./SonarrIntegration');
const EventEmitter = require('events');

/**
 * Factory function to create a Sonarr integration instance
 * @param {Object} config - Configuration options
 * @returns {SonarrIntegration} Configured Sonarr integration instance
 */
function createSonarrIntegration(config = {}) {
    return new SonarrIntegration(config);
}

/**
 * Default configuration for Sonarr integration
 */
const defaultConfig = {
    baseURL: process.env.SONARR_URL || 'http://localhost:8989',
    apiKey: process.env.SONARR_API_KEY,
    timeout: 30000,
    retries: 3,
    webhookEnabled: true,
    version: 'v3'
};

/**
 * Quick setup function for common use cases
 * @param {Object} options - Setup options
 * @returns {Promise<SonarrIntegration>} Configured and authenticated integration
 */
async function quickSetup(options = {}) {
    const config = { ...defaultConfig, ...options };
    const sonarr = new SonarrIntegration(config);
    
    try {
        // Test connection
        const connectionResult = await sonarr.testConnection();
        if (connectionResult.success) {
            logger.info('✅ Sonarr integration setup successfully');
        } else {
            logger.warn('⚠️ Sonarr connection test failed:', connectionResult.error);
        }
        
        return sonarr;
    } catch (error) {
        logger.error('❌ Sonarr quick setup failed:', error.message);
        throw error;
    }
}

/**
 * Utility functions for common Sonarr operations
 */
const utils = {
    /**
     * Format series info for display
     * @param {Object} series - Series object from Sonarr
     * @returns {Object} Formatted series info
     */
    formatSeriesInfo(series) {
        return {
            id: series.id,
            title: series.title,
            sortTitle: series.sortTitle,
            status: series.status,
            overview: series.overview,
            network: series.network,
            airTime: series.airTime,
            seasons: series.seasonCount || 0,
            episodeFileCount: series.episodeFileCount || 0,
            totalEpisodeCount: series.totalEpisodeCount || 0,
            sizeOnDisk: series.sizeOnDisk || 0,
            qualityProfile: series.qualityProfileId,
            languageProfile: series.languageProfileId,
            monitored: series.monitored,
            year: series.year,
            path: series.path,
            tvdbId: series.tvdbId,
            imdbId: series.imdbId,
            tags: series.tags || [],
            images: series.images || []
        };
    },

    /**
     * Format episode info for display
     * @param {Object} episode - Episode object from Sonarr
     * @returns {Object} Formatted episode info
     */
    formatEpisodeInfo(episode) {
        return {
            id: episode.id,
            seriesId: episode.seriesId,
            title: episode.title,
            episodeNumber: episode.episodeNumber,
            seasonNumber: episode.seasonNumber,
            overview: episode.overview,
            airDate: episode.airDate,
            airDateUtc: episode.airDateUtc,
            hasFile: episode.hasFile,
            monitored: episode.monitored,
            absoluteEpisodeNumber: episode.absoluteEpisodeNumber,
            sceneEpisodeNumber: episode.sceneEpisodeNumber,
            sceneSeasonNumber: episode.sceneSeasonNumber,
            tvDbEpisodeId: episode.tvDbEpisodeId,
            grabDate: episode.grabDate
        };
    },

    /**
     * Parse Sonarr quality profile
     * @param {Object} profile - Quality profile object
     * @returns {Object} Parsed quality profile
     */
    parseQualityProfile(profile) {
        return {
            id: profile.id,
            name: profile.name,
            cutoff: profile.cutoff,
            items: profile.items?.map(item => ({
                id: item.id,
                name: item.quality?.name,
                allowed: item.allowed,
                quality: item.quality
            })) || []
        };
    },

    /**
     * Parse Sonarr webhook payload
     * @param {Object} payload - Webhook payload
     * @returns {Object} Parsed webhook data
     */
    parseWebhookPayload(payload) {
        return {
            event: payload.eventType,
            timestamp: new Date(payload.dateTime || Date.now()),
            series: payload.series ? {
                id: payload.series.id,
                title: payload.series.title,
                tvdbId: payload.series.tvdbId,
                imdbId: payload.series.imdbId
            } : null,
            episodes: payload.episodes?.map(ep => ({
                id: ep.id,
                title: ep.title,
                episodeNumber: ep.episodeNumber,
                seasonNumber: ep.seasonNumber,
                airDate: ep.airDate
            })) || [],
            episodeFile: payload.episodeFile ? {
                id: payload.episodeFile.id,
                relativePath: payload.episodeFile.relativePath,
                path: payload.episodeFile.path,
                quality: payload.episodeFile.quality,
                size: payload.episodeFile.size
            } : null,
            release: payload.release ? {
                quality: payload.release.quality,
                qualityWeight: payload.release.qualityWeight,
                size: payload.release.size,
                title: payload.release.title,
                indexer: payload.release.indexer,
                releaseGroup: payload.release.releaseGroup
            } : null
        };
    },

    /**
     * Calculate series completion percentage
     * @param {Object} series - Series object
     * @returns {number} Completion percentage
     */
    getSeriesCompletion(series) {
        if (!series.totalEpisodeCount || series.totalEpisodeCount === 0) {
            return 0;
        }
        return Math.round((series.episodeFileCount / series.totalEpisodeCount) * 100);
    },

    /**
     * Format file size
     * @param {number} bytes - Size in bytes
     * @returns {string} Formatted size
     */
    formatFileSize(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
};

/**
 * Health check function
 * @param {Object} config - Sonarr configuration
 * @returns {Promise<Object>} Health check result
 */
async function healthCheck(config = {}) {
    try {
        const sonarr = createSonarrIntegration(config);
        const result = await sonarr.testConnection();
        
        return {
            service: 'sonarr',
            healthy: result.success,
            timestamp: new Date(),
            response_time: result.responseTime,
            version: result.version,
            error: result.success ? null : result.error
        };
    } catch (error) {
        return {
            service: 'sonarr',
            healthy: false,
            timestamp: new Date(),
            error: error.message
        };
    }
}

/**
 * Series management utilities
 */
const seriesManager = {
    /**
     * Add series with recommended settings
     * @param {SonarrIntegration} sonarr - Sonarr integration instance
     * @param {Object} seriesData - Series data to add
     * @param {Object} options - Addition options
     * @returns {Promise<Object>} Added series
     */
    async addSeriesWithDefaults(sonarr, seriesData, options = {}) {
        const profiles = await sonarr.getQualityProfiles();
        const rootFolders = await sonarr.getRootFolders();
        
        const defaultProfile = profiles.find(p => p.name.toLowerCase().includes('hd')) || profiles[0];
        const defaultRootFolder = rootFolders[0];
        
        return sonarr.addSeries({
            tvdbId: seriesData.tvdbId,
            title: seriesData.title,
            qualityProfileId: options.qualityProfileId || defaultProfile.id,
            languageProfileId: options.languageProfileId || 1,
            rootFolderPath: options.rootFolderPath || defaultRootFolder.path,
            monitored: options.monitored !== undefined ? options.monitored : true,
            seasonFolder: options.seasonFolder !== undefined ? options.seasonFolder : true,
            addOptions: {
                searchForMissingEpisodes: options.searchForMissing !== undefined ? options.searchForMissing : true,
                searchForCutoffUnmetEpisodes: false
            },
            ...seriesData
        });
    },

    /**
     * Bulk monitor seasons
     * @param {SonarrIntegration} sonarr - Sonarr integration instance
     * @param {number} seriesId - Series ID
     * @param {Array} seasonNumbers - Season numbers to monitor
     * @param {boolean} monitored - Monitor status
     * @returns {Promise<Array>} Updated seasons
     */
    async bulkMonitorSeasons(sonarr, seriesId, seasonNumbers, monitored = true) {
        const series = await sonarr.getSeries(seriesId);
        const updatedSeasons = series.seasons.map(season => ({
            ...season,
            monitored: seasonNumbers.includes(season.seasonNumber) ? monitored : season.monitored
        }));
        
        return sonarr.updateSeries({
            ...series,
            seasons: updatedSeasons
        });
    },

    /**
     * Get series statistics
     * @param {SonarrIntegration} sonarr - Sonarr integration instance
     * @returns {Promise<Object>} Series statistics
     */
    async getStatistics(sonarr) {
        const series = await sonarr.getAllSeries();
        const totalSeries = series.length;
        const monitoredSeries = series.filter(s => s.monitored).length;
        const completedSeries = series.filter(s => utils.getSeriesCompletion(s) === 100).length;
        const totalSize = series.reduce((sum, s) => sum + (s.sizeOnDisk || 0), 0);
        const totalEpisodes = series.reduce((sum, s) => sum + (s.totalEpisodeCount || 0), 0);
        const downloadedEpisodes = series.reduce((sum, s) => sum + (s.episodeFileCount || 0), 0);
        
        return {
            totalSeries,
            monitoredSeries,
            completedSeries,
            totalSize: utils.formatFileSize(totalSize),
            totalEpisodes,
            downloadedEpisodes,
            completionPercentage: totalEpisodes > 0 ? Math.round((downloadedEpisodes / totalEpisodes) * 100) : 0
        };
    }
};

module.exports = {
    SonarrIntegration,
    createSonarrIntegration,
    quickSetup,
    defaultConfig,
    utils,
    healthCheck,
    seriesManager,
    
    // Aliases for convenience
    create: createSonarrIntegration,
    setup: quickSetup,
    Integration: SonarrIntegration
};