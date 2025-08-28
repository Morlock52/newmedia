/**
 * Radarr Integration Wrapper
 * Simplified interface for Radarr movie management
 */

const RadarrIntegration = require('./RadarrIntegration');
const EventEmitter = require('events');

/**
 * Factory function to create a Radarr integration instance
 * @param {Object} config - Configuration options
 * @returns {RadarrIntegration} Configured Radarr integration instance
 */
function createRadarrIntegration(config = {}) {
    return new RadarrIntegration(config);
}

/**
 * Default configuration for Radarr integration
 */
const defaultConfig = {
    baseURL: process.env.RADARR_URL || 'http://localhost:7878',
    apiKey: process.env.RADARR_API_KEY,
    timeout: 30000,
    retries: 3,
    webhookEnabled: true,
    version: 'v3'
};

/**
 * Quick setup function for common use cases
 * @param {Object} options - Setup options
 * @returns {Promise<RadarrIntegration>} Configured and authenticated integration
 */
async function quickSetup(options = {}) {
    const config = { ...defaultConfig, ...options };
    const radarr = new RadarrIntegration(config);
    
    try {
        // Test connection
        const connectionResult = await radarr.testConnection();
        if (connectionResult.success) {
            console.log('✅ Radarr integration setup successfully');
        } else {
            console.warn('⚠️ Radarr connection test failed:', connectionResult.error);
        }
        
        return radarr;
    } catch (error) {
        console.error('❌ Radarr quick setup failed:', error.message);
        throw error;
    }
}

/**
 * Utility functions for common Radarr operations
 */
const utils = {
    /**
     * Format movie info for display
     * @param {Object} movie - Movie object from Radarr
     * @returns {Object} Formatted movie info
     */
    formatMovieInfo(movie) {
        return {
            id: movie.id,
            title: movie.title,
            originalTitle: movie.originalTitle,
            sortTitle: movie.sortTitle,
            status: movie.status,
            overview: movie.overview,
            inCinemas: movie.inCinemas,
            physicalRelease: movie.physicalRelease,
            digitalRelease: movie.digitalRelease,
            runtime: movie.runtime,
            year: movie.year,
            tmdbId: movie.tmdbId,
            imdbId: movie.imdbId,
            titleSlug: movie.titleSlug,
            genres: movie.genres || [],
            tags: movie.tags || [],
            images: movie.images || [],
            website: movie.website,
            qualityProfile: movie.qualityProfileId,
            monitored: movie.monitored,
            minimumAvailability: movie.minimumAvailability,
            hasFile: movie.hasFile,
            path: movie.path,
            rootFolderPath: movie.rootFolderPath,
            folderName: movie.folderName,
            sizeOnDisk: movie.sizeOnDisk || 0,
            movieFile: movie.movieFile
        };
    },

    /**
     * Format movie file info for display
     * @param {Object} movieFile - Movie file object from Radarr
     * @returns {Object} Formatted movie file info
     */
    formatMovieFileInfo(movieFile) {
        return {
            id: movieFile.id,
            movieId: movieFile.movieId,
            relativePath: movieFile.relativePath,
            path: movieFile.path,
            size: movieFile.size,
            dateAdded: movieFile.dateAdded,
            sceneName: movieFile.sceneName,
            releaseGroup: movieFile.releaseGroup,
            quality: movieFile.quality,
            indexerFlags: movieFile.indexerFlags,
            mediaInfo: movieFile.mediaInfo,
            originalFilePath: movieFile.originalFilePath,
            qualityCutoffNotMet: movieFile.qualityCutoffNotMet
        };
    },

    /**
     * Parse Radarr quality profile
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
            })) || [],
            minFormatScore: profile.minFormatScore,
            cutoffFormatScore: profile.cutoffFormatScore,
            formatItems: profile.formatItems || []
        };
    },

    /**
     * Parse Radarr webhook payload
     * @param {Object} payload - Webhook payload
     * @returns {Object} Parsed webhook data
     */
    parseWebhookPayload(payload) {
        return {
            event: payload.eventType,
            timestamp: new Date(payload.dateTime || Date.now()),
            movie: payload.movie ? {
                id: payload.movie.id,
                title: payload.movie.title,
                year: payload.movie.year,
                tmdbId: payload.movie.tmdbId,
                imdbId: payload.movie.imdbId,
                folderPath: payload.movie.folderPath
            } : null,
            movieFile: payload.movieFile ? {
                id: payload.movieFile.id,
                relativePath: payload.movieFile.relativePath,
                path: payload.movieFile.path,
                quality: payload.movieFile.quality,
                size: payload.movieFile.size,
                sceneName: payload.movieFile.sceneName,
                releaseGroup: payload.movieFile.releaseGroup
            } : null,
            release: payload.release ? {
                quality: payload.release.quality,
                qualityWeight: payload.release.qualityWeight,
                size: payload.release.size,
                title: payload.release.title,
                indexer: payload.release.indexer,
                releaseGroup: payload.release.releaseGroup
            } : null,
            remoteMovie: payload.remoteMovie ? {
                tmdbId: payload.remoteMovie.tmdbId,
                imdbId: payload.remoteMovie.imdbId,
                title: payload.remoteMovie.title,
                year: payload.remoteMovie.year
            } : null
        };
    },

    /**
     * Calculate movie availability status
     * @param {Object} movie - Movie object
     * @returns {string} Availability status
     */
    getAvailabilityStatus(movie) {
        const now = new Date();
        const inCinemas = movie.inCinemas ? new Date(movie.inCinemas) : null;
        const physicalRelease = movie.physicalRelease ? new Date(movie.physicalRelease) : null;
        const digitalRelease = movie.digitalRelease ? new Date(movie.digitalRelease) : null;
        
        if (movie.hasFile) {
            return 'Downloaded';
        }
        
        switch (movie.minimumAvailability) {
            case 'inCinemas':
                return inCinemas && inCinemas <= now ? 'Available' : 'Coming Soon';
            case 'physicalRelease':
                return physicalRelease && physicalRelease <= now ? 'Available' : 'Coming Soon';
            case 'released':
                return (digitalRelease && digitalRelease <= now) || (physicalRelease && physicalRelease <= now) ? 'Available' : 'Coming Soon';
            default:
                return 'Available';
        }
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
    },

    /**
     * Get movie poster URL
     * @param {Object} movie - Movie object
     * @param {string} baseURL - Radarr base URL
     * @returns {string|null} Poster URL
     */
    getPosterURL(movie, baseURL) {
        const poster = movie.images?.find(img => img.coverType === 'poster');
        return poster ? `${baseURL}${poster.url}` : null;
    }
};

/**
 * Health check function
 * @param {Object} config - Radarr configuration
 * @returns {Promise<Object>} Health check result
 */
async function healthCheck(config = {}) {
    try {
        const radarr = createRadarrIntegration(config);
        const result = await radarr.testConnection();
        
        return {
            service: 'radarr',
            healthy: result.success,
            timestamp: new Date(),
            response_time: result.responseTime,
            version: result.version,
            error: result.success ? null : result.error
        };
    } catch (error) {
        return {
            service: 'radarr',
            healthy: false,
            timestamp: new Date(),
            error: error.message
        };
    }
}

/**
 * Movie management utilities
 */
const movieManager = {
    /**
     * Add movie with recommended settings
     * @param {RadarrIntegration} radarr - Radarr integration instance
     * @param {Object} movieData - Movie data to add
     * @param {Object} options - Addition options
     * @returns {Promise<Object>} Added movie
     */
    async addMovieWithDefaults(radarr, movieData, options = {}) {
        const profiles = await radarr.getQualityProfiles();
        const rootFolders = await radarr.getRootFolders();
        
        const defaultProfile = profiles.find(p => p.name.toLowerCase().includes('hd')) || profiles[0];
        const defaultRootFolder = rootFolders[0];
        
        return radarr.addMovie({
            tmdbId: movieData.tmdbId,
            title: movieData.title,
            year: movieData.year,
            qualityProfileId: options.qualityProfileId || defaultProfile.id,
            rootFolderPath: options.rootFolderPath || defaultRootFolder.path,
            monitored: options.monitored !== undefined ? options.monitored : true,
            minimumAvailability: options.minimumAvailability || 'released',
            addOptions: {
                searchForMovie: options.searchForMovie !== undefined ? options.searchForMovie : true
            },
            ...movieData
        });
    },

    /**
     * Bulk update movie monitoring
     * @param {RadarrIntegration} radarr - Radarr integration instance
     * @param {Array} movieIds - Array of movie IDs
     * @param {boolean} monitored - Monitor status
     * @returns {Promise<Array>} Update results
     */
    async bulkMonitorMovies(radarr, movieIds, monitored = true) {
        const updatePromises = movieIds.map(async (id) => {
            try {
                const movie = await radarr.getMovie(id);
                return radarr.updateMovie({ ...movie, monitored });
            } catch (error) {
                return { id, error: error.message };
            }
        });
        
        return Promise.allSettled(updatePromises);
    },

    /**
     * Get movie statistics
     * @param {RadarrIntegration} radarr - Radarr integration instance
     * @returns {Promise<Object>} Movie statistics
     */
    async getStatistics(radarr) {
        const movies = await radarr.getAllMovies();
        const totalMovies = movies.length;
        const monitoredMovies = movies.filter(m => m.monitored).length;
        const downloadedMovies = movies.filter(m => m.hasFile).length;
        const availableMovies = movies.filter(m => utils.getAvailabilityStatus(m) === 'Available').length;
        const totalSize = movies.reduce((sum, m) => sum + (m.sizeOnDisk || 0), 0);
        
        // Group by status
        const statusBreakdown = movies.reduce((acc, movie) => {
            const status = utils.getAvailabilityStatus(movie);
            acc[status] = (acc[status] || 0) + 1;
            return acc;
        }, {});
        
        // Group by year
        const yearBreakdown = movies.reduce((acc, movie) => {
            const year = movie.year || 'Unknown';
            acc[year] = (acc[year] || 0) + 1;
            return acc;
        }, {});
        
        return {
            totalMovies,
            monitoredMovies,
            downloadedMovies,
            availableMovies,
            totalSize: utils.formatFileSize(totalSize),
            completionPercentage: totalMovies > 0 ? Math.round((downloadedMovies / totalMovies) * 100) : 0,
            statusBreakdown,
            yearBreakdown
        };
    },

    /**
     * Search for missing movies
     * @param {RadarrIntegration} radarr - Radarr integration instance
     * @param {Object} options - Search options
     * @returns {Promise<Object>} Search results
     */
    async searchMissingMovies(radarr, options = {}) {
        const movies = await radarr.getAllMovies();
        const missingMovies = movies.filter(m => 
            m.monitored && 
            !m.hasFile && 
            utils.getAvailabilityStatus(m) === 'Available'
        );
        
        if (options.search) {
            const searchPromises = missingMovies
                .slice(0, options.limit || 10)
                .map(movie => radarr.searchMovie(movie.id).catch(err => ({ 
                    movieId: movie.id, 
                    title: movie.title, 
                    error: err.message 
                })));
            
            const searchResults = await Promise.allSettled(searchPromises);
            return {
                totalMissing: missingMovies.length,
                searched: searchResults.length,
                results: searchResults
            };
        }
        
        return {
            totalMissing: missingMovies.length,
            movies: missingMovies.map(m => ({
                id: m.id,
                title: m.title,
                year: m.year,
                minimumAvailability: m.minimumAvailability
            }))
        };
    }
};

module.exports = {
    RadarrIntegration,
    createRadarrIntegration,
    quickSetup,
    defaultConfig,
    utils,
    healthCheck,
    movieManager,
    
    // Aliases for convenience
    create: createRadarrIntegration,
    setup: quickSetup,
    Integration: RadarrIntegration
};