/**
 * Jellyseerr Integration Wrapper
 * Simplified interface for Jellyseerr request management
 */

const JellyseerrIntegration = require('./JellyseerrIntegration');
const EventEmitter = require('events');

/**
 * Factory function to create a Jellyseerr integration instance
 * @param {Object} config - Configuration options
 * @returns {JellyseerrIntegration} Configured Jellyseerr integration instance
 */
function createJellyseerrIntegration(config = {}) {
    return new JellyseerrIntegration(config);
}

/**
 * Default configuration for Jellyseerr integration
 */
const defaultConfig = {
    baseURL: process.env.JELLYSEERR_URL || 'http://localhost:5055',
    apiKey: process.env.JELLYSEERR_API_KEY,
    timeout: 30000,
    retries: 3,
    webhookEnabled: true,
    version: 'v1'
};

/**
 * Quick setup function for common use cases
 * @param {Object} options - Setup options
 * @returns {Promise<JellyseerrIntegration>} Configured and authenticated integration
 */
async function quickSetup(options = {}) {
    const config = { ...defaultConfig, ...options };
    const jellyseerr = new JellyseerrIntegration(config);
    
    try {
        // Test connection
        const connectionResult = await jellyseerr.testConnection();
        if (connectionResult.success) {
            console.log('✅ Jellyseerr integration setup successfully');
        } else {
            console.warn('⚠️ Jellyseerr connection test failed:', connectionResult.error);
        }
        
        return jellyseerr;
    } catch (error) {
        console.error('❌ Jellyseerr quick setup failed:', error.message);
        throw error;
    }
}

/**
 * Utility functions for common Jellyseerr operations
 */
const utils = {
    /**
     * Format request info for display
     * @param {Object} request - Request object from Jellyseerr
     * @returns {Object} Formatted request info
     */
    formatRequestInfo(request) {
        return {
            id: request.id,
            status: request.status,
            createdAt: request.createdAt,
            updatedAt: request.updatedAt,
            type: request.type,
            is4k: request.is4k,
            serverId: request.serverId,
            profileId: request.profileId,
            rootFolder: request.rootFolder,
            languageProfileId: request.languageProfileId,
            tags: request.tags || [],
            requestedBy: request.requestedBy ? {
                id: request.requestedBy.id,
                email: request.requestedBy.email,
                username: request.requestedBy.plexUsername || request.requestedBy.jellyfinUsername,
                displayName: request.requestedBy.displayName,
                avatar: request.requestedBy.avatar
            } : null,
            modifiedBy: request.modifiedBy ? {
                id: request.modifiedBy.id,
                email: request.modifiedBy.email,
                username: request.modifiedBy.plexUsername || request.modifiedBy.jellyfinUsername,
                displayName: request.modifiedBy.displayName
            } : null,
            media: request.media ? {
                id: request.media.id,
                mediaType: request.media.mediaType,
                tmdbId: request.media.tmdbId,
                tvdbId: request.media.tvdbId,
                imdbId: request.media.imdbId,
                status: request.media.status,
                status4k: request.media.status4k,
                createdAt: request.media.createdAt,
                updatedAt: request.media.updatedAt
            } : null
        };
    },

    /**
     * Format media info for display
     * @param {Object} media - Media object from Jellyseerr
     * @returns {Object} Formatted media info
     */
    formatMediaInfo(media) {
        return {
            id: media.id,
            mediaType: media.mediaType,
            tmdbId: media.tmdbId,
            tvdbId: media.tvdbId,
            imdbId: media.imdbId,
            status: media.status,
            status4k: media.status4k,
            createdAt: media.createdAt,
            updatedAt: media.updatedAt,
            lastSeasonSearched: media.lastSeasonSearched,
            serviceId: media.serviceId,
            serviceId4k: media.serviceId4k,
            externalServiceId: media.externalServiceId,
            externalServiceId4k: media.externalServiceId4k,
            externalServiceSlug: media.externalServiceSlug,
            externalServiceSlug4k: media.externalServiceSlug4k,
            ratingKey: media.ratingKey,
            ratingKey4k: media.ratingKey4k,
            requests: media.requests?.map(r => utils.formatRequestInfo(r)) || []
        };
    },

    /**
     * Format user info for display
     * @param {Object} user - User object from Jellyseerr
     * @returns {Object} Formatted user info
     */
    formatUserInfo(user) {
        return {
            id: user.id,
            email: user.email,
            username: user.plexUsername || user.jellyfinUsername,
            displayName: user.displayName,
            avatar: user.avatar,
            userType: user.userType,
            permissions: user.permissions,
            movieQuotaLimit: user.movieQuotaLimit,
            movieQuotaDays: user.movieQuotaDays,
            tvQuotaLimit: user.tvQuotaLimit,
            tvQuotaDays: user.tvQuotaDays,
            createdAt: user.createdAt,
            updatedAt: user.updatedAt,
            requestCount: user.requestCount,
            settings: user.settings
        };
    },

    /**
     * Parse Jellyseerr webhook payload
     * @param {Object} payload - Webhook payload
     * @returns {Object} Parsed webhook data
     */
    parseWebhookPayload(payload) {
        return {
            notificationType: payload.notification_type,
            event: payload.event,
            subject: payload.subject,
            message: payload.message,
            image: payload.image,
            timestamp: new Date(),
            request: payload.request ? {
                id: payload.request.request_id,
                type: payload.request.requestedBy_requestType,
                status: payload.request.request_status,
                is4k: payload.request.request_is4k,
                requestedBy: payload.request.requestedBy_username || payload.request.requestedBy_email
            } : null,
            media: payload.media ? {
                mediaType: payload.media.media_type,
                tmdbId: payload.media.tmdbid,
                tvdbId: payload.media.tvdbid,
                imdbId: payload.media.imdbid,
                status: payload.media.media_status,
                status4k: payload.media.media_status4k
            } : null,
            issue: payload.issue ? {
                id: payload.issue.issue_id,
                type: payload.issue.issue_type,
                status: payload.issue.issue_status,
                message: payload.issue.issue_message,
                createdBy: payload.issue.reportedBy_username || payload.issue.reportedBy_email
            } : null,
            comment: payload.comment ? {
                id: payload.comment.comment_id,
                message: payload.comment.comment_message,
                user: payload.comment.commentedBy_username || payload.comment.commentedBy_email
            } : null
        };
    },

    /**
     * Get request status description
     * @param {number} status - Status code
     * @returns {string} Status description
     */
    getRequestStatusDescription(status) {
        const statusMap = {
            1: 'Pending',
            2: 'Approved',
            3: 'Declined',
            4: 'Processing'
        };
        return statusMap[status] || 'Unknown';
    },

    /**
     * Get media status description
     * @param {number} status - Status code
     * @returns {string} Status description
     */
    getMediaStatusDescription(status) {
        const statusMap = {
            1: 'Unknown',
            2: 'Pending',
            3: 'Processing',
            4: 'Partially Available',
            5: 'Available'
        };
        return statusMap[status] || 'Unknown';
    },

    /**
     * Calculate user quota usage
     * @param {Object} user - User object
     * @param {Array} requests - User's requests
     * @returns {Object} Quota usage info
     */
    calculateQuotaUsage(user, requests) {
        const now = new Date();
        const movieQuotaDays = user.movieQuotaDays || 30;
        const tvQuotaDays = user.tvQuotaDays || 30;
        
        const movieCutoff = new Date(now.getTime() - (movieQuotaDays * 24 * 60 * 60 * 1000));
        const tvCutoff = new Date(now.getTime() - (tvQuotaDays * 24 * 60 * 60 * 1000));
        
        const recentMovieRequests = requests.filter(r => 
            r.type === 'movie' && new Date(r.createdAt) > movieCutoff
        ).length;
        
        const recentTvRequests = requests.filter(r => 
            r.type === 'tv' && new Date(r.createdAt) > tvCutoff
        ).length;
        
        return {
            movie: {
                used: recentMovieRequests,
                limit: user.movieQuotaLimit || null,
                days: movieQuotaDays,
                unlimited: !user.movieQuotaLimit,
                remaining: user.movieQuotaLimit ? Math.max(0, user.movieQuotaLimit - recentMovieRequests) : null
            },
            tv: {
                used: recentTvRequests,
                limit: user.tvQuotaLimit || null,
                days: tvQuotaDays,
                unlimited: !user.tvQuotaLimit,
                remaining: user.tvQuotaLimit ? Math.max(0, user.tvQuotaLimit - recentTvRequests) : null
            }
        };
    }
};

/**
 * Health check function
 * @param {Object} config - Jellyseerr configuration
 * @returns {Promise<Object>} Health check result
 */
async function healthCheck(config = {}) {
    try {
        const jellyseerr = createJellyseerrIntegration(config);
        const result = await jellyseerr.testConnection();
        
        return {
            service: 'jellyseerr',
            healthy: result.success,
            timestamp: new Date(),
            response_time: result.responseTime,
            version: result.version,
            error: result.success ? null : result.error
        };
    } catch (error) {
        return {
            service: 'jellyseerr',
            healthy: false,
            timestamp: new Date(),
            error: error.message
        };
    }
}

/**
 * Request management utilities
 */
const requestManager = {
    /**
     * Get request statistics
     * @param {JellyseerrIntegration} jellyseerr - Jellyseerr integration instance
     * @returns {Promise<Object>} Request statistics
     */
    async getRequestStats(jellyseerr) {
        try {
            const requests = await jellyseerr.getRequests({ take: 1000 });
            const allRequests = requests.results || [];
            
            const totalRequests = allRequests.length;
            const pendingRequests = allRequests.filter(r => r.status === 1).length;
            const approvedRequests = allRequests.filter(r => r.status === 2).length;
            const declinedRequests = allRequests.filter(r => r.status === 3).length;
            const processingRequests = allRequests.filter(r => r.status === 4).length;
            
            const movieRequests = allRequests.filter(r => r.type === 'movie').length;
            const tvRequests = allRequests.filter(r => r.type === 'tv').length;
            const fourKRequests = allRequests.filter(r => r.is4k).length;
            
            // Recent requests (last 30 days)
            const thirtyDaysAgo = new Date(Date.now() - (30 * 24 * 60 * 60 * 1000));
            const recentRequests = allRequests.filter(r => 
                new Date(r.createdAt) > thirtyDaysAgo
            ).length;
            
            return {
                totalRequests,
                pendingRequests,
                approvedRequests,
                declinedRequests,
                processingRequests,
                movieRequests,
                tvRequests,
                fourKRequests,
                recentRequests,
                statusBreakdown: {
                    pending: pendingRequests,
                    approved: approvedRequests,
                    declined: declinedRequests,
                    processing: processingRequests
                },
                typeBreakdown: {
                    movie: movieRequests,
                    tv: tvRequests
                }
            };
        } catch (error) {
            return {
                error: error.message,
                totalRequests: 0
            };
        }
    },

    /**
     * Process pending requests
     * @param {JellyseerrIntegration} jellyseerr - Jellyseerr integration instance
     * @param {Object} options - Processing options
     * @returns {Promise<Object>} Processing results
     */
    async processPendingRequests(jellyseerr, options = {}) {
        try {
            const requests = await jellyseerr.getRequests({ 
                filter: 'pending',
                take: options.limit || 50
            });
            
            const pendingRequests = requests.results || [];
            const processed = [];
            const errors = [];
            
            for (const request of pendingRequests) {
                try {
                    if (options.autoApprove) {
                        const result = await jellyseerr.updateRequestStatus(request.id, {
                            status: 2 // Approved
                        });
                        processed.push({
                            requestId: request.id,
                            action: 'approved',
                            result
                        });
                    }
                } catch (error) {
                    errors.push({
                        requestId: request.id,
                        error: error.message
                    });
                }
            }
            
            return {
                totalPending: pendingRequests.length,
                processed: processed.length,
                errors: errors.length,
                results: processed,
                errorDetails: errors
            };
        } catch (error) {
            return {
                error: error.message,
                processed: 0
            };
        }
    },

    /**
     * Get user request summary
     * @param {JellyseerrIntegration} jellyseerr - Jellyseerr integration instance
     * @param {number} userId - User ID
     * @returns {Promise<Object>} User request summary
     */
    async getUserRequestSummary(jellyseerr, userId) {
        try {
            const user = await jellyseerr.getUser(userId);
            const userRequests = await jellyseerr.getUserRequests(userId);
            
            const quotaUsage = utils.calculateQuotaUsage(user, userRequests);
            
            const requestsByStatus = userRequests.reduce((acc, req) => {
                const status = utils.getRequestStatusDescription(req.status);
                acc[status] = (acc[status] || 0) + 1;
                return acc;
            }, {});
            
            const requestsByType = userRequests.reduce((acc, req) => {
                acc[req.type] = (acc[req.type] || 0) + 1;
                return acc;
            }, {});
            
            return {
                user: utils.formatUserInfo(user),
                totalRequests: userRequests.length,
                quotaUsage,
                requestsByStatus,
                requestsByType,
                recentRequests: userRequests
                    .sort((a, b) => new Date(b.createdAt) - new Date(a.createdAt))
                    .slice(0, 10)
                    .map(r => utils.formatRequestInfo(r))
            };
        } catch (error) {
            return {
                error: error.message,
                userId
            };
        }
    }
};

module.exports = {
    JellyseerrIntegration,
    createJellyseerrIntegration,
    quickSetup,
    defaultConfig,
    utils,
    healthCheck,
    requestManager,
    
    // Aliases for convenience
    create: createJellyseerrIntegration,
    setup: quickSetup,
    Integration: JellyseerrIntegration
};