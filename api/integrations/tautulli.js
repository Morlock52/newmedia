const logger = require('../../middleware/logger.js');
/**
 * Tautulli Integration Wrapper
 * Simplified interface for Tautulli analytics and monitoring
 */

const TautulliIntegration = require('./TautulliIntegration');
const EventEmitter = require('events');

/**
 * Factory function to create a Tautulli integration instance
 * @param {Object} config - Configuration options
 * @returns {TautulliIntegration} Configured Tautulli integration instance
 */
function createTautulliIntegration(config = {}) {
    return new TautulliIntegration(config);
}

/**
 * Default configuration for Tautulli integration
 */
const defaultConfig = {
    baseURL: process.env.TAUTULLI_URL || 'http://localhost:8181',
    apiKey: process.env.TAUTULLI_API_KEY,
    timeout: 30000,
    retries: 3,
    webhookEnabled: true
};

/**
 * Quick setup function for common use cases
 * @param {Object} options - Setup options
 * @returns {Promise<TautulliIntegration>} Configured and authenticated integration
 */
async function quickSetup(options = {}) {
    const config = { ...defaultConfig, ...options };
    const tautulli = new TautulliIntegration(config);
    
    try {
        // Test connection
        const connectionResult = await tautulli.testConnection();
        if (connectionResult.success) {
            logger.info('✅ Tautulli integration setup successfully');
        } else {
            logger.warn('⚠️ Tautulli connection test failed:', connectionResult.error);
        }
        
        return tautulli;
    } catch (error) {
        logger.error('❌ Tautulli quick setup failed:', error.message);
        throw error;
    }
}

/**
 * Utility functions for common Tautulli operations
 */
const utils = {
    /**
     * Format activity info for display
     * @param {Object} activity - Activity object from Tautulli
     * @returns {Object} Formatted activity info
     */
    formatActivityInfo(activity) {
        return {
            sessionKey: activity.session_key,
            sessionId: activity.session_id,
            username: activity.username,
            friendlyName: activity.friendly_name,
            userThumb: activity.user_thumb,
            title: activity.title,
            parentTitle: activity.parent_title,
            grandparentTitle: activity.grandparent_title,
            originalTitle: activity.original_title,
            year: activity.year,
            mediaType: activity.media_type,
            rating: activity.rating,
            audienceRating: activity.audience_rating,
            userRating: activity.user_rating,
            duration: activity.duration,
            viewOffset: activity.view_offset,
            progressPercent: activity.progress_percent,
            state: activity.state,
            mediaIndex: activity.media_index,
            parentMediaIndex: activity.parent_media_index,
            thumb: activity.thumb,
            parentThumb: activity.parent_thumb,
            grandparentThumb: activity.grandparent_thumb,
            art: activity.art,
            parentArt: activity.parent_art,
            grandparentArt: activity.grandparent_art,
            player: activity.player,
            platform: activity.platform,
            machineId: activity.machine_id,
            ipAddress: activity.ip_address,
            location: activity.location,
            secure: activity.secure === 1,
            relayed: activity.relayed === 1,
            qualityProfile: activity.quality_profile,
            syncedVersion: activity.synced_version,
            optimizedVersion: activity.optimized_version,
            channelStream: activity.channel_stream,
            channelCallSign: activity.channel_call_sign,
            channelIdentifier: activity.channel_identifier,
            channelThumb: activity.channel_thumb
        };
    },

    /**
     * Format history item for display
     * @param {Object} item - History item from Tautulli
     * @returns {Object} Formatted history item
     */
    formatHistoryItem(item) {
        return {
            id: item.id,
            date: item.date,
            startedTime: item.started,
            stoppedTime: item.stopped,
            pausedCounter: item.paused_counter,
            username: item.username,
            friendlyName: item.friendly_name,
            userThumb: item.user_thumb,
            ipAddress: item.ip_address,
            platform: item.platform,
            player: item.player,
            machineId: item.machine_id,
            title: item.full_title || item.title,
            year: item.year,
            mediaType: item.media_type,
            ratingKey: item.rating_key,
            parentRatingKey: item.parent_rating_key,
            grandparentRatingKey: item.grandparent_rating_key,
            thumb: item.thumb,
            parentThumb: item.parent_thumb,
            grandparentThumb: item.grandparent_thumb,
            duration: item.duration,
            watchedStatus: item.watched_status,
            percentComplete: item.percent_complete,
            marked: item.marked_watched === 1,
            groupCount: item.group_count,
            groupIds: item.group_ids,
            state: item.state,
            sessionKey: item.session_key
        };
    },

    /**
     * Format user statistics
     * @param {Object} stats - User stats from Tautulli
     * @returns {Object} Formatted user stats
     */
    formatUserStats(stats) {
        return {
            username: stats.username,
            friendlyName: stats.friendly_name,
            userThumb: stats.user_thumb,
            totalPlays: stats.total_plays,
            totalTime: stats.total_time,
            totalDuration: stats.total_duration,
            playCount: {
                movies: stats.total_plays_movies || 0,
                tv: stats.total_plays_tv || 0,
                music: stats.total_plays_music || 0,
                live: stats.total_plays_live || 0,
                other: stats.total_plays_other || 0
            },
            duration: {
                movies: stats.total_duration_movies || 0,
                tv: stats.total_duration_tv || 0,
                music: stats.total_duration_music || 0,
                live: stats.total_duration_live || 0,
                other: stats.total_duration_other || 0
            },
            lastSeen: stats.last_seen,
            lastPlayed: stats.last_played,
            platformStats: stats.platform_stats || [],
            playerStats: stats.player_stats || []
        };
    },

    /**
     * Parse Tautulli webhook payload
     * @param {Object} payload - Webhook payload
     * @returns {Object} Parsed webhook data
     */
    parseWebhookPayload(payload) {
        return {
            action: payload.action,
            timestamp: new Date(),
            session: {
                sessionKey: payload.session_key,
                sessionId: payload.session_id,
                username: payload.username,
                title: payload.title,
                year: payload.year,
                mediaType: payload.media_type,
                thumb: payload.thumb,
                player: payload.player,
                platform: payload.platform,
                ipAddress: payload.ip_address,
                state: payload.state,
                progressPercent: payload.progress_percent,
                duration: payload.duration,
                viewOffset: payload.view_offset
            },
            server: {
                machineIdentifier: payload.machine_identifier,
                serverName: payload.server_name,
                plexServerOwner: payload.plex_server_owner
            },
            user: {
                userId: payload.user_id,
                username: payload.username,
                email: payload.email,
                thumb: payload.user_thumb
            }
        };
    },

    /**
     * Format duration to human readable
     * @param {number} seconds - Duration in seconds
     * @returns {string} Formatted duration
     */
    formatDuration(seconds) {
        if (!seconds || seconds === 0) return '0m';
        
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        const secs = seconds % 60;
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        } else if (minutes > 0) {
            return `${minutes}m ${secs}s`;
        } else {
            return `${secs}s`;
        }
    },

    /**
     * Format bytes to human readable
     * @param {number} bytes - Size in bytes
     * @returns {string} Formatted size
     */
    formatBytes(bytes) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    },

    /**
     * Calculate watch time percentage
     * @param {number} viewOffset - Current position in seconds
     * @param {number} duration - Total duration in seconds
     * @returns {number} Percentage watched
     */
    calculateWatchPercentage(viewOffset, duration) {
        if (!duration || duration === 0) return 0;
        return Math.round((viewOffset / duration) * 100);
    },

    /**
     * Get quality description
     * @param {string} qualityProfile - Quality profile string
     * @returns {Object} Parsed quality info
     */
    parseQualityProfile(qualityProfile) {
        if (!qualityProfile) {
            return { resolution: 'Unknown', quality: 'Unknown' };
        }
        
        const parts = qualityProfile.split(' - ');
        const resolution = parts[0] || 'Unknown';
        const quality = parts[1] || 'Unknown';
        
        return {
            resolution,
            quality,
            full: qualityProfile
        };
    }
};

/**
 * Health check function
 * @param {Object} config - Tautulli configuration
 * @returns {Promise<Object>} Health check result
 */
async function healthCheck(config = {}) {
    try {
        const tautulli = createTautulliIntegration(config);
        const result = await tautulli.testConnection();
        
        return {
            service: 'tautulli',
            healthy: result.success,
            timestamp: new Date(),
            response_time: result.responseTime,
            version: result.version,
            error: result.success ? null : result.error
        };
    } catch (error) {
        return {
            service: 'tautulli',
            healthy: false,
            timestamp: new Date(),
            error: error.message
        };
    }
}

/**
 * Analytics and reporting utilities
 */
const analytics = {
    /**
     * Get comprehensive dashboard data
     * @param {TautulliIntegration} tautulli - Tautulli integration instance
     * @param {Object} options - Options for data retrieval
     * @returns {Promise<Object>} Dashboard data
     */
    async getDashboardData(tautulli, options = {}) {
        try {
            const [activity, stats, recentlyAdded] = await Promise.all([
                tautulli.getActivity(),
                tautulli.getServerStats(),
                tautulli.getRecentlyAdded({ count: 10 })
            ]);
            
            const currentStreams = activity.sessions?.length || 0;
            const totalBandwidth = activity.sessions?.reduce((sum, session) => 
                sum + (parseInt(session.bandwidth) || 0), 0) || 0;
            
            return {
                timestamp: new Date(),
                currentActivity: {
                    streamCount: currentStreams,
                    totalBandwidth: utils.formatBytes(totalBandwidth),
                    sessions: activity.sessions?.map(s => utils.formatActivityInfo(s)) || []
                },
                serverStats: {
                    totalPlays: stats.total_plays || 0,
                    totalUsers: stats.total_users || 0,
                    totalDuration: utils.formatDuration(stats.total_duration || 0),
                    totalSize: utils.formatBytes(stats.total_file_size || 0),
                    movieCount: stats.count_movies || 0,
                    showCount: stats.count_shows || 0,
                    episodeCount: stats.count_episodes || 0,
                    artistCount: stats.count_artists || 0,
                    albumCount: stats.count_albums || 0,
                    trackCount: stats.count_tracks || 0
                },
                recentActivity: recentlyAdded?.recently_added?.map(item => ({
                    title: item.title,
                    type: item.media_type,
                    addedAt: item.added_at,
                    thumb: item.thumb
                })) || []
            };
        } catch (error) {
            return {
                error: error.message,
                timestamp: new Date()
            };
        }
    },

    /**
     * Generate user activity report
     * @param {TautulliIntegration} tautulli - Tautulli integration instance
     * @param {Object} options - Report options
     * @returns {Promise<Object>} User activity report
     */
    async getUserActivityReport(tautulli, options = {}) {
        try {
            const days = options.days || 30;
            const topCount = options.topCount || 10;
            
            const [userStats, topMovies, topShows, topMusic] = await Promise.all([
                tautulli.getUsersStats({ grouping: 1, time_range: days }),
                tautulli.getPopularMovies({ time_range: days, y_axis: 'plays' }),
                tautulli.getPopularTVShows({ time_range: days, y_axis: 'plays' }),
                tautulli.getPopularMusic({ time_range: days, y_axis: 'plays' })
            ]);
            
            const topUsers = userStats.data?.slice(0, topCount).map(user => 
                utils.formatUserStats(user)
            ) || [];
            
            return {
                reportPeriod: `${days} days`,
                generatedAt: new Date(),
                summary: {
                    totalUsers: userStats.data?.length || 0,
                    activeUsers: topUsers.length,
                    topMoviesCount: topMovies.data?.length || 0,
                    topShowsCount: topShows.data?.length || 0,
                    topMusicCount: topMusic.data?.length || 0
                },
                topUsers,
                popularContent: {
                    movies: topMovies.data?.slice(0, topCount) || [],
                    shows: topShows.data?.slice(0, topCount) || [],
                    music: topMusic.data?.slice(0, topCount) || []
                }
            };
        } catch (error) {
            return {
                error: error.message,
                reportPeriod: options.days || 30,
                generatedAt: new Date()
            };
        }
    },

    /**
     * Get streaming statistics
     * @param {TautulliIntegration} tautulli - Tautulli integration instance
     * @param {Object} options - Statistics options
     * @returns {Promise<Object>} Streaming statistics
     */
    async getStreamingStats(tautulli, options = {}) {
        try {
            const days = options.days || 7;
            
            const [playsStats, durationStats, platformStats] = await Promise.all([
                tautulli.getPlaysHistory({ time_range: days }),
                tautulli.getDurationStats({ time_range: days }),
                tautulli.getPlatformStats({ time_range: days })
            ]);
            
            // Calculate daily averages
            const totalPlays = playsStats.data?.reduce((sum, day) => sum + (day.plays || 0), 0) || 0;
            const totalDuration = durationStats.data?.reduce((sum, day) => sum + (day.duration || 0), 0) || 0;
            const avgPlaysPerDay = Math.round(totalPlays / days);
            const avgDurationPerDay = Math.round(totalDuration / days);
            
            return {
                period: `${days} days`,
                summary: {
                    totalPlays,
                    totalDuration: utils.formatDuration(totalDuration),
                    avgPlaysPerDay,
                    avgDurationPerDay: utils.formatDuration(avgDurationPerDay),
                    avgPlayDuration: totalPlays > 0 ? utils.formatDuration(Math.round(totalDuration / totalPlays)) : '0m'
                },
                dailyBreakdown: playsStats.data?.map(day => ({
                    date: day.date,
                    plays: day.plays || 0,
                    duration: utils.formatDuration(day.duration || 0)
                })) || [],
                platformBreakdown: platformStats.data?.map(platform => ({
                    platform: platform.platform_name,
                    plays: platform.total_plays || 0,
                    duration: utils.formatDuration(platform.total_duration || 0),
                    percentage: platform.percentage || 0
                })) || []
            };
        } catch (error) {
            return {
                error: error.message,
                period: options.days || 7
            };
        }
    },

    /**
     * Monitor server performance
     * @param {TautulliIntegration} tautulli - Tautulli integration instance
     * @returns {Promise<Object>} Performance metrics
     */
    async getPerformanceMetrics(tautulli) {
        try {
            const [serverInfo, activity, diskSpace] = await Promise.all([
                tautulli.getServerInfo(),
                tautulli.getActivity(),
                tautulli.getServerResources().catch(() => null) // Optional endpoint
            ]);
            
            const currentStreams = activity.sessions?.length || 0;
            const transcodingSessions = activity.sessions?.filter(s => 
                s.transcode_decision === 'transcode'
            ).length || 0;
            
            const totalBandwidth = activity.sessions?.reduce((sum, session) => 
                sum + (parseInt(session.bandwidth) || 0), 0) || 0;
            
            return {
                timestamp: new Date(),
                server: {
                    version: serverInfo.version,
                    platform: serverInfo.platform,
                    platformVersion: serverInfo.platform_version,
                    uptime: serverInfo.uptime
                },
                currentLoad: {
                    activeStreams: currentStreams,
                    transcodingStreams: transcodingSessions,
                    totalBandwidth: utils.formatBytes(totalBandwidth),
                    avgBandwidthPerStream: currentStreams > 0 ? 
                        utils.formatBytes(Math.round(totalBandwidth / currentStreams)) : '0 Bytes'
                },
                resources: diskSpace ? {
                    diskSpace: utils.formatBytes(diskSpace.total_space || 0),
                    freeSpace: utils.formatBytes(diskSpace.free_space || 0),
                    usedPercentage: diskSpace.used_percentage || 0
                } : null
            };
        } catch (error) {
            return {
                error: error.message,
                timestamp: new Date()
            };
        }
    }
};

module.exports = {
    TautulliIntegration,
    createTautulliIntegration,
    quickSetup,
    defaultConfig,
    utils,
    healthCheck,
    analytics,
    
    // Aliases for convenience
    create: createTautulliIntegration,
    setup: quickSetup,
    Integration: TautulliIntegration
};