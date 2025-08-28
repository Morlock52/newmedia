/**
 * Tautulli Integration Module
 * Complete Tautulli API integration for Plex statistics and monitoring
 */

const axios = require('axios');
const EventEmitter = require('events');

class TautulliIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.baseURL = config.baseURL || process.env.TAUTULLI_URL || 'http://localhost:8181';
        this.apiKey = config.apiKey || process.env.TAUTULLI_API_KEY;
        
        if (!this.apiKey) {
            throw new Error('Tautulli API key is required');
        }
        
        this.client = axios.create({
            baseURL: `${this.baseURL}/api/v2`,
            timeout: 30000,
            params: {
                apikey: this.apiKey,
                out: 'json'
            }
        });

        this._setupInterceptors();
    }

    _setupInterceptors() {
        this.client.interceptors.response.use(
            (response) => {
                if (response.data?.response?.result === 'success') {
                    return { ...response, data: response.data.response.data };
                }
                return response;
            },
            (error) => {
                this.emit('error', error);
                throw error;
            }
        );
    }

    /**
     * Get server info
     */
    async getServerInfo() {
        try {
            const response = await this.client.get('', { params: { cmd: 'get_server_info' } });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get server info: ${error.message}`);
        }
    }

    /**
     * Get server identity
     */
    async getServerIdentity() {
        try {
            const response = await this.client.get('', { params: { cmd: 'get_server_identity' } });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get server identity: ${error.message}`);
        }
    }

    /**
     * Get current activity
     */
    async getActivity(sessionKey = null, sessionId = null) {
        try {
            const params = { cmd: 'get_activity' };
            if (sessionKey) params.session_key = sessionKey;
            if (sessionId) params.session_id = sessionId;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get activity: ${error.message}`);
        }
    }

    /**
     * Get history
     */
    async getHistory(grouping = null, user = null, userId = null, ratingKey = null, parentRatingKey = null, grandparentRatingKey = null, startDate = null, sectionId = null, mediaType = null, transcodingDecision = null, guid = null, orderColumn = null, orderDir = null, start = 0, length = 25, search = null) {
        try {
            const params = {
                cmd: 'get_history',
                start,
                length
            };
            
            if (grouping) params.grouping = grouping;
            if (user) params.user = user;
            if (userId) params.user_id = userId;
            if (ratingKey) params.rating_key = ratingKey;
            if (parentRatingKey) params.parent_rating_key = parentRatingKey;
            if (grandparentRatingKey) params.grandparent_rating_key = grandparentRatingKey;
            if (startDate) params.start_date = startDate;
            if (sectionId) params.section_id = sectionId;
            if (mediaType) params.media_type = mediaType;
            if (transcodingDecision) params.transcode_decision = transcodingDecision;
            if (guid) params.guid = guid;
            if (orderColumn) params.order_column = orderColumn;
            if (orderDir) params.order_dir = orderDir;
            if (search) params.search = search;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get history: ${error.message}`);
        }
    }

    /**
     * Get libraries
     */
    async getLibraries() {
        try {
            const response = await this.client.get('', { params: { cmd: 'get_libraries' } });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get libraries: ${error.message}`);
        }
    }

    /**
     * Get library details
     */
    async getLibrary(sectionId) {
        try {
            const response = await this.client.get('', {
                params: { cmd: 'get_library', section_id: sectionId }
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get library: ${error.message}`);
        }
    }

    /**
     * Get library media info
     */
    async getLibraryMediaInfo(sectionId, orderColumn = null, orderDir = null, start = 0, length = 25, search = null) {
        try {
            const params = {
                cmd: 'get_library_media_info',
                section_id: sectionId,
                start,
                length
            };
            
            if (orderColumn) params.order_column = orderColumn;
            if (orderDir) params.order_dir = orderDir;
            if (search) params.search = search;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get library media info: ${error.message}`);
        }
    }

    /**
     * Get users
     */
    async getUsers() {
        try {
            const response = await this.client.get('', { params: { cmd: 'get_users' } });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get users: ${error.message}`);
        }
    }

    /**
     * Get user details
     */
    async getUser(userId) {
        try {
            const response = await this.client.get('', {
                params: { cmd: 'get_user', user_id: userId }
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get user: ${error.message}`);
        }
    }

    /**
     * Get user watch time stats
     */
    async getUserWatchTimeStats(userId, grouping = null, queryDays = null) {
        try {
            const params = {
                cmd: 'get_user_watch_time_stats',
                user_id: userId
            };
            
            if (grouping) params.grouping = grouping;
            if (queryDays) params.query_days = queryDays;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get user watch time stats: ${error.message}`);
        }
    }

    /**
     * Get user player stats
     */
    async getUserPlayerStats(userId, grouping = null, queryDays = null) {
        try {
            const params = {
                cmd: 'get_user_player_stats',
                user_id: userId
            };
            
            if (grouping) params.grouping = grouping;
            if (queryDays) params.query_days = queryDays;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get user player stats: ${error.message}`);
        }
    }

    /**
     * Get plays by date
     */
    async getPlaysByDate(timeRange = 30, userId = null, grouping = null) {
        try {
            const params = {
                cmd: 'get_plays_by_date',
                time_range: timeRange
            };
            
            if (userId) params.user_id = userId;
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get plays by date: ${error.message}`);
        }
    }

    /**
     * Get plays by hour of day
     */
    async getPlaysByHour(timeRange = 30, userId = null, grouping = null) {
        try {
            const params = {
                cmd: 'get_plays_by_hourofday',
                time_range: timeRange
            };
            
            if (userId) params.user_id = userId;
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get plays by hour: ${error.message}`);
        }
    }

    /**
     * Get plays by day of week
     */
    async getPlaysByDayOfWeek(timeRange = 30, userId = null, grouping = null) {
        try {
            const params = {
                cmd: 'get_plays_by_dayofweek',
                time_range: timeRange
            };
            
            if (userId) params.user_id = userId;
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get plays by day of week: ${error.message}`);
        }
    }

    /**
     * Get plays by top 10 platforms
     */
    async getPlaysByTop10Platforms(timeRange = 30, userId = null, grouping = null) {
        try {
            const params = {
                cmd: 'get_plays_by_top_10_platforms',
                time_range: timeRange
            };
            
            if (userId) params.user_id = userId;
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get plays by top 10 platforms: ${error.message}`);
        }
    }

    /**
     * Get plays by top 10 users
     */
    async getPlaysByTop10Users(timeRange = 30, userId = null, grouping = null) {
        try {
            const params = {
                cmd: 'get_plays_by_top_10_users',
                time_range: timeRange
            };
            
            if (userId) params.user_id = userId;
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get plays by top 10 users: ${error.message}`);
        }
    }

    /**
     * Get home stats
     */
    async getHomeStats(timeRange = 30, statsType = 0, statsCount = 5) {
        try {
            const params = {
                cmd: 'get_home_stats',
                time_range: timeRange,
                stats_type: statsType,
                stats_count: statsCount
            };
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get home stats: ${error.message}`);
        }
    }

    /**
     * Get recently added
     */
    async getRecentlyAdded(count = 25, start = 0, mediaType = null, sectionId = null) {
        try {
            const params = {
                cmd: 'get_recently_added',
                count,
                start
            };
            
            if (mediaType) params.media_type = mediaType;
            if (sectionId) params.section_id = sectionId;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get recently added: ${error.message}`);
        }
    }

    /**
     * Get metadata
     */
    async getMetadata(ratingKey, mediaInfo = false) {
        try {
            const params = {
                cmd: 'get_metadata',
                rating_key: ratingKey,
                media_info: mediaInfo
            };
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get metadata: ${error.message}`);
        }
    }

    /**
     * Get children metadata
     */
    async getChildrenMetadata(ratingKey) {
        try {
            const response = await this.client.get('', {
                params: { cmd: 'get_children_metadata', rating_key: ratingKey }
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get children metadata: ${error.message}`);
        }
    }

    /**
     * Get item watch time stats
     */
    async getItemWatchTimeStats(ratingKey, grouping = null) {
        try {
            const params = {
                cmd: 'get_item_watch_time_stats',
                rating_key: ratingKey
            };
            
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get item watch time stats: ${error.message}`);
        }
    }

    /**
     * Get item user stats
     */
    async getItemUserStats(ratingKey, grouping = null) {
        try {
            const params = {
                cmd: 'get_item_user_stats',
                rating_key: ratingKey
            };
            
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get item user stats: ${error.message}`);
        }
    }

    /**
     * Get stream type by top 10 platforms
     */
    async getStreamTypeByTop10Platforms(timeRange = 30, userId = null, grouping = null) {
        try {
            const params = {
                cmd: 'get_stream_type_by_top_10_platforms',
                time_range: timeRange
            };
            
            if (userId) params.user_id = userId;
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get stream type by top 10 platforms: ${error.message}`);
        }
    }

    /**
     * Get stream type by top 10 users
     */
    async getStreamTypeByTop10Users(timeRange = 30, userId = null, grouping = null) {
        try {
            const params = {
                cmd: 'get_stream_type_by_top_10_users',
                time_range: timeRange
            };
            
            if (userId) params.user_id = userId;
            if (grouping) params.grouping = grouping;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get stream type by top 10 users: ${error.message}`);
        }
    }

    /**
     * Get notifications
     */
    async getNotifications() {
        try {
            const response = await this.client.get('', { params: { cmd: 'get_notifications' } });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get notifications: ${error.message}`);
        }
    }

    /**
     * Get logs
     */
    async getLogs(sort = null, search = null, order = null, regex = null, start = 0, end = 25) {
        try {
            const params = {
                cmd: 'get_logs',
                start,
                end
            };
            
            if (sort) params.sort = sort;
            if (search) params.search = search;
            if (order) params.order = order;
            if (regex) params.regex = regex;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get logs: ${error.message}`);
        }
    }

    /**
     * Get Plex log
     */
    async getPlexLog(window = 1000, logLevel = null) {
        try {
            const params = {
                cmd: 'get_plex_log',
                window
            };
            
            if (logLevel) params.log_level = logLevel;
            
            const response = await this.client.get('', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get Plex log: ${error.message}`);
        }
    }

    /**
     * Get server resources
     */
    async getServerResources() {
        try {
            const response = await this.client.get('', { params: { cmd: 'get_server_resources' } });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get server resources: ${error.message}`);
        }
    }

    /**
     * Refresh libraries list
     */
    async refreshLibrariesList() {
        try {
            const response = await this.client.get('', { params: { cmd: 'refresh_libraries_list' } });
            this.emit('librariesRefreshed');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to refresh libraries list: ${error.message}`);
        }
    }

    /**
     * Terminate session
     */
    async terminateSession(sessionKey, message = '') {
        try {
            const params = {
                cmd: 'terminate_session',
                session_key: sessionKey
            };
            
            if (message) params.message = message;
            
            const response = await this.client.get('', { params });
            this.emit('sessionTerminated', sessionKey);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to terminate session: ${error.message}`);
        }
    }

    /**
     * Delete image cache
     */
    async deleteImageCache() {
        try {
            const response = await this.client.get('', { params: { cmd: 'delete_image_cache' } });
            this.emit('imageCacheDeleted');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to delete image cache: ${error.message}`);
        }
    }

    /**
     * Get comprehensive statistics
     */
    async getStatistics() {
        try {
            const [serverInfo, libraries, users, activity, homeStats] = await Promise.all([
                this.getServerInfo(),
                this.getLibraries(),
                this.getUsers(),
                this.getActivity(),
                this.getHomeStats()
            ]);
            
            const stats = {
                server: {
                    name: serverInfo.pms_name,
                    version: serverInfo.pms_version,
                    platform: serverInfo.pms_platform,
                    uptime: serverInfo.pms_uptime
                },
                libraries: {
                    total: libraries.length,
                    sections: libraries.map(lib => ({
                        id: lib.section_id,
                        name: lib.section_name,
                        type: lib.section_type,
                        count: lib.count
                    }))
                },
                users: {
                    total: users.length,
                    active: users.filter(u => u.is_active).length
                },
                activity: {
                    streamCount: activity.stream_count,
                    streamCountDirectPlay: activity.stream_count_direct_play,
                    streamCountDirectStream: activity.stream_count_direct_stream,
                    streamCountTranscode: activity.stream_count_transcode,
                    totalBandwidth: activity.total_bandwidth,
                    lanBandwidth: activity.lan_bandwidth,
                    wanBandwidth: activity.wan_bandwidth
                },
                homeStats: homeStats
            };
            
            return stats;
        } catch (error) {
            throw new Error(`Failed to get statistics: ${error.message}`);
        }
    }

    /**
     * Test connection
     */
    async testConnection() {
        try {
            const info = await this.getServerInfo();
            return {
                success: true,
                serverName: info.pms_name,
                version: info.pms_version,
                platform: info.pms_platform
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Setup webhook endpoint for Plex webhooks (forwarded through Tautulli)
     */
    setupWebhook(app, path = '/tautulli/webhook') {
        app.post(path, (req, res) => {
            try {
                const event = req.body;
                this.emit('webhook', event);
                
                // Emit specific events based on action
                switch (event.action) {
                    case 'play':
                        this.emit('playbackStarted', event);
                        break;
                    case 'stop':
                        this.emit('playbackStopped', event);
                        break;
                    case 'pause':
                        this.emit('playbackPaused', event);
                        break;
                    case 'resume':
                        this.emit('playbackResumed', event);
                        break;
                    case 'buffer':
                        this.emit('playbackBuffering', event);
                        break;
                    case 'error':
                        this.emit('playbackError', event);
                        break;
                    case 'concurrent':
                        this.emit('concurrentStreams', event);
                        break;
                    case 'newdevice':
                        this.emit('newDevice', event);
                        break;
                    case 'playbacklocation':
                        this.emit('playbackLocation', event);
                        break;
                    default:
                        this.emit('unknownEvent', event);
                }
                
                res.status(200).json({ success: true });
            } catch (error) {
                console.error('Tautulli webhook error:', error);
                res.status(500).json({ error: 'Webhook processing failed' });
            }
        });
    }
}

module.exports = TautulliIntegration;