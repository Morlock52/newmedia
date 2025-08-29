const logger = require('../../middleware/logger.js');
/**
 * Jellyseerr Integration Module
 * Complete Jellyseerr API integration with request management and approval workflow
 */

const axios = require('axios');
const EventEmitter = require('events');

class JellyseerrIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.baseURL = config.baseURL || process.env.JELLYSEERR_URL || 'http://localhost:5055';
        this.apiKey = config.apiKey || process.env.JELLYSEERR_API_KEY;
        
        if (!this.apiKey) {
            throw new Error('Jellyseerr API key is required');
        }
        
        this.client = axios.create({
            baseURL: `${this.baseURL}/api/v1`,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'X-Api-Key': this.apiKey
            }
        });

        this._setupInterceptors();
    }

    _setupInterceptors() {
        this.client.interceptors.response.use(
            (response) => response,
            (error) => {
                this.emit('error', error);
                throw error;
            }
        );
    }

    /**
     * Get system status
     */
    async getStatus() {
        try {
            const response = await this.client.get('/status');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get status: ${error.message}`);
        }
    }

    /**
     * Get all requests
     */
    async getRequests(take = 20, skip = 0, filter = 'all', sort = 'added') {
        try {
            const params = { take, skip, filter, sort };
            const response = await this.client.get('/request', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get requests: ${error.message}`);
        }
    }

    /**
     * Get request by ID
     */
    async getRequestById(requestId) {
        try {
            const response = await this.client.get(`/request/${requestId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get request: ${error.message}`);
        }
    }

    /**
     * Create new request
     */
    async createRequest(mediaData) {
        try {
            const requestData = {
                mediaType: mediaData.mediaType, // 'movie' or 'tv'
                mediaId: mediaData.mediaId,
                tvdbId: mediaData.tvdbId,
                seasons: mediaData.seasons || 'all',
                is4k: mediaData.is4k || false,
                serverId: mediaData.serverId,
                profileId: mediaData.profileId,
                rootFolder: mediaData.rootFolder,
                languageProfileId: mediaData.languageProfileId,
                tags: mediaData.tags || []
            };

            const response = await this.client.post('/request', requestData);
            this.emit('requestCreated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to create request: ${error.message}`);
        }
    }

    /**
     * Update request
     */
    async updateRequest(requestId, updates) {
        try {
            const response = await this.client.put(`/request/${requestId}`, updates);
            this.emit('requestUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update request: ${error.message}`);
        }
    }

    /**
     * Delete request
     */
    async deleteRequest(requestId) {
        try {
            await this.client.delete(`/request/${requestId}`);
            this.emit('requestDeleted', requestId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete request: ${error.message}`);
        }
    }

    /**
     * Approve request
     */
    async approveRequest(requestId) {
        try {
            const response = await this.client.post(`/request/${requestId}/approve`);
            this.emit('requestApproved', { requestId, data: response.data });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to approve request: ${error.message}`);
        }
    }

    /**
     * Decline request
     */
    async declineRequest(requestId) {
        try {
            const response = await this.client.post(`/request/${requestId}/decline`);
            this.emit('requestDeclined', { requestId, data: response.data });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to decline request: ${error.message}`);
        }
    }

    /**
     * Retry request
     */
    async retryRequest(requestId) {
        try {
            const response = await this.client.post(`/request/${requestId}/retry`);
            this.emit('requestRetried', { requestId, data: response.data });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to retry request: ${error.message}`);
        }
    }

    /**
     * Search for media
     */
    async searchMedia(query, page = 1, language = 'en') {
        try {
            const params = { query, page, language };
            const response = await this.client.get('/search', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to search media: ${error.message}`);
        }
    }

    /**
     * Get media details
     */
    async getMediaDetails(mediaType, mediaId, language = 'en') {
        try {
            const params = { language };
            const response = await this.client.get(`/${mediaType}/${mediaId}`, { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get media details: ${error.message}`);
        }
    }

    /**
     * Get trending media
     */
    async getTrending(mediaType = 'all', page = 1, language = 'en') {
        try {
            const params = { page, language };
            const response = await this.client.get(`/discover/${mediaType}/trending`, { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get trending: ${error.message}`);
        }
    }

    /**
     * Get popular media
     */
    async getPopular(mediaType = 'movie', page = 1, language = 'en') {
        try {
            const params = { page, language };
            const response = await this.client.get(`/discover/${mediaType}/popular`, { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get popular: ${error.message}`);
        }
    }

    /**
     * Get upcoming media
     */
    async getUpcoming(mediaType = 'movie', page = 1, language = 'en') {
        try {
            const params = { page, language };
            const response = await this.client.get(`/discover/${mediaType}/upcoming`, { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get upcoming: ${error.message}`);
        }
    }

    /**
     * Get all users
     */
    async getUsers(take = 20, skip = 0, sort = 'created') {
        try {
            const params = { take, skip, sort };
            const response = await this.client.get('/user', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get users: ${error.message}`);
        }
    }

    /**
     * Get user by ID
     */
    async getUserById(userId) {
        try {
            const response = await this.client.get(`/user/${userId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get user: ${error.message}`);
        }
    }

    /**
     * Create user
     */
    async createUser(userData) {
        try {
            const response = await this.client.post('/user', userData);
            this.emit('userCreated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to create user: ${error.message}`);
        }
    }

    /**
     * Update user
     */
    async updateUser(userId, updates) {
        try {
            const response = await this.client.put(`/user/${userId}`, updates);
            this.emit('userUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update user: ${error.message}`);
        }
    }

    /**
     * Delete user
     */
    async deleteUser(userId) {
        try {
            await this.client.delete(`/user/${userId}`);
            this.emit('userDeleted', userId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete user: ${error.message}`);
        }
    }

    /**
     * Get user quota
     */
    async getUserQuota(userId) {
        try {
            const response = await this.client.get(`/user/${userId}/quota`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get user quota: ${error.message}`);
        }
    }

    /**
     * Get all issues
     */
    async getIssues(take = 20, skip = 0, sort = 'created', filter = 'all') {
        try {
            const params = { take, skip, sort, filter };
            const response = await this.client.get('/issue', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get issues: ${error.message}`);
        }
    }

    /**
     * Get issue by ID
     */
    async getIssueById(issueId) {
        try {
            const response = await this.client.get(`/issue/${issueId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get issue: ${error.message}`);
        }
    }

    /**
     * Create issue
     */
    async createIssue(issueData) {
        try {
            const response = await this.client.post('/issue', issueData);
            this.emit('issueCreated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to create issue: ${error.message}`);
        }
    }

    /**
     * Update issue
     */
    async updateIssue(issueId, updates) {
        try {
            const response = await this.client.put(`/issue/${issueId}`, updates);
            this.emit('issueUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update issue: ${error.message}`);
        }
    }

    /**
     * Delete issue
     */
    async deleteIssue(issueId) {
        try {
            await this.client.delete(`/issue/${issueId}`);
            this.emit('issueDeleted', issueId);
            return true;
        } catch (error) {
            throw new Error(`Failed to delete issue: ${error.message}`);
        }
    }

    /**
     * Get settings
     */
    async getSettings() {
        try {
            const response = await this.client.get('/settings/main');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get settings: ${error.message}`);
        }
    }

    /**
     * Update settings
     */
    async updateSettings(settings) {
        try {
            const response = await this.client.post('/settings/main', settings);
            this.emit('settingsUpdated', response.data);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to update settings: ${error.message}`);
        }
    }

    /**
     * Get Plex servers
     */
    async getPlexServers() {
        try {
            const response = await this.client.get('/settings/plex');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get Plex servers: ${error.message}`);
        }
    }

    /**
     * Get Sonarr settings
     */
    async getSonarrSettings() {
        try {
            const response = await this.client.get('/settings/sonarr');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get Sonarr settings: ${error.message}`);
        }
    }

    /**
     * Get Radarr settings
     */
    async getRadarrSettings() {
        try {
            const response = await this.client.get('/settings/radarr');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get Radarr settings: ${error.message}`);
        }
    }

    /**
     * Test service connection
     */
    async testService(serviceType, config) {
        try {
            const response = await this.client.post(`/settings/${serviceType}/test`, config);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to test ${serviceType}: ${error.message}`);
        }
    }

    /**
     * Get watchlist
     */
    async getWatchlist(userId) {
        try {
            const response = await this.client.get(`/user/${userId}/watchlist`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get watchlist: ${error.message}`);
        }
    }

    /**
     * Add to watchlist
     */
    async addToWatchlist(userId, mediaData) {
        try {
            const response = await this.client.post(`/user/${userId}/watchlist`, mediaData);
            this.emit('watchlistItemAdded', { userId, data: response.data });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to add to watchlist: ${error.message}`);
        }
    }

    /**
     * Remove from watchlist
     */
    async removeFromWatchlist(userId, mediaId) {
        try {
            await this.client.delete(`/user/${userId}/watchlist/${mediaId}`);
            this.emit('watchlistItemRemoved', { userId, mediaId });
            return true;
        } catch (error) {
            throw new Error(`Failed to remove from watchlist: ${error.message}`);
        }
    }

    /**
     * Get comprehensive statistics
     */
    async getStatistics() {
        try {
            const [requests, users, issues] = await Promise.all([
                this.getRequests(1000),
                this.getUsers(1000),
                this.getIssues(1000)
            ]);
            
            const stats = {
                requests: {
                    total: requests.pageInfo.results,
                    pending: requests.results.filter(r => r.status === 1).length,
                    approved: requests.results.filter(r => r.status === 2).length,
                    available: requests.results.filter(r => r.status === 3).length,
                    partiallyAvailable: requests.results.filter(r => r.status === 4).length,
                    processing: requests.results.filter(r => r.status === 5).length,
                    failed: requests.results.filter(r => r.status === 6).length,
                    byType: {
                        movies: requests.results.filter(r => r.type === 'movie').length,
                        tv: requests.results.filter(r => r.type === 'tv').length
                    }
                },
                users: {
                    total: users.pageInfo.results,
                    active: users.results.filter(u => !u.disabled).length,
                    disabled: users.results.filter(u => u.disabled).length,
                    admins: users.results.filter(u => u.permissions & 2).length
                },
                issues: {
                    total: issues.pageInfo.results,
                    open: issues.results.filter(i => i.status === 1).length,
                    resolved: issues.results.filter(i => i.status === 2).length,
                    byType: this._groupIssuesByType(issues.results)
                }
            };
            
            return stats;
        } catch (error) {
            throw new Error(`Failed to get statistics: ${error.message}`);
        }
    }

    /**
     * Group issues by type
     */
    _groupIssuesByType(issues) {
        const grouped = {};
        issues.forEach(issue => {
            const type = issue.issueType || 'other';
            if (!grouped[type]) grouped[type] = 0;
            grouped[type]++;
        });
        return grouped;
    }

    /**
     * Test connection
     */
    async testConnection() {
        try {
            const status = await this.getStatus();
            return {
                success: true,
                version: status.version,
                commitTag: status.commitTag,
                updateAvailable: status.updateAvailable
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Setup webhook endpoint
     */
    setupWebhook(app, path = '/jellyseerr/webhook') {
        app.post(path, (req, res) => {
            try {
                const event = req.body;
                this.emit('webhook', event);
                
                // Emit specific events based on notification type
                switch (event.notification_type) {
                    case 'MEDIA_PENDING':
                        this.emit('mediaPending', event);
                        break;
                    case 'MEDIA_APPROVED':
                        this.emit('mediaApproved', event);
                        break;
                    case 'MEDIA_AUTO_APPROVED':
                        this.emit('mediaAutoApproved', event);
                        break;
                    case 'MEDIA_AVAILABLE':
                        this.emit('mediaAvailable', event);
                        break;
                    case 'MEDIA_DECLINED':
                        this.emit('mediaDeclined', event);
                        break;
                    case 'MEDIA_FAILED':
                        this.emit('mediaFailed', event);
                        break;
                    case 'ISSUE_CREATED':
                        this.emit('issueCreated', event);
                        break;
                    case 'ISSUE_COMMENT':
                        this.emit('issueComment', event);
                        break;
                    case 'ISSUE_RESOLVED':
                        this.emit('issueResolved', event);
                        break;
                    case 'ISSUE_REOPENED':
                        this.emit('issueReopened', event);
                        break;
                    case 'TEST_NOTIFICATION':
                        this.emit('webhookTest', event);
                        break;
                    default:
                        this.emit('unknownEvent', event);
                }
                
                res.status(200).json({ success: true });
            } catch (error) {
                logger.error('Jellyseerr webhook error:', error);
                res.status(500).json({ error: 'Webhook processing failed' });
            }
        });
    }
}

module.exports = JellyseerrIntegration;