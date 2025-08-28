/**
 * Jellyfin Integration Module
 * Complete Jellyfin API integration with MediaBrowser token authentication
 */

const axios = require('axios');
const EventEmitter = require('events');

class JellyfinIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.baseURL = config.baseURL || process.env.JELLYFIN_URL || 'http://localhost:8096';
        this.apiKey = config.apiKey || process.env.JELLYFIN_API_KEY;
        this.username = config.username || process.env.JELLYFIN_USERNAME;
        this.password = config.password || process.env.JELLYFIN_PASSWORD;
        this.accessToken = null;
        this.userId = null;
        
        this.client = axios.create({
            baseURL: this.baseURL,
            timeout: 30000,
            headers: {
                'Content-Type': 'application/json',
                'User-Agent': 'MediaServer-API/1.0.0',
                'X-Emby-Authorization': this._getAuthHeader()
            }
        });

        this._setupInterceptors();
    }

    _getAuthHeader() {
        const deviceId = 'media-server-api';
        const device = 'MediaServer';
        const client = 'MediaServer API';
        const version = '1.0.0';
        
        let authString = `MediaBrowser Client="${client}", Device="${device}", DeviceId="${deviceId}", Version="${version}"`;
        
        if (this.accessToken) {
            authString += `, Token="${this.accessToken}"`;
        }
        
        return authString;
    }

    _setupInterceptors() {
        this.client.interceptors.request.use(
            (config) => {
                config.headers['X-Emby-Authorization'] = this._getAuthHeader();
                return config;
            },
            (error) => Promise.reject(error)
        );

        this.client.interceptors.response.use(
            (response) => response,
            async (error) => {
                if (error.response?.status === 401 && this.username && this.password) {
                    try {
                        await this.authenticate();
                        const originalRequest = error.config;
                        originalRequest.headers['X-Emby-Authorization'] = this._getAuthHeader();
                        return this.client.request(originalRequest);
                    } catch (authError) {
                        this.emit('error', authError);
                        throw authError;
                    }
                }
                this.emit('error', error);
                throw error;
            }
        );
    }

    /**
     * Authenticate with Jellyfin server
     */
    async authenticate(username = this.username, password = this.password) {
        try {
            const response = await this.client.post('/Users/authenticatebyname', {
                Username: username,
                Pw: password
            });

            this.accessToken = response.data.AccessToken;
            this.userId = response.data.User.Id;
            this.emit('authenticated', { userId: this.userId, token: this.accessToken });
            
            return response.data;
        } catch (error) {
            this.emit('authenticationFailed', error);
            throw new Error(`Jellyfin authentication failed: ${error.message}`);
        }
    }

    /**
     * Get server information
     */
    async getServerInfo() {
        try {
            const response = await this.client.get('/System/Info');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get server info: ${error.message}`);
        }
    }

    /**
     * Get all users
     */
    async getUsers() {
        try {
            const response = await this.client.get('/Users');
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
            const response = await this.client.get(`/Users/${userId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get user: ${error.message}`);
        }
    }

    /**
     * Get libraries
     */
    async getLibraries(userId = this.userId) {
        try {
            const response = await this.client.get(`/Users/${userId}/Views`);
            return response.data.Items || [];
        } catch (error) {
            throw new Error(`Failed to get libraries: ${error.message}`);
        }
    }

    /**
     * Get items from library
     */
    async getLibraryItems(libraryId, options = {}) {
        try {
            const params = {
                ParentId: libraryId,
                UserId: this.userId,
                Fields: 'BasicSyncInfo,CanDelete,PrimaryImageAspectRatio,ProductionYear,Status,EndDate',
                ...options
            };

            const response = await this.client.get('/Items', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get library items: ${error.message}`);
        }
    }

    /**
     * Search for items
     */
    async searchItems(query, options = {}) {
        try {
            const params = {
                searchTerm: query,
                UserId: this.userId,
                IncludeItemTypes: 'Movie,Series,Episode,Audio,MusicAlbum,Book',
                Fields: 'BasicSyncInfo,CanDelete,PrimaryImageAspectRatio,ProductionYear',
                ...options
            };

            const response = await this.client.get('/Items', { params });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to search items: ${error.message}`);
        }
    }

    /**
     * Get item details
     */
    async getItem(itemId) {
        try {
            const response = await this.client.get(`/Users/${this.userId}/Items/${itemId}`);
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get item: ${error.message}`);
        }
    }

    /**
     * Get similar items
     */
    async getSimilarItems(itemId, limit = 20) {
        try {
            const response = await this.client.get(`/Items/${itemId}/Similar`, {
                params: { UserId: this.userId, Limit: limit }
            });
            return response.data.Items || [];
        } catch (error) {
            throw new Error(`Failed to get similar items: ${error.message}`);
        }
    }

    /**
     * Get latest media
     */
    async getLatestMedia(libraryId, limit = 16) {
        try {
            const response = await this.client.get(`/Users/${this.userId}/Items/Latest`, {
                params: { ParentId: libraryId, Limit: limit }
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get latest media: ${error.message}`);
        }
    }

    /**
     * Get resume items
     */
    async getResumeItems(limit = 12) {
        try {
            const response = await this.client.get(`/Users/${this.userId}/Items/Resume`, {
                params: { Limit: limit, MediaTypes: 'Video' }
            });
            return response.data.Items || [];
        } catch (error) {
            throw new Error(`Failed to get resume items: ${error.message}`);
        }
    }

    /**
     * Get next up episodes
     */
    async getNextUpEpisodes(limit = 12) {
        try {
            const response = await this.client.get('/Shows/NextUp', {
                params: { UserId: this.userId, Limit: limit }
            });
            return response.data.Items || [];
        } catch (error) {
            throw new Error(`Failed to get next up episodes: ${error.message}`);
        }
    }

    /**
     * Mark item as played
     */
    async markPlayed(itemId) {
        try {
            await this.client.post(`/Users/${this.userId}/PlayedItems/${itemId}`);
            this.emit('itemPlayed', itemId);
            return true;
        } catch (error) {
            throw new Error(`Failed to mark item as played: ${error.message}`);
        }
    }

    /**
     * Mark item as unplayed
     */
    async markUnplayed(itemId) {
        try {
            await this.client.delete(`/Users/${this.userId}/PlayedItems/${itemId}`);
            this.emit('itemUnplayed', itemId);
            return true;
        } catch (error) {
            throw new Error(`Failed to mark item as unplayed: ${error.message}`);
        }
    }

    /**
     * Add to favorites
     */
    async addToFavorites(itemId) {
        try {
            await this.client.post(`/Users/${this.userId}/FavoriteItems/${itemId}`);
            this.emit('itemFavorited', itemId);
            return true;
        } catch (error) {
            throw new Error(`Failed to add to favorites: ${error.message}`);
        }
    }

    /**
     * Remove from favorites
     */
    async removeFromFavorites(itemId) {
        try {
            await this.client.delete(`/Users/${this.userId}/FavoriteItems/${itemId}`);
            this.emit('itemUnfavorited', itemId);
            return true;
        } catch (error) {
            throw new Error(`Failed to remove from favorites: ${error.message}`);
        }
    }

    /**
     * Report playback start
     */
    async reportPlaybackStart(itemId, sessionId = null) {
        try {
            const data = {
                ItemId: itemId,
                UserId: this.userId,
                MediaSourceId: itemId,
                CanSeek: true
            };

            if (sessionId) data.SessionId = sessionId;

            await this.client.post('/Sessions/Playing', data);
            this.emit('playbackStarted', { itemId, sessionId });
            return true;
        } catch (error) {
            throw new Error(`Failed to report playback start: ${error.message}`);
        }
    }

    /**
     * Report playback progress
     */
    async reportPlaybackProgress(itemId, positionTicks, sessionId = null) {
        try {
            const data = {
                ItemId: itemId,
                UserId: this.userId,
                PositionTicks: positionTicks,
                MediaSourceId: itemId,
                CanSeek: true,
                IsPaused: false
            };

            if (sessionId) data.SessionId = sessionId;

            await this.client.post('/Sessions/Playing/Progress', data);
            this.emit('playbackProgress', { itemId, positionTicks, sessionId });
            return true;
        } catch (error) {
            throw new Error(`Failed to report playback progress: ${error.message}`);
        }
    }

    /**
     * Report playback stop
     */
    async reportPlaybackStop(itemId, positionTicks, sessionId = null) {
        try {
            const data = {
                ItemId: itemId,
                UserId: this.userId,
                PositionTicks: positionTicks,
                MediaSourceId: itemId
            };

            if (sessionId) data.SessionId = sessionId;

            await this.client.post('/Sessions/Playing/Stopped', data);
            this.emit('playbackStopped', { itemId, positionTicks, sessionId });
            return true;
        } catch (error) {
            throw new Error(`Failed to report playback stop: ${error.message}`);
        }
    }

    /**
     * Get active sessions
     */
    async getSessions() {
        try {
            const response = await this.client.get('/Sessions');
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get sessions: ${error.message}`);
        }
    }

    /**
     * Get server activity
     */
    async getActivity(startIndex = 0, limit = 20) {
        try {
            const response = await this.client.get('/System/ActivityLog/Entries', {
                params: { StartIndex: startIndex, Limit: limit }
            });
            return response.data;
        } catch (error) {
            throw new Error(`Failed to get activity: ${error.message}`);
        }
    }

    /**
     * Get library statistics
     */
    async getLibraryStats() {
        try {
            const libraries = await this.getLibraries();
            const stats = {};

            for (const library of libraries) {
                const items = await this.getLibraryItems(library.Id, { Recursive: true });
                stats[library.Name] = {
                    id: library.Id,
                    type: library.CollectionType,
                    itemCount: items.TotalRecordCount || 0,
                    items: items.Items || []
                };
            }

            return stats;
        } catch (error) {
            throw new Error(`Failed to get library stats: ${error.message}`);
        }
    }

    /**
     * Scan library
     */
    async scanLibrary(libraryId) {
        try {
            await this.client.post(`/Items/${libraryId}/Refresh`, {
                Recursive: true,
                ImageRefreshMode: 'Default',
                MetadataRefreshMode: 'Default'
            });
            this.emit('libraryScanStarted', libraryId);
            return true;
        } catch (error) {
            throw new Error(`Failed to scan library: ${error.message}`);
        }
    }

    /**
     * Get image URL for item
     */
    getImageUrl(itemId, imageType = 'Primary', maxWidth = null, maxHeight = null) {
        let url = `${this.baseURL}/Items/${itemId}/Images/${imageType}`;
        const params = new URLSearchParams();
        
        if (maxWidth) params.append('maxWidth', maxWidth);
        if (maxHeight) params.append('maxHeight', maxHeight);
        
        const queryString = params.toString();
        if (queryString) url += `?${queryString}`;
        
        return url;
    }

    /**
     * Test connection
     */
    async testConnection() {
        try {
            const info = await this.getServerInfo();
            return {
                success: true,
                serverName: info.ServerName,
                version: info.Version,
                serverId: info.Id
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
    setupWebhook(app, path = '/jellyfin/webhook') {
        app.post(path, (req, res) => {
            try {
                const event = req.body;
                this.emit('webhook', event);
                
                // Emit specific events based on notification type
                switch (event.NotificationType) {
                    case 'ItemAdded':
                        this.emit('itemAdded', event);
                        break;
                    case 'PlaybackStart':
                        this.emit('playbackStarted', event);
                        break;
                    case 'PlaybackStop':
                        this.emit('playbackStopped', event);
                        break;
                    case 'UserDataSaved':
                        this.emit('userDataSaved', event);
                        break;
                    default:
                        this.emit('unknownEvent', event);
                }
                
                res.status(200).json({ success: true });
            } catch (error) {
                console.error('Jellyfin webhook error:', error);
                res.status(500).json({ error: 'Webhook processing failed' });
            }
        });
    }
}

module.exports = JellyfinIntegration;