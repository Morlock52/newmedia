const logger = require('../../middleware/logger.js');
/**
 * Plex Integration Module
 * Complete Plex API integration with X-Plex-Token authentication
 */

const axios = require('axios');
const EventEmitter = require('events');
const xml2js = require('xml2js');

class PlexIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.baseURL = config.baseURL || process.env.PLEX_URL || 'http://localhost:32400';
        this.token = config.token || process.env.PLEX_TOKEN;
        this.username = config.username || process.env.PLEX_USERNAME;
        this.password = config.password || process.env.PLEX_PASSWORD;
        this.clientId = config.clientId || 'media-server-api';
        this.product = config.product || 'MediaServer API';
        this.version = config.version || '1.0.0';
        this.device = config.device || 'MediaServer';
        
        this.client = axios.create({
            baseURL: this.baseURL,
            timeout: 30000,
            headers: {
                'Accept': 'application/json',
                'Content-Type': 'application/json',
                'X-Plex-Product': this.product,
                'X-Plex-Version': this.version,
                'X-Plex-Client-Identifier': this.clientId,
                'X-Plex-Platform': 'NodeJS',
                'X-Plex-Device': this.device,
                'X-Plex-Device-Name': this.device
            }
        });

        this._setupInterceptors();
        this.parser = new xml2js.Parser({ explicitArray: false, mergeAttrs: true });
    }

    _setupInterceptors() {
        this.client.interceptors.request.use(
            (config) => {
                if (this.token) {
                    config.headers['X-Plex-Token'] = this.token;
                }
                return config;
            },
            (error) => Promise.reject(error)
        );

        this.client.interceptors.response.use(
            async (response) => {
                // Handle XML responses
                if (response.headers['content-type']?.includes('application/xml') ||
                    response.headers['content-type']?.includes('text/xml')) {
                    try {
                        const parsed = await this.parser.parseStringPromise(response.data);
                        response.data = parsed;
                    } catch (error) {
                        logger.warn('Failed to parse XML response:', error);
                    }
                }
                return response;
            },
            async (error) => {
                if (error.response?.status === 401 && this.username && this.password) {
                    try {
                        await this.authenticate();
                        const originalRequest = error.config;
                        originalRequest.headers['X-Plex-Token'] = this.token;
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
     * Authenticate with Plex
     */
    async authenticate(username = this.username, password = this.password) {
        try {
            const authClient = axios.create({
                baseURL: 'https://plex.tv',
                headers: {
                    'X-Plex-Product': this.product,
                    'X-Plex-Version': this.version,
                    'X-Plex-Client-Identifier': this.clientId,
                    'Content-Type': 'application/x-www-form-urlencoded'
                }
            });

            const response = await authClient.post('/users/sign_in.json', 
                `user[login]=${encodeURIComponent(username)}&user[password]=${encodeURIComponent(password)}`
            );

            this.token = response.data.user.authToken;
            this.emit('authenticated', { token: this.token, user: response.data.user });
            
            return response.data;
        } catch (error) {
            this.emit('authenticationFailed', error);
            throw new Error(`Plex authentication failed: ${error.message}`);
        }
    }

    /**
     * Get server information
     */
    async getServerInfo() {
        try {
            const response = await this.client.get('/');
            return response.data.MediaContainer || response.data;
        } catch (error) {
            throw new Error(`Failed to get server info: ${error.message}`);
        }
    }

    /**
     * Get all libraries
     */
    async getLibraries() {
        try {
            const response = await this.client.get('/library/sections');
            const container = response.data.MediaContainer || response.data;
            return container.Directory || [];
        } catch (error) {
            throw new Error(`Failed to get libraries: ${error.message}`);
        }
    }

    /**
     * Get library content
     */
    async getLibraryContent(sectionId, options = {}) {
        try {
            const params = new URLSearchParams();
            Object.entries(options).forEach(([key, value]) => {
                if (value !== undefined) params.append(key, value);
            });

            const queryString = params.toString();
            const url = `/library/sections/${sectionId}/all${queryString ? '?' + queryString : ''}`;
            
            const response = await this.client.get(url);
            const container = response.data.MediaContainer || response.data;
            return {
                totalSize: container.totalSize || 0,
                size: container.size || 0,
                items: container.Metadata || container.Video || container.Track || []
            };
        } catch (error) {
            throw new Error(`Failed to get library content: ${error.message}`);
        }
    }

    /**
     * Search across all libraries
     */
    async search(query, options = {}) {
        try {
            const params = new URLSearchParams({
                query: query,
                ...options
            });

            const response = await this.client.get(`/search?${params.toString()}`);
            const container = response.data.MediaContainer || response.data;
            return container.Metadata || [];
        } catch (error) {
            throw new Error(`Failed to search: ${error.message}`);
        }
    }

    /**
     * Get item metadata
     */
    async getMetadata(ratingKey) {
        try {
            const response = await this.client.get(`/library/metadata/${ratingKey}`);
            const container = response.data.MediaContainer || response.data;
            return container.Metadata?.[0] || container.Metadata || null;
        } catch (error) {
            throw new Error(`Failed to get metadata: ${error.message}`);
        }
    }

    /**
     * Get item children (seasons, episodes, etc.)
     */
    async getChildren(ratingKey) {
        try {
            const response = await this.client.get(`/library/metadata/${ratingKey}/children`);
            const container = response.data.MediaContainer || response.data;
            return container.Metadata || [];
        } catch (error) {
            throw new Error(`Failed to get children: ${error.message}`);
        }
    }

    /**
     * Get recently added items
     */
    async getRecentlyAdded(sectionId = null, limit = 50) {
        try {
            let url = '/library/recentlyAdded';
            const params = new URLSearchParams();
            
            if (sectionId) params.append('sectionId', sectionId);
            if (limit) params.append('X-Plex-Container-Size', limit);
            
            const queryString = params.toString();
            if (queryString) url += '?' + queryString;

            const response = await this.client.get(url);
            const container = response.data.MediaContainer || response.data;
            return container.Metadata || [];
        } catch (error) {
            throw new Error(`Failed to get recently added: ${error.message}`);
        }
    }

    /**
     * Get on deck (continue watching)
     */
    async getOnDeck(limit = 50) {
        try {
            const params = new URLSearchParams();
            if (limit) params.append('X-Plex-Container-Size', limit);

            const response = await this.client.get(`/library/onDeck?${params.toString()}`);
            const container = response.data.MediaContainer || response.data;
            return container.Metadata || [];
        } catch (error) {
            throw new Error(`Failed to get on deck: ${error.message}`);
        }
    }

    /**
     * Get active sessions
     */
    async getSessions() {
        try {
            const response = await this.client.get('/status/sessions');
            const container = response.data.MediaContainer || response.data;
            return container.Metadata || [];
        } catch (error) {
            throw new Error(`Failed to get sessions: ${error.message}`);
        }
    }

    /**
     * Get server statistics
     */
    async getStatistics() {
        try {
            const response = await this.client.get('/library/sections/all');
            const container = response.data.MediaContainer || response.data;
            const sections = container.Directory || [];
            
            const stats = {
                totalSections: sections.length,
                sections: {}
            };

            for (const section of sections) {
                stats.sections[section.title] = {
                    key: section.key,
                    type: section.type,
                    agent: section.agent,
                    scanner: section.scanner,
                    language: section.language,
                    refreshing: section.refreshing === '1',
                    updatedAt: section.updatedAt
                };
            }

            return stats;
        } catch (error) {
            throw new Error(`Failed to get statistics: ${error.message}`);
        }
    }

    /**
     * Mark item as watched
     */
    async markWatched(ratingKey) {
        try {
            await this.client.get(`/:/scrobble?key=${ratingKey}&identifier=com.plexapp.plugins.library`);
            this.emit('itemWatched', ratingKey);
            return true;
        } catch (error) {
            throw new Error(`Failed to mark as watched: ${error.message}`);
        }
    }

    /**
     * Mark item as unwatched
     */
    async markUnwatched(ratingKey) {
        try {
            await this.client.get(`/:/unscrobble?key=${ratingKey}&identifier=com.plexapp.plugins.library`);
            this.emit('itemUnwatched', ratingKey);
            return true;
        } catch (error) {
            throw new Error(`Failed to mark as unwatched: ${error.message}`);
        }
    }

    /**
     * Rate item
     */
    async rateItem(ratingKey, rating) {
        try {
            await this.client.get(`/:/rate?key=${ratingKey}&rating=${rating}&identifier=com.plexapp.plugins.library`);
            this.emit('itemRated', { ratingKey, rating });
            return true;
        } catch (error) {
            throw new Error(`Failed to rate item: ${error.message}`);
        }
    }

    /**
     * Update library section
     */
    async updateLibrary(sectionId) {
        try {
            await this.client.get(`/library/sections/${sectionId}/refresh`);
            this.emit('libraryUpdateStarted', sectionId);
            return true;
        } catch (error) {
            throw new Error(`Failed to update library: ${error.message}`);
        }
    }

    /**
     * Get server preferences
     */
    async getPreferences() {
        try {
            const response = await this.client.get('/:/prefs');
            const container = response.data.MediaContainer || response.data;
            return container.Setting || [];
        } catch (error) {
            throw new Error(`Failed to get preferences: ${error.message}`);
        }
    }

    /**
     * Get server activities
     */
    async getActivities() {
        try {
            const response = await this.client.get('/activities');
            const container = response.data.MediaContainer || response.data;
            return container.Activity || [];
        } catch (error) {
            throw new Error(`Failed to get activities: ${error.message}`);
        }
    }

    /**
     * Get playlists
     */
    async getPlaylists() {
        try {
            const response = await this.client.get('/playlists');
            const container = response.data.MediaContainer || response.data;
            return container.Metadata || [];
        } catch (error) {
            throw new Error(`Failed to get playlists: ${error.message}`);
        }
    }

    /**
     * Create playlist
     */
    async createPlaylist(title, type = 'video', items = []) {
        try {
            const params = new URLSearchParams({
                title: title,
                type: type,
                smart: '0'
            });

            if (items.length > 0) {
                params.append('uri', `library://uuid/items?${items.map(id => `key=${id}`).join('&')}`);
            }

            const response = await this.client.post(`/playlists?${params.toString()}`);
            const container = response.data.MediaContainer || response.data;
            const playlist = container.Metadata?.[0] || container.Metadata;
            
            this.emit('playlistCreated', playlist);
            return playlist;
        } catch (error) {
            throw new Error(`Failed to create playlist: ${error.message}`);
        }
    }

    /**
     * Get transcode sessions
     */
    async getTranscodeSessions() {
        try {
            const response = await this.client.get('/transcode/sessions');
            const container = response.data.MediaContainer || response.data;
            return container.TranscodeSession || [];
        } catch (error) {
            throw new Error(`Failed to get transcode sessions: ${error.message}`);
        }
    }

    /**
     * Get image URL
     */
    getImageUrl(imagePath, width = null, height = null) {
        if (!imagePath) return null;
        
        let url = `${this.baseURL}${imagePath}`;
        const params = new URLSearchParams();
        
        if (this.token) params.append('X-Plex-Token', this.token);
        if (width) params.append('width', width);
        if (height) params.append('height', height);
        
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
                serverName: info.friendlyName || info.machineIdentifier,
                version: info.version,
                platform: info.platform
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
    setupWebhook(app, path = '/plex/webhook') {
        app.post(path, async (req, res) => {
            try {
                const payload = JSON.parse(req.body.payload || '{}');
                this.emit('webhook', payload);
                
                // Emit specific events based on event type
                switch (payload.event) {
                    case 'media.play':
                        this.emit('playbackStarted', payload);
                        break;
                    case 'media.pause':
                        this.emit('playbackPaused', payload);
                        break;
                    case 'media.resume':
                        this.emit('playbackResumed', payload);
                        break;
                    case 'media.stop':
                        this.emit('playbackStopped', payload);
                        break;
                    case 'media.scrobble':
                        this.emit('mediaScrobbled', payload);
                        break;
                    case 'library.new':
                        this.emit('libraryNew', payload);
                        break;
                    case 'admin.database.backup':
                        this.emit('databaseBackup', payload);
                        break;
                    case 'admin.database.corrupted':
                        this.emit('databaseCorrupted', payload);
                        break;
                    default:
                        this.emit('unknownEvent', payload);
                }
                
                res.status(200).json({ success: true });
            } catch (error) {
                logger.error('Plex webhook error:', error);
                res.status(500).json({ error: 'Webhook processing failed' });
            }
        });
    }
}

module.exports = PlexIntegration;