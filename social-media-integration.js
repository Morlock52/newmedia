/**
 * Safe Social Media Integration Service
 * Secure social media features with content filtering and privacy protection
 */

const express = require('express');
const axios = require('axios');
const crypto = require('crypto');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const { body, validationResult } = require('express-validator');
const winston = require('winston');
const sqlite3 = require('sqlite3').verbose();
const WebSocket = require('ws');
const NodeCache = require('node-cache');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const cors = require('cors');
const sanitizeHtml = require('sanitize-html');
const profanityFilter = require('bad-words');

// Configure logging
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    defaultMeta: { service: 'social-media-integration' },
    transports: [
        new winston.transports.File({ filename: 'social-media-error.log', level: 'error' }),
        new winston.transports.File({ filename: 'social-media.log' }),
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        })
    ]
});

class SafeSocialMediaIntegration {
    constructor() {
        this.db = null;
        this.cache = new NodeCache({ stdTTL: 3600 }); // 1 hour cache
        this.profanityFilter = new profanityFilter();
        this.websockets = new Set();
        
        // Social media platforms configuration
        this.platforms = {
            youtube: {
                apiKey: process.env.YOUTUBE_API_KEY,
                baseUrl: 'https://www.googleapis.com/youtube/v3',
                rateLimitPerMinute: 100
            },
            twitter: {
                bearerToken: process.env.TWITTER_BEARER_TOKEN,
                baseUrl: 'https://api.twitter.com/2',
                rateLimitPerMinute: 300
            },
            reddit: {
                clientId: process.env.REDDIT_CLIENT_ID,
                clientSecret: process.env.REDDIT_CLIENT_SECRET,
                userAgent: 'MediaServer/1.0',
                baseUrl: 'https://oauth.reddit.com',
                rateLimitPerMinute: 60
            }
        };
        
        // Content filtering rules
        this.contentFilters = {
            maxTextLength: 500,
            allowedImageTypes: ['image/jpeg', 'image/png', 'image/gif'],
            maxImageSize: 5 * 1024 * 1024, // 5MB
            blockedKeywords: [
                'spam', 'scam', 'phishing', 'malware', 'virus',
                'hack', 'crack', 'piracy', 'illegal', 'drugs'
            ],
            sensitiveTopics: [
                'politics', 'religion', 'controversial', 'nsfw',
                'adult', 'violence', 'hate', 'discrimination'
            ]
        };
        
        // Privacy and safety settings
        this.privacySettings = {
            dataRetentionDays: 30,
            anonymizeUserData: true,
            encryptSensitiveData: true,
            logUserActivities: false,
            shareDataWithThirdParties: false
        };
        
        this.initDatabase();
        this.setupProfanityFilter();
    }
    
    async initDatabase() {
        return new Promise((resolve, reject) => {
            this.db = new sqlite3.Database('social_media.db', (err) => {
                if (err) {
                    logger.error('Database connection error:', err);
                    reject(err);
                    return;
                }
                
                this.db.serialize(() => {
                    // Social media posts table
                    this.db.run(`
                        CREATE TABLE IF NOT EXISTS social_posts (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            post_id TEXT UNIQUE,
                            platform TEXT,
                            user_id TEXT,
                            content TEXT,
                            media_urls TEXT,
                            hashtags TEXT,
                            mentions TEXT,
                            engagement_score REAL,
                            safety_score REAL,
                            content_warnings TEXT,
                            moderation_status TEXT DEFAULT 'pending',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    `);
                    
                    // User social profiles table
                    this.db.run(`
                        CREATE TABLE IF NOT EXISTS user_social_profiles (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT UNIQUE,
                            connected_platforms TEXT,
                            privacy_settings TEXT,
                            content_preferences TEXT,
                            blocked_users TEXT,
                            blocked_hashtags TEXT,
                            safety_level TEXT DEFAULT 'moderate',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    `);
                    
                    // Social interactions table
                    this.db.run(`
                        CREATE TABLE IF NOT EXISTS social_interactions (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            user_id TEXT,
                            post_id TEXT,
                            interaction_type TEXT,
                            platform TEXT,
                            content_shared TEXT,
                            safety_checked BOOLEAN DEFAULT FALSE,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    `);
                    
                    // Trending topics table
                    this.db.run(`
                        CREATE TABLE IF NOT EXISTS trending_topics (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            topic TEXT,
                            platform TEXT,
                            mention_count INTEGER,
                            safety_score REAL,
                            content_category TEXT,
                            trend_score REAL,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    `);
                    
                    // Content moderation log
                    this.db.run(`
                        CREATE TABLE IF NOT EXISTS moderation_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            content_id TEXT,
                            platform TEXT,
                            moderation_action TEXT,
                            reason TEXT,
                            moderator_type TEXT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    `);
                });
                
                logger.info('Social media database initialized');
                resolve();
            });
        });
    }
    
    setupProfanityFilter() {
        // Add custom words to profanity filter
        this.profanityFilter.addWords(...this.contentFilters.blockedKeywords);
    }
    
    // Content Safety and Moderation
    async moderateContent(content, contentType = 'text', platform = 'unknown') {
        const moderation = {
            isApproved: false,
            safetyScore: 0.0,
            contentWarnings: [],
            reasoning: [],
            suggestedActions: []
        };
        
        try {
            // Text content moderation
            if (contentType === 'text') {
                const textModeration = await this.moderateText(content);
                moderation.safetyScore = textModeration.safetyScore;
                moderation.contentWarnings.push(...textModeration.warnings);
                moderation.reasoning.push(...textModeration.reasoning);
            }
            
            // Media content moderation
            if (contentType === 'media' && Array.isArray(content)) {
                const mediaModeration = await this.moderateMedia(content);
                moderation.safetyScore = Math.min(moderation.safetyScore, mediaModeration.safetyScore);
                moderation.contentWarnings.push(...mediaModeration.warnings);
                moderation.reasoning.push(...mediaModeration.reasoning);
            }
            
            // Platform-specific moderation
            const platformModeration = await this.applyPlatformRules(content, platform);
            moderation.safetyScore = Math.min(moderation.safetyScore, platformModeration.safetyScore);
            moderation.reasoning.push(...platformModeration.reasoning);
            
            // Determine approval
            moderation.isApproved = moderation.safetyScore >= 0.7;
            
            if (!moderation.isApproved) {
                moderation.suggestedActions = [
                    'Content requires review',
                    'Consider adding content warnings',
                    'Remove inappropriate elements'
                ];
            } else if (moderation.safetyScore < 0.9) {
                moderation.suggestedActions = [
                    'Content approved with warnings',
                    'Monitor engagement for safety'
                ];
            }
            
            // Log moderation action
            await this.logModerationAction(content, platform, moderation);
            
        } catch (error) {
            logger.error('Content moderation error:', error);
            moderation.contentWarnings.push('Moderation system error - manual review required');
            moderation.suggestedActions.push('Escalate to human moderator');
        }
        
        return moderation;
    }
    
    async moderateText(text) {
        const moderation = {
            safetyScore: 1.0,
            warnings: [],
            reasoning: []
        };
        
        // Profanity detection
        if (this.profanityFilter.isProfane(text)) {
            moderation.safetyScore -= 0.4;
            moderation.warnings.push('Profanity detected');
            moderation.reasoning.push('Text contains inappropriate language');
        }
        
        // Blocked keywords check
        const blockedFound = this.contentFilters.blockedKeywords.filter(keyword =>
            text.toLowerCase().includes(keyword.toLowerCase())
        );
        if (blockedFound.length > 0) {
            moderation.safetyScore -= 0.3 * blockedFound.length;
            moderation.warnings.push(`Blocked keywords: ${blockedFound.join(', ')}`);
            moderation.reasoning.push('Text contains prohibited keywords');
        }
        
        // Sensitive topics detection
        const sensitiveFound = this.contentFilters.sensitiveTopics.filter(topic =>
            text.toLowerCase().includes(topic.toLowerCase())
        );
        if (sensitiveFound.length > 0) {
            moderation.safetyScore -= 0.2;
            moderation.warnings.push(`Sensitive topics: ${sensitiveFound.join(', ')}`);
            moderation.reasoning.push('Text discusses sensitive topics');
        }
        
        // Length check
        if (text.length > this.contentFilters.maxTextLength) {
            moderation.safetyScore -= 0.1;
            moderation.warnings.push('Content too long');
            moderation.reasoning.push(`Text exceeds ${this.contentFilters.maxTextLength} characters`);
        }
        
        // URL analysis
        const urls = this.extractUrls(text);
        if (urls.length > 0) {
            const urlSafety = await this.checkUrlSafety(urls);
            if (!urlSafety.allSafe) {
                moderation.safetyScore -= 0.3;
                moderation.warnings.push('Suspicious URLs detected');
                moderation.reasoning.push('Text contains potentially unsafe links');
            }
        }
        
        moderation.safetyScore = Math.max(0, moderation.safetyScore);
        return moderation;
    }
    
    async moderateMedia(mediaUrls) {
        const moderation = {
            safetyScore: 1.0,
            warnings: [],
            reasoning: []
        };
        
        for (const mediaUrl of mediaUrls) {
            try {
                // Check file type
                const response = await axios.head(mediaUrl, { timeout: 5000 });
                const contentType = response.headers['content-type'];
                
                if (!this.contentFilters.allowedImageTypes.includes(contentType)) {
                    moderation.safetyScore -= 0.2;
                    moderation.warnings.push(`Unsupported media type: ${contentType}`);
                    continue;
                }
                
                // Check file size
                const contentLength = parseInt(response.headers['content-length'] || '0');
                if (contentLength > this.contentFilters.maxImageSize) {
                    moderation.safetyScore -= 0.1;
                    moderation.warnings.push('Media file too large');
                }
                
                // Basic image safety check (would integrate with AI vision API)
                const imageSafety = await this.checkImageSafety(mediaUrl);
                moderation.safetyScore = Math.min(moderation.safetyScore, imageSafety.safetyScore);
                moderation.warnings.push(...imageSafety.warnings);
                
            } catch (error) {
                moderation.safetyScore -= 0.3;
                moderation.warnings.push('Unable to verify media safety');
                moderation.reasoning.push(`Media verification failed: ${error.message}`);
            }
        }
        
        return moderation;
    }
    
    async applyPlatformRules(content, platform) {
        const moderation = {
            safetyScore: 1.0,
            reasoning: []
        };
        
        switch (platform.toLowerCase()) {
            case 'youtube':
                // YouTube-specific content rules
                if (typeof content === 'string' && content.length > 5000) {
                    moderation.safetyScore -= 0.1;
                    moderation.reasoning.push('Content too long for YouTube description');
                }
                break;
                
            case 'twitter':
                // Twitter-specific rules
                if (typeof content === 'string' && content.length > 280) {
                    moderation.safetyScore -= 0.2;
                    moderation.reasoning.push('Content exceeds Twitter character limit');
                }
                break;
                
            case 'reddit':
                // Reddit-specific rules
                if (typeof content === 'string' && content.includes('reddit.com')) {
                    moderation.safetyScore -= 0.1;
                    moderation.reasoning.push('Self-promotion may violate Reddit rules');
                }
                break;
        }
        
        return moderation;
    }
    
    async checkImageSafety(imageUrl) {
        // Placeholder for image safety check
        // In production, this would use Google Vision API, Azure Cognitive Services, etc.
        return {
            safetyScore: 0.8,
            warnings: []
        };
    }
    
    extractUrls(text) {
        const urlRegex = /(https?:\/\/[^\s]+)/g;
        return text.match(urlRegex) || [];
    }
    
    async checkUrlSafety(urls) {
        const results = {
            allSafe: true,
            unsafeUrls: [],
            reasoning: []
        };
        
        for (const url of urls) {
            try {
                // Basic domain check
                const domain = new URL(url).hostname;
                
                // Check against known malicious domains (would use threat intelligence API)
                const knownBadDomains = ['malicious-site.com', 'phishing-domain.org'];
                if (knownBadDomains.includes(domain)) {
                    results.allSafe = false;
                    results.unsafeUrls.push(url);
                    results.reasoning.push(`Blocked domain: ${domain}`);
                }
                
                // Check for suspicious patterns
                if (url.includes('bit.ly') || url.includes('tinyurl.com')) {
                    // URL shorteners require additional verification
                    results.reasoning.push('URL shortener detected - requires verification');
                }
                
            } catch (error) {
                results.reasoning.push(`Invalid URL format: ${url}`);
            }
        }
        
        return results;
    }
    
    async logModerationAction(content, platform, moderation) {
        const contentId = crypto.createHash('md5').update(JSON.stringify(content)).digest('hex');
        
        this.db.run(`
            INSERT INTO moderation_log 
            (content_id, platform, moderation_action, reason, moderator_type)
            VALUES (?, ?, ?, ?, ?)
        `, [
            contentId,
            platform,
            moderation.isApproved ? 'approved' : 'rejected',
            JSON.stringify(moderation.reasoning),
            'ai'
        ], (err) => {
            if (err) {
                logger.error('Failed to log moderation action:', err);
            }
        });
    }
    
    // Social Media Platform Integration
    async searchYouTubeContent(query, maxResults = 10, safetyLevel = 'moderate') {
        try {
            const cacheKey = `youtube_search_${crypto.createHash('md5').update(query + safetyLevel).digest('hex')}`;
            const cached = this.cache.get(cacheKey);
            if (cached) return cached;
            
            const response = await axios.get(`${this.platforms.youtube.baseUrl}/search`, {
                params: {
                    key: this.platforms.youtube.apiKey,
                    q: query,
                    part: 'snippet',
                    maxResults,
                    safeSearch: safetyLevel === 'strict' ? 'strict' : 'moderate',
                    type: 'video',
                    videoEmbeddable: 'true',
                    videoCategoryId: '27' // Education category for safer content
                }
            });
            
            const results = await this.processYouTubeResults(response.data.items || []);
            this.cache.set(cacheKey, results, 1800); // Cache for 30 minutes
            
            return results;
            
        } catch (error) {
            logger.error('YouTube search error:', error);
            return { error: 'YouTube search failed', results: [] };
        }
    }
    
    async processYouTubeResults(items) {
        const processedResults = [];
        
        for (const item of items) {
            // Content safety assessment
            const contentText = `${item.snippet.title} ${item.snippet.description}`;
            const moderation = await this.moderateContent(contentText, 'text', 'youtube');
            
            if (moderation.isApproved) {
                processedResults.push({
                    id: item.id.videoId,
                    title: sanitizeHtml(item.snippet.title),
                    description: sanitizeHtml(item.snippet.description),
                    channelTitle: sanitizeHtml(item.snippet.channelTitle),
                    publishedAt: item.snippet.publishedAt,
                    thumbnails: item.snippet.thumbnails,
                    safetyScore: moderation.safetyScore,
                    contentWarnings: moderation.contentWarnings,
                    url: `https://www.youtube.com/watch?v=${item.id.videoId}`
                });
            }
        }
        
        return {
            results: processedResults,
            totalProcessed: items.length,
            safeResults: processedResults.length
        };
    }
    
    async getTwitterTrends(location = 1) { // 1 = worldwide
        try {
            const cacheKey = `twitter_trends_${location}`;
            const cached = this.cache.get(cacheKey);
            if (cached) return cached;
            
            const response = await axios.get(`${this.platforms.twitter.baseUrl}/trends/place.json`, {
                params: { id: location },
                headers: {
                    'Authorization': `Bearer ${this.platforms.twitter.bearerToken}`
                }
            });
            
            const trends = response.data[0]?.trends || [];
            const processedTrends = await this.processTwitterTrends(trends);
            
            this.cache.set(cacheKey, processedTrends, 900); // Cache for 15 minutes
            return processedTrends;
            
        } catch (error) {
            logger.error('Twitter trends error:', error);
            return { error: 'Twitter trends unavailable', trends: [] };
        }
    }
    
    async processTwitterTrends(trends) {
        const safeTrends = [];
        
        for (const trend of trends) {
            const moderation = await this.moderateContent(trend.name, 'text', 'twitter');
            
            if (moderation.isApproved && moderation.safetyScore > 0.6) {
                // Store trend in database
                await this.storeTrendingTopic(trend.name, 'twitter', trend.tweet_volume || 0, moderation.safetyScore);
                
                safeTrends.push({
                    name: sanitizeHtml(trend.name),
                    url: trend.url,
                    volume: trend.tweet_volume,
                    safetyScore: moderation.safetyScore,
                    contentWarnings: moderation.contentWarnings
                });
            }
        }
        
        return {
            trends: safeTrends,
            totalTrends: trends.length,
            safeTrends: safeTrends.length
        };
    }
    
    async searchRedditContent(subreddit, query = '', limit = 10, sort = 'hot') {
        try {
            const cacheKey = `reddit_${subreddit}_${crypto.createHash('md5').update(query + sort).digest('hex')}`;
            const cached = this.cache.get(cacheKey);
            if (cached) return cached;
            
            // Get Reddit access token
            const accessToken = await this.getRedditAccessToken();
            
            const endpoint = query ? 
                `${this.platforms.reddit.baseUrl}/r/${subreddit}/search` :
                `${this.platforms.reddit.baseUrl}/r/${subreddit}/${sort}`;
            
            const response = await axios.get(endpoint, {
                params: {
                    q: query,
                    limit,
                    sort,
                    restrict_sr: true
                },
                headers: {
                    'Authorization': `Bearer ${accessToken}`,
                    'User-Agent': this.platforms.reddit.userAgent
                }
            });
            
            const results = await this.processRedditResults(response.data.data.children || []);
            this.cache.set(cacheKey, results, 1200); // Cache for 20 minutes
            
            return results;
            
        } catch (error) {
            logger.error('Reddit search error:', error);
            return { error: 'Reddit search failed', results: [] };
        }
    }
    
    async getRedditAccessToken() {
        // Simplified token retrieval - in production, implement proper OAuth flow
        const cacheKey = 'reddit_access_token';
        let token = this.cache.get(cacheKey);
        
        if (!token) {
            try {
                const response = await axios.post('https://www.reddit.com/api/v1/access_token', 
                    'grant_type=client_credentials',
                    {
                        auth: {
                            username: this.platforms.reddit.clientId,
                            password: this.platforms.reddit.clientSecret
                        },
                        headers: {
                            'Content-Type': 'application/x-www-form-urlencoded',
                            'User-Agent': this.platforms.reddit.userAgent
                        }
                    }
                );
                
                token = response.data.access_token;
                this.cache.set(cacheKey, token, response.data.expires_in - 300); // Cache with buffer
            } catch (error) {
                logger.error('Reddit token error:', error);
                throw new Error('Failed to get Reddit access token');
            }
        }
        
        return token;
    }
    
    async processRedditResults(posts) {
        const processedResults = [];
        
        for (const post of posts) {
            const data = post.data;
            
            // Content safety assessment
            const contentText = `${data.title} ${data.selftext || ''}`;
            const moderation = await this.moderateContent(contentText, 'text', 'reddit');
            
            // Additional Reddit-specific safety checks
            if (data.over_18 || data.nsfw) {
                moderation.safetyScore = Math.min(moderation.safetyScore, 0.3);
                moderation.contentWarnings.push('NSFW content');
            }
            
            if (moderation.isApproved && moderation.safetyScore > 0.5) {
                processedResults.push({
                    id: data.id,
                    title: sanitizeHtml(data.title),
                    content: sanitizeHtml(data.selftext || ''),
                    author: data.author,
                    subreddit: data.subreddit,
                    score: data.score,
                    comments: data.num_comments,
                    created: new Date(data.created_utc * 1000),
                    url: `https://reddit.com${data.permalink}`,
                    safetyScore: moderation.safetyScore,
                    contentWarnings: moderation.contentWarnings
                });
            }
        }
        
        return {
            results: processedResults,
            totalProcessed: posts.length,
            safeResults: processedResults.length
        };
    }
    
    async storeTrendingTopic(topic, platform, mentionCount, safetyScore) {
        const category = await this.categorizeTopic(topic);
        
        this.db.run(`
            INSERT OR REPLACE INTO trending_topics 
            (topic, platform, mention_count, safety_score, content_category, trend_score)
            VALUES (?, ?, ?, ?, ?, ?)
        `, [
            topic,
            platform,
            mentionCount,
            safetyScore,
            category,
            mentionCount * safetyScore // Simple trend score calculation
        ]);
    }
    
    async categorizeTopic(topic) {
        // Simplified topic categorization
        const categories = {
            'technology': ['tech', 'ai', 'software', 'computer', 'internet'],
            'entertainment': ['movie', 'music', 'game', 'tv', 'celebrity'],
            'news': ['breaking', 'news', 'update', 'report'],
            'sports': ['football', 'basketball', 'soccer', 'baseball', 'sports'],
            'education': ['learn', 'study', 'education', 'science', 'research']
        };
        
        const lowerTopic = topic.toLowerCase();
        
        for (const [category, keywords] of Object.entries(categories)) {
            if (keywords.some(keyword => lowerTopic.includes(keyword))) {
                return category;
            }
        }
        
        return 'general';
    }
    
    // User Management and Privacy
    async createUserSocialProfile(userId, preferences = {}) {
        const defaultPreferences = {
            safetyLevel: 'moderate',
            connectedPlatforms: [],
            privacySettings: {
                shareActivity: false,
                allowTagging: false,
                publicProfile: false
            },
            contentPreferences: {
                categories: ['education', 'entertainment'],
                languages: ['en']
            },
            blockedUsers: [],
            blockedHashtags: []
        };
        
        const userPreferences = { ...defaultPreferences, ...preferences };
        
        this.db.run(`
            INSERT OR REPLACE INTO user_social_profiles 
            (user_id, connected_platforms, privacy_settings, content_preferences, 
             blocked_users, blocked_hashtags, safety_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        `, [
            userId,
            JSON.stringify(userPreferences.connectedPlatforms),
            JSON.stringify(userPreferences.privacySettings),
            JSON.stringify(userPreferences.contentPreferences),
            JSON.stringify(userPreferences.blockedUsers),
            JSON.stringify(userPreferences.blockedHashtags),
            userPreferences.safetyLevel
        ], (err) => {
            if (err) {
                logger.error('Failed to create user social profile:', err);
                throw new Error('Profile creation failed');
            }
        });
        
        return { success: true, userId, preferences: userPreferences };
    }
    
    async getUserSocialProfile(userId) {
        return new Promise((resolve, reject) => {
            this.db.get(`
                SELECT * FROM user_social_profiles WHERE user_id = ?
            `, [userId], (err, row) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                if (!row) {
                    resolve(null);
                    return;
                }
                
                resolve({
                    userId: row.user_id,
                    connectedPlatforms: JSON.parse(row.connected_platforms || '[]'),
                    privacySettings: JSON.parse(row.privacy_settings || '{}'),
                    contentPreferences: JSON.parse(row.content_preferences || '{}'),
                    blockedUsers: JSON.parse(row.blocked_users || '[]'),
                    blockedHashtags: JSON.parse(row.blocked_hashtags || '[]'),
                    safetyLevel: row.safety_level,
                    createdAt: row.created_at,
                    updatedAt: row.updated_at
                });
            });
        });
    }
    
    async recordSocialInteraction(userId, postId, interactionType, platform, contentShared = null) {
        // Safety check on shared content
        let safetyChecked = false;
        if (contentShared) {
            const moderation = await this.moderateContent(contentShared, 'text', platform);
            safetyChecked = moderation.isApproved;
            
            if (!safetyChecked) {
                logger.warn(`Unsafe content sharing attempt by user ${userId}:`, moderation.contentWarnings);
                return { success: false, reason: 'Content failed safety check' };
            }
        }
        
        this.db.run(`
            INSERT INTO social_interactions 
            (user_id, post_id, interaction_type, platform, content_shared, safety_checked)
            VALUES (?, ?, ?, ?, ?, ?)
        `, [
            userId, postId, interactionType, platform, 
            contentShared, safetyChecked
        ]);
        
        return { success: true, safetyChecked };
    }
    
    // Analytics and Insights
    async getSocialMediaInsights(userId = null, days = 7) {
        return new Promise((resolve, reject) => {
            const query = userId ? 
                'SELECT * FROM social_interactions WHERE user_id = ? AND timestamp > datetime("now", "-" || ? || " days")' :
                'SELECT * FROM social_interactions WHERE timestamp > datetime("now", "-" || ? || " days")';
            
            const params = userId ? [userId, days] : [days];
            
            this.db.all(query, params, (err, rows) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                const insights = this.analyzeInteractions(rows);
                resolve(insights);
            });
        });
    }
    
    analyzeInteractions(interactions) {
        const insights = {
            totalInteractions: interactions.length,
            platformBreakdown: {},
            interactionTypes: {},
            safetyStats: {
                safeInteractions: 0,
                unsafeAttempts: 0
            },
            trendsAnalysis: {}
        };
        
        interactions.forEach(interaction => {
            // Platform breakdown
            insights.platformBreakdown[interaction.platform] = 
                (insights.platformBreakdown[interaction.platform] || 0) + 1;
            
            // Interaction types
            insights.interactionTypes[interaction.interaction_type] = 
                (insights.interactionTypes[interaction.interaction_type] || 0) + 1;
            
            // Safety stats
            if (interaction.safety_checked) {
                insights.safetyStats.safeInteractions++;
            } else if (interaction.content_shared) {
                insights.safetyStats.unsafeAttempts++;
            }
        });
        
        return insights;
    }
    
    async getTrendingTopicsSafe(platform = null, limit = 10) {
        return new Promise((resolve, reject) => {
            let query = `
                SELECT * FROM trending_topics 
                WHERE safety_score > 0.6
            `;
            const params = [];
            
            if (platform) {
                query += ' AND platform = ?';
                params.push(platform);
            }
            
            query += ' ORDER BY trend_score DESC LIMIT ?';
            params.push(limit);
            
            this.db.all(query, params, (err, rows) => {
                if (err) {
                    reject(err);
                    return;
                }
                
                const trends = rows.map(row => ({
                    topic: row.topic,
                    platform: row.platform,
                    mentionCount: row.mention_count,
                    safetyScore: row.safety_score,
                    category: row.content_category,
                    trendScore: row.trend_score,
                    createdAt: row.created_at
                }));
                
                resolve(trends);
            });
        });
    }
    
    // WebSocket real-time features
    setupWebSocket(server) {
        const wss = new WebSocket.Server({ server });
        
        wss.on('connection', (ws, request) => {
            logger.info('WebSocket client connected');
            this.websockets.add(ws);
            
            ws.on('message', async (message) => {
                try {
                    const data = JSON.parse(message);
                    await this.handleWebSocketMessage(ws, data);
                } catch (error) {
                    ws.send(JSON.stringify({ error: 'Invalid message format' }));
                }
            });
            
            ws.on('close', () => {
                this.websockets.delete(ws);
                logger.info('WebSocket client disconnected');
            });
        });
        
        // Periodic trending topics broadcast
        setInterval(async () => {
            try {
                const trends = await this.getTrendingTopicsSafe(null, 5);
                this.broadcastToAll({
                    type: 'trending_update',
                    data: trends,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                logger.error('Error broadcasting trends:', error);
            }
        }, 300000); // Every 5 minutes
    }
    
    async handleWebSocketMessage(ws, data) {
        switch (data.type) {
            case 'search_content':
                const results = await this.handleContentSearch(data.payload);
                ws.send(JSON.stringify({
                    type: 'search_results',
                    data: results,
                    requestId: data.requestId
                }));
                break;
                
            case 'moderate_content':
                const moderation = await this.moderateContent(
                    data.payload.content,
                    data.payload.contentType,
                    data.payload.platform
                );
                ws.send(JSON.stringify({
                    type: 'moderation_result',
                    data: moderation,
                    requestId: data.requestId
                }));
                break;
                
            default:
                ws.send(JSON.stringify({ error: 'Unknown message type' }));
        }
    }
    
    async handleContentSearch(payload) {
        const { platform, query, userId } = payload;
        
        // Get user preferences for safer results
        let userProfile = null;
        if (userId) {
            userProfile = await this.getUserSocialProfile(userId);
        }
        
        const safetyLevel = userProfile?.safetyLevel || 'moderate';
        
        switch (platform.toLowerCase()) {
            case 'youtube':
                return await this.searchYouTubeContent(query, 10, safetyLevel);
            case 'twitter':
                return await this.getTwitterTrends();
            case 'reddit':
                return await this.searchRedditContent('all', query, 10);
            default:
                return { error: 'Unsupported platform' };
        }
    }
    
    broadcastToAll(message) {
        const messageString = JSON.stringify(message);
        this.websockets.forEach(ws => {
            if (ws.readyState === WebSocket.OPEN) {
                ws.send(messageString);
            }
        });
    }
    
    // Cleanup and maintenance
    async performMaintenance() {
        logger.info('Starting social media integration maintenance');
        
        // Clean old data based on privacy settings
        const retentionDays = this.privacySettings.dataRetentionDays;
        
        this.db.run(`
            DELETE FROM social_interactions 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        `, [retentionDays]);
        
        this.db.run(`
            DELETE FROM trending_topics 
            WHERE created_at < datetime('now', '-' || ? || ' days')
        `, [retentionDays]);
        
        this.db.run(`
            DELETE FROM moderation_log 
            WHERE timestamp < datetime('now', '-' || ? || ' days')
        `, [retentionDays]);
        
        // Clear cache
        this.cache.flushAll();
        
        logger.info('Maintenance completed');
    }
}

// Express.js API Server
class SocialMediaAPI {
    constructor() {
        this.app = express();
        this.integration = new SafeSocialMediaIntegration();
        
        this.setupMiddleware();
        this.setupRoutes();
    }
    
    setupMiddleware() {
        // Security middleware
        this.app.use(helmet({
            contentSecurityPolicy: {
                directives: {
                    defaultSrc: ["'self'"],
                    styleSrc: ["'self'", "'unsafe-inline'"],
                    scriptSrc: ["'self'"],
                    imgSrc: ["'self'", "data:", "https:"],
                    connectSrc: ["'self'", "wss:"]
                }
            }
        }));
        
        this.app.use(cors({
            origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
            credentials: true
        }));
        
        // Rate limiting
        const limiter = rateLimit({
            windowMs: 15 * 60 * 1000, // 15 minutes
            max: 200, // limit each IP to 200 requests per windowMs
            message: 'Too many requests from this IP'
        });
        this.app.use(limiter);
        
        // Body parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));
        
        // Logging
        this.app.use((req, res, next) => {
            logger.info(`${req.method} ${req.path} - ${req.ip}`);
            next();
        });
    }
    
    setupRoutes() {
        // Health check
        this.app.get('/health', (req, res) => {
            res.json({ status: 'healthy', timestamp: new Date().toISOString() });
        });
        
        // Search social media content
        this.app.post('/api/social/search/:platform',
            body('query').isLength({ min: 1, max: 200 }),
            body('maxResults').optional().isInt({ min: 1, max: 50 }),
            this.validateRequest,
            async (req, res) => {
                try {
                    const { platform } = req.params;
                    const { query, maxResults = 10, userId } = req.body;
                    
                    let results;
                    switch (platform.toLowerCase()) {
                        case 'youtube':
                            results = await this.integration.searchYouTubeContent(query, maxResults);
                            break;
                        case 'twitter':
                            results = await this.integration.getTwitterTrends();
                            break;
                        case 'reddit':
                            const subreddit = req.body.subreddit || 'all';
                            results = await this.integration.searchRedditContent(subreddit, query, maxResults);
                            break;
                        default:
                            return res.status(400).json({ error: 'Unsupported platform' });
                    }
                    
                    res.json(results);
                } catch (error) {
                    logger.error('Social search error:', error);
                    res.status(500).json({ error: 'Search failed' });
                }
            }
        );
        
        // Moderate content
        this.app.post('/api/social/moderate',
            body('content').isLength({ min: 1 }),
            body('contentType').optional().isIn(['text', 'media']),
            body('platform').optional().isLength({ min: 1, max: 20 }),
            this.validateRequest,
            async (req, res) => {
                try {
                    const { content, contentType = 'text', platform = 'unknown' } = req.body;
                    const moderation = await this.integration.moderateContent(content, contentType, platform);
                    res.json(moderation);
                } catch (error) {
                    logger.error('Content moderation error:', error);
                    res.status(500).json({ error: 'Moderation failed' });
                }
            }
        );
        
        // Create user social profile
        this.app.post('/api/social/profile',
            body('userId').isLength({ min: 1, max: 50 }),
            body('preferences').optional().isObject(),
            this.validateRequest,
            async (req, res) => {
                try {
                    const { userId, preferences } = req.body;
                    const result = await this.integration.createUserSocialProfile(userId, preferences);
                    res.json(result);
                } catch (error) {
                    logger.error('Profile creation error:', error);
                    res.status(500).json({ error: 'Profile creation failed' });
                }
            }
        );
        
        // Get user social profile
        this.app.get('/api/social/profile/:userId', async (req, res) => {
            try {
                const { userId } = req.params;
                const profile = await this.integration.getUserSocialProfile(userId);
                
                if (!profile) {
                    return res.status(404).json({ error: 'Profile not found' });
                }
                
                res.json(profile);
            } catch (error) {
                logger.error('Profile retrieval error:', error);
                res.status(500).json({ error: 'Failed to get profile' });
            }
        });
        
        // Record social interaction
        this.app.post('/api/social/interaction',
            body('userId').isLength({ min: 1, max: 50 }),
            body('postId').isLength({ min: 1, max: 100 }),
            body('interactionType').isIn(['like', 'share', 'comment', 'view']),
            body('platform').isLength({ min: 1, max: 20 }),
            body('contentShared').optional().isLength({ max: 500 }),
            this.validateRequest,
            async (req, res) => {
                try {
                    const { userId, postId, interactionType, platform, contentShared } = req.body;
                    const result = await this.integration.recordSocialInteraction(
                        userId, postId, interactionType, platform, contentShared
                    );
                    res.json(result);
                } catch (error) {
                    logger.error('Interaction recording error:', error);
                    res.status(500).json({ error: 'Failed to record interaction' });
                }
            }
        );
        
        // Get trending topics
        this.app.get('/api/social/trending', async (req, res) => {
            try {
                const { platform, limit = 10 } = req.query;
                const trends = await this.integration.getTrendingTopicsSafe(platform, parseInt(limit));
                res.json({ trends });
            } catch (error) {
                logger.error('Trending topics error:', error);
                res.status(500).json({ error: 'Failed to get trending topics' });
            }
        });
        
        // Get social media insights
        this.app.get('/api/social/insights', async (req, res) => {
            try {
                const { userId, days = 7 } = req.query;
                const insights = await this.integration.getSocialMediaInsights(userId, parseInt(days));
                res.json(insights);
            } catch (error) {
                logger.error('Insights error:', error);
                res.status(500).json({ error: 'Failed to get insights' });
            }
        });
    }
    
    validateRequest(req, res, next) {
        const errors = validationResult(req);
        if (!errors.isEmpty()) {
            return res.status(400).json({ errors: errors.array() });
        }
        next();
    }
    
    start(port = 8082) {
        const server = this.app.listen(port, () => {
            logger.info(`Social Media Integration API started on port ${port}`);
        });
        
        // Setup WebSocket
        this.integration.setupWebSocket(server);
        
        // Schedule maintenance
        setInterval(() => {
            this.integration.performMaintenance();
        }, 24 * 60 * 60 * 1000); // Daily maintenance
        
        return server;
    }
}

// Start the API server
if (require.main === module) {
    const api = new SocialMediaAPI();
    api.start(process.env.PORT || 8082);
}

module.exports = { SafeSocialMediaIntegration, SocialMediaAPI };