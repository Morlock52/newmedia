/**
 * Content Moderation Service - JavaScript/Node.js Implementation
 * Real-time content filtering and moderation with AI guardrails
 */

const express = require('express');
const axios = require('axios');
const multer = require('multer');
const sharp = require('sharp');
const ffmpeg = require('fluent-ffmpeg');
const natural = require('natural');
const crypto = require('crypto');
const sqlite3 = require('sqlite3').verbose();
const WebSocket = require('ws');
const rateLimit = require('express-rate-limit');
const helmet = require('helmet');
const cors = require('cors');
const { body, validationResult } = require('express-validator');
const winston = require('winston');
const path = require('path');
const fs = require('fs').promises;

// Configure logging
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    defaultMeta: { service: 'content-moderation' },
    transports: [
        new winston.transports.File({ filename: 'moderation-error.log', level: 'error' }),
        new winston.transports.File({ filename: 'moderation.log' }),
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        })
    ]
});

class ContentModerator {
    constructor() {
        this.db = null;
        this.blockedHashes = new Set();
        this.trustedSources = new Set(['youtube.com', 'vimeo.com', 'github.com']);
        this.blockedDomains = new Set(['malicious-site.com', 'spam-domain.org']);
        
        // Content analysis models and rules
        this.nsfwKeywords = [
            'explicit', 'adult', 'nsfw', 'xxx', 'porn', 'nude', 'naked',
            'sexual', 'erotic', 'mature', '18+', 'uncensored'
        ];
        
        this.harmfulKeywords = [
            'hate', 'violence', 'threat', 'harassment', 'abuse',
            'discrimination', 'terrorism', 'extremism', 'toxic'
        ];
        
        this.copyrightKeywords = [
            'copyright', '©', 'all rights reserved', 'unauthorized',
            'piracy', 'illegal download', 'torrent', 'crack'
        ];
        
        this.safetyThresholds = {
            nsfw: 0.3,
            toxicity: 0.4,
            spam: 0.5,
            copyright: 0.6,
            overall: 0.5
        };
        
        this.initDatabase();
        this.loadBlockedContent();
    }
    
    async initDatabase() {
        return new Promise((resolve, reject) => {
            this.db = new sqlite3.Database('content_moderation.db', (err) => {
                if (err) {
                    logger.error('Database connection error:', err);
                    reject(err);
                    return;
                }
                
                // Create tables
                this.db.serialize(() => {
                    this.db.run(`
                        CREATE TABLE IF NOT EXISTS moderated_content (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            content_hash TEXT UNIQUE,
                            content_type TEXT,
                            file_path TEXT,
                            source_url TEXT,
                            nsfw_score REAL,
                            toxicity_score REAL,
                            spam_score REAL,
                            copyright_score REAL,
                            overall_score REAL,
                            moderation_action TEXT,
                            reasoning TEXT,
                            moderator_id TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    `);
                    
                    this.db.run(`
                        CREATE TABLE IF NOT EXISTS blocked_content (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            content_hash TEXT UNIQUE,
                            block_reason TEXT,
                            blocked_by TEXT,
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    `);
                    
                    this.db.run(`
                        CREATE TABLE IF NOT EXISTS user_reports (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            content_hash TEXT,
                            user_id TEXT,
                            report_reason TEXT,
                            report_details TEXT,
                            status TEXT DEFAULT 'pending',
                            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    `);
                });
                
                logger.info('Content moderation database initialized');
                resolve();
            });
        });
    }
    
    async loadBlockedContent() {
        return new Promise((resolve) => {
            this.db.all('SELECT content_hash FROM blocked_content', (err, rows) => {
                if (err) {
                    logger.error('Error loading blocked content:', err);
                    resolve();
                    return;
                }
                
                this.blockedHashes.clear();
                rows.forEach(row => {
                    this.blockedHashes.add(row.content_hash);
                });
                
                logger.info(`Loaded ${this.blockedHashes.size} blocked content hashes`);
                resolve();
            });
        });
    }
    
    generateContentHash(content) {
        const hash = crypto.createHash('sha256');
        
        if (typeof content === 'string') {
            hash.update(content);
        } else if (Buffer.isBuffer(content)) {
            hash.update(content);
        } else {
            hash.update(JSON.stringify(content));
        }
        
        return hash.digest('hex');
    }
    
    async moderateText(text, sourceUrl = null) {
        const contentHash = this.generateContentHash(text);
        
        // Check if already blocked
        if (this.blockedHashes.has(contentHash)) {
            return {
                contentHash,
                action: 'block',
                reason: 'Previously flagged content',
                scores: { overall: 0.0 }
            };
        }
        
        const analysis = await this.analyzeText(text);
        const sourceAnalysis = sourceUrl ? await this.analyzeSource(sourceUrl) : { score: 1.0, reasoning: [] };
        
        // Calculate overall score
        const overallScore = this.calculateOverallScore(analysis, sourceAnalysis);
        
        // Determine moderation action
        const action = this.determineModerationAction(overallScore, analysis);
        
        // Store results
        await this.storeModeration(contentHash, 'text', null, sourceUrl, analysis, overallScore, action);
        
        return {
            contentHash,
            action,
            scores: {
                nsfw: analysis.nsfw,
                toxicity: analysis.toxicity,
                spam: analysis.spam,
                copyright: analysis.copyright,
                overall: overallScore
            },
            reasoning: analysis.reasoning,
            warnings: analysis.warnings
        };
    }
    
    async moderateImage(imagePath, sourceUrl = null) {
        try {
            const imageBuffer = await fs.readFile(imagePath);
            const contentHash = this.generateContentHash(imageBuffer);
            
            // Check if already blocked
            if (this.blockedHashes.has(contentHash)) {
                return {
                    contentHash,
                    action: 'block',
                    reason: 'Previously flagged content',
                    scores: { overall: 0.0 }
                };
            }
            
            const analysis = await this.analyzeImage(imagePath, imageBuffer);
            const sourceAnalysis = sourceUrl ? await this.analyzeSource(sourceUrl) : { score: 1.0, reasoning: [] };
            
            const overallScore = this.calculateOverallScore(analysis, sourceAnalysis);
            const action = this.determineModerationAction(overallScore, analysis);
            
            await this.storeModeration(contentHash, 'image', imagePath, sourceUrl, analysis, overallScore, action);
            
            return {
                contentHash,
                action,
                scores: {
                    nsfw: analysis.nsfw,
                    inappropriate: analysis.inappropriate,
                    overall: overallScore
                },
                reasoning: analysis.reasoning,
                warnings: analysis.warnings
            };
            
        } catch (error) {
            logger.error('Image moderation error:', error);
            return {
                contentHash: null,
                action: 'error',
                reason: 'Failed to process image',
                scores: { overall: 0.0 }
            };
        }
    }
    
    async moderateVideo(videoPath, sourceUrl = null) {
        try {
            // Extract keyframes and audio for analysis
            const keyframes = await this.extractKeyframes(videoPath);
            const audioPath = await this.extractAudio(videoPath);
            
            const videoBuffer = await fs.readFile(videoPath);
            const contentHash = this.generateContentHash(videoBuffer);
            
            if (this.blockedHashes.has(contentHash)) {
                return {
                    contentHash,
                    action: 'block',
                    reason: 'Previously flagged content',
                    scores: { overall: 0.0 }
                };
            }
            
            // Analyze keyframes
            const frameAnalyses = await Promise.all(
                keyframes.map(frame => this.analyzeImage(frame))
            );
            
            // Analyze audio (placeholder - would need speech-to-text)
            const audioAnalysis = { nsfw: 0.1, inappropriate: 0.1, reasoning: ['Audio analysis placeholder'] };
            
            // Combine analyses
            const videoAnalysis = this.combineVideoAnalyses(frameAnalyses, audioAnalysis);
            const sourceAnalysis = sourceUrl ? await this.analyzeSource(sourceUrl) : { score: 1.0, reasoning: [] };
            
            const overallScore = this.calculateOverallScore(videoAnalysis, sourceAnalysis);
            const action = this.determineModerationAction(overallScore, videoAnalysis);
            
            await this.storeModeration(contentHash, 'video', videoPath, sourceUrl, videoAnalysis, overallScore, action);
            
            // Cleanup temporary files
            await this.cleanupTempFiles([audioPath, ...keyframes]);
            
            return {
                contentHash,
                action,
                scores: {
                    nsfw: videoAnalysis.nsfw,
                    inappropriate: videoAnalysis.inappropriate,
                    overall: overallScore
                },
                reasoning: videoAnalysis.reasoning,
                warnings: videoAnalysis.warnings
            };
            
        } catch (error) {
            logger.error('Video moderation error:', error);
            return {
                contentHash: null,
                action: 'error',
                reason: 'Failed to process video',
                scores: { overall: 0.0 }
            };
        }
    }
    
    async analyzeText(text) {
        const reasoning = [];
        const warnings = [];
        
        // NSFW detection
        let nsfwScore = 0;
        const nsfwMatches = this.nsfwKeywords.filter(keyword => 
            text.toLowerCase().includes(keyword.toLowerCase())
        );
        if (nsfwMatches.length > 0) {
            nsfwScore = Math.min(nsfwMatches.length * 0.3, 1.0);
            reasoning.push(`NSFW keywords detected: ${nsfwMatches.join(', ')}`);
        }
        
        // Toxicity detection
        let toxicityScore = 0;
        const harmfulMatches = this.harmfulKeywords.filter(keyword =>
            text.toLowerCase().includes(keyword.toLowerCase())
        );
        if (harmfulMatches.length > 0) {
            toxicityScore = Math.min(harmfulMatches.length * 0.4, 1.0);
            reasoning.push(`Harmful content indicators: ${harmfulMatches.join(', ')}`);
        }
        
        // Spam detection
        const spamScore = this.detectSpam(text);
        if (spamScore > 0.3) {
            reasoning.push(`Potential spam detected (score: ${spamScore.toFixed(3)})`);
        }
        
        // Copyright detection
        let copyrightScore = 0;
        const copyrightMatches = this.copyrightKeywords.filter(keyword =>
            text.toLowerCase().includes(keyword.toLowerCase())
        );
        if (copyrightMatches.length > 0) {
            copyrightScore = Math.min(copyrightMatches.length * 0.2, 0.8);
            reasoning.push(`Copyright indicators: ${copyrightMatches.join(', ')}`);
        }
        
        // Language analysis
        const sentiment = natural.SentimentAnalyzer.getSentiment(['negative', 'words', 'detected']);
        if (sentiment < -0.5) {
            reasoning.push('Negative sentiment detected');
            warnings.push('Content may contain negative sentiment');
        }
        
        return {
            nsfw: nsfwScore,
            toxicity: toxicityScore,
            spam: spamScore,
            copyright: copyrightScore,
            reasoning,
            warnings
        };
    }
    
    async analyzeImage(imagePath, imageBuffer = null) {
        const reasoning = [];
        const warnings = [];
        
        try {
            if (!imageBuffer) {
                imageBuffer = await fs.readFile(imagePath);
            }
            
            // Image metadata analysis
            const metadata = await sharp(imageBuffer).metadata();
            reasoning.push(`Image analysis: ${metadata.width}x${metadata.height}, format: ${metadata.format}`);
            
            // Basic NSFW detection (placeholder - would use ML model in production)
            let nsfwScore = 0.1; // Default low score
            
            // Detect skin tones (basic approximation)
            const skinToneAnalysis = await this.analyzeSkinTones(imageBuffer);
            if (skinToneAnalysis.skinPixelRatio > 0.4) {
                nsfwScore += 0.3;
                reasoning.push(`High skin tone ratio detected: ${(skinToneAnalysis.skinPixelRatio * 100).toFixed(1)}%`);
            }
            
            // Image hash comparison (for known inappropriate content)
            const imageHash = this.generateImageHash(imageBuffer);
            reasoning.push(`Image hash: ${imageHash.substring(0, 16)}...`);
            
            // File size and aspect ratio checks
            if (metadata.width && metadata.height) {
                const aspectRatio = metadata.width / metadata.height;
                if (aspectRatio < 0.5 || aspectRatio > 2.0) {
                    warnings.push('Unusual aspect ratio detected');
                }
            }
            
            return {
                nsfw: nsfwScore,
                inappropriate: nsfwScore * 0.8,
                reasoning,
                warnings,
                metadata: {
                    width: metadata.width,
                    height: metadata.height,
                    format: metadata.format,
                    size: imageBuffer.length
                }
            };
            
        } catch (error) {
            logger.error('Image analysis error:', error);
            return {
                nsfw: 0.5, // Default to moderate risk on error
                inappropriate: 0.5,
                reasoning: [`Image analysis failed: ${error.message}`],
                warnings: ['Unable to complete full image analysis']
            };
        }
    }
    
    async analyzeSkinTones(imageBuffer) {
        try {
            // Simple skin tone detection using color ranges
            const { data, info } = await sharp(imageBuffer)
                .raw()
                .toBuffer({ resolveWithObject: true });
            
            let skinPixels = 0;
            const totalPixels = info.width * info.height;
            
            // Iterate through pixels (R, G, B values)
            for (let i = 0; i < data.length; i += 3) {
                const r = data[i];
                const g = data[i + 1];
                const b = data[i + 2];
                
                // Basic skin tone detection (simplified)
                if (this.isSkinTone(r, g, b)) {
                    skinPixels++;
                }
            }
            
            return {
                skinPixelRatio: skinPixels / totalPixels,
                totalPixels,
                skinPixels
            };
            
        } catch (error) {
            return {
                skinPixelRatio: 0,
                totalPixels: 0,
                skinPixels: 0
            };
        }
    }
    
    isSkinTone(r, g, b) {
        // Simplified skin tone detection
        return (
            r > 95 && g > 40 && b > 20 &&
            r > g && r > b &&
            Math.abs(r - g) > 15 &&
            r - g > 15 &&
            r - b > 15
        );
    }
    
    generateImageHash(imageBuffer) {
        // Simple image hash (would use perceptual hashing in production)
        return crypto.createHash('md5').update(imageBuffer).digest('hex');
    }
    
    detectSpam(text) {
        let spamScore = 0;
        
        // Excessive capitalization
        const upperCaseRatio = (text.match(/[A-Z]/g) || []).length / text.length;
        if (upperCaseRatio > 0.5) {
            spamScore += 0.3;
        }
        
        // Excessive punctuation
        const punctuationRatio = (text.match(/[!?@#$%^&*]/g) || []).length / text.length;
        if (punctuationRatio > 0.1) {
            spamScore += 0.2;
        }
        
        // Repeated characters
        if (/(.)\1{4,}/.test(text)) {
            spamScore += 0.3;
        }
        
        // Common spam phrases
        const spamPhrases = ['click here', 'free money', 'limited time', 'act now', 'guarantee'];
        const spamMatches = spamPhrases.filter(phrase => 
            text.toLowerCase().includes(phrase)
        );
        spamScore += spamMatches.length * 0.2;
        
        return Math.min(spamScore, 1.0);
    }
    
    async analyzeSource(sourceUrl) {
        try {
            const url = new URL(sourceUrl);
            const domain = url.hostname;
            
            if (this.blockedDomains.has(domain)) {
                return {
                    score: 0.0,
                    reasoning: [`Blocked domain: ${domain}`]
                };
            }
            
            if (this.trustedSources.has(domain)) {
                return {
                    score: 1.0,
                    reasoning: [`Trusted source: ${domain}`]
                };
            }
            
            // Basic domain reputation check
            const domainAge = await this.checkDomainAge(domain);
            const isHttps = url.protocol === 'https:';
            
            let score = 0.5; // Neutral score for unknown domains
            const reasoning = [];
            
            if (isHttps) {
                score += 0.2;
                reasoning.push('HTTPS protocol used');
            } else {
                score -= 0.3;
                reasoning.push('Non-HTTPS protocol');
            }
            
            if (domainAge && domainAge > 365) {
                score += 0.2;
                reasoning.push(`Domain age: ${domainAge} days`);
            } else {
                score -= 0.1;
                reasoning.push('New or unknown domain');
            }
            
            return {
                score: Math.max(0, Math.min(1, score)),
                reasoning
            };
            
        } catch (error) {
            return {
                score: 0.3, // Low score for invalid URLs
                reasoning: [`Invalid source URL: ${error.message}`]
            };
        }
    }
    
    async checkDomainAge(domain) {
        // Placeholder for domain age check
        // In production, this would use a domain reputation API
        return Math.random() * 1000; // Random age between 0-1000 days
    }
    
    calculateOverallScore(analysis, sourceAnalysis) {
        // Weighted scoring
        const weights = {
            nsfw: 0.3,
            toxicity: 0.25,
            spam: 0.15,
            copyright: 0.2,
            source: 0.1
        };
        
        let score = 1.0;
        
        // Subtract risk scores
        score -= (analysis.nsfw || 0) * weights.nsfw;
        score -= (analysis.toxicity || 0) * weights.toxicity;
        score -= (analysis.spam || 0) * weights.spam;
        score -= (analysis.copyright || 0) * weights.copyright;
        score -= (1.0 - sourceAnalysis.score) * weights.source;
        
        return Math.max(0, Math.min(1, score));
    }
    
    determineModerationAction(overallScore, analysis) {
        if (overallScore < 0.2) {
            return 'block';
        } else if (overallScore < 0.4) {
            return 'restrict';
        } else if (overallScore < 0.7) {
            return 'warn';
        } else {
            return 'allow';
        }
    }
    
    async storeModeration(contentHash, contentType, filePath, sourceUrl, analysis, overallScore, action) {
        return new Promise((resolve, reject) => {
            const reasoning = JSON.stringify({
                analysis,
                timestamp: new Date().toISOString()
            });
            
            this.db.run(`
                INSERT OR REPLACE INTO moderated_content 
                (content_hash, content_type, file_path, source_url, 
                 nsfw_score, toxicity_score, spam_score, copyright_score, 
                 overall_score, moderation_action, reasoning, moderator_id)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            `, [
                contentHash, contentType, filePath, sourceUrl,
                analysis.nsfw || 0, analysis.toxicity || 0, 
                analysis.spam || 0, analysis.copyright || 0,
                overallScore, action, reasoning, 'ai-moderator'
            ], (err) => {
                if (err) {
                    logger.error('Store moderation error:', err);
                    reject(err);
                } else {
                    resolve();
                }
            });
        });
    }
    
    async extractKeyframes(videoPath, count = 5) {
        return new Promise((resolve, reject) => {
            const keyframes = [];
            const tempDir = path.join(__dirname, 'temp');
            
            // Create temp directory if it doesn't exist
            fs.mkdir(tempDir, { recursive: true }).then(() => {
                ffmpeg(videoPath)
                    .screenshots({
                        count,
                        folder: tempDir,
                        filename: `frame-%i-${Date.now()}.png`
                    })
                    .on('end', async () => {
                        try {
                            const files = await fs.readdir(tempDir);
                            const frameFiles = files
                                .filter(file => file.startsWith('frame-'))
                                .map(file => path.join(tempDir, file));
                            
                            resolve(frameFiles);
                        } catch (error) {
                            reject(error);
                        }
                    })
                    .on('error', reject);
            });
        });
    }
    
    async extractAudio(videoPath) {
        return new Promise((resolve, reject) => {
            const audioPath = path.join(__dirname, 'temp', `audio-${Date.now()}.wav`);
            
            ffmpeg(videoPath)
                .output(audioPath)
                .audioCodec('pcm_s16le')
                .on('end', () => resolve(audioPath))
                .on('error', reject)
                .run();
        });
    }
    
    combineVideoAnalyses(frameAnalyses, audioAnalysis) {
        const reasoning = ['Video analysis combining keyframes and audio'];
        const warnings = [];
        
        // Average frame analysis scores
        const avgNsfw = frameAnalyses.reduce((sum, frame) => sum + frame.nsfw, 0) / frameAnalyses.length;
        const avgInappropriate = frameAnalyses.reduce((sum, frame) => sum + frame.inappropriate, 0) / frameAnalyses.length;
        
        // Combine with audio analysis
        const combinedNsfw = (avgNsfw + audioAnalysis.nsfw) / 2;
        const combinedInappropriate = (avgInappropriate + audioAnalysis.inappropriate) / 2;
        
        reasoning.push(`Analyzed ${frameAnalyses.length} keyframes`);
        reasoning.push(`Average NSFW score: ${avgNsfw.toFixed(3)}`);
        reasoning.push(`Combined with audio analysis`);
        
        // Collect all reasoning from frame analyses
        frameAnalyses.forEach((frame, index) => {
            if (frame.warnings && frame.warnings.length > 0) {
                warnings.push(`Frame ${index + 1}: ${frame.warnings.join(', ')}`);
            }
        });
        
        return {
            nsfw: combinedNsfw,
            inappropriate: combinedInappropriate,
            reasoning,
            warnings
        };
    }
    
    async cleanupTempFiles(filePaths) {
        for (const filePath of filePaths) {
            try {
                await fs.unlink(filePath);
            } catch (error) {
                logger.warn(`Failed to cleanup temp file ${filePath}:`, error);
            }
        }
    }
    
    async getModerationStats() {
        return new Promise((resolve, reject) => {
            this.db.all(`
                SELECT 
                    moderation_action,
                    content_type,
                    COUNT(*) as count,
                    AVG(overall_score) as avg_score
                FROM moderated_content 
                GROUP BY moderation_action, content_type
                ORDER BY count DESC
            `, (err, rows) => {
                if (err) {
                    reject(err);
                } else {
                    resolve(rows);
                }
            });
        });
    }
    
    async reportContent(contentHash, userId, reason, details) {
        return new Promise((resolve, reject) => {
            this.db.run(`
                INSERT INTO user_reports 
                (content_hash, user_id, report_reason, report_details)
                VALUES (?, ?, ?, ?)
            `, [contentHash, userId, reason, details], (err) => {
                if (err) {
                    reject(err);
                } else {
                    resolve();
                }
            });
        });
    }
    
    async blockContent(contentHash, reason, moderatorId) {
        return new Promise((resolve, reject) => {
            this.db.serialize(() => {
                this.db.run(`
                    INSERT OR IGNORE INTO blocked_content 
                    (content_hash, block_reason, blocked_by)
                    VALUES (?, ?, ?)
                `, [contentHash, reason, moderatorId]);
                
                this.db.run(`
                    UPDATE moderated_content 
                    SET moderation_action = 'block', updated_at = CURRENT_TIMESTAMP
                    WHERE content_hash = ?
                `, [contentHash], (err) => {
                    if (err) {
                        reject(err);
                    } else {
                        this.blockedHashes.add(contentHash);
                        resolve();
                    }
                });
            });
        });
    }
}

// Express.js API Server
class ModerationAPI {
    constructor() {
        this.app = express();
        this.moderator = new ContentModerator();
        this.wss = new WebSocket.Server({ port: 8081 });
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
    }
    
    setupMiddleware() {
        // Security middleware
        this.app.use(helmet());
        this.app.use(cors({
            origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:3000'],
            credentials: true
        }));
        
        // Rate limiting
        const limiter = rateLimit({
            windowMs: 15 * 60 * 1000, // 15 minutes
            max: 100, // limit each IP to 100 requests per windowMs
            message: 'Too many requests from this IP'
        });
        this.app.use(limiter);
        
        // Body parsing
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.urlencoded({ extended: true }));
        
        // File upload
        const upload = multer({
            dest: 'uploads/',
            limits: {
                fileSize: 100 * 1024 * 1024 // 100MB limit
            }
        });
        this.app.use('/api/moderate', upload.single('file'));
        
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
        
        // Moderate text content
        this.app.post('/api/moderate/text', 
            body('text').isLength({ min: 1, max: 10000 }),
            body('sourceUrl').optional().isURL(),
            this.validateRequest,
            async (req, res) => {
                try {
                    const { text, sourceUrl } = req.body;
                    const result = await this.moderator.moderateText(text, sourceUrl);
                    
                    this.broadcastModeration(result);
                    res.json(result);
                } catch (error) {
                    logger.error('Text moderation error:', error);
                    res.status(500).json({ error: 'Moderation failed' });
                }
            }
        );
        
        // Moderate file content
        this.app.post('/api/moderate/file',
            this.validateRequest,
            async (req, res) => {
                try {
                    if (!req.file) {
                        return res.status(400).json({ error: 'No file provided' });
                    }
                    
                    const { sourceUrl } = req.body;
                    const filePath = req.file.path;
                    const mimeType = req.file.mimetype;
                    
                    let result;
                    if (mimeType.startsWith('image/')) {
                        result = await this.moderator.moderateImage(filePath, sourceUrl);
                    } else if (mimeType.startsWith('video/')) {
                        result = await this.moderator.moderateVideo(filePath, sourceUrl);
                    } else {
                        result = { error: 'Unsupported file type' };
                    }
                    
                    // Cleanup uploaded file
                    await fs.unlink(filePath).catch(err => 
                        logger.warn('File cleanup error:', err)
                    );
                    
                    this.broadcastModeration(result);
                    res.json(result);
                } catch (error) {
                    logger.error('File moderation error:', error);
                    res.status(500).json({ error: 'Moderation failed' });
                }
            }
        );
        
        // Get moderation statistics
        this.app.get('/api/stats', async (req, res) => {
            try {
                const stats = await this.moderator.getModerationStats();
                res.json(stats);
            } catch (error) {
                logger.error('Stats error:', error);
                res.status(500).json({ error: 'Failed to get statistics' });
            }
        });
        
        // Report content
        this.app.post('/api/report',
            body('contentHash').isLength({ min: 1, max: 64 }),
            body('userId').isLength({ min: 1, max: 50 }),
            body('reason').isIn(['spam', 'inappropriate', 'copyright', 'other']),
            body('details').isLength({ max: 1000 }),
            this.validateRequest,
            async (req, res) => {
                try {
                    const { contentHash, userId, reason, details } = req.body;
                    await this.moderator.reportContent(contentHash, userId, reason, details);
                    res.json({ success: true });
                } catch (error) {
                    logger.error('Report error:', error);
                    res.status(500).json({ error: 'Failed to report content' });
                }
            }
        );
        
        // Block content (admin endpoint)
        this.app.post('/api/admin/block',
            body('contentHash').isLength({ min: 1, max: 64 }),
            body('reason').isLength({ min: 1, max: 500 }),
            // Add authentication middleware here
            this.validateRequest,
            async (req, res) => {
                try {
                    const { contentHash, reason } = req.body;
                    const moderatorId = req.headers['x-moderator-id'] || 'admin';
                    
                    await this.moderator.blockContent(contentHash, reason, moderatorId);
                    res.json({ success: true });
                } catch (error) {
                    logger.error('Block content error:', error);
                    res.status(500).json({ error: 'Failed to block content' });
                }
            }
        );
    }
    
    setupWebSocket() {
        this.wss.on('connection', (ws) => {
            logger.info('WebSocket client connected');
            
            ws.on('message', (message) => {
                try {
                    const data = JSON.parse(message);
                    // Handle real-time moderation requests
                    if (data.type === 'moderate') {
                        this.handleRealtimeModeration(ws, data);
                    }
                } catch (error) {
                    ws.send(JSON.stringify({ error: 'Invalid message format' }));
                }
            });
            
            ws.on('close', () => {
                logger.info('WebSocket client disconnected');
            });
        });
    }
    
    async handleRealtimeModeration(ws, data) {
        try {
            const { text, type } = data;
            let result;
            
            if (type === 'text') {
                result = await this.moderator.moderateText(text);
            } else {
                result = { error: 'Unsupported content type for real-time moderation' };
            }
            
            ws.send(JSON.stringify({
                type: 'moderation_result',
                result
            }));
        } catch (error) {
            ws.send(JSON.stringify({
                type: 'error',
                message: 'Real-time moderation failed'
            }));
        }
    }
    
    broadcastModeration(result) {
        const message = JSON.stringify({
            type: 'moderation_broadcast',
            result: {
                action: result.action,
                contentHash: result.contentHash,
                timestamp: new Date().toISOString()
            }
        });
        
        this.wss.clients.forEach((client) => {
            if (client.readyState === WebSocket.OPEN) {
                client.send(message);
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
    
    start(port = 8080) {
        this.app.listen(port, () => {
            logger.info(`Content Moderation API started on port ${port}`);
            logger.info(`WebSocket server running on port 8081`);
        });
    }
}

// Start the moderation API server
if (require.main === module) {
    const api = new ModerationAPI();
    api.start(process.env.PORT || 8080);
}

module.exports = { ContentModerator, ModerationAPI };