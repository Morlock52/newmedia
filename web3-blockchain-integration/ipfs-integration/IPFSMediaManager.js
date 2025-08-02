/**
 * IPFS Media Manager
 * Handles distributed content storage, retrieval, and streaming for the Web3 media platform
 * Integrates with existing Jellyfin server and provides decentralized content distribution
 */

const IPFS = require('ipfs-core');
const { create } = require('ipfs-http-client');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const EventEmitter = require('events');

class IPFSMediaManager extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            // IPFS node configuration
            ipfsNode: null,
            ipfsClient: null,
            
            // Storage configuration
            contentPath: config.contentPath || './ipfs-content',
            metadataPath: config.metadataPath || './ipfs-metadata',
            pinningServices: config.pinningServices || [],
            
            // Performance settings
            chunksSize: config.chunksSize || 8 * 1024 * 1024, // 8MB chunks
            concurrentUploads: config.concurrentUploads || 3,
            preloadCache: config.preloadCache || true,
            
            // Integration settings
            jellyfinApi: config.jellyfinApi || null,
            web3Provider: config.web3Provider || null,
            contractAddress: config.contractAddress || null,
            
            // Security settings
            encryption: config.encryption || true,
            accessControl: config.accessControl || true,
            
            // Network settings
            swarmAddresses: config.swarmAddresses || [
                '/ip4/104.131.131.82/tcp/4001/p2p/QmaCpDMGvV2BGHeYERUEnRQAwe3N8SzbUtfsmvsqQLuvuJ',
                '/ip4/104.236.179.241/tcp/4001/p2p/QmSoLPppuBtQSGwKDZT2M73ULpjvfd3aZ6ha4oFGL1KrGM'
            ]
        };
        
        // Content registry
        this.contentRegistry = new Map();
        this.pinningQueue = [];
        this.retrievalCache = new Map();
        
        // Performance tracking
        this.metrics = {
            uploadsCompleted: 0,
            downloadsCompleted: 0,
            totalStorageUsed: 0,
            averageUploadTime: 0,
            averageDownloadTime: 0
        };
    }
    
    /**
     * Initialize IPFS node and connect to network
     */
    async initialize() {
        try {
            console.log('Initializing IPFS Media Manager...');
            
            // Create IPFS node
            this.config.ipfsNode = await IPFS.create({
                repo: './ipfs-repo',
                config: {
                    Addresses: {
                        Swarm: [
                            '/ip4/0.0.0.0/tcp/4001',
                            '/ip4/127.0.0.1/tcp/4002/ws'
                        ],
                        API: '/ip4/127.0.0.1/tcp/5001',
                        Gateway: '/ip4/127.0.0.1/tcp/8080'
                    },
                    Bootstrap: this.config.swarmAddresses
                }
            });
            
            // Create HTTP client for easier API access
            this.config.ipfsClient = create({
                url: 'http://127.0.0.1:5001'
            });
            
            // Ensure directories exist
            await this._ensureDirectories();
            
            // Load existing content registry
            await this._loadContentRegistry();
            
            // Start background services
            this._startPinningService();
            this._startMetricsCollection();
            
            console.log('IPFS Media Manager initialized successfully');
            this.emit('initialized');
            
        } catch (error) {
            console.error('Failed to initialize IPFS Media Manager:', error);
            throw error;
        }
    }
    
    /**
     * Upload media content to IPFS with metadata and encryption
     * @param {string} filePath - Path to the media file
     * @param {Object} metadata - Content metadata
     * @param {Object} options - Upload options
     * @returns {Object} Upload result with IPFS hash and metadata
     */
    async uploadContent(filePath, metadata, options = {}) {
        const startTime = Date.now();
        
        try {
            console.log(`Uploading content: ${filePath}`);
            
            // Validate file
            const fileStats = await fs.stat(filePath);
            if (!fileStats.isFile()) {
                throw new Error('Invalid file path');
            }
            
            // Generate content ID
            const contentId = crypto.randomUUID();
            const fileBuffer = await fs.readFile(filePath);
            
            // Encrypt content if enabled
            let processedContent = fileBuffer;
            let encryptionKey = null;
            
            if (this.config.encryption && options.encrypt !== false) {
                const encryption = this._encryptContent(fileBuffer);
                processedContent = encryption.encrypted;
                encryptionKey = encryption.key;
            }
            
            // Add to IPFS
            const addResult = await this.config.ipfsClient.add(processedContent, {
                chunker: 'size-' + this.config.chunksSize,
                pin: true,
                hashAlg: 'sha2-256'
            });
            
            const ipfsHash = addResult.cid.toString();
            
            // Create comprehensive metadata
            const contentMetadata = {
                contentId,
                ipfsHash,
                originalFilename: path.basename(filePath),
                fileSize: fileStats.size,
                mimeType: this._getMimeType(filePath),
                uploadTimestamp: Date.now(),
                encryptionKey: encryptionKey ? encryptionKey.toString('hex') : null,
                ...metadata,
                
                // Technical metadata
                chunks: Math.ceil(fileStats.size / this.config.chunksSize),
                verification: {
                    hash: crypto.createHash('sha256').update(fileBuffer).digest('hex'),
                    algorithm: 'sha256'
                },
                
                // IPFS specific
                ipfsVersion: await this.config.ipfsNode.version(),
                pinned: true,
                
                // Access control
                accessLevel: options.accessLevel || 'public',
                allowedUsers: options.allowedUsers || [],
                licenseRequired: options.licenseRequired || false
            };
            
            // Store metadata
            await this._storeMetadata(contentId, contentMetadata);
            
            // Register content
            this.contentRegistry.set(contentId, {
                ipfsHash,
                filePath: filePath,
                metadata: contentMetadata,
                uploadTime: startTime,
                accessCount: 0
            });
            
            // Queue for additional pinning services
            if (this.config.pinningServices.length > 0) {
                this.pinningQueue.push({
                    ipfsHash,
                    contentId,
                    priority: options.priority || 'normal'
                });
            }
            
            // Update metrics
            const uploadTime = Date.now() - startTime;
            this._updateUploadMetrics(fileStats.size, uploadTime);
            
            console.log(`Content uploaded successfully: ${ipfsHash}`);
            this.emit('contentUploaded', { contentId, ipfsHash, metadata: contentMetadata });
            
            return {
                success: true,
                contentId,
                ipfsHash,
                metadata: contentMetadata,
                uploadTime
            };
            
        } catch (error) {
            console.error('Upload failed:', error);
            throw error;
        }
    }
    
    /**
     * Retrieve and decrypt content from IPFS
     * @param {string} contentId - Content identifier
     * @param {Object} options - Retrieval options
     * @returns {Buffer} Content buffer
     */
    async retrieveContent(contentId, options = {}) {
        const startTime = Date.now();
        
        try {
            // Get content metadata
            const contentInfo = this.contentRegistry.get(contentId);
            if (!contentInfo) {
                throw new Error('Content not found in registry');
            }
            
            const { ipfsHash, metadata } = contentInfo;
            
            // Check access permissions
            if (this.config.accessControl && !this._checkAccess(contentId, options.user)) {
                throw new Error('Access denied');
            }
            
            // Check cache first
            if (this.retrievalCache.has(ipfsHash)) {
                console.log(`Retrieved from cache: ${ipfsHash}`);
                return this.retrievalCache.get(ipfsHash);
            }
            
            console.log(`Retrieving content from IPFS: ${ipfsHash}`);
            
            // Retrieve from IPFS
            const chunks = [];
            for await (const chunk of this.config.ipfsClient.cat(ipfsHash)) {
                chunks.push(chunk);
            }
            
            let content = Buffer.concat(chunks);
            
            // Decrypt if necessary
            if (metadata.encryptionKey) {
                const decryptionKey = Buffer.from(metadata.encryptionKey, 'hex');
                content = this._decryptContent(content, decryptionKey);
            }
            
            // Cache content if enabled
            if (this.config.preloadCache && content.length < 100 * 1024 * 1024) { // Cache files < 100MB
                this.retrievalCache.set(ipfsHash, content);
                
                // Cleanup cache if too large
                if (this.retrievalCache.size > 50) {
                    const firstKey = this.retrievalCache.keys().next().value;
                    this.retrievalCache.delete(firstKey);
                }
            }
            
            // Update access metrics
            contentInfo.accessCount++;
            const retrievalTime = Date.now() - startTime;
            this._updateDownloadMetrics(content.length, retrievalTime);
            
            console.log(`Content retrieved successfully: ${ipfsHash}`);
            this.emit('contentRetrieved', { contentId, ipfsHash, size: content.length });
            
            return content;
            
        } catch (error) {
            console.error('Retrieval failed:', error);
            throw error;
        }
    }
    
    /**
     * Stream content from IPFS with range support
     * @param {string} contentId - Content identifier
     * @param {Object} range - Range specification (start, end)
     * @param {Object} options - Streaming options
     * @returns {AsyncIterable} Content stream
     */
    async *streamContent(contentId, range = {}, options = {}) {
        try {
            const contentInfo = this.contentRegistry.get(contentId);
            if (!contentInfo) {
                throw new Error('Content not found in registry');
            }
            
            const { ipfsHash, metadata } = contentInfo;
            
            // Check access permissions
            if (this.config.accessControl && !this._checkAccess(contentId, options.user)) {
                throw new Error('Access denied');
            }
            
            console.log(`Streaming content: ${ipfsHash}`);
            
            // Stream from IPFS with range support
            const streamOptions = {};
            if (range.start !== undefined) {
                streamOptions.offset = range.start;
            }
            if (range.end !== undefined) {
                streamOptions.length = range.end - (range.start || 0) + 1;
            }
            
            let chunkCount = 0;
            for await (const chunk of this.config.ipfsClient.cat(ipfsHash, streamOptions)) {
                // Decrypt chunk if necessary
                let processedChunk = chunk;
                if (metadata.encryptionKey && chunkCount === 0) {
                    // For streaming, we need to handle encryption differently
                    // This is a simplified approach - production would need proper streaming decryption
                    const decryptionKey = Buffer.from(metadata.encryptionKey, 'hex');
                    processedChunk = this._decryptContent(chunk, decryptionKey);
                }
                
                yield processedChunk;
                chunkCount++;
            }
            
            this.emit('contentStreamed', { contentId, ipfsHash, range });
            
        } catch (error) {
            console.error('Streaming failed:', error);
            throw error;
        }
    }
    
    /**
     * Get content metadata
     * @param {string} contentId - Content identifier
     * @returns {Object} Content metadata
     */
    async getContentMetadata(contentId) {
        const contentInfo = this.contentRegistry.get(contentId);
        if (!contentInfo) {
            throw new Error('Content not found');
        }
        
        return contentInfo.metadata;
    }
    
    /**
     * List all content in registry
     * @param {Object} filters - Filter options
     * @returns {Array} Array of content info
     */
    async listContent(filters = {}) {
        const results = [];
        
        for (const [contentId, contentInfo] of this.contentRegistry) {
            let include = true;
            
            // Apply filters
            if (filters.contentType && contentInfo.metadata.contentType !== filters.contentType) {
                include = false;
            }
            
            if (filters.uploadedAfter && contentInfo.metadata.uploadTimestamp < filters.uploadedAfter) {
                include = false;
            }
            
            if (filters.accessLevel && contentInfo.metadata.accessLevel !== filters.accessLevel) {
                include = false;
            }
            
            if (include) {
                results.push({
                    contentId,
                    ipfsHash: contentInfo.ipfsHash,
                    metadata: contentInfo.metadata,
                    accessCount: contentInfo.accessCount
                });
            }
        }
        
        return results;
    }
    
    /**
     * Pin content to additional IPFS nodes or pinning services
     * @param {string} contentId - Content identifier
     * @param {Array} services - Pinning services to use
     */
    async pinContent(contentId, services = []) {
        const contentInfo = this.contentRegistry.get(contentId);
        if (!contentInfo) {
            throw new Error('Content not found');
        }
        
        const { ipfsHash } = contentInfo;
        
        // Pin to local node
        await this.config.ipfsClient.pin.add(ipfsHash);
        
        // Pin to external services
        for (const service of services) {
            try {
                await this._pinToService(ipfsHash, service);
                console.log(`Content pinned to ${service.name}: ${ipfsHash}`);
            } catch (error) {
                console.error(`Failed to pin to ${service.name}:`, error);
            }
        }
        
        this.emit('contentPinned', { contentId, ipfsHash, services });
    }
    
    /**
     * Encrypt content buffer
     * @param {Buffer} content - Content to encrypt
     * @returns {Object} Encrypted content and key
     */
    _encryptContent(content) {
        const algorithm = 'aes-256-gcm';
        const key = crypto.randomBytes(32);
        const iv = crypto.randomBytes(16);
        
        const cipher = crypto.createCipher(algorithm, key);
        cipher.setAAD(Buffer.from('ipfs-media-content'));
        
        const encrypted = Buffer.concat([
            cipher.update(content),
            cipher.final(),
            cipher.getAuthTag()
        ]);
        
        return {
            encrypted: Buffer.concat([iv, encrypted]),
            key
        };
    }
    
    /**
     * Decrypt content buffer
     * @param {Buffer} encryptedContent - Encrypted content
     * @param {Buffer} key - Decryption key
     * @returns {Buffer} Decrypted content
     */
    _decryptContent(encryptedContent, key) {
        const algorithm = 'aes-256-gcm';
        const iv = encryptedContent.slice(0, 16);
        const authTag = encryptedContent.slice(-16);
        const encrypted = encryptedContent.slice(16, -16);
        
        const decipher = crypto.createDecipher(algorithm, key);
        decipher.setAAD(Buffer.from('ipfs-media-content'));
        decipher.setAuthTag(authTag);
        
        return Buffer.concat([
            decipher.update(encrypted),
            decipher.final()
        ]);
    }
    
    /**
     * Check access permissions for content
     * @param {string} contentId - Content identifier
     * @param {string} user - User address or identifier
     * @returns {boolean} Access allowed
     */
    _checkAccess(contentId, user) {
        const contentInfo = this.contentRegistry.get(contentId);
        if (!contentInfo) return false;
        
        const { metadata } = contentInfo;
        
        // Public content
        if (metadata.accessLevel === 'public') return true;
        
        // Private content - check allowed users
        if (metadata.accessLevel === 'private') {
            return metadata.allowedUsers.includes(user);
        }
        
        // License required - check with smart contract
        if (metadata.licenseRequired && this.config.web3Provider) {
            // This would check the blockchain for valid license
            return this._checkLicenseOnChain(contentId, user);
        }
        
        return false;
    }
    
    /**
     * Get MIME type from file extension
     * @param {string} filePath - File path
     * @returns {string} MIME type
     */
    _getMimeType(filePath) {
        const ext = path.extname(filePath).toLowerCase();
        const mimeTypes = {
            '.mp4': 'video/mp4',
            '.mkv': 'video/x-matroska',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mp3': 'audio/mpeg',
            '.flac': 'audio/flac',
            '.wav': 'audio/wav',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.pdf': 'application/pdf',
            '.epub': 'application/epub+zip'
        };
        
        return mimeTypes[ext] || 'application/octet-stream';
    }
    
    /**
     * Ensure required directories exist
     */
    async _ensureDirectories() {
        await fs.mkdir(this.config.contentPath, { recursive: true });
        await fs.mkdir(this.config.metadataPath, { recursive: true });
    }
    
    /**
     * Store content metadata to filesystem
     * @param {string} contentId - Content identifier
     * @param {Object} metadata - Metadata object
     */
    async _storeMetadata(contentId, metadata) {
        const metadataPath = path.join(this.config.metadataPath, `${contentId}.json`);
        await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));
    }
    
    /**
     * Load existing content registry from metadata files
     */
    async _loadContentRegistry() {
        try {
            const files = await fs.readdir(this.config.metadataPath);
            
            for (const file of files) {
                if (file.endsWith('.json')) {
                    const contentId = path.basename(file, '.json');
                    const metadataPath = path.join(this.config.metadataPath, file);
                    const metadata = JSON.parse(await fs.readFile(metadataPath, 'utf8'));
                    
                    this.contentRegistry.set(contentId, {
                        ipfsHash: metadata.ipfsHash,
                        filePath: metadata.originalFilename,
                        metadata,
                        uploadTime: metadata.uploadTimestamp,
                        accessCount: 0
                    });
                }
            }
            
            console.log(`Loaded ${this.contentRegistry.size} content entries from registry`);
            
        } catch (error) {
            console.error('Failed to load content registry:', error);
        }
    }
    
    /**
     * Start background pinning service
     */
    _startPinningService() {
        setInterval(async () => {
            if (this.pinningQueue.length > 0) {
                const item = this.pinningQueue.shift();
                try {
                    await this.pinContent(item.contentId, this.config.pinningServices);
                } catch (error) {
                    console.error('Background pinning failed:', error);
                }
            }
        }, 30000); // Every 30 seconds
    }
    
    /**
     * Start metrics collection
     */
    _startMetricsCollection() {
        setInterval(() => {
            this.emit('metrics', this.metrics);
        }, 60000); // Every minute
    }
    
    /**
     * Update upload metrics
     * @param {number} fileSize - Size of uploaded file
     * @param {number} uploadTime - Time taken to upload
     */
    _updateUploadMetrics(fileSize, uploadTime) {
        this.metrics.uploadsCompleted++;
        this.metrics.totalStorageUsed += fileSize;
        this.metrics.averageUploadTime = 
            (this.metrics.averageUploadTime * (this.metrics.uploadsCompleted - 1) + uploadTime) / 
            this.metrics.uploadsCompleted;
    }
    
    /**
     * Update download metrics
     * @param {number} fileSize - Size of downloaded file
     * @param {number} downloadTime - Time taken to download
     */
    _updateDownloadMetrics(fileSize, downloadTime) {
        this.metrics.downloadsCompleted++;
        this.metrics.averageDownloadTime = 
            (this.metrics.averageDownloadTime * (this.metrics.downloadsCompleted - 1) + downloadTime) / 
            this.metrics.downloadsCompleted;
    }
    
    /**
     * Get current performance metrics
     * @returns {Object} Performance metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            contentCount: this.contentRegistry.size,
            cacheSize: this.retrievalCache.size,
            pinningQueueSize: this.pinningQueue.length
        };
    }
    
    /**
     * Cleanup resources and shutdown
     */
    async shutdown() {
        console.log('Shutting down IPFS Media Manager...');
        
        // Stop IPFS node
        if (this.config.ipfsNode) {
            await this.config.ipfsNode.stop();
        }
        
        // Clear caches
        this.retrievalCache.clear();
        this.pinningQueue.length = 0;
        
        console.log('IPFS Media Manager shutdown complete');
        this.emit('shutdown');
    }
}

module.exports = IPFSMediaManager;