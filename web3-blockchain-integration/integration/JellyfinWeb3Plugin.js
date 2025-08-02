/**
 * Jellyfin Web3 Integration Plugin
 * Integrates blockchain features with existing Jellyfin media server
 * Provides NFT-based content access, licensing verification, and Web3 authentication
 */

const { EventEmitter } = require('events');
const { ethers } = require('ethers');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

class JellyfinWeb3Plugin extends EventEmitter {
    constructor(config = {}) {
        super();
        
        this.config = {
            // Jellyfin server configuration
            jellyfinUrl: config.jellyfinUrl || 'http://localhost:8096',
            jellyfinApiKey: config.jellyfinApiKey || '',
            
            // Web3 configuration
            web3Provider: config.web3Provider || 'https://rpc.ankr.com/eth',
            contractAddresses: {
                contentOwnership: config.contentOwnershipAddress || '',
                mediaDAO: config.mediaDAOAddress || '',
                marketplace: config.marketplaceAddress || ''
            },
            
            // IPFS configuration
            ipfsApiUrl: config.ipfsApiUrl || 'http://localhost:5001',
            ipfsGateway: config.ipfsGateway || 'http://localhost:8080',
            
            // Plugin settings
            enableNFTAccess: config.enableNFTAccess || true,
            enableLicenseVerification: config.enableLicenseVerification || true,
            enableWeb3Auth: config.enableWeb3Auth || true,
            enableIPFSContent: config.enableIPFSContent || true,
            
            // Cache settings
            licenseCache: new Map(),
            licenseCacheTTL: config.licenseCacheTTL || 5 * 60 * 1000, // 5 minutes
            
            // Database path for plugin data
            pluginDataPath: config.pluginDataPath || './jellyfin-web3-data'
        };
        
        // Initialize providers and contracts
        this.provider = null;
        this.contracts = {};
        this.jellyfinApi = null;
        
        // Plugin state
        this.isInitialized = false;
        this.contentMappings = new Map(); // Map Jellyfin items to blockchain content
        this.userSessions = new Map(); // Track Web3 authenticated users
        
        // Performance metrics
        this.metrics = {
            licenseVerifications: 0,
            nftAccessGrants: 0,
            ipfsContentServed: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
    }
    
    /**
     * Initialize the Web3 plugin
     */
    async initialize() {
        try {
            console.log('Initializing Jellyfin Web3 Plugin...');
            
            // Initialize Web3 provider
            this.provider = new ethers.JsonRpcProvider(this.config.web3Provider);
            
            // Initialize smart contracts
            await this._initializeContracts();
            
            // Initialize Jellyfin API client
            this._initializeJellyfinApi();
            
            // Ensure plugin data directory exists
            await fs.mkdir(this.config.pluginDataPath, { recursive: true });
            
            // Load existing content mappings
            await this._loadContentMappings();
            
            // Start background services
            this._startLicenseCacheCleanup();
            this._startContentSync();
            this._startMetricsReporting();
            
            this.isInitialized = true;
            console.log('Jellyfin Web3 Plugin initialized successfully');
            this.emit('initialized');
            
        } catch (error) {
            console.error('Failed to initialize Jellyfin Web3 Plugin:', error);
            throw error;
        }
    }
    
    /**
     * Verify user access to content based on NFT ownership or license
     * @param {string} userId - Jellyfin user ID
     * @param {string} itemId - Jellyfin item ID
     * @param {string} walletAddress - User's Web3 wallet address
     * @returns {Object} Access verification result
     */
    async verifyContentAccess(userId, itemId, walletAddress) {
        try {
            // Get content mapping
            const contentMapping = this.contentMappings.get(itemId);
            if (!contentMapping) {
                // No Web3 restrictions, allow normal access
                return { 
                    allowed: true, 
                    type: 'standard',
                    message: 'Standard Jellyfin access'
                };
            }
            
            const { tokenId, requiresLicense, requiresNFT } = contentMapping;
            
            // Check NFT ownership
            if (requiresNFT) {
                const isOwner = await this._checkNFTOwnership(tokenId, walletAddress);
                if (isOwner) {
                    this.metrics.nftAccessGrants++;
                    return {
                        allowed: true,
                        type: 'nft_owner',
                        message: 'Access granted as NFT owner'
                    };
                }
            }
            
            // Check license
            if (requiresLicense) {
                const hasLicense = await this._checkLicense(tokenId, walletAddress);
                if (hasLicense) {
                    this.metrics.licenseVerifications++;
                    return {
                        allowed: true,
                        type: 'licensed',
                        message: 'Access granted via license'
                    };
                }
            }
            
            // Access denied
            return {
                allowed: false,
                type: 'restricted',
                message: 'Web3 access verification failed',
                requirements: {
                    nft: requiresNFT,
                    license: requiresLicense,
                    tokenId: tokenId
                }
            };
            
        } catch (error) {
            console.error('Access verification failed:', error);
            return {
                allowed: false,
                type: 'error',
                message: 'Verification error occurred'
            };
        }
    }
    
    /**
     * Link Jellyfin content item to blockchain NFT
     * @param {string} itemId - Jellyfin item ID
     * @param {string} tokenId - NFT token ID
     * @param {Object} options - Linking options
     */
    async linkContentToNFT(itemId, tokenId, options = {}) {
        try {
            // Get NFT metadata from blockchain
            const nftMetadata = await this.contracts.contentOwnership.getContentMetadata(tokenId);
            
            // Get Jellyfin item details
            const jellyfinItem = await this._getJellyfinItem(itemId);
            
            // Create content mapping
            const contentMapping = {
                itemId,
                tokenId,
                ipfsHash: nftMetadata.ipfsHash,
                requiresLicense: options.requiresLicense || nftMetadata.isLicensable,
                requiresNFT: options.requiresNFT || false,
                createdAt: Date.now(),
                jellyfinPath: jellyfinItem.Path,
                originalFilename: jellyfinItem.Name,
                contentType: nftMetadata.contentType,
                creator: nftMetadata.creator
            };
            
            // Store mapping
            this.contentMappings.set(itemId, contentMapping);
            await this._saveContentMappings();
            
            console.log(`Linked content ${itemId} to NFT ${tokenId}`);
            this.emit('contentLinked', { itemId, tokenId, mapping: contentMapping });
            
            return contentMapping;
            
        } catch (error) {
            console.error('Failed to link content to NFT:', error);
            throw error;
        }
    }
    
    /**
     * Serve IPFS content through Jellyfin
     * @param {string} itemId - Jellyfin item ID
     * @param {string} walletAddress - User wallet address
     * @param {Object} requestOptions - Request options (range, quality, etc.)
     * @returns {Stream} Content stream
     */
    async serveIPFSContent(itemId, walletAddress, requestOptions = {}) {
        try {
            const contentMapping = this.contentMappings.get(itemId);
            if (!contentMapping) {
                throw new Error('Content not found');
            }
            
            // Verify access
            const accessResult = await this.verifyContentAccess(null, itemId, walletAddress);
            if (!accessResult.allowed) {
                throw new Error('Access denied');
            }
            
            const { ipfsHash } = contentMapping;
            
            // Construct IPFS URL
            let ipfsUrl = `${this.config.ipfsGateway}/ipfs/${ipfsHash}`;
            
            // Add range header if specified
            const headers = {};
            if (requestOptions.range) {
                headers['Range'] = requestOptions.range;
            }
            
            // Stream content from IPFS
            const response = await axios({
                method: 'GET',
                url: ipfsUrl,
                headers,
                responseType: 'stream'
            });
            
            this.metrics.ipfsContentServed++;
            this.emit('ipfsContentServed', { itemId, ipfsHash, walletAddress });
            
            return response.data;
            
        } catch (error) {
            console.error('Failed to serve IPFS content:', error);
            throw error;
        }
    }
    
    /**
     * Authenticate user with Web3 wallet signature
     * @param {string} walletAddress - User's wallet address
     * @param {string} signature - Signed message
     * @param {string} message - Original message
     * @returns {Object} Authentication result
     */
    async authenticateWeb3User(walletAddress, signature, message) {
        try {
            // Verify signature
            const expectedMessage = `Login to Jellyfin Media Server\nTimestamp: ${message}`;
            const recoveredAddress = ethers.verifyMessage(expectedMessage, signature);
            
            if (recoveredAddress.toLowerCase() !== walletAddress.toLowerCase()) {
                throw new Error('Invalid signature');
            }
            
            // Check if timestamp is recent (within 5 minutes)
            const timestamp = parseInt(message);
            const now = Date.now();
            if (now - timestamp > 5 * 60 * 1000) {
                throw new Error('Message too old');
            }
            
            // Create session
            const sessionId = ethers.id(`${walletAddress}-${timestamp}`);
            const session = {
                walletAddress,
                authenticated: true,
                authTime: now,
                sessionId
            };
            
            this.userSessions.set(sessionId, session);
            
            // Try to find or create Jellyfin user
            const jellyfinUser = await this._findOrCreateJellyfinUser(walletAddress);
            
            console.log(`Web3 user authenticated: ${walletAddress}`);
            this.emit('userAuthenticated', { walletAddress, sessionId, jellyfinUser });
            
            return {
                success: true,
                sessionId,
                jellyfinUser,
                message: 'Authentication successful'
            };
            
        } catch (error) {
            console.error('Web3 authentication failed:', error);
            return {
                success: false,
                message: error.message
            };
        }
    }
    
    /**
     * Get user's Web3 content library
     * @param {string} walletAddress - User's wallet address
     * @returns {Array} Array of accessible content
     */
    async getUserWeb3Library(walletAddress) {
        try {
            const library = [];
            
            // Get user's owned NFTs
            const ownedTokens = await this.contracts.contentOwnership.getCreatorContent(walletAddress);
            
            // Get user's licensed content
            const licensedContent = await this._getUserLicensedContent(walletAddress);
            
            // Process owned NFTs
            for (const tokenId of ownedTokens) {
                const metadata = await this.contracts.contentOwnership.getContentMetadata(tokenId);
                const jellyfinMapping = this._findJellyfinMapping(tokenId.toString());
                
                library.push({
                    tokenId: tokenId.toString(),
                    type: 'owned',
                    title: metadata.title,
                    description: metadata.description,
                    contentType: metadata.contentType,
                    ipfsHash: metadata.ipfsHash,
                    jellyfinItem: jellyfinMapping
                });
            }
            
            // Process licensed content
            for (const license of licensedContent) {
                const metadata = await this.contracts.contentOwnership.getContentMetadata(license.tokenId);
                const jellyfinMapping = this._findJellyfinMapping(license.tokenId);
                
                library.push({
                    tokenId: license.tokenId,
                    type: 'licensed',
                    title: metadata.title,
                    description: metadata.description,
                    contentType: metadata.contentType,
                    ipfsHash: metadata.ipfsHash,
                    license: license,
                    jellyfinItem: jellyfinMapping
                });
            }
            
            return library;
            
        } catch (error) {
            console.error('Failed to get user Web3 library:', error);
            return [];
        }
    }
    
    /**
     * Sync blockchain content with Jellyfin library
     */
    async syncBlockchainContent() {
        try {
            console.log('Syncing blockchain content with Jellyfin...');
            
            // This would typically be called periodically to sync new content
            // Get recent NFT mint events
            const currentBlock = await this.provider.getBlockNumber();
            const fromBlock = currentBlock - 10000; // Last ~2 hours on Ethereum
            
            const filter = this.contracts.contentOwnership.filters.ContentMinted();
            const events = await this.contracts.contentOwnership.queryFilter(
                filter,
                fromBlock,
                currentBlock
            );
            
            let syncedCount = 0;
            
            for (const event of events) {
                const { tokenId, creator, ipfsHash, contentType } = event.args;
                
                // Check if already mapped
                const existingMapping = this._findJellyfinMappingByToken(tokenId.toString());
                if (existingMapping) continue;
                
                // Get NFT metadata
                const metadata = await this.contracts.contentOwnership.getContentMetadata(tokenId);
                
                // Download content from IPFS and add to Jellyfin
                const jellyfinItem = await this._importIPFSToJellyfin(metadata);
                
                if (jellyfinItem) {
                    // Create mapping
                    await this.linkContentToNFT(
                        jellyfinItem.Id,
                        tokenId.toString(),
                        { requiresLicense: metadata.isLicensable }
                    );
                    syncedCount++;
                }
            }
            
            console.log(`Synced ${syncedCount} new blockchain content items`);
            this.emit('contentSynced', { count: syncedCount });
            
        } catch (error) {
            console.error('Content sync failed:', error);
        }
    }
    
    /**
     * Check if user owns specific NFT
     * @param {string} tokenId - NFT token ID
     * @param {string} walletAddress - User wallet address
     * @returns {boolean} Ownership status
     */
    async _checkNFTOwnership(tokenId, walletAddress) {
        try {
            const owner = await this.contracts.contentOwnership.ownerOf(tokenId);
            return owner.toLowerCase() === walletAddress.toLowerCase();
        } catch (error) {
            return false;
        }
    }
    
    /**
     * Check if user has valid license for content
     * @param {string} tokenId - NFT token ID
     * @param {string} walletAddress - User wallet address
     * @returns {boolean} License status
     */
    async _checkLicense(tokenId, walletAddress) {
        try {
            // Check cache first
            const cacheKey = `${tokenId}-${walletAddress}`;
            const cached = this.config.licenseCache.get(cacheKey);
            
            if (cached && Date.now() - cached.timestamp < this.config.licenseCacheTTL) {
                this.metrics.cacheHits++;
                return cached.hasLicense;
            }
            
            // Check blockchain
            const hasLicense = await this.contracts.contentOwnership.hasValidLicense(tokenId, walletAddress);
            
            // Cache result
            this.config.licenseCache.set(cacheKey, {
                hasLicense,
                timestamp: Date.now()
            });
            
            this.metrics.cacheMisses++;
            return hasLicense;
            
        } catch (error) {
            console.error('License check failed:', error);
            return false;
        }
    }
    
    /**
     * Initialize smart contracts
     */
    async _initializeContracts() {
        const contentOwnershipABI = [
            "function ownerOf(uint256 tokenId) view returns (address)",
            "function hasValidLicense(uint256 tokenId, address user) view returns (bool)",
            "function getContentMetadata(uint256 tokenId) view returns (tuple(address creator, string ipfsHash, string contentType, uint256 creationDate, uint256 fileSize, string title, string description, string[] tags, tuple(uint256 licensePrice, uint256 royaltyPercentage, uint256 licenseDuration, bool commercialUse, bool resaleAllowed, bool modificationAllowed) license, bool isLicensable, uint256 totalSupply, uint256 currentSupply))",
            "function getCreatorContent(address creator) view returns (uint256[])",
            "event ContentMinted(uint256 indexed tokenId, address indexed creator, string ipfsHash, string contentType)"
        ];
        
        this.contracts.contentOwnership = new ethers.Contract(
            this.config.contractAddresses.contentOwnership,
            contentOwnershipABI,
            this.provider
        );
    }
    
    /**
     * Initialize Jellyfin API client
     */
    _initializeJellyfinApi() {
        this.jellyfinApi = axios.create({
            baseURL: this.config.jellyfinUrl,
            headers: {
                'X-Emby-Authorization': `MediaBrowser Token="${this.config.jellyfinApiKey}"`
            }
        });
    }
    
    /**
     * Load content mappings from storage
     */
    async _loadContentMappings() {
        try {
            const mappingsPath = path.join(this.config.pluginDataPath, 'content-mappings.json');
            const data = await fs.readFile(mappingsPath, 'utf8');
            const mappings = JSON.parse(data);
            
            for (const [itemId, mapping] of Object.entries(mappings)) {
                this.contentMappings.set(itemId, mapping);
            }
            
            console.log(`Loaded ${this.contentMappings.size} content mappings`);
            
        } catch (error) {
            // File doesn't exist yet, that's okay
            console.log('No existing content mappings found');
        }
    }
    
    /**
     * Save content mappings to storage
     */
    async _saveContentMappings() {
        try {
            const mappingsPath = path.join(this.config.pluginDataPath, 'content-mappings.json');
            const mappings = Object.fromEntries(this.contentMappings);
            await fs.writeFile(mappingsPath, JSON.stringify(mappings, null, 2));
        } catch (error) {
            console.error('Failed to save content mappings:', error);
        }
    }
    
    /**
     * Start license cache cleanup service
     */
    _startLicenseCacheCleanup() {
        setInterval(() => {
            const now = Date.now();
            for (const [key, cached] of this.config.licenseCache.entries()) {
                if (now - cached.timestamp > this.config.licenseCacheTTL) {
                    this.config.licenseCache.delete(key);
                }
            }
        }, 60000); // Every minute
    }
    
    /**
     * Start content sync service
     */
    _startContentSync() {
        // Sync every 10 minutes
        setInterval(() => {
            this.syncBlockchainContent();
        }, 10 * 60 * 1000);
    }
    
    /**
     * Start metrics reporting
     */
    _startMetricsReporting() {
        setInterval(() => {
            this.emit('metrics', this.metrics);
        }, 60000); // Every minute
    }
    
    /**
     * Get plugin metrics
     * @returns {Object} Plugin performance metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            contentMappings: this.contentMappings.size,
            activeSessions: this.userSessions.size,
            licenseCache: this.config.licenseCache.size
        };
    }
    
    /**
     * Shutdown the plugin
     */
    async shutdown() {
        console.log('Shutting down Jellyfin Web3 Plugin...');
        
        // Save current state
        await this._saveContentMappings();
        
        // Clear caches and sessions
        this.config.licenseCache.clear();
        this.userSessions.clear();
        this.contentMappings.clear();
        
        console.log('Jellyfin Web3 Plugin shutdown complete');
        this.emit('shutdown');
    }
}

module.exports = JellyfinWeb3Plugin;