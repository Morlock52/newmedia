/**
 * Web3 Media Service
 * Integrates blockchain functionality with the existing media server
 */

const { ethers } = require('ethers');
const express = require('express');
const multer = require('multer');
const cors = require('cors');
const jwt = require('jsonwebtoken');
const IPFSManager = require('./IPFSManager');
const MediaNFTABI = require('../contracts/MediaNFTOptimized.json');
const PaymentProcessorABI = require('../contracts/CrossChainPaymentProcessor.json');

class Web3MediaService {
    constructor(config) {
        this.config = config;
        
        // Initialize Express app
        this.app = express();
        this.app.use(cors());
        this.app.use(express.json());
        
        // Initialize blockchain providers
        this.providers = {
            ethereum: new ethers.providers.JsonRpcProvider(config.ethereum.rpcUrl),
            polygon: new ethers.providers.JsonRpcProvider(config.polygon.rpcUrl),
            arbitrum: new ethers.providers.JsonRpcProvider(config.arbitrum.rpcUrl),
            optimism: new ethers.providers.JsonRpcProvider(config.optimism.rpcUrl)
        };
        
        // Initialize contracts
        this.contracts = {};
        this.initializeContracts();
        
        // Initialize IPFS manager
        this.ipfsManager = new IPFSManager({
            ipfsHost: config.ipfs.host,
            ipfsPort: config.ipfs.port,
            encryptionEnabled: config.ipfs.encryptionEnabled,
            pinningServices: config.ipfs.pinningServices,
            clusterNodes: config.ipfs.clusterNodes
        });
        
        // File upload configuration
        const storage = multer.memoryStorage();
        this.upload = multer({ 
            storage: storage,
            limits: {
                fileSize: config.maxFileSize || 5 * 1024 * 1024 * 1024 // 5GB default
            }
        });
        
        // Setup routes
        this.setupRoutes();
        
        // Jellyfin integration
        this.jellyfinApiUrl = config.jellyfin.apiUrl;
        this.jellyfinApiKey = config.jellyfin.apiKey;
        
        // Cache for blockchain data
        this.cache = new Map();
        this.cacheExpiry = config.cacheExpiry || 300000; // 5 minutes
    }
    
    /**
     * Initialize smart contracts on all supported chains
     */
    initializeContracts() {
        Object.keys(this.providers).forEach(chain => {
            const provider = this.providers[chain];
            const signer = new ethers.Wallet(this.config[chain].privateKey, provider);
            
            this.contracts[chain] = {
                mediaNFT: new ethers.Contract(
                    this.config[chain].contracts.mediaNFT,
                    MediaNFTABI.abi,
                    signer
                ),
                paymentProcessor: new ethers.Contract(
                    this.config[chain].contracts.paymentProcessor,
                    PaymentProcessorABI.abi,
                    signer
                )
            };
        });
    }
    
    /**
     * Setup API routes
     */
    setupRoutes() {
        // Health check
        this.app.get('/api/health', (req, res) => {
            res.json({ status: 'ok', service: 'Web3MediaService' });
        });
        
        // IPFS routes
        this.app.post('/api/ipfs/upload', 
            this.authenticateRequest.bind(this),
            this.upload.single('file'), 
            this.handleIPFSUpload.bind(this)
        );
        
        this.app.get('/api/ipfs/content/:hash', this.handleIPFSRetrieve.bind(this));
        this.app.get('/api/ipfs/metadata/:hash', this.handleMetadataRetrieve.bind(this));
        this.app.get('/api/ipfs/thumbnail/:hash', this.handleThumbnailRetrieve.bind(this));
        
        // NFT routes
        this.app.post('/api/nft/mint', 
            this.authenticateRequest.bind(this),
            this.handleNFTMint.bind(this)
        );
        
        this.app.get('/api/nft/content/:tokenId', this.handleNFTContent.bind(this));
        this.app.get('/api/nft/owned/:address', this.handleOwnedNFTs.bind(this));
        this.app.post('/api/nft/verify', this.handleNFTVerification.bind(this));
        
        // Payment routes
        this.app.post('/api/payment/process', 
            this.authenticateRequest.bind(this),
            this.handlePayment.bind(this)
        );
        
        this.app.post('/api/payment/subscribe', 
            this.authenticateRequest.bind(this),
            this.handleSubscription.bind(this)
        );
        
        this.app.get('/api/payment/tokens', this.handleSupportedTokens.bind(this));
        this.app.get('/api/payment/price/:contentId', this.handlePriceQuote.bind(this));
        
        // Integration routes
        this.app.post('/api/jellyfin/web3-auth', this.handleJellyfinAuth.bind(this));
        this.app.get('/api/jellyfin/web3-content', this.handleJellyfinContent.bind(this));
        this.app.post('/api/jellyfin/sync-nft', this.handleJellyfinNFTSync.bind(this));
        
        // Analytics routes
        this.app.get('/api/analytics/user/:address', this.handleUserAnalytics.bind(this));
        this.app.get('/api/analytics/content/:contentId', this.handleContentAnalytics.bind(this));
    }
    
    /**
     * Authenticate requests
     */
    authenticateRequest(req, res, next) {
        const token = req.headers.authorization?.split(' ')[1];
        
        if (!token) {
            return res.status(401).json({ error: 'No token provided' });
        }
        
        try {
            const decoded = jwt.verify(token, this.config.jwtSecret);
            req.user = decoded;
            next();
        } catch (error) {
            return res.status(401).json({ error: 'Invalid token' });
        }
    }
    
    /**
     * Handle IPFS upload
     */
    async handleIPFSUpload(req, res) {
        try {
            if (!req.file) {
                return res.status(400).json({ error: 'No file provided' });
            }
            
            const metadata = JSON.parse(req.body.metadata || '{}');
            
            const result = await this.ipfsManager.uploadContent(req.file.buffer, {
                name: req.file.originalname,
                type: req.file.mimetype,
                creator: req.user.address,
                encrypt: req.body.encrypt === 'true',
                ...metadata
            });
            
            // Store in database for indexing
            await this.storeContentMetadata(result);
            
            res.json({
                success: true,
                ...result
            });
            
        } catch (error) {
            console.error('IPFS upload error:', error);
            res.status(500).json({ error: error.message });
        }
    }
    
    /**
     * Handle IPFS content retrieval
     */
    async handleIPFSRetrieve(req, res) {
        try {
            const { hash } = req.params;
            const { encryptionKey, encryptionIV } = req.query;
            
            // Check if user has access rights
            const hasAccess = await this.checkContentAccess(hash, req.headers.authorization);
            
            if (!hasAccess) {
                return res.status(403).json({ error: 'Access denied' });
            }
            
            const content = await this.ipfsManager.retrieveContent(hash, {
                encryptionKey,
                encryptionIV
            });
            
            // Set appropriate content type
            const metadata = await this.getContentMetadata(hash);
            res.setHeader('Content-Type', metadata.type || 'application/octet-stream');
            res.setHeader('Content-Disposition', `inline; filename="${metadata.name}"`);
            
            res.send(content);
            
        } catch (error) {
            console.error('IPFS retrieve error:', error);
            res.status(500).json({ error: error.message });
        }
    }
    
    /**
     * Handle NFT minting
     */
    async handleNFTMint(req, res) {
        try {
            const {
                ipfsHash,
                contentType,
                fileSize,
                title,
                description,
                tags,
                licenseType,
                royaltyPercentage,
                chain = 'polygon' // Default to Polygon for lower gas
            } = req.body;
            
            // Validate chain
            if (!this.contracts[chain]) {
                return res.status(400).json({ error: 'Unsupported chain' });
            }
            
            const contract = this.contracts[chain].mediaNFT;
            
            // Estimate gas
            const gasEstimate = await contract.estimateGas.mintMedia(
                ipfsHash,
                contentType,
                fileSize,
                licenseType,
                false, // isEncrypted
                '', // encryptionKey
                royaltyPercentage || 250 // 2.5% default
            );
            
            // Get gas price
            const gasPrice = await this.providers[chain].getGasPrice();
            
            // Execute transaction
            const tx = await contract.mintMedia(
                ipfsHash,
                contentType,
                fileSize,
                licenseType,
                false,
                '',
                royaltyPercentage || 250,
                {
                    gasLimit: gasEstimate.mul(110).div(100), // 10% buffer
                    gasPrice: gasPrice
                }
            );
            
            // Wait for confirmation
            const receipt = await tx.wait();
            
            // Get token ID from events
            const event = receipt.events.find(e => e.event === 'MediaMinted');
            const tokenId = event.args.tokenId.toString();
            
            // Create Jellyfin entry
            await this.createJellyfinEntry({
                tokenId,
                ipfsHash,
                title,
                description,
                chain
            });
            
            res.json({
                success: true,
                tokenId,
                transactionHash: receipt.transactionHash,
                chain,
                gasUsed: receipt.gasUsed.toString(),
                blockNumber: receipt.blockNumber
            });
            
        } catch (error) {
            console.error('NFT mint error:', error);
            res.status(500).json({ error: error.message });
        }
    }
    
    /**
     * Handle payment processing
     */
    async handlePayment(req, res) {
        try {
            const {
                contentId,
                recipient,
                paymentToken,
                chain = 'polygon'
            } = req.body;
            
            const contract = this.contracts[chain].paymentProcessor;
            
            // Get content price
            const priceUSD = await contract.contentPrices(contentId);
            
            if (priceUSD.eq(0)) {
                return res.status(400).json({ error: 'Content not priced' });
            }
            
            // Calculate token amount
            const tokenAmount = await contract.calculateTokenAmount(
                priceUSD,
                paymentToken
            );
            
            res.json({
                success: true,
                priceUSD: ethers.utils.formatUnits(priceUSD, 6),
                tokenAmount: ethers.utils.formatEther(tokenAmount),
                paymentToken,
                recipient,
                chain
            });
            
        } catch (error) {
            console.error('Payment processing error:', error);
            res.status(500).json({ error: error.message });
        }
    }
    
    /**
     * Handle Jellyfin Web3 authentication
     */
    async handleJellyfinAuth(req, res) {
        try {
            const { address, signature, message } = req.body;
            
            // Verify signature
            const recoveredAddress = ethers.utils.verifyMessage(message, signature);
            
            if (recoveredAddress.toLowerCase() !== address.toLowerCase()) {
                return res.status(401).json({ error: 'Invalid signature' });
            }
            
            // Check if user has active subscription or owned content
            const hasAccess = await this.checkUserAccess(address);
            
            if (!hasAccess) {
                return res.status(403).json({ error: 'No active subscription or owned content' });
            }
            
            // Generate JWT token
            const token = jwt.sign(
                { 
                    address,
                    type: 'web3',
                    timestamp: Date.now()
                },
                this.config.jwtSecret,
                { expiresIn: '24h' }
            );
            
            // Create or update Jellyfin user
            const jellyfinUser = await this.createJellyfinUser(address);
            
            res.json({
                success: true,
                token,
                jellyfinUserId: jellyfinUser.Id,
                jellyfinToken: jellyfinUser.AccessToken
            });
            
        } catch (error) {
            console.error('Jellyfin auth error:', error);
            res.status(500).json({ error: error.message });
        }
    }
    
    /**
     * Check user access across all chains
     */
    async checkUserAccess(address) {
        // Check cache first
        const cacheKey = `access:${address}`;
        const cached = this.cache.get(cacheKey);
        
        if (cached && cached.expiry > Date.now()) {
            return cached.hasAccess;
        }
        
        // Check subscription on all chains
        for (const chain of Object.keys(this.contracts)) {
            try {
                const contract = this.contracts[chain].paymentProcessor;
                const hasSubscription = await contract.hasActiveSubscription(address);
                
                if (hasSubscription) {
                    this.cache.set(cacheKey, {
                        hasAccess: true,
                        expiry: Date.now() + this.cacheExpiry
                    });
                    return true;
                }
            } catch (error) {
                console.error(`Error checking subscription on ${chain}:`, error);
            }
        }
        
        // Check owned NFTs
        const ownedNFTs = await this.getOwnedNFTs(address);
        const hasAccess = ownedNFTs.length > 0;
        
        this.cache.set(cacheKey, {
            hasAccess,
            expiry: Date.now() + this.cacheExpiry
        });
        
        return hasAccess;
    }
    
    /**
     * Get owned NFTs across all chains
     */
    async getOwnedNFTs(address) {
        const ownedNFTs = [];
        
        for (const chain of Object.keys(this.contracts)) {
            try {
                const contract = this.contracts[chain].mediaNFT;
                const tokens = await contract.getCreatorTokens(address);
                
                for (const tokenId of tokens) {
                    const metadata = await contract.getMediaData(tokenId);
                    ownedNFTs.push({
                        chain,
                        tokenId: tokenId.toString(),
                        ipfsHash: metadata.ipfsHash,
                        metadata
                    });
                }
            } catch (error) {
                console.error(`Error fetching NFTs on ${chain}:`, error);
            }
        }
        
        return ownedNFTs;
    }
    
    /**
     * Create Jellyfin user for Web3 address
     */
    async createJellyfinUser(address) {
        try {
            const response = await fetch(`${this.jellyfinApiUrl}/Users/New`, {
                method: 'POST',
                headers: {
                    'X-Emby-Token': this.jellyfinApiKey,
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    Name: `web3_${address.slice(0, 8)}`,
                    Password: '',
                    PasswordResetProviderId: 'Web3Auth'
                })
            });
            
            return await response.json();
            
        } catch (error) {
            console.error('Error creating Jellyfin user:', error);
            throw error;
        }
    }
    
    /**
     * Start the service
     */
    start(port = 3333) {
        this.app.listen(port, () => {
            console.log(`Web3 Media Service running on port ${port}`);
        });
    }
}

module.exports = Web3MediaService;