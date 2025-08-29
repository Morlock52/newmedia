const logger = require('../../middleware/logger.js');
/**
 * Web3Service - Web3 wallet connection, NFT media collections, IPFS streaming, smart contracts
 * Provides blockchain integration for media ownership, NFT collections, and decentralized storage
 */

const axios = require('axios');
const EventEmitter = require('events');

class Web3Service extends EventEmitter {
    constructor(config = {}) {
        super();
        this.config = {
            web3Provider: config.web3Provider || process.env.WEB3_PROVIDER_URL || 'https://mainnet.infura.io/v3/',
            infuraKey: config.infuraKey || process.env.INFURA_API_KEY,
            ipfsGateway: config.ipfsGateway || process.env.IPFS_GATEWAY || 'https://gateway.pinata.cloud/ipfs/',
            pinataKey: config.pinataKey || process.env.PINATA_API_KEY,
            pinataSecret: config.pinataSecret || process.env.PINATA_SECRET_KEY,
            contractAddress: config.contractAddress || process.env.NFT_CONTRACT_ADDRESS,
            chainId: config.chainId || process.env.CHAIN_ID || 1, // Ethereum mainnet
            gasLimit: config.gasLimit || 500000,
            ...config
        };

        this.connectedWallets = new Map();
        this.nftCollections = new Map();
        this.ipfsCache = new Map();
        this.isInitialized = false;
        this.supportedChains = {
            1: 'Ethereum',
            137: 'Polygon',
            42161: 'Arbitrum',
            10: 'Optimism',
            56: 'BSC'
        };
    }

    /**
     * Initialize Web3 service
     */
    async initialize() {
        try {
            logger.info('🔗 Initializing Web3Service...');
            
            // Validate configuration
            await this.validateConfig();
            
            // Initialize IPFS connection
            await this.initializeIPFS();
            
            // Load cached NFT collections
            await this.loadNFTCollections();
            
            this.isInitialized = true;
            this.emit('initialized');
            logger.info('✅ Web3Service initialized successfully');
            
            return { success: true, message: 'Web3Service initialized' };
        } catch (error) {
            logger.error('❌ Web3Service initialization failed:', error);
            this.emit('error', error);
            throw error;
        }
    }

    /**
     * Validate service configuration
     */
    async validateConfig() {
        const required = ['infuraKey', 'pinataKey', 'pinataSecret'];
        const missing = required.filter(key => !this.config[key]);
        
        if (missing.length > 0) {
            throw new Error(`Missing required Web3 configuration: ${missing.join(', ')}`);
        }

        // Test API connections
        try {
            await this.testInfuraConnection();
            await this.testPinataConnection();
        } catch (error) {
            throw new Error(`Web3 service connection test failed: ${error.message}`);
        }
    }

    /**
     * Test Infura connection
     */
    async testInfuraConnection() {
        const url = `${this.config.web3Provider}${this.config.infuraKey}`;
        const response = await axios.post(url, {
            jsonrpc: '2.0',
            method: 'eth_blockNumber',
            params: [],
            id: 1
        }, {
            timeout: 5000,
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.data.result) {
            throw new Error('Invalid Infura API response');
        }
    }

    /**
     * Test Pinata IPFS connection
     */
    async testPinataConnection() {
        const response = await axios.get('https://api.pinata.cloud/data/testAuthentication', {
            headers: {
                'pinata_api_key': this.config.pinataKey,
                'pinata_secret_api_key': this.config.pinataSecret
            },
            timeout: 5000
        });

        if (!response.data.message || !response.data.message.includes('Congratulations')) {
            throw new Error('Pinata authentication failed');
        }
    }

    /**
     * Initialize IPFS connection
     */
    async initializeIPFS() {
        try {
            // Test IPFS gateway connectivity
            const testHash = 'QmYwAPJzv5CZsnA625s3Xf2nemtYgPpHdWEz79ojWnPbdG'; // Hello World
            const response = await axios.get(`${this.config.ipfsGateway}${testHash}`, {
                timeout: 10000
            });
            
            if (!response.data || !response.data.includes('Hello')) {
                throw new Error('IPFS gateway test failed');
            }
            
            logger.info('✅ IPFS gateway connection verified');
        } catch (error) {
            logger.warn('⚠️ IPFS gateway test failed, using fallback:', error.message);
            this.config.ipfsGateway = 'https://ipfs.io/ipfs/';
        }
    }

    /**
     * Connect wallet (simulate wallet connection)
     */
    async connectWallet(walletType, address, signature) {
        try {
            if (!address || !signature) {
                throw new Error('Address and signature required for wallet connection');
            }

            // Validate wallet address format
            if (!/^0x[a-fA-F0-9]{40}$/.test(address)) {
                throw new Error('Invalid Ethereum address format');
            }

            const wallet = {
                address,
                type: walletType,
                connected: true,
                connectedAt: new Date(),
                chainId: this.config.chainId,
                nftCollections: [],
                balance: '0'
            };

            // Get wallet balance and NFTs
            await this.updateWalletData(wallet);

            this.connectedWallets.set(address, wallet);
            this.emit('walletConnected', wallet);

            logger.info(`✅ Wallet connected: ${address} (${walletType})`);
            return {
                success: true,
                wallet,
                message: 'Wallet connected successfully'
            };
        } catch (error) {
            logger.error('❌ Wallet connection failed:', error);
            throw error;
        }
    }

    /**
     * Update wallet data (balance, NFTs)
     */
    async updateWalletData(wallet) {
        try {
            // Get ETH balance
            const balanceResult = await this.getWalletBalance(wallet.address);
            wallet.balance = balanceResult.balance;

            // Get NFT collections
            const nfts = await this.getNFTsByWallet(wallet.address);
            wallet.nftCollections = nfts.collections;
            wallet.totalNFTs = nfts.total;

            return wallet;
        } catch (error) {
            logger.warn('⚠️ Failed to update wallet data:', error.message);
            return wallet;
        }
    }

    /**
     * Get wallet balance
     */
    async getWalletBalance(address) {
        try {
            const url = `${this.config.web3Provider}${this.config.infuraKey}`;
            const response = await axios.post(url, {
                jsonrpc: '2.0',
                method: 'eth_getBalance',
                params: [address, 'latest'],
                id: 1
            });

            const balanceWei = BigInt(response.data.result);
            const balanceEth = Number(balanceWei) / Math.pow(10, 18);

            return {
                balance: balanceEth.toFixed(6),
                balanceWei: balanceWei.toString(),
                currency: 'ETH'
            };
        } catch (error) {
            logger.error('❌ Failed to get wallet balance:', error);
            return { balance: '0', balanceWei: '0', currency: 'ETH' };
        }
    }

    /**
     * Get NFTs by wallet address
     */
    async getNFTsByWallet(address) {
        try {
            // Use OpenSea API or Alchemy NFT API
            const response = await axios.get(`https://api.opensea.io/api/v1/assets`, {
                params: {
                    owner: address,
                    limit: 50,
                    format: 'json'
                },
                headers: {
                    'X-API-KEY': process.env.OPENSEA_API_KEY || ''
                },
                timeout: 10000
            });

            const collections = new Map();
            const nfts = response.data.assets || [];

            nfts.forEach(nft => {
                const collectionName = nft.collection?.name || 'Unknown';
                if (!collections.has(collectionName)) {
                    collections.set(collectionName, {
                        name: collectionName,
                        slug: nft.collection?.slug,
                        items: [],
                        floorPrice: nft.collection?.stats?.floor_price || 0
                    });
                }

                collections.get(collectionName).items.push({
                    tokenId: nft.token_id,
                    name: nft.name,
                    description: nft.description,
                    imageUrl: nft.image_url,
                    animationUrl: nft.animation_url,
                    traits: nft.traits,
                    contract: nft.asset_contract
                });
            });

            return {
                collections: Array.from(collections.values()),
                total: nfts.length,
                timestamp: new Date()
            };
        } catch (error) {
            logger.warn('⚠️ Failed to fetch NFTs:', error.message);
            return { collections: [], total: 0, timestamp: new Date() };
        }
    }

    /**
     * Upload media to IPFS
     */
    async uploadToIPFS(mediaBuffer, metadata = {}) {
        try {
            const formData = new FormData();
            formData.append('file', mediaBuffer, {
                filename: metadata.filename || `media_${Date.now()}`,
                contentType: metadata.contentType || 'application/octet-stream'
            });

            const pinataMetadata = JSON.stringify({
                name: metadata.name || 'Media File',
                description: metadata.description || 'Uploaded via Web3Service',
                keyvalues: {
                    uploadedAt: new Date().toISOString(),
                    service: 'Web3Service',
                    ...metadata.keyvalues
                }
            });

            formData.append('pinataMetadata', pinataMetadata);

            const response = await axios.post('https://api.pinata.cloud/pinning/pinFileToIPFS', formData, {
                headers: {
                    'Content-Type': 'multipart/form-data',
                    'pinata_api_key': this.config.pinataKey,
                    'pinata_secret_api_key': this.config.pinataSecret
                },
                timeout: 30000
            });

            const ipfsHash = response.data.IpfsHash;
            const ipfsUrl = `${this.config.ipfsGateway}${ipfsHash}`;

            // Cache the upload
            this.ipfsCache.set(ipfsHash, {
                url: ipfsUrl,
                metadata,
                uploadedAt: new Date(),
                size: mediaBuffer.length
            });

            this.emit('ipfsUpload', { hash: ipfsHash, url: ipfsUrl, metadata });

            logger.info(`✅ Media uploaded to IPFS: ${ipfsHash}`);
            return {
                success: true,
                hash: ipfsHash,
                url: ipfsUrl,
                gateway: this.config.ipfsGateway,
                metadata: response.data
            };
        } catch (error) {
            logger.error('❌ IPFS upload failed:', error);
            throw error;
        }
    }

    /**
     * Stream media from IPFS
     */
    async streamFromIPFS(ipfsHash) {
        try {
            const url = `${this.config.ipfsGateway}${ipfsHash}`;
            
            // Check cache first
            if (this.ipfsCache.has(ipfsHash)) {
                const cached = this.ipfsCache.get(ipfsHash);
                cached.lastAccessed = new Date();
            }

            const response = await axios({
                method: 'GET',
                url,
                responseType: 'stream',
                timeout: 30000
            });

            return {
                success: true,
                stream: response.data,
                contentType: response.headers['content-type'],
                contentLength: response.headers['content-length'],
                url
            };
        } catch (error) {
            logger.error('❌ IPFS streaming failed:', error);
            throw error;
        }
    }

    /**
     * Create NFT metadata
     */
    async createNFTMetadata(mediaData) {
        try {
            const metadata = {
                name: mediaData.title || 'Media NFT',
                description: mediaData.description || 'Media content as NFT',
                image: mediaData.imageUrl,
                animation_url: mediaData.animationUrl,
                external_url: mediaData.externalUrl,
                attributes: [
                    { trait_type: 'Media Type', value: mediaData.type || 'Video' },
                    { trait_type: 'Duration', value: mediaData.duration || 'Unknown' },
                    { trait_type: 'Quality', value: mediaData.quality || 'HD' },
                    { trait_type: 'Genre', value: mediaData.genre || 'Entertainment' },
                    { trait_type: 'Created', value: new Date().toISOString() }
                ],
                properties: {
                    category: 'Media',
                    subcategory: mediaData.subcategory || 'Video',
                    creator: mediaData.creator || 'Anonymous'
                }
            };

            // Upload metadata to IPFS
            const metadataBuffer = Buffer.from(JSON.stringify(metadata, null, 2));
            const ipfsResult = await this.uploadToIPFS(metadataBuffer, {
                filename: `metadata_${Date.now()}.json`,
                contentType: 'application/json',
                name: 'NFT Metadata'
            });

            return {
                success: true,
                metadata,
                metadataUri: ipfsResult.url,
                ipfsHash: ipfsResult.hash
            };
        } catch (error) {
            logger.error('❌ NFT metadata creation failed:', error);
            throw error;
        }
    }

    /**
     * Load NFT collections cache
     */
    async loadNFTCollections() {
        try {
            // Load from persistent storage if available
            logger.info('📚 Loading NFT collections...');
            
            // For now, initialize with empty collections
            // In production, load from database or cache
            this.nftCollections.clear();
            
            logger.info('✅ NFT collections loaded');
        } catch (error) {
            logger.warn('⚠️ Failed to load NFT collections:', error.message);
        }
    }

    /**
     * Get service status
     */
    getStatus() {
        return {
            initialized: this.isInitialized,
            connectedWallets: this.connectedWallets.size,
            nftCollections: this.nftCollections.size,
            ipfsCache: this.ipfsCache.size,
            supportedChains: Object.keys(this.supportedChains).length,
            config: {
                chainId: this.config.chainId,
                ipfsGateway: this.config.ipfsGateway,
                web3Provider: this.config.web3Provider.split('/').slice(0, 3).join('/')
            },
            lastUpdate: new Date()
        };
    }

    /**
     * Disconnect wallet
     */
    async disconnectWallet(address) {
        try {
            if (this.connectedWallets.has(address)) {
                this.connectedWallets.delete(address);
                this.emit('walletDisconnected', { address });
                logger.info(`✅ Wallet disconnected: ${address}`);
                return { success: true, message: 'Wallet disconnected' };
            }
            
            return { success: false, message: 'Wallet not found' };
        } catch (error) {
            logger.error('❌ Wallet disconnection failed:', error);
            throw error;
        }
    }

    /**
     * Cleanup resources
     */
    async cleanup() {
        try {
            logger.info('🧹 Cleaning up Web3Service...');
            
            this.connectedWallets.clear();
            this.nftCollections.clear();
            this.ipfsCache.clear();
            this.removeAllListeners();
            
            this.isInitialized = false;
            logger.info('✅ Web3Service cleanup completed');
        } catch (error) {
            logger.error('❌ Web3Service cleanup failed:', error);
        }
    }
}

module.exports = Web3Service;