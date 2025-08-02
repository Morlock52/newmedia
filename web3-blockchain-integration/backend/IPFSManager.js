/**
 * IPFS Manager for Decentralized Media Storage
 * Handles IPFS operations with encryption, pinning, and clustering
 */

const { create } = require('ipfs-http-client');
const { CID } = require('multiformats/cid');
const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');
const FormData = require('form-data');
const axios = require('axios');

class IPFSManager {
    constructor(config = {}) {
        // IPFS node configuration
        this.ipfsConfig = {
            host: config.ipfsHost || 'localhost',
            port: config.ipfsPort || 5001,
            protocol: config.ipfsProtocol || 'http',
            timeout: config.timeout || 30000
        };
        
        // Pinning service configuration (Pinata, Infura, etc.)
        this.pinningServices = config.pinningServices || [];
        
        // Encryption settings
        this.encryptionEnabled = config.encryptionEnabled || false;
        this.encryptionAlgorithm = config.encryptionAlgorithm || 'aes-256-gcm';
        
        // Initialize IPFS client
        this.ipfs = create({
            host: this.ipfsConfig.host,
            port: this.ipfsConfig.port,
            protocol: this.ipfsConfig.protocol,
            timeout: this.ipfsConfig.timeout
        });
        
        // Cluster configuration for redundancy
        this.clusterNodes = config.clusterNodes || [];
        
        // Cache for frequently accessed content
        this.cache = new Map();
        this.cacheMaxSize = config.cacheMaxSize || 100;
        
        // Metrics
        this.metrics = {
            uploads: 0,
            downloads: 0,
            pins: 0,
            cacheHits: 0,
            cacheMisses: 0
        };
    }
    
    /**
     * Upload content to IPFS with optional encryption
     * @param {Buffer|Stream} content - Content to upload
     * @param {Object} options - Upload options
     * @returns {Object} Upload result with IPFS hash and metadata
     */
    async uploadContent(content, options = {}) {
        try {
            let uploadData = content;
            let encryptionKey = null;
            let encryptionIV = null;
            
            // Encrypt content if enabled
            if (this.encryptionEnabled || options.encrypt) {
                const encrypted = await this.encryptContent(content);
                uploadData = encrypted.encryptedData;
                encryptionKey = encrypted.key;
                encryptionIV = encrypted.iv;
            }
            
            // Create metadata object
            const metadata = {
                name: options.name || 'untitled',
                type: options.type || 'application/octet-stream',
                size: uploadData.length,
                timestamp: new Date().toISOString(),
                creator: options.creator || 'anonymous',
                description: options.description || '',
                tags: options.tags || [],
                encrypted: this.encryptionEnabled || options.encrypt,
                contentType: options.contentType || 'unknown',
                duration: options.duration || null,
                resolution: options.resolution || null,
                format: options.format || null
            };
            
            // Upload to IPFS
            const fileResult = await this.ipfs.add(uploadData, {
                pin: true,
                wrapWithDirectory: false,
                progress: options.onProgress
            });
            
            // Upload metadata
            const metadataResult = await this.ipfs.add(JSON.stringify(metadata), {
                pin: true
            });
            
            // Create DAG link between content and metadata
            const dagNode = {
                content: fileResult.cid.toString(),
                metadata: metadataResult.cid.toString(),
                version: '1.0',
                timestamp: Date.now()
            };
            
            const dagResult = await this.ipfs.dag.put(dagNode);
            
            // Pin to additional services if configured
            if (this.pinningServices.length > 0) {
                await this.pinToServices(fileResult.cid.toString(), metadata);
            }
            
            // Distribute to cluster nodes
            if (this.clusterNodes.length > 0) {
                await this.distributeToCluster(fileResult.cid.toString());
            }
            
            // Update metrics
            this.metrics.uploads++;
            
            return {
                contentHash: fileResult.cid.toString(),
                metadataHash: metadataResult.cid.toString(),
                dagHash: dagResult.toString(),
                size: fileResult.size,
                encryptionKey: encryptionKey ? encryptionKey.toString('hex') : null,
                encryptionIV: encryptionIV ? encryptionIV.toString('hex') : null,
                metadata: metadata,
                pinned: true,
                distributedTo: this.clusterNodes.length
            };
            
        } catch (error) {
            console.error('IPFS upload error:', error);
            throw new Error(`Failed to upload to IPFS: ${error.message}`);
        }
    }
    
    /**
     * Retrieve content from IPFS with decryption
     * @param {string} ipfsHash - IPFS content hash
     * @param {Object} options - Retrieval options
     * @returns {Buffer} Content data
     */
    async retrieveContent(ipfsHash, options = {}) {
        try {
            // Check cache first
            if (this.cache.has(ipfsHash)) {
                this.metrics.cacheHits++;
                return this.cache.get(ipfsHash);
            }
            
            this.metrics.cacheMisses++;
            
            // Retrieve from IPFS
            const chunks = [];
            for await (const chunk of this.ipfs.cat(ipfsHash, {
                timeout: options.timeout || 30000
            })) {
                chunks.push(chunk);
            }
            
            let content = Buffer.concat(chunks);
            
            // Decrypt if needed
            if (options.encryptionKey && options.encryptionIV) {
                content = await this.decryptContent(
                    content,
                    Buffer.from(options.encryptionKey, 'hex'),
                    Buffer.from(options.encryptionIV, 'hex')
                );
            }
            
            // Add to cache
            this.addToCache(ipfsHash, content);
            
            // Update metrics
            this.metrics.downloads++;
            
            return content;
            
        } catch (error) {
            // Try cluster nodes if main node fails
            if (this.clusterNodes.length > 0) {
                return await this.retrieveFromCluster(ipfsHash, options);
            }
            throw new Error(`Failed to retrieve from IPFS: ${error.message}`);
        }
    }
    
    /**
     * Get content metadata
     * @param {string} metadataHash - IPFS metadata hash
     * @returns {Object} Content metadata
     */
    async getMetadata(metadataHash) {
        try {
            const chunks = [];
            for await (const chunk of this.ipfs.cat(metadataHash)) {
                chunks.push(chunk);
            }
            
            const metadataJson = Buffer.concat(chunks).toString();
            return JSON.parse(metadataJson);
            
        } catch (error) {
            console.error('Failed to retrieve metadata:', error);
            throw error;
        }
    }
    
    /**
     * Pin content to additional pinning services
     * @param {string} ipfsHash - Content hash to pin
     * @param {Object} metadata - Content metadata
     */
    async pinToServices(ipfsHash, metadata) {
        const pinPromises = this.pinningServices.map(async (service) => {
            try {
                if (service.type === 'pinata') {
                    return await this.pinToPinata(ipfsHash, metadata, service);
                } else if (service.type === 'infura') {
                    return await this.pinToInfura(ipfsHash, service);
                } else if (service.type === 'filebase') {
                    return await this.pinToFilebase(ipfsHash, metadata, service);
                }
            } catch (error) {
                console.error(`Failed to pin to ${service.type}:`, error);
                return { success: false, service: service.type, error: error.message };
            }
        });
        
        const results = await Promise.allSettled(pinPromises);
        this.metrics.pins += results.filter(r => r.status === 'fulfilled').length;
        
        return results;
    }
    
    /**
     * Pin to Pinata service
     */
    async pinToPinata(ipfsHash, metadata, config) {
        const url = 'https://api.pinata.cloud/pinning/pinByHash';
        const body = {
            hashToPin: ipfsHash,
            pinataMetadata: {
                name: metadata.name,
                keyvalues: {
                    type: metadata.type,
                    creator: metadata.creator,
                    timestamp: metadata.timestamp
                }
            }
        };
        
        const response = await axios.post(url, body, {
            headers: {
                'pinata_api_key': config.apiKey,
                'pinata_secret_api_key': config.secretKey
            }
        });
        
        return response.data;
    }
    
    /**
     * Encrypt content using AES
     */
    async encryptContent(content) {
        const key = crypto.randomBytes(32); // 256-bit key
        const iv = crypto.randomBytes(16);  // 128-bit IV
        
        const cipher = crypto.createCipheriv(this.encryptionAlgorithm, key, iv);
        
        const encrypted = Buffer.concat([
            cipher.update(content),
            cipher.final()
        ]);
        
        // Get the auth tag for GCM mode
        const authTag = cipher.getAuthTag();
        
        return {
            encryptedData: Buffer.concat([authTag, encrypted]),
            key: key,
            iv: iv
        };
    }
    
    /**
     * Decrypt content
     */
    async decryptContent(encryptedContent, key, iv) {
        const authTag = encryptedContent.slice(0, 16);
        const encrypted = encryptedContent.slice(16);
        
        const decipher = crypto.createDecipheriv(this.encryptionAlgorithm, key, iv);
        decipher.setAuthTag(authTag);
        
        const decrypted = Buffer.concat([
            decipher.update(encrypted),
            decipher.final()
        ]);
        
        return decrypted;
    }
    
    /**
     * Distribute content to cluster nodes
     */
    async distributeToCluster(ipfsHash) {
        const distributionPromises = this.clusterNodes.map(async (node) => {
            try {
                const nodeClient = create({
                    host: node.host,
                    port: node.port,
                    protocol: node.protocol || 'http'
                });
                
                // Pin on remote node
                await nodeClient.pin.add(ipfsHash);
                
                return { node: node.host, success: true };
            } catch (error) {
                return { node: node.host, success: false, error: error.message };
            }
        });
        
        return await Promise.allSettled(distributionPromises);
    }
    
    /**
     * Retrieve content from cluster nodes
     */
    async retrieveFromCluster(ipfsHash, options) {
        for (const node of this.clusterNodes) {
            try {
                const nodeClient = create({
                    host: node.host,
                    port: node.port,
                    protocol: node.protocol || 'http'
                });
                
                const chunks = [];
                for await (const chunk of nodeClient.cat(ipfsHash)) {
                    chunks.push(chunk);
                }
                
                return Buffer.concat(chunks);
            } catch (error) {
                continue; // Try next node
            }
        }
        
        throw new Error('Failed to retrieve from any cluster node');
    }
    
    /**
     * Generate streaming URL for content
     */
    generateStreamingURL(ipfsHash, options = {}) {
        const gateway = options.gateway || `http://${this.ipfsConfig.host}:8080`;
        return `${gateway}/ipfs/${ipfsHash}`;
    }
    
    /**
     * Check if content is available
     */
    async isContentAvailable(ipfsHash) {
        try {
            const stats = await this.ipfs.object.stat(ipfsHash, { timeout: 5000 });
            return stats && stats.Hash === ipfsHash;
        } catch (error) {
            return false;
        }
    }
    
    /**
     * Calculate content hash without uploading
     */
    async calculateHash(content) {
        const result = await this.ipfs.add(content, {
            onlyHash: true,
            pin: false
        });
        
        return result.cid.toString();
    }
    
    /**
     * Add content to cache
     */
    addToCache(hash, content) {
        if (this.cache.size >= this.cacheMaxSize) {
            // Remove oldest entry
            const firstKey = this.cache.keys().next().value;
            this.cache.delete(firstKey);
        }
        
        this.cache.set(hash, content);
    }
    
    /**
     * Get metrics
     */
    getMetrics() {
        return {
            ...this.metrics,
            cacheSize: this.cache.size,
            clusterNodes: this.clusterNodes.length,
            pinningServices: this.pinningServices.length
        };
    }
    
    /**
     * Clear cache
     */
    clearCache() {
        this.cache.clear();
    }
}

module.exports = IPFSManager;