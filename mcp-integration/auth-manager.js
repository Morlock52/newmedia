#!/usr/bin/env node

/**
 * MCP Authentication Manager
 * Handles secure authentication for external services
 */

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');

class MCPAuthManager {
  constructor(configPath = './secrets') {
    this.secretsPath = configPath;
    this.tokenCache = new Map();
    this.keyCache = new Map();
  }

  async initialize() {
    console.log('🔐 Initializing MCP Authentication Manager...');
    
    // Ensure secrets directory exists
    try {
      await fs.access(this.secretsPath);
    } catch (error) {
      await fs.mkdir(this.secretsPath, { recursive: true });
      console.log(`📁 Created secrets directory: ${this.secretsPath}`);
    }
    
    // Load existing tokens and keys
    await this.loadCredentials();
  }

  async loadCredentials() {
    try {
      const files = await fs.readdir(this.secretsPath);
      
      for (const file of files) {
        if (file.endsWith('_api_key.txt')) {
          const service = file.replace('_api_key.txt', '');
          const keyPath = path.join(this.secretsPath, file);
          const apiKey = await fs.readFile(keyPath, 'utf8');
          this.keyCache.set(service, apiKey.trim());
          console.log(`🔑 Loaded API key for ${service}`);
        }
      }
    } catch (error) {
      console.warn('⚠️ No existing credentials found, starting fresh');
    }
  }

  async storeAPIKey(service, apiKey) {
    if (!apiKey || apiKey.trim() === '') {
      throw new Error(`Invalid API key provided for ${service}`);
    }

    const keyPath = path.join(this.secretsPath, `${service}_api_key.txt`);
    await fs.writeFile(keyPath, apiKey.trim(), { mode: 0o600 });
    this.keyCache.set(service, apiKey.trim());
    
    console.log(`✅ Stored API key for ${service}`);
  }

  async getAPIKey(service) {
    const key = this.keyCache.get(service);
    if (!key) {
      throw new Error(`No API key found for ${service}. Please configure it first.`);
    }
    return key;
  }

  async generateServiceToken(service, ttlMinutes = 60) {
    const payload = {
      service,
      issued: Date.now(),
      expires: Date.now() + (ttlMinutes * 60 * 1000)
    };

    const token = crypto.randomBytes(32).toString('hex');
    this.tokenCache.set(token, payload);
    
    // Auto-cleanup expired tokens
    setTimeout(() => {
      this.tokenCache.delete(token);
    }, ttlMinutes * 60 * 1000);

    return token;
  }

  async validateToken(token) {
    const payload = this.tokenCache.get(token);
    if (!payload) {
      throw new Error('Invalid or expired token');
    }

    if (Date.now() > payload.expires) {
      this.tokenCache.delete(token);
      throw new Error('Token has expired');
    }

    return payload;
  }

  async createAuthHeaders(service, customHeaders = {}) {
    const apiKey = await this.getAPIKey(service);
    
    const headers = {
      'User-Agent': 'MCP-Gateway/1.0',
      'Content-Type': 'application/json',
      ...customHeaders
    };

    // Service-specific auth header formats
    switch (service) {
      case 'sonarr':
      case 'radarr':
      case 'lidarr':
      case 'prowlarr':
        headers['X-Api-Key'] = apiKey;
        break;
      case 'jellyfin':
        headers['Authorization'] = `MediaBrowser Token="${apiKey}"`;
        break;
      case 'plex':
        headers['X-Plex-Token'] = apiKey;
        break;
      default:
        headers['Authorization'] = `Bearer ${apiKey}`;
    }

    return headers;
  }

  async encryptSensitiveData(data, service) {
    const key = crypto.scryptSync(service, 'salt', 32);
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher('aes-256-cbc', key);
    
    let encrypted = cipher.update(JSON.stringify(data), 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    return {
      encrypted,
      iv: iv.toString('hex')
    };
  }

  async decryptSensitiveData(encryptedData, service) {
    const key = crypto.scryptSync(service, 'salt', 32);
    const decipher = crypto.createDecipher('aes-256-cbc', key);
    
    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return JSON.parse(decrypted);
  }

  async rotateAPIKey(service) {
    console.log(`🔄 Rotating API key for ${service}...`);
    
    // Backup old key
    const oldKey = this.keyCache.get(service);
    if (oldKey) {
      const backupPath = path.join(this.secretsPath, `${service}_api_key_backup.txt`);
      await fs.writeFile(backupPath, oldKey, { mode: 0o600 });
    }

    // Clear current key
    this.keyCache.delete(service);
    
    console.log(`⚠️ API key rotation initiated for ${service}. Please update with new key.`);
  }

  getAuthStatus() {
    const services = Array.from(this.keyCache.keys());
    const tokens = this.tokenCache.size;
    
    return {
      configuredServices: services,
      activeTokens: tokens,
      lastUpdated: Date.now()
    };
  }
}

// Export for use in other modules
module.exports = MCPAuthManager;

// CLI usage
if (require.main === module) {
  const authManager = new MCPAuthManager();
  
  authManager.initialize().then(() => {
    console.log('🔐 MCP Authentication Manager ready!');
    console.log('📊 Status:', authManager.getAuthStatus());
  });
}