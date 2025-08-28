#!/usr/bin/env node

/**
 * Secure Environment Configuration Manager
 * Fixes: Exposed API keys, insecure credential management
 * Author: Security Manager Agent
 * Date: 2025-08-03
 */

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');
const readline = require('readline');

class SecureEnvironmentManager {
  constructor() {
    this.encryptionKey = null;
    this.configPath = path.join(process.cwd(), '.env.secure');
    this.keyPath = path.join(process.cwd(), '.security', 'master.key');
    this.algorithm = 'aes-256-gcm'; // Using AES-256-GCM for authenticated encryption
  }

  /**
   * Initialize secure environment management
   */
  async initialize() {
    try {
      await fs.mkdir(path.dirname(this.keyPath), { recursive: true, mode: 0o700 });
      
      if (!(await this.keyExists())) {
        await this.generateMasterKey();
        console.log('✅ Master encryption key generated');
      }
      
      await this.loadMasterKey();
      console.log('✅ Secure environment manager initialized');
    } catch (error) {
      throw new Error(`Failed to initialize secure environment: ${error.message}`);
    }
  }

  /**
   * Generate a new master encryption key
   */
  async generateMasterKey() {
    const key = crypto.randomBytes(32);
    await fs.writeFile(this.keyPath, key, { mode: 0o600 });
  }

  /**
   * Check if master key exists
   */
  async keyExists() {
    try {
      await fs.access(this.keyPath);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Load master key from secure storage
   */
  async loadMasterKey() {
    try {
      this.encryptionKey = await fs.readFile(this.keyPath);
    } catch (error) {
      throw new Error(`Failed to load master key: ${error.message}`);
    }
  }

  /**
   * Encrypt sensitive configuration data
   */
  encrypt(plaintext) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher(this.algorithm, this.encryptionKey);
    cipher.setAAD(Buffer.from('secure-env-data'));
    
    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  }

  /**
   * Decrypt sensitive configuration data
   */
  decrypt(encryptedData) {
    const { encrypted, iv, authTag } = encryptedData;
    
    const decipher = crypto.createDecipher(this.algorithm, this.encryptionKey);
    decipher.setAAD(Buffer.from('secure-env-data'));
    decipher.setAuthTag(Buffer.from(authTag, 'hex'));
    
    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }

  /**
   * Secure environment variable storage
   */
  async setSecureVariable(key, value, category = 'general') {
    const config = await this.loadSecureConfig();
    
    if (!config[category]) {
      config[category] = {};
    }
    
    config[category][key] = this.encrypt(value);
    await this.saveSecureConfig(config);
  }

  /**
   * Retrieve secure environment variable
   */
  async getSecureVariable(key, category = 'general') {
    const config = await this.loadSecureConfig();
    
    if (!config[category] || !config[category][key]) {
      return null;
    }
    
    return this.decrypt(config[category][key]);
  }

  /**
   * Load secure configuration file
   */
  async loadSecureConfig() {
    try {
      const data = await fs.readFile(this.configPath, 'utf8');
      return JSON.parse(data);
    } catch {
      return {};
    }
  }

  /**
   * Save secure configuration file
   */
  async saveSecureConfig(config) {
    await fs.writeFile(
      this.configPath, 
      JSON.stringify(config, null, 2), 
      { mode: 0o600 }
    );
  }

  /**
   * Generate secure API keys
   */
  generateApiKey(length = 64) {
    return crypto.randomBytes(length).toString('hex');
  }

  /**
   * Generate secure JWT secret
   */
  generateJwtSecret(length = 128) {
    return crypto.randomBytes(length).toString('base64url');
  }

  /**
   * Generate secure database password
   */
  generateDatabasePassword(length = 32) {
    const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?';
    let password = '';
    
    for (let i = 0; i < length; i++) {
      password += charset.charAt(Math.floor(Math.random() * charset.length));
    }
    
    return password;
  }

  /**
   * Migrate existing .env file to secure storage
   */
  async migrateEnvironmentFile(envPath) {
    try {
      const envContent = await fs.readFile(envPath, 'utf8');
      const lines = envContent.split('\n');
      
      const sensitiveKeys = [
        'PASSWORD', 'SECRET', 'KEY', 'TOKEN', 'PRIVATE',
        'DB_PASS', 'API_KEY', 'JWT', 'AUTH', 'CREDENTIAL'
      ];
      
      for (const line of lines) {
        if (line.includes('=') && !line.startsWith('#')) {
          const [key, ...valueParts] = line.split('=');
          const value = valueParts.join('=');
          
          const isSensitive = sensitiveKeys.some(sensitive => 
            key.toUpperCase().includes(sensitive)
          );
          
          if (isSensitive && value && value !== '') {
            const category = this.categorizeKey(key);
            await this.setSecureVariable(key, value, category);
            console.log(`✅ Migrated sensitive key: ${key}`);
          }
        }
      }
      
      // Create backup of original file
      const backupPath = `${envPath}.backup.${Date.now()}`;
      await fs.copyFile(envPath, backupPath);
      console.log(`📁 Backup created: ${backupPath}`);
      
    } catch (error) {
      throw new Error(`Failed to migrate environment file: ${error.message}`);
    }
  }

  /**
   * Categorize environment keys for better organization
   */
  categorizeKey(key) {
    const upperKey = key.toUpperCase();
    
    if (upperKey.includes('DB') || upperKey.includes('DATABASE')) return 'database';
    if (upperKey.includes('API')) return 'api';
    if (upperKey.includes('JWT') || upperKey.includes('AUTH')) return 'auth';
    if (upperKey.includes('SMTP') || upperKey.includes('EMAIL')) return 'email';
    if (upperKey.includes('VPN') || upperKey.includes('CLOUDFLARE')) return 'network';
    
    return 'general';
  }

  /**
   * Generate secure .env file with public variables only
   */
  async generatePublicEnvFile(outputPath) {
    const publicTemplate = `# Ultimate Media Server 2025 - Public Configuration
# Generated: ${new Date().toISOString()}
# SECURITY: Sensitive values are stored in secure encrypted storage

# Timezone and User Settings
TZ=America/New_York
PUID=1000
PGID=1000

# Paths (Update these to match your system)
CONFIG_PATH=/Users/morlock/fun/newmedia/config
MEDIA_PATH=/Volumes/Plex
DOWNLOADS_PATH=/Volumes/Plex/downloads
LOGS_PATH=/Users/morlock/fun/newmedia/logs
BACKUP_PATH=/Users/morlock/fun/newmedia/backups

# Domain Configuration
DOMAIN=media.local
PUBLIC_DOMAIN=media.yourdomain.com

# Performance Settings
COMPOSE_PARALLEL_LIMIT=10
DOCKER_CLIENT_TIMEOUT=120
COMPOSE_HTTP_TIMEOUT=120

# Features
ENABLE_AI_RECOMMENDATIONS=true
ENABLE_8K_SUPPORT=true
ENABLE_HARDWARE_ACCELERATION=true
ENABLE_DISTRIBUTED_TRANSCODING=true

# SECURITY WARNING: All sensitive credentials (passwords, API keys, tokens)
# are now stored in encrypted secure storage. Use the secure-env-manager to
# access and manage these values.

# To retrieve a secure value: node security/secure-env-manager.js get <key>
# To set a secure value: node security/secure-env-manager.js set <key> <value>
`;

    await fs.writeFile(outputPath, publicTemplate, { mode: 0o644 });
    console.log(`✅ Generated secure public environment file: ${outputPath}`);
  }

  /**
   * CLI interface for secure environment management
   */
  async runCLI() {
    const args = process.argv.slice(2);
    const command = args[0];
    
    await this.initialize();
    
    switch (command) {
      case 'set':
        if (args.length < 3) {
          console.error('Usage: node secure-env-manager.js set <key> <value> [category]');
          process.exit(1);
        }
        await this.setSecureVariable(args[1], args[2], args[3] || 'general');
        console.log(`✅ Secure variable set: ${args[1]}`);
        break;
        
      case 'get':
        if (args.length < 2) {
          console.error('Usage: node secure-env-manager.js get <key> [category]');
          process.exit(1);
        }
        const value = await this.getSecureVariable(args[1], args[2] || 'general');
        if (value) {
          console.log(value);
        } else {
          console.error(`Variable not found: ${args[1]}`);
          process.exit(1);
        }
        break;
        
      case 'migrate':
        if (args.length < 2) {
          console.error('Usage: node secure-env-manager.js migrate <env-file-path>');
          process.exit(1);
        }
        await this.migrateEnvironmentFile(args[1]);
        console.log('✅ Environment file migration completed');
        break;
        
      case 'generate':
        const type = args[1] || 'api-key';
        switch (type) {
          case 'api-key':
            console.log(this.generateApiKey());
            break;
          case 'jwt-secret':
            console.log(this.generateJwtSecret());
            break;
          case 'db-password':
            console.log(this.generateDatabasePassword());
            break;
          default:
            console.error('Unknown generation type. Use: api-key, jwt-secret, db-password');
        }
        break;
        
      case 'public-env':
        const outputPath = args[1] || '.env.public';
        await this.generatePublicEnvFile(outputPath);
        break;
        
      default:
        console.log(`
Secure Environment Manager
Usage: node secure-env-manager.js <command> [args]

Commands:
  set <key> <value> [category]  - Store a secure environment variable
  get <key> [category]          - Retrieve a secure environment variable
  migrate <env-file>            - Migrate existing .env file to secure storage
  generate <type>               - Generate secure credentials (api-key, jwt-secret, db-password)
  public-env [output-file]      - Generate public environment template

Categories: general, database, api, auth, email, network
        `);
    }
  }
}

// Run CLI if called directly
if (require.main === module) {
  const manager = new SecureEnvironmentManager();
  manager.runCLI().catch(console.error);
}

module.exports = SecureEnvironmentManager;