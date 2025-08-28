#!/usr/bin/env node

/**
 * Secure Secrets Management System
 * Fixes: Hardcoded secrets, insecure credential storage
 * Author: Security Manager Agent
 * Date: 2025-08-03
 */

const crypto = require('crypto');
const fs = require('fs').promises;
const path = require('path');
const readline = require('readline');

class SecretsManager {
  constructor(options = {}) {
    this.secretsDir = options.secretsDir || './security/secrets';
    this.vaultFile = path.join(this.secretsDir, 'vault.encrypted');
    this.keyFile = path.join(this.secretsDir, '.vault.key');
    this.algorithm = 'aes-256-gcm';
    this.keyDerivationRounds = 100000;
    this.masterKey = null;
    this.vault = {};
  }

  /**
   * Initialize secrets manager
   */
  async initialize(masterPassword = null) {
    try {
      await fs.mkdir(this.secretsDir, { recursive: true, mode: 0o700 });
      
      if (await this.vaultExists()) {
        await this.unlockVault(masterPassword);
      } else {
        await this.createVault(masterPassword);
      }
      
      console.log('✅ Secrets manager initialized');
    } catch (error) {
      throw new Error(`Failed to initialize secrets manager: ${error.message}`);
    }
  }

  /**
   * Check if vault exists
   */
  async vaultExists() {
    try {
      await fs.access(this.vaultFile);
      return true;
    } catch {
      return false;
    }
  }

  /**
   * Create new vault
   */
  async createVault(masterPassword = null) {
    if (!masterPassword) {
      masterPassword = await this.promptForPassword('Create master password: ');
    }
    
    // Derive key from password
    const salt = crypto.randomBytes(32);
    this.masterKey = crypto.pbkdf2Sync(masterPassword, salt, this.keyDerivationRounds, 32, 'sha256');
    
    // Store salt for future key derivation
    await fs.writeFile(this.keyFile, salt, { mode: 0o600 });
    
    // Initialize empty vault
    this.vault = {
      version: '1.0',
      created: new Date().toISOString(),
      secrets: {}
    };
    
    await this.saveVault();
    console.log('✅ New vault created');
  }

  /**
   * Unlock existing vault
   */
  async unlockVault(masterPassword = null) {
    if (!masterPassword) {
      masterPassword = await this.promptForPassword('Enter master password: ');
    }
    
    // Load salt and derive key
    const salt = await fs.readFile(this.keyFile);
    this.masterKey = crypto.pbkdf2Sync(masterPassword, salt, this.keyDerivationRounds, 32, 'sha256');
    
    await this.loadVault();
    console.log('✅ Vault unlocked');
  }

  /**
   * Save vault to encrypted file
   */
  async saveVault() {
    const vaultData = JSON.stringify(this.vault);
    const encrypted = this.encrypt(vaultData);
    
    const vaultContainer = {
      iv: encrypted.iv,
      authTag: encrypted.authTag,
      data: encrypted.encrypted
    };
    
    await fs.writeFile(this.vaultFile, JSON.stringify(vaultContainer), { mode: 0o600 });
  }

  /**
   * Load vault from encrypted file
   */
  async loadVault() {
    try {
      const vaultContainer = JSON.parse(await fs.readFile(this.vaultFile, 'utf8'));
      const decrypted = this.decrypt({
        encrypted: vaultContainer.data,
        iv: vaultContainer.iv,
        authTag: vaultContainer.authTag
      });
      
      this.vault = JSON.parse(decrypted);
    } catch (error) {
      throw new Error('Failed to decrypt vault - incorrect password or corrupted file');
    }
  }

  /**
   * Encrypt data
   */
  encrypt(plaintext) {
    const iv = crypto.randomBytes(16);
    const cipher = crypto.createCipher(this.algorithm, this.masterKey);
    cipher.setAAD(Buffer.from('secrets-vault'));
    
    let encrypted = cipher.update(plaintext, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: cipher.getAuthTag().toString('hex')
    };
  }

  /**
   * Decrypt data
   */
  decrypt(encryptedData) {
    const decipher = crypto.createDecipher(this.algorithm, this.masterKey);
    decipher.setAAD(Buffer.from('secrets-vault'));
    decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));
    
    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }

  /**
   * Store secret
   */
  async storeSecret(name, value, category = 'general', metadata = {}) {
    if (!this.vault.secrets[category]) {
      this.vault.secrets[category] = {};
    }
    
    this.vault.secrets[category][name] = {
      value,
      created: new Date().toISOString(),
      lastModified: new Date().toISOString(),
      metadata
    };
    
    await this.saveVault();
    console.log(`✅ Secret stored: ${category}/${name}`);
  }

  /**
   * Retrieve secret
   */
  getSecret(name, category = 'general') {
    if (!this.vault.secrets[category] || !this.vault.secrets[category][name]) {
      return null;
    }
    
    return this.vault.secrets[category][name].value;
  }

  /**
   * List all secrets
   */
  listSecrets(category = null) {
    if (category) {
      return Object.keys(this.vault.secrets[category] || {});
    }
    
    const allSecrets = {};
    for (const [cat, secrets] of Object.entries(this.vault.secrets)) {
      allSecrets[cat] = Object.keys(secrets);
    }
    return allSecrets;
  }

  /**
   * Delete secret
   */
  async deleteSecret(name, category = 'general') {
    if (this.vault.secrets[category] && this.vault.secrets[category][name]) {
      delete this.vault.secrets[category][name];
      await this.saveVault();
      console.log(`✅ Secret deleted: ${category}/${name}`);
      return true;
    }
    return false;
  }

  /**
   * Generate secure passwords/keys
   */
  generateSecret(type = 'password', length = 32) {
    switch (type) {
      case 'password':
        return this.generatePassword(length);
      case 'api-key':
        return crypto.randomBytes(length).toString('hex');
      case 'jwt-secret':
        return crypto.randomBytes(64).toString('base64url');
      case 'encryption-key':
        return crypto.randomBytes(32).toString('hex');
      case 'uuid':
        return crypto.randomUUID();
      default:
        throw new Error(`Unknown secret type: ${type}`);
    }
  }

  /**
   * Generate secure password
   */
  generatePassword(length = 32) {
    const lowercase = 'abcdefghijklmnopqrstuvwxyz';
    const uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
    const numbers = '0123456789';
    const symbols = '!@#$%^&*()_+-=[]{}|;:,.<>?';
    
    const allChars = lowercase + uppercase + numbers + symbols;
    let password = '';
    
    // Ensure at least one character from each set
    password += lowercase[Math.floor(Math.random() * lowercase.length)];
    password += uppercase[Math.floor(Math.random() * uppercase.length)];
    password += numbers[Math.floor(Math.random() * numbers.length)];
    password += symbols[Math.floor(Math.random() * symbols.length)];
    
    // Fill the rest randomly
    for (let i = 4; i < length; i++) {
      password += allChars[Math.floor(Math.random() * allChars.length)];
    }
    
    // Shuffle the password
    return password.split('').sort(() => Math.random() - 0.5).join('');
  }

  /**
   * Export secrets to files for Docker secrets
   */
  async exportDockerSecrets(outputDir = './secrets') {
    await fs.mkdir(outputDir, { recursive: true, mode: 0o700 });
    
    const exported = [];
    
    for (const [category, secrets] of Object.entries(this.vault.secrets)) {
      for (const [name, secretData] of Object.entries(secrets)) {
        const filename = `${category}_${name}`.toLowerCase().replace(/[^a-z0-9_]/g, '_');
        const filepath = path.join(outputDir, `${filename}.txt`);
        
        await fs.writeFile(filepath, secretData.value, { mode: 0o600 });
        exported.push({ category, name, filename, filepath });
      }
    }
    
    console.log(`✅ Exported ${exported.length} secrets to ${outputDir}`);
    return exported;
  }

  /**
   * Import secrets from environment variables
   */
  async importFromEnv(envFile) {
    try {
      const envContent = await fs.readFile(envFile, 'utf8');
      const lines = envContent.split('\n');
      
      const sensitivePatterns = [
        /password/i, /secret/i, /key/i, /token/i, /private/i,
        /credential/i, /auth/i, /api/i, /jwt/i
      ];
      
      let imported = 0;
      
      for (const line of lines) {
        if (line.includes('=') && !line.startsWith('#')) {
          const [key, ...valueParts] = line.split('=');
          const value = valueParts.join('=').trim();
          
          if (value && sensitivePatterns.some(pattern => pattern.test(key))) {
            const category = this.categorizeEnvKey(key);
            await this.storeSecret(key, value, category, {
              importedFrom: envFile,
              importedAt: new Date().toISOString()
            });
            imported++;
          }
        }
      }
      
      console.log(`✅ Imported ${imported} secrets from ${envFile}`);
    } catch (error) {
      throw new Error(`Failed to import from env file: ${error.message}`);
    }
  }

  /**
   * Categorize environment variable key
   */
  categorizeEnvKey(key) {
    const upperKey = key.toUpperCase();
    
    if (upperKey.includes('DB') || upperKey.includes('DATABASE')) return 'database';
    if (upperKey.includes('API')) return 'api';
    if (upperKey.includes('JWT') || upperKey.includes('AUTH') || upperKey.includes('SESSION')) return 'auth';
    if (upperKey.includes('SMTP') || upperKey.includes('EMAIL')) return 'email';
    if (upperKey.includes('VPN') || upperKey.includes('CLOUDFLARE')) return 'network';
    if (upperKey.includes('REDIS') || upperKey.includes('CACHE')) return 'cache';
    
    return 'general';
  }

  /**
   * Rotate all secrets
   */
  async rotateSecrets(categories = null) {
    const categoriesToRotate = categories || Object.keys(this.vault.secrets);
    const rotated = [];
    
    for (const category of categoriesToRotate) {
      if (!this.vault.secrets[category]) continue;
      
      for (const [name, secretData] of Object.entries(this.vault.secrets[category])) {
        // Determine secret type and generate new value
        let newValue;
        const oldValue = secretData.value;
        
        if (name.toLowerCase().includes('password')) {
          newValue = this.generatePassword();
        } else if (name.toLowerCase().includes('api') || name.toLowerCase().includes('key')) {
          newValue = this.generateSecret('api-key');
        } else if (name.toLowerCase().includes('jwt') || name.toLowerCase().includes('secret')) {
          newValue = this.generateSecret('jwt-secret');
        } else {
          // Skip rotation for unknown types
          continue;
        }
        
        // Update secret
        this.vault.secrets[category][name] = {
          ...secretData,
          value: newValue,
          lastModified: new Date().toISOString(),
          rotated: true,
          previousValue: oldValue // For rollback if needed
        };
        
        rotated.push({ category, name });
      }
    }
    
    await this.saveVault();
    console.log(`✅ Rotated ${rotated.length} secrets`);
    return rotated;
  }

  /**
   * Create vault backup
   */
  async createBackup(backupPath = null) {
    if (!backupPath) {
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      backupPath = path.join(this.secretsDir, `vault-backup-${timestamp}.encrypted`);
    }
    
    await fs.copyFile(this.vaultFile, backupPath);
    console.log(`✅ Vault backup created: ${backupPath}`);
    return backupPath;
  }

  /**
   * Prompt for password (CLI only)
   */
  async promptForPassword(prompt) {
    if (process.env.NODE_ENV === 'test') {
      return 'test-password';
    }
    
    const rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    
    return new Promise((resolve) => {
      // Hide password input
      const stdin = process.openStdin();
      process.stdout.write(prompt);
      stdin.setRawMode(true);
      stdin.resume();
      stdin.setEncoding('utf8');
      
      let password = '';
      stdin.on('data', (char) => {
        char = char + '';
        
        switch (char) {
          case '\n':
          case '\r':
          case '\u0004': // Ctrl+D
            stdin.setRawMode(false);
            stdin.pause();
            process.stdout.write('\n');
            rl.close();
            resolve(password);
            break;
          case '\u0003': // Ctrl+C
            process.exit();
            break;
          case '\u007f': // Backspace
            if (password.length > 0) {
              password = password.slice(0, -1);
              process.stdout.write('\b \b');
            }
            break;
          default:
            password += char;
            process.stdout.write('*');
            break;
        }
      });
    });
  }

  /**
   * CLI interface
   */
  async runCLI() {
    const args = process.argv.slice(2);
    const command = args[0];
    
    try {
      await this.initialize();
      
      switch (command) {
        case 'store':
          if (args.length < 3) {
            console.error('Usage: secrets-manager store <name> <value> [category]');
            process.exit(1);
          }
          await this.storeSecret(args[1], args[2], args[3] || 'general');
          break;
          
        case 'get':
          if (args.length < 2) {
            console.error('Usage: secrets-manager get <name> [category]');
            process.exit(1);
          }
          const value = this.getSecret(args[1], args[2] || 'general');
          if (value) {
            console.log(value);
          } else {
            console.error(`Secret not found: ${args[1]}`);
            process.exit(1);
          }
          break;
          
        case 'list':
          const secrets = this.listSecrets(args[1]);
          console.log(JSON.stringify(secrets, null, 2));
          break;
          
        case 'generate':
          const type = args[1] || 'password';
          const length = parseInt(args[2]) || 32;
          console.log(this.generateSecret(type, length));
          break;
          
        case 'delete':
          if (args.length < 2) {
            console.error('Usage: secrets-manager delete <name> [category]');
            process.exit(1);
          }
          const deleted = await this.deleteSecret(args[1], args[2] || 'general');
          if (!deleted) {
            console.error(`Secret not found: ${args[1]}`);
            process.exit(1);
          }
          break;
          
        case 'import':
          if (args.length < 2) {
            console.error('Usage: secrets-manager import <env-file>');
            process.exit(1);
          }
          await this.importFromEnv(args[1]);
          break;
          
        case 'export':
          const outputDir = args[1] || './secrets';
          await this.exportDockerSecrets(outputDir);
          break;
          
        case 'rotate':
          const categories = args.slice(1);
          await this.rotateSecrets(categories.length > 0 ? categories : null);
          break;
          
        case 'backup':
          await this.createBackup(args[1]);
          break;
          
        default:
          console.log(`
Secrets Manager
Usage: node secrets-manager.js <command> [args]

Commands:
  store <name> <value> [category]   - Store a secret
  get <name> [category]             - Retrieve a secret
  list [category]                   - List all secrets
  generate <type> [length]          - Generate secure credentials
  delete <name> [category]          - Delete a secret
  import <env-file>                 - Import from .env file
  export [output-dir]               - Export secrets for Docker
  rotate [categories...]            - Rotate secrets
  backup [backup-path]              - Create vault backup

Types: password, api-key, jwt-secret, encryption-key, uuid
Categories: general, database, api, auth, email, network, cache
          `);
      }
    } catch (error) {
      console.error('❌ Error:', error.message);
      process.exit(1);
    }
  }
}

// Run CLI if called directly
if (require.main === module) {
  const manager = new SecretsManager();
  manager.runCLI().catch(console.error);
}

module.exports = SecretsManager;