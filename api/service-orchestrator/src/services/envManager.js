import crypto from 'crypto';
import fs from 'fs/promises';
import path from 'path';
import { promisify } from 'util';
import { logger } from '../utils/logger.js';
import { VaultClient } from '../integrations/vault.js';
import { ValidationError } from '../utils/errors.js';

const randomBytes = promisify(crypto.randomBytes);

export class EnvironmentManager {
  constructor(configManager, eventBus) {
    this.configManager = configManager;
    this.eventBus = eventBus;
    this.vault = new VaultClient();
    this.envCache = new Map();
    this.encryptionKey = null;
    this.rotationSchedule = new Map();
  }

  async initialize() {
    // Initialize encryption key
    await this.initializeEncryption();
    
    // Load existing environment configurations
    await this.loadEnvironments();
    
    // Start secret rotation scheduler
    this.startRotationScheduler();
    
    logger.info('Environment Manager initialized');
  }

  async initializeEncryption() {
    try {
      // Try to load existing key from vault
      const key = await this.vault.read('config/encryption-key');
      if (key) {
        this.encryptionKey = Buffer.from(key, 'hex');
      } else {
        // Generate new key
        this.encryptionKey = await randomBytes(32);
        await this.vault.write('config/encryption-key', this.encryptionKey.toString('hex'));
      }
    } catch (error) {
      logger.error('Failed to initialize encryption:', error);
      // Fallback to environment variable
      if (process.env.ENCRYPTION_KEY) {
        this.encryptionKey = Buffer.from(process.env.ENCRYPTION_KEY, 'hex');
      } else {
        throw new Error('No encryption key available');
      }
    }
  }

  async createEnvironment(serviceName, environment = 'production', variables = {}) {
    logger.info(`Creating environment for ${serviceName} (${environment})`);

    // Validate environment name
    if (!['development', 'staging', 'production'].includes(environment)) {
      throw new ValidationError(`Invalid environment: ${environment}`);
    }

    // Prepare environment configuration
    const envConfig = {
      service: serviceName,
      environment: environment,
      variables: {},
      secrets: {},
      created: new Date(),
      lastModified: new Date(),
      version: 1
    };

    // Separate secrets from regular variables
    for (const [key, value] of Object.entries(variables)) {
      if (this.isSecret(key)) {
        envConfig.secrets[key] = await this.encryptValue(value);
      } else {
        envConfig.variables[key] = value;
      }
    }

    // Add default variables
    envConfig.variables = {
      ...this.getDefaultVariables(serviceName),
      ...envConfig.variables
    };

    // Store in vault
    const path = `environments/${serviceName}/${environment}`;
    await this.vault.write(path, envConfig);

    // Update cache
    const cacheKey = `${serviceName}:${environment}`;
    this.envCache.set(cacheKey, envConfig);

    // Emit event
    await this.eventBus.emit('environment:created', {
      service: serviceName,
      environment: environment
    });

    return envConfig;
  }

  async getEnvironment(serviceName, environment = 'production') {
    const cacheKey = `${serviceName}:${environment}`;
    
    // Check cache first
    if (this.envCache.has(cacheKey)) {
      return this.decryptEnvironment(this.envCache.get(cacheKey));
    }

    // Load from vault
    const path = `environments/${serviceName}/${environment}`;
    const envConfig = await this.vault.read(path);
    
    if (!envConfig) {
      // Return defaults if no specific config exists
      return this.getDefaultVariables(serviceName);
    }

    // Cache the config
    this.envCache.set(cacheKey, envConfig);

    return this.decryptEnvironment(envConfig);
  }

  async updateEnvironment(serviceName, environment, updates) {
    logger.info(`Updating environment for ${serviceName} (${environment})`);

    const envConfig = await this.vault.read(`environments/${serviceName}/${environment}`);
    if (!envConfig) {
      throw new Error(`Environment not found: ${serviceName}/${environment}`);
    }

    // Backup current config
    await this.backupEnvironment(serviceName, environment, envConfig);

    // Apply updates
    for (const [key, value] of Object.entries(updates)) {
      if (value === null || value === undefined) {
        // Remove variable
        delete envConfig.variables[key];
        delete envConfig.secrets[key];
      } else if (this.isSecret(key)) {
        // Update secret
        envConfig.secrets[key] = await this.encryptValue(value);
        delete envConfig.variables[key]; // Remove from variables if it was there
      } else {
        // Update variable
        envConfig.variables[key] = value;
        delete envConfig.secrets[key]; // Remove from secrets if it was there
      }
    }

    // Update metadata
    envConfig.lastModified = new Date();
    envConfig.version = (envConfig.version || 1) + 1;

    // Store updated config
    const path = `environments/${serviceName}/${environment}`;
    await this.vault.write(path, envConfig);

    // Update cache
    const cacheKey = `${serviceName}:${environment}`;
    this.envCache.set(cacheKey, envConfig);

    // Emit event
    await this.eventBus.emit('environment:updated', {
      service: serviceName,
      environment: environment,
      changes: Object.keys(updates)
    });

    return this.decryptEnvironment(envConfig);
  }

  async deleteEnvironment(serviceName, environment) {
    logger.info(`Deleting environment for ${serviceName} (${environment})`);

    // Backup before deletion
    const envConfig = await this.vault.read(`environments/${serviceName}/${environment}`);
    if (envConfig) {
      await this.backupEnvironment(serviceName, environment, envConfig);
    }

    // Delete from vault
    await this.vault.delete(`environments/${serviceName}/${environment}`);

    // Remove from cache
    const cacheKey = `${serviceName}:${environment}`;
    this.envCache.delete(cacheKey);

    // Emit event
    await this.eventBus.emit('environment:deleted', {
      service: serviceName,
      environment: environment
    });
  }

  async rotateSecret(serviceName, secretKey, environment = 'production') {
    logger.info(`Rotating secret ${secretKey} for ${serviceName} (${environment})`);

    const envConfig = await this.vault.read(`environments/${serviceName}/${environment}`);
    if (!envConfig || !envConfig.secrets[secretKey]) {
      throw new Error(`Secret not found: ${secretKey}`);
    }

    // Generate new secret value
    const newValue = await this.generateSecretValue(secretKey);

    // Update the secret
    const oldEncrypted = envConfig.secrets[secretKey];
    envConfig.secrets[secretKey] = await this.encryptValue(newValue);
    
    // Update version tracking
    if (!envConfig.secretVersions) {
      envConfig.secretVersions = {};
    }
    
    envConfig.secretVersions[secretKey] = {
      current: envConfig.secrets[secretKey],
      previous: oldEncrypted,
      rotatedAt: new Date(),
      version: (envConfig.secretVersions[secretKey]?.version || 0) + 1
    };

    // Save updated config
    const path = `environments/${serviceName}/${environment}`;
    await this.vault.write(path, envConfig);

    // Update cache
    const cacheKey = `${serviceName}:${environment}`;
    this.envCache.set(cacheKey, envConfig);

    // Notify service of secret rotation
    await this.eventBus.emit('secret:rotated', {
      service: serviceName,
      environment: environment,
      secret: secretKey,
      version: envConfig.secretVersions[secretKey].version
    });

    return newValue;
  }

  async scheduleSecretRotation(serviceName, secretKey, intervalDays, environment = 'production') {
    const rotationKey = `${serviceName}:${environment}:${secretKey}`;
    
    // Clear existing schedule if any
    if (this.rotationSchedule.has(rotationKey)) {
      clearInterval(this.rotationSchedule.get(rotationKey));
    }

    // Schedule rotation
    const intervalMs = intervalDays * 24 * 60 * 60 * 1000;
    const intervalId = setInterval(async () => {
      try {
        await this.rotateSecret(serviceName, secretKey, environment);
        logger.info(`Scheduled rotation completed for ${rotationKey}`);
      } catch (error) {
        logger.error(`Failed to rotate secret ${rotationKey}:`, error);
        
        // Alert on rotation failure
        await this.eventBus.emit('alert:critical', {
          type: 'secret_rotation_failed',
          service: serviceName,
          secret: secretKey,
          error: error.message
        });
      }
    }, intervalMs);

    this.rotationSchedule.set(rotationKey, intervalId);
    
    logger.info(`Scheduled secret rotation for ${rotationKey} every ${intervalDays} days`);
  }

  async generateEnvFile(serviceName, environment = 'production', format = 'docker') {
    const env = await this.getEnvironment(serviceName, environment);
    
    let content = '';
    
    switch (format) {
      case 'docker':
        content = this.formatDockerEnv(env);
        break;
      case 'systemd':
        content = this.formatSystemdEnv(env);
        break;
      case 'shell':
        content = this.formatShellEnv(env);
        break;
      case 'kubernetes':
        content = this.formatKubernetesEnv(serviceName, env);
        break;
      default:
        throw new ValidationError(`Unknown format: ${format}`);
    }

    return content;
  }

  async validateEnvironment(serviceName, environment = 'production') {
    const env = await this.getEnvironment(serviceName, environment);
    const errors = [];
    const warnings = [];

    // Get service-specific validation rules
    const rules = this.getValidationRules(serviceName);

    for (const rule of rules) {
      const value = env[rule.key];

      // Check required fields
      if (rule.required && !value) {
        errors.push(`Missing required variable: ${rule.key}`);
        continue;
      }

      if (!value) continue;

      // Type validation
      if (rule.type) {
        if (!this.validateType(value, rule.type)) {
          errors.push(`Invalid type for ${rule.key}: expected ${rule.type}`);
        }
      }

      // Pattern validation
      if (rule.pattern) {
        const regex = new RegExp(rule.pattern);
        if (!regex.test(value)) {
          errors.push(`Invalid format for ${rule.key}`);
        }
      }

      // Range validation
      if (rule.min !== undefined || rule.max !== undefined) {
        const numValue = parseFloat(value);
        if (rule.min !== undefined && numValue < rule.min) {
          errors.push(`${rule.key} is below minimum value: ${rule.min}`);
        }
        if (rule.max !== undefined && numValue > rule.max) {
          errors.push(`${rule.key} is above maximum value: ${rule.max}`);
        }
      }

      // Custom validation
      if (rule.validate) {
        const result = rule.validate(value, env);
        if (result.error) {
          errors.push(result.error);
        }
        if (result.warning) {
          warnings.push(result.warning);
        }
      }
    }

    // Check for sensitive data in non-secret fields
    for (const [key, value] of Object.entries(env)) {
      if (!this.isSecret(key) && this.containsSensitiveData(value)) {
        warnings.push(`Possible sensitive data in non-secret field: ${key}`);
      }
    }

    return {
      valid: errors.length === 0,
      errors,
      warnings
    };
  }

  async exportEnvironment(serviceName, environment = 'production', includeSecrets = false) {
    const env = await this.getEnvironment(serviceName, environment);
    
    const exported = {
      service: serviceName,
      environment: environment,
      exportedAt: new Date(),
      variables: { ...env }
    };

    if (!includeSecrets) {
      // Mask secrets
      for (const key of Object.keys(exported.variables)) {
        if (this.isSecret(key)) {
          exported.variables[key] = '***REDACTED***';
        }
      }
    }

    return exported;
  }

  async importEnvironment(serviceName, environment, data) {
    logger.info(`Importing environment for ${serviceName} (${environment})`);

    // Validate import data
    if (!data.variables) {
      throw new ValidationError('Import data must contain variables');
    }

    // Create or update environment
    await this.createEnvironment(serviceName, environment, data.variables);

    // Emit event
    await this.eventBus.emit('environment:imported', {
      service: serviceName,
      environment: environment,
      variableCount: Object.keys(data.variables).length
    });
  }

  // Private helper methods

  isSecret(key) {
    const secretPatterns = [
      /password/i,
      /secret/i,
      /key/i,
      /token/i,
      /api[-_]?key/i,
      /private/i,
      /credential/i,
      /auth/i
    ];

    return secretPatterns.some(pattern => pattern.test(key));
  }

  async encryptValue(value) {
    const iv = await randomBytes(16);
    const cipher = crypto.createCipheriv('aes-256-gcm', this.encryptionKey, iv);
    
    let encrypted = cipher.update(value, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    const authTag = cipher.getAuthTag();
    
    return {
      encrypted,
      iv: iv.toString('hex'),
      authTag: authTag.toString('hex')
    };
  }

  async decryptValue(encryptedData) {
    const decipher = crypto.createDecipheriv(
      'aes-256-gcm',
      this.encryptionKey,
      Buffer.from(encryptedData.iv, 'hex')
    );
    
    decipher.setAuthTag(Buffer.from(encryptedData.authTag, 'hex'));
    
    let decrypted = decipher.update(encryptedData.encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');
    
    return decrypted;
  }

  async decryptEnvironment(envConfig) {
    const decrypted = { ...envConfig.variables };

    // Decrypt secrets
    for (const [key, encryptedValue] of Object.entries(envConfig.secrets || {})) {
      try {
        decrypted[key] = await this.decryptValue(encryptedValue);
      } catch (error) {
        logger.error(`Failed to decrypt secret ${key}:`, error);
        decrypted[key] = null;
      }
    }

    return decrypted;
  }

  getDefaultVariables(serviceName) {
    const defaults = {
      // Common variables
      TZ: process.env.TZ || 'UTC',
      PUID: process.env.PUID || '1000',
      PGID: process.env.PGID || '1000',
      NODE_ENV: process.env.NODE_ENV || 'production',
      
      // Service-specific defaults
      [`${serviceName.toUpperCase()}_LOG_LEVEL`]: 'info',
      [`${serviceName.toUpperCase()}_DATA_DIR`]: `/config/${serviceName}`
    };

    // Add service-specific defaults
    const serviceDefaults = this.getServiceDefaults(serviceName);
    
    return { ...defaults, ...serviceDefaults };
  }

  getServiceDefaults(serviceName) {
    const serviceDefaults = {
      jellyfin: {
        JELLYFIN_PublishedServerUrl: 'http://localhost:8096',
        JELLYFIN_CACHE_DIR: '/cache'
      },
      sonarr: {
        SONARR_BRANCH: 'main',
        SONARR_API_KEY: () => this.generateApiKey()
      },
      radarr: {
        RADARR_BRANCH: 'master',
        RADARR_API_KEY: () => this.generateApiKey()
      },
      qbittorrent: {
        WEBUI_PORT: '8080',
        TORRENTING_PORT: '6881'
      },
      grafana: {
        GF_SECURITY_ADMIN_USER: 'admin',
        GF_SECURITY_ADMIN_PASSWORD: () => this.generatePassword(),
        GF_INSTALL_PLUGINS: 'grafana-clock-panel,grafana-simple-json-datasource'
      }
    };

    const defaults = serviceDefaults[serviceName] || {};
    
    // Resolve function values
    const resolved = {};
    for (const [key, value] of Object.entries(defaults)) {
      resolved[key] = typeof value === 'function' ? value() : value;
    }
    
    return resolved;
  }

  async generateSecretValue(key) {
    const secretType = this.getSecretType(key);
    
    switch (secretType) {
      case 'password':
        return this.generatePassword();
      case 'api_key':
        return this.generateApiKey();
      case 'token':
        return this.generateToken();
      case 'certificate':
        return this.generateCertificate();
      default:
        return this.generateRandomString(32);
    }
  }

  getSecretType(key) {
    if (/password/i.test(key)) return 'password';
    if (/api[-_]?key/i.test(key)) return 'api_key';
    if (/token/i.test(key)) return 'token';
    if (/cert/i.test(key)) return 'certificate';
    return 'generic';
  }

  generatePassword(length = 16) {
    const charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()_+-=[]{}|;:,.<>?';
    let password = '';
    
    const randomValues = crypto.randomBytes(length);
    for (let i = 0; i < length; i++) {
      password += charset[randomValues[i] % charset.length];
    }
    
    return password;
  }

  generateApiKey() {
    return crypto.randomBytes(32).toString('hex');
  }

  generateToken() {
    return crypto.randomBytes(24).toString('base64url');
  }

  generateCertificate() {
    // This would normally generate a proper certificate
    // For now, return a placeholder
    return '-----BEGIN CERTIFICATE-----\n[CERTIFICATE_DATA]\n-----END CERTIFICATE-----';
  }

  generateRandomString(length) {
    return crypto.randomBytes(Math.ceil(length / 2)).toString('hex').slice(0, length);
  }

  formatDockerEnv(env) {
    let content = '# Generated by Ultimate Media Server Environment Manager\n';
    content += `# Generated at: ${new Date().toISOString()}\n\n`;
    
    for (const [key, value] of Object.entries(env)) {
      if (value !== null && value !== undefined) {
        content += `${key}=${this.escapeEnvValue(value)}\n`;
      }
    }
    
    return content;
  }

  formatSystemdEnv(env) {
    let content = '';
    
    for (const [key, value] of Object.entries(env)) {
      if (value !== null && value !== undefined) {
        content += `Environment="${key}=${this.escapeEnvValue(value)}"\n`;
      }
    }
    
    return content;
  }

  formatShellEnv(env) {
    let content = '#!/bin/bash\n';
    content += '# Generated by Ultimate Media Server Environment Manager\n\n';
    
    for (const [key, value] of Object.entries(env)) {
      if (value !== null && value !== undefined) {
        content += `export ${key}="${this.escapeShellValue(value)}"\n`;
      }
    }
    
    return content;
  }

  formatKubernetesEnv(serviceName, env) {
    const config = {
      apiVersion: 'v1',
      kind: 'ConfigMap',
      metadata: {
        name: `${serviceName}-config`,
        labels: {
          app: serviceName,
          'managed-by': 'media-server-orchestrator'
        }
      },
      data: {}
    };

    const secret = {
      apiVersion: 'v1',
      kind: 'Secret',
      metadata: {
        name: `${serviceName}-secrets`,
        labels: {
          app: serviceName,
          'managed-by': 'media-server-orchestrator'
        }
      },
      type: 'Opaque',
      data: {}
    };

    for (const [key, value] of Object.entries(env)) {
      if (this.isSecret(key)) {
        secret.data[key] = Buffer.from(value).toString('base64');
      } else {
        config.data[key] = value;
      }
    }

    return `${JSON.stringify(config, null, 2)}\n---\n${JSON.stringify(secret, null, 2)}`;
  }

  escapeEnvValue(value) {
    // Escape special characters for .env files
    if (typeof value !== 'string') {
      value = String(value);
    }
    
    if (value.includes(' ') || value.includes('"') || value.includes("'")) {
      return `"${value.replace(/"/g, '\\"')}"`;
    }
    
    return value;
  }

  escapeShellValue(value) {
    // Escape special characters for shell scripts
    if (typeof value !== 'string') {
      value = String(value);
    }
    
    return value.replace(/"/g, '\\"').replace(/\$/g, '\\$');
  }

  validateType(value, type) {
    switch (type) {
      case 'string':
        return typeof value === 'string';
      case 'number':
        return !isNaN(parseFloat(value));
      case 'boolean':
        return ['true', 'false', '1', '0'].includes(value.toLowerCase());
      case 'url':
        try {
          new URL(value);
          return true;
        } catch {
          return false;
        }
      case 'email':
        return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(value);
      case 'port':
        const port = parseInt(value);
        return !isNaN(port) && port > 0 && port <= 65535;
      default:
        return true;
    }
  }

  containsSensitiveData(value) {
    if (typeof value !== 'string') return false;
    
    const sensitivePatterns = [
      /^[a-zA-Z0-9]{32,}$/, // Long random strings
      /^[A-Za-z0-9+/]{20,}={0,2}$/, // Base64 encoded
      /^-----BEGIN/, // Certificates/Keys
      /^ey[A-Za-z0-9_-]+\.ey[A-Za-z0-9_-]+\./, // JWT tokens
    ];
    
    return sensitivePatterns.some(pattern => pattern.test(value));
  }

  getValidationRules(serviceName) {
    const commonRules = [
      { key: 'TZ', type: 'string', pattern: '^[A-Za-z]+/[A-Za-z_]+$' },
      { key: 'PUID', type: 'number', min: 0 },
      { key: 'PGID', type: 'number', min: 0 }
    ];

    const serviceRules = {
      jellyfin: [
        { key: 'JELLYFIN_PublishedServerUrl', type: 'url', required: true }
      ],
      sonarr: [
        { key: 'SONARR_API_KEY', type: 'string', pattern: '^[a-f0-9]{32}$', required: true }
      ],
      radarr: [
        { key: 'RADARR_API_KEY', type: 'string', pattern: '^[a-f0-9]{32}$', required: true }
      ],
      grafana: [
        { key: 'GF_SECURITY_ADMIN_USER', type: 'string', required: true },
        { key: 'GF_SECURITY_ADMIN_PASSWORD', type: 'string', required: true, 
          validate: (value) => {
            if (value.length < 8) {
              return { error: 'Admin password must be at least 8 characters' };
            }
            if (value === 'admin' || value === 'password') {
              return { warning: 'Weak admin password detected' };
            }
            return {};
          }
        }
      ],
      traefik: [
        { key: 'CF_API_EMAIL', type: 'email', required: true },
        { key: 'CF_API_KEY', type: 'string', required: true }
      ]
    };

    return [...commonRules, ...(serviceRules[serviceName] || [])];
  }

  async backupEnvironment(serviceName, environment, config) {
    const backupPath = `environments/${serviceName}/${environment}/backups/${Date.now()}`;
    await this.vault.write(backupPath, config);
    
    // Keep only last 10 backups
    const backups = await this.vault.list(`environments/${serviceName}/${environment}/backups`);
    if (backups.length > 10) {
      const toDelete = backups.sort().slice(0, backups.length - 10);
      for (const backup of toDelete) {
        await this.vault.delete(`environments/${serviceName}/${environment}/backups/${backup}`);
      }
    }
  }

  async loadEnvironments() {
    try {
      const services = await this.vault.list('environments');
      
      for (const service of services) {
        const environments = await this.vault.list(`environments/${service}`);
        
        for (const env of environments) {
          if (env !== 'backups') {
            const config = await this.vault.read(`environments/${service}/${env}`);
            if (config) {
              const cacheKey = `${service}:${env}`;
              this.envCache.set(cacheKey, config);
            }
          }
        }
      }
      
      logger.info(`Loaded ${this.envCache.size} environment configurations`);
    } catch (error) {
      logger.error('Failed to load environments:', error);
    }
  }

  startRotationScheduler() {
    // Check for scheduled rotations in configuration
    setInterval(async () => {
      try {
        const rotationConfigs = await this.vault.list('config/rotations');
        
        for (const configName of rotationConfigs) {
          const config = await this.vault.read(`config/rotations/${configName}`);
          if (config && config.enabled) {
            await this.scheduleSecretRotation(
              config.service,
              config.secret,
              config.intervalDays,
              config.environment
            );
          }
        }
      } catch (error) {
        logger.error('Failed to process rotation schedules:', error);
      }
    }, 3600000); // Check every hour
  }
}