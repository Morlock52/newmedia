/**
 * Configuration Manager Service
 * Comprehensive configuration management with .env parsing, validation, and templates
 */

const fs = require('fs').promises;
const path = require('path');
const Joi = require('joi');
const crypto = require('crypto');

class ConfigManager {
    constructor() {
        this.projectPath = process.env.DOCKER_PROJECT_PATH || path.join(__dirname, '../../');
        this.envPath = path.join(this.projectPath, '.env');
        this.envTemplatePath = path.join(this.projectPath, 'env.template');
        this.configCache = new Map();
        this.secretsPath = path.join(this.projectPath, 'secrets');
        
        // Configuration schema for validation
        this.configSchema = Joi.object({
            general: Joi.object({
                PUID: Joi.number().integer().min(1).default(1000),
                PGID: Joi.number().integer().min(1).default(1000),
                TZ: Joi.string().default('America/New_York'),
                DOMAIN: Joi.string().domain().optional()
            }),
            paths: Joi.object({
                MEDIA_PATH: Joi.string().required(),
                DOWNLOADS_PATH: Joi.string().required(),
                USENET_PATH: Joi.string().optional()
            }),
            authentication: Joi.object({
                AUTHELIA_JWT_SECRET: Joi.string().length(64).optional(),
                AUTHELIA_SESSION_SECRET: Joi.string().length(64).optional(),
                AUTHELIA_STORAGE_ENCRYPTION_KEY: Joi.string().length(64).optional(),
                AUTHELIA_DEFAULT_USER: Joi.string().default('admin'),
                AUTHELIA_DEFAULT_PASSWORD_HASH: Joi.string().optional()
            }),
            apiKeys: Joi.object({
                SONARR_API_KEY: Joi.string().optional(),
                RADARR_API_KEY: Joi.string().optional(),
                PROWLARR_API_KEY: Joi.string().optional(),
                JELLYFIN_API_KEY: Joi.string().optional(),
                TMDB_API_KEY: Joi.string().optional()
            }),
            cloudflare: Joi.object({
                CLOUDFLARE_EMAIL: Joi.string().email().optional(),
                CLOUDFLARE_API_TOKEN: Joi.string().optional()
            }),
            monitoring: Joi.object({
                GRAFANA_USER: Joi.string().default('admin'),
                GRAFANA_PASSWORD: Joi.string().min(8).default('changeme')
            }),
            vpn: Joi.object({
                VPN_PROVIDER: Joi.string().optional(),
                VPN_USER: Joi.string().optional(),
                VPN_PASSWORD: Joi.string().optional(),
                VPN_REGION: Joi.string().optional(),
                VPN_PRIVATE_KEY: Joi.string().optional(),
                VPN_ADDRESSES: Joi.string().optional()
            }),
            smtp: Joi.object({
                SMTP_HOST: Joi.string().optional(),
                SMTP_PORT: Joi.number().port().optional(),
                SMTP_USER: Joi.string().optional(),
                SMTP_PASSWORD: Joi.string().optional()
            })
        });

        // Configuration templates for different setups
        this.templates = {
            minimal: {
                description: 'Minimal setup with essential services only',
                config: {
                    general: { PUID: 1000, PGID: 1000, TZ: 'America/New_York' },
                    paths: { MEDIA_PATH: './media-data', DOWNLOADS_PATH: './media-data/downloads' }
                }
            },
            standard: {
                description: 'Standard setup with media management',
                config: {
                    general: { PUID: 1000, PGID: 1000, TZ: 'America/New_York' },
                    paths: { MEDIA_PATH: './media-data', DOWNLOADS_PATH: './media-data/downloads' },
                    monitoring: { GRAFANA_USER: 'admin', GRAFANA_PASSWORD: 'changeme' }
                }
            },
            advanced: {
                description: 'Advanced setup with all features',
                config: {
                    general: { PUID: 1000, PGID: 1000, TZ: 'America/New_York', DOMAIN: 'example.com' },
                    paths: { MEDIA_PATH: './media-data', DOWNLOADS_PATH: './media-data/downloads' },
                    authentication: { AUTHELIA_DEFAULT_USER: 'admin' },
                    monitoring: { GRAFANA_USER: 'admin', GRAFANA_PASSWORD: 'changeme' }
                }
            }
        };
    }

    async initialize() {
        try {
            // Ensure secrets directory exists
            await this.ensureSecretsDirectory();
            
            // Load current configuration
            await this.loadConfiguration();
            
            console.log('ConfigManager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize ConfigManager:', error);
            throw error;
        }
    }

    async ensureSecretsDirectory() {
        try {
            await fs.access(this.secretsPath);
        } catch (error) {
            await fs.mkdir(this.secretsPath, { recursive: true });
            console.log('Created secrets directory');
        }
    }

    async loadConfiguration() {
        try {
            // Try to read existing .env file
            let envContent = '';
            try {
                envContent = await fs.readFile(this.envPath, 'utf8');
            } catch (error) {
                // If .env doesn't exist, try to use template
                try {
                    envContent = await fs.readFile(this.envTemplatePath, 'utf8');
                    console.log('Using template configuration');
                } catch (templateError) {
                    console.log('No configuration file found, using defaults');
                }
            }

            // Parse environment variables
            this.currentConfig = this.parseEnvContent(envContent);
            
            // Cache the configuration
            this.configCache.set('current', {
                data: this.currentConfig,
                timestamp: Date.now()
            });

            return this.currentConfig;
        } catch (error) {
            throw new Error('Failed to load configuration: ' + error.message);
        }
    }

    parseEnvContent(content) {
        const config = {
            general: {},
            paths: {},
            authentication: {},
            apiKeys: {},
            cloudflare: {},
            monitoring: {},
            vpn: {},
            smtp: {}
        };

        const lines = content.split('\n');
        
        for (const line of lines) {
            const trimmed = line.trim();
            
            // Skip comments and empty lines
            if (!trimmed || trimmed.startsWith('#')) continue;
            
            const [key, ...valueParts] = trimmed.split('=');
            const value = valueParts.join('=').trim();
            
            // Remove quotes if present
            const cleanValue = value.replace(/^["']|["']$/g, '');
            
            // Categorize the configuration key
            this.categorizeConfigKey(key.trim(), cleanValue, config);
        }

        return config;
    }

    categorizeConfigKey(key, value, config) {
        // General settings
        if (['PUID', 'PGID', 'TZ', 'DOMAIN'].includes(key)) {
            config.general[key] = key === 'PUID' || key === 'PGID' ? parseInt(value) || 1000 : value;
        }
        // Paths
        else if (key.includes('PATH')) {
            config.paths[key] = value;
        }
        // Authentication
        else if (key.includes('AUTHELIA')) {
            config.authentication[key] = value;
        }
        // API Keys
        else if (key.includes('API_KEY')) {
            config.apiKeys[key] = value;
        }
        // Cloudflare
        else if (key.includes('CLOUDFLARE')) {
            config.cloudflare[key] = value;
        }
        // Monitoring
        else if (key.includes('GRAFANA')) {
            config.monitoring[key] = value;
        }
        // VPN
        else if (key.includes('VPN')) {
            config.vpn[key] = value;
        }
        // SMTP
        else if (key.includes('SMTP')) {
            config.smtp[key] = key === 'SMTP_PORT' ? parseInt(value) || 587 : value;
        }
        // Unknown keys go to general
        else {
            config.general[key] = value;
        }
    }

    async getConfiguration() {
        // Check cache first
        const cached = this.configCache.get('current');
        if (cached && Date.now() - cached.timestamp < 30000) { // 30 second cache
            return cached.data;
        }

        return await this.loadConfiguration();
    }

    async updateConfiguration(newConfig) {
        try {
            // Validate the configuration
            const validation = await this.validateConfiguration(newConfig);
            if (!validation.valid) {
                throw new Error('Configuration validation failed: ' + JSON.stringify(validation.errors));
            }

            // Merge with existing configuration
            const currentConfig = await this.getConfiguration();
            const mergedConfig = this.mergeConfigurations(currentConfig, newConfig);

            // Generate secrets if needed
            await this.generateMissingSecrets(mergedConfig);

            // Convert back to .env format
            const envContent = this.configToEnvFormat(mergedConfig);

            // Backup existing .env file
            await this.backupConfiguration();

            // Write new .env file
            await fs.writeFile(this.envPath, envContent, 'utf8');

            // Update cache
            this.configCache.set('current', {
                data: mergedConfig,
                timestamp: Date.now()
            });

            console.log('Configuration updated successfully');

            return {
                success: true,
                config: mergedConfig,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to update configuration: ' + error.message);
        }
    }

    async validateConfiguration(config) {
        try {
            const { error, value } = this.configSchema.validate(config, {
                allowUnknown: true,
                stripUnknown: false
            });

            if (error) {
                return {
                    valid: false,
                    errors: error.details.map(detail => ({
                        field: detail.path.join('.'),
                        message: detail.message
                    }))
                };
            }

            // Additional validation checks
            const additionalErrors = await this.performAdditionalValidation(value);

            return {
                valid: additionalErrors.length === 0,
                errors: additionalErrors,
                validatedConfig: value
            };
        } catch (error) {
            return {
                valid: false,
                errors: [{ field: 'general', message: error.message }]
            };
        }
    }

    async performAdditionalValidation(config) {
        const errors = [];

        // Check if paths exist and are accessible
        if (config.paths) {
            for (const [key, path] of Object.entries(config.paths)) {
                if (path && !path.startsWith('./')) {
                    try {
                        await fs.access(path);
                    } catch (error) {
                        errors.push({
                            field: `paths.${key}`,
                            message: `Path does not exist or is not accessible: ${path}`
                        });
                    }
                }
            }
        }

        // Validate secrets format
        if (config.authentication) {
            const secrets = ['AUTHELIA_JWT_SECRET', 'AUTHELIA_SESSION_SECRET', 'AUTHELIA_STORAGE_ENCRYPTION_KEY'];
            for (const secret of secrets) {
                const value = config.authentication[secret];
                if (value && (value.length !== 64 || !/^[a-f0-9]+$/i.test(value))) {
                    errors.push({
                        field: `authentication.${secret}`,
                        message: 'Secret must be 64 characters hexadecimal string'
                    });
                }
            }
        }

        return errors;
    }

    mergeConfigurations(current, updates) {
        const merged = JSON.parse(JSON.stringify(current)); // Deep clone

        for (const [category, values] of Object.entries(updates)) {
            if (!merged[category]) {
                merged[category] = {};
            }
            
            for (const [key, value] of Object.entries(values)) {
                merged[category][key] = value;
            }
        }

        return merged;
    }

    async generateMissingSecrets(config) {
        if (!config.authentication) {
            config.authentication = {};
        }

        const secrets = [
            'AUTHELIA_JWT_SECRET',
            'AUTHELIA_SESSION_SECRET',
            'AUTHELIA_STORAGE_ENCRYPTION_KEY'
        ];

        for (const secretKey of secrets) {
            if (!config.authentication[secretKey]) {
                const secret = crypto.randomBytes(32).toString('hex');
                config.authentication[secretKey] = secret;
                
                // Save to secrets file
                const secretFile = path.join(this.secretsPath, `${secretKey.toLowerCase()}.txt`);
                await fs.writeFile(secretFile, secret, 'utf8');
                
                console.log(`Generated ${secretKey}`);
            }
        }
    }

    configToEnvFormat(config) {
        let envContent = '# Media Server Environment Configuration\n';
        envContent += `# Generated on ${new Date().toISOString()}\n\n`;

        const categoryMap = {
            general: 'GENERAL SETTINGS',
            paths: 'PATHS',
            authentication: 'AUTHENTICATION',
            apiKeys: 'API KEYS',
            cloudflare: 'CLOUDFLARE (for SSL/TLS)',
            monitoring: 'MONITORING',
            vpn: 'VPN CONFIGURATION',
            smtp: 'SMTP CONFIGURATION'
        };

        for (const [category, title] of Object.entries(categoryMap)) {
            if (config[category] && Object.keys(config[category]).length > 0) {
                envContent += `# ===========================================\n`;
                envContent += `# ${title}\n`;
                envContent += `# ===========================================\n\n`;

                for (const [key, value] of Object.entries(config[category])) {
                    if (value !== undefined && value !== null && value !== '') {
                        envContent += `${key}=${value}\n`;
                    }
                }
                envContent += '\n';
            }
        }

        return envContent;
    }

    async backupConfiguration() {
        try {
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const backupPath = `${this.envPath}.backup.${timestamp}`;
            
            await fs.copyFile(this.envPath, backupPath);
            console.log(`Configuration backed up to: ${backupPath}`);
        } catch (error) {
            // If backup fails, continue anyway
            console.warn('Failed to backup configuration:', error.message);
        }
    }

    async getEnvironmentVariables() {
        const config = await this.getConfiguration();
        const envVars = {};

        // Flatten the configuration
        for (const [category, values] of Object.entries(config)) {
            for (const [key, value] of Object.entries(values)) {
                envVars[key] = value;
            }
        }

        return envVars;
    }

    async applyTemplate(templateName) {
        if (!this.templates[templateName]) {
            throw new Error(`Template '${templateName}' not found`);
        }

        const template = this.templates[templateName];
        return await this.updateConfiguration(template.config);
    }

    getTemplates() {
        return Object.keys(this.templates).map(name => ({
            name,
            description: this.templates[name].description
        }));
    }

    async generatePasswordHash(password) {
        // This would typically use the same hashing method as Authelia
        // For now, we'll use a basic implementation
        const hash = crypto.createHash('sha256');
        hash.update(password + 'salt'); // In production, use proper salt
        return hash.digest('hex');
    }

    async getSecrets() {
        try {
            const secretFiles = await fs.readdir(this.secretsPath);
            const secrets = {};

            for (const file of secretFiles) {
                if (file.endsWith('.txt')) {
                    const secretName = file.replace('.txt', '').toUpperCase();
                    const secretPath = path.join(this.secretsPath, file);
                    const secretValue = await fs.readFile(secretPath, 'utf8');
                    secrets[secretName] = secretValue.trim();
                }
            }

            return secrets;
        } catch (error) {
            console.error('Failed to load secrets:', error);
            return {};
        }
    }

    async validatePaths() {
        const config = await this.getConfiguration();
        const results = {};

        if (config.paths) {
            for (const [key, path] of Object.entries(config.paths)) {
                try {
                    const stats = await fs.stat(path);
                    results[key] = {
                        path,
                        exists: true,
                        isDirectory: stats.isDirectory(),
                        size: stats.size,
                        modified: stats.mtime
                    };
                } catch (error) {
                    results[key] = {
                        path,
                        exists: false,
                        error: error.message
                    };
                }
            }
        }

        return results;
    }

    async createDirectories() {
        const config = await this.getConfiguration();
        const results = {};

        if (config.paths) {
            for (const [key, path] of Object.entries(config.paths)) {
                try {
                    await fs.mkdir(path, { recursive: true });
                    results[key] = { path, created: true };
                } catch (error) {
                    results[key] = { path, created: false, error: error.message };
                }
            }
        }

        return results;
    }
}

module.exports = ConfigManager;