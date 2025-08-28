/**
 * Configuration Manager Service
 * Handles application configuration management
 */

const fs = require('fs').promises;
const path = require('path');
const Joi = require('joi');

class ConfigManager {
    constructor() {
        this.configPath = process.env.CONFIG_PATH || path.join(__dirname, '../../config.json');
        this.config = {
            general: {
                theme: 'dark',
                language: 'en',
                timezone: 'UTC',
                autoStart: true
            },
            services: {
                jellyfin: {
                    enabled: true,
                    port: 8096,
                    autostart: true,
                    healthCheck: true
                },
                sonarr: {
                    enabled: true,
                    port: 8989,
                    autostart: true,
                    healthCheck: true
                },
                radarr: {
                    enabled: true,
                    port: 7878,
                    autostart: true,
                    healthCheck: true
                },
                prowlarr: {
                    enabled: true,
                    port: 9696,
                    autostart: true,
                    healthCheck: true
                }
            },
            security: {
                requireAuth: true,
                sessionTimeout: 3600,
                jwtExpiry: '24h',
                rateLimiting: true
            },
            logging: {
                level: 'info',
                maxFiles: 10,
                maxSize: '10m'
            },
            notifications: {
                email: {
                    enabled: false,
                    smtp: {
                        host: '',
                        port: 587,
                        secure: false,
                        user: '',
                        pass: ''
                    }
                },
                discord: {
                    enabled: false,
                    webhook: ''
                },
                telegram: {
                    enabled: false,
                    botToken: '',
                    chatId: ''
                }
            }
        };
        this.initialized = false;
    }

    async initialize() {
        try {
            await this.loadConfiguration();
            this.initialized = true;
            console.log('ConfigManager initialized successfully');
        } catch (error) {
            console.error('Failed to initialize ConfigManager:', error);
            // Use default config if loading fails
            this.initialized = true;
        }
    }

    async loadConfiguration() {
        try {
            const configData = await fs.readFile(this.configPath, 'utf8');
            const loadedConfig = JSON.parse(configData);
            
            // Merge with defaults to ensure all required fields exist
            this.config = this.mergeConfig(this.config, loadedConfig);
            
            console.log('Configuration loaded from:', this.configPath);
            return this.config;
        } catch (error) {
            if (error.code === 'ENOENT') {
                console.log('Config file not found, creating default configuration');
                await this.saveConfiguration();
            } else {
                console.error('Failed to load configuration:', error);
                throw error;
            }
        }
    }

    async saveConfiguration() {
        try {
            const configDir = path.dirname(this.configPath);
            await fs.mkdir(configDir, { recursive: true });
            
            await fs.writeFile(
                this.configPath, 
                JSON.stringify(this.config, null, 2),
                'utf8'
            );
            
            console.log('Configuration saved to:', this.configPath);
            return true;
        } catch (error) {
            console.error('Failed to save configuration:', error);
            throw error;
        }
    }

    getConfiguration() {
        return this.config;
    }

    async updateConfiguration(updates) {
        try {
            // Validate updates
            const validation = this.validateConfiguration(updates);
            if (!validation.valid) {
                throw new Error('Invalid configuration: ' + validation.errors.join(', '));
            }

            // Merge updates with existing config
            this.config = this.mergeConfig(this.config, updates);
            
            // Save to disk
            await this.saveConfiguration();
            
            return {
                success: true,
                config: this.config,
                timestamp: new Date().toISOString()
            };
        } catch (error) {
            throw new Error('Failed to update configuration: ' + error.message);
        }
    }

    validateConfiguration(config) {
        const schema = Joi.object({
            general: Joi.object({
                theme: Joi.string().valid('light', 'dark', 'auto').optional(),
                language: Joi.string().min(2).max(5).optional(),
                timezone: Joi.string().optional(),
                autoStart: Joi.boolean().optional()
            }).optional(),
            services: Joi.object().pattern(
                Joi.string(),
                Joi.object({
                    enabled: Joi.boolean().optional(),
                    port: Joi.number().port().optional(),
                    autostart: Joi.boolean().optional(),
                    healthCheck: Joi.boolean().optional()
                })
            ).optional(),
            security: Joi.object({
                requireAuth: Joi.boolean().optional(),
                sessionTimeout: Joi.number().min(300).max(86400).optional(),
                jwtExpiry: Joi.string().optional(),
                rateLimiting: Joi.boolean().optional()
            }).optional(),
            logging: Joi.object({
                level: Joi.string().valid('error', 'warn', 'info', 'debug').optional(),
                maxFiles: Joi.number().min(1).max(100).optional(),
                maxSize: Joi.string().optional()
            }).optional(),
            notifications: Joi.object().optional()
        });

        const { error } = schema.validate(config);
        
        return {
            valid: !error,
            errors: error ? error.details.map(d => d.message) : []
        };
    }

    getEnvironmentVariables() {
        const envVars = {};
        
        // Extract relevant environment variables
        const relevantEnvs = [
            'NODE_ENV', 'API_PORT', 'JWT_SECRET', 'ADMIN_PASSWORD',
            'DOCKER_PROJECT_PATH', 'DOCKER_COMPOSE_FILE',
            'PLEX_URL', 'PLEX_TOKEN',
            'JELLYFIN_URL', 'JELLYFIN_TOKEN',
            'SONARR_URL', 'SONARR_API_KEY',
            'RADARR_URL', 'RADARR_API_KEY',
            'PROWLARR_URL', 'PROWLARR_API_KEY'
        ];
        
        for (const env of relevantEnvs) {
            if (process.env[env]) {
                // Mask sensitive values
                if (env.includes('SECRET') || env.includes('PASSWORD') || env.includes('TOKEN') || env.includes('KEY')) {
                    envVars[env] = process.env[env].replace(/.(?=.{4})/g, '*');
                } else {
                    envVars[env] = process.env[env];
                }
            }
        }
        
        return envVars;
    }

    getServiceConfiguration(serviceName) {
        return this.config.services[serviceName] || null;
    }

    async updateServiceConfiguration(serviceName, serviceConfig) {
        if (!this.config.services[serviceName]) {
            this.config.services[serviceName] = {};
        }
        
        this.config.services[serviceName] = {
            ...this.config.services[serviceName],
            ...serviceConfig
        };
        
        await this.saveConfiguration();
        return this.config.services[serviceName];
    }

    mergeConfig(target, source) {
        const result = { ...target };
        
        for (const key in source) {
            if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
                result[key] = this.mergeConfig(result[key] || {}, source[key]);
            } else {
                result[key] = source[key];
            }
        }
        
        return result;
    }

    // Backup and restore
    async createBackup() {
        try {
            const backupPath = this.configPath + '.backup.' + Date.now();
            await fs.copyFile(this.configPath, backupPath);
            return backupPath;
        } catch (error) {
            throw new Error('Failed to create backup: ' + error.message);
        }
    }

    async restoreBackup(backupPath) {
        try {
            await fs.copyFile(backupPath, this.configPath);
            await this.loadConfiguration();
            return true;
        } catch (error) {
            throw new Error('Failed to restore backup: ' + error.message);
        }
    }

    // Reset to defaults
    async resetToDefaults() {
        const defaultConfig = {
            general: {
                theme: 'dark',
                language: 'en',
                timezone: 'UTC',
                autoStart: true
            },
            services: {},
            security: {
                requireAuth: true,
                sessionTimeout: 3600,
                jwtExpiry: '24h',
                rateLimiting: true
            },
            logging: {
                level: 'info',
                maxFiles: 10,
                maxSize: '10m'
            },
            notifications: {
                email: { enabled: false },
                discord: { enabled: false },
                telegram: { enabled: false }
            }
        };
        
        this.config = defaultConfig;
        await this.saveConfiguration();
        return this.config;
    }
}

module.exports = ConfigManager;
