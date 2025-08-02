/**
 * Advanced Environment Manager API Client
 * Handles all backend communication for the .env management interface
 */

class EnvManagerAPI {
    constructor(options = {}) {
        this.baseURL = options.baseURL || 'http://localhost:3001/api';
        this.timeout = options.timeout || 10000;
        this.retryAttempts = options.retryAttempts || 3;
        this.retryDelay = options.retryDelay || 1000;
        
        // Event emitter for real-time updates
        this.listeners = {};
        
        // Security context
        this.encryptionEnabled = options.encryptionEnabled !== false;
        this.encryptionKey = null;
        
        this.initializeEncryption();
        this.initializeWebSocket();
    }

    // Event system
    on(event, callback) {
        if (!this.listeners[event]) {
            this.listeners[event] = [];
        }
        this.listeners[event].push(callback);
    }

    emit(event, data) {
        if (this.listeners[event]) {
            this.listeners[event].forEach(callback => callback(data));
        }
    }

    // Initialize client-side encryption for sensitive values
    async initializeEncryption() {
        if (!this.encryptionEnabled || !window.crypto || !window.crypto.subtle) {
            console.warn('Encryption not available or disabled');
            return;
        }

        try {
            // Generate or retrieve encryption key
            const stored = localStorage.getItem('env-manager-key');
            if (stored) {
                this.encryptionKey = await window.crypto.subtle.importKey(
                    'jwk',
                    JSON.parse(stored),
                    { name: 'AES-GCM' },
                    true,
                    ['encrypt', 'decrypt']
                );
            } else {
                this.encryptionKey = await window.crypto.subtle.generateKey(
                    { name: 'AES-GCM', length: 256 },
                    true,
                    ['encrypt', 'decrypt']
                );
                
                const exported = await window.crypto.subtle.exportKey('jwk', this.encryptionKey);
                localStorage.setItem('env-manager-key', JSON.stringify(exported));
            }
        } catch (error) {
            console.error('Failed to initialize encryption:', error);
            this.encryptionEnabled = false;
        }
    }

    // WebSocket for real-time updates
    initializeWebSocket() {
        try {
            const wsUrl = this.baseURL.replace('http', 'ws') + '/env-updates';
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected for real-time updates');
                this.emit('connection-status', { connected: true });
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleRealtimeUpdate(data);
                } catch (error) {
                    console.error('Failed to parse WebSocket message:', error);
                }
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket disconnected');
                this.emit('connection-status', { connected: false });
                
                // Attempt to reconnect after 5 seconds
                setTimeout(() => this.initializeWebSocket(), 5000);
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
            };
        } catch (error) {
            console.error('Failed to initialize WebSocket:', error);
        }
    }

    handleRealtimeUpdate(data) {
        switch (data.type) {
            case 'env-file-changed':
                this.emit('env-file-updated', data.content);
                break;
            case 'service-status-changed':
                this.emit('service-status-updated', data.services);
                break;
            case 'validation-result':
                this.emit('validation-updated', data.result);
                break;
            case 'deployment-status':
                this.emit('deployment-status', data.status);
                break;
        }
    }

    // Encrypt sensitive values
    async encryptValue(value) {
        if (!this.encryptionEnabled || !this.encryptionKey) {
            return value;
        }

        try {
            const iv = window.crypto.getRandomValues(new Uint8Array(12));
            const encoded = new TextEncoder().encode(value);
            
            const encrypted = await window.crypto.subtle.encrypt(
                { name: 'AES-GCM', iv },
                this.encryptionKey,
                encoded
            );
            
            return {
                encrypted: Array.from(new Uint8Array(encrypted)),
                iv: Array.from(iv),
                _encrypted: true
            };
        } catch (error) {
            console.error('Encryption failed:', error);
            return value;
        }
    }

    // Decrypt sensitive values
    async decryptValue(encryptedData) {
        if (!this.encryptionEnabled || !encryptedData._encrypted || !this.encryptionKey) {
            return encryptedData;
        }

        try {
            const encrypted = new Uint8Array(encryptedData.encrypted);
            const iv = new Uint8Array(encryptedData.iv);
            
            const decrypted = await window.crypto.subtle.decrypt(
                { name: 'AES-GCM', iv },
                this.encryptionKey,
                encrypted
            );
            
            return new TextDecoder().decode(decrypted);
        } catch (error) {
            console.error('Decryption failed:', error);
            return '[Decryption Error]';
        }
    }

    // HTTP request wrapper with retry logic
    async request(method, endpoint, data = null, options = {}) {
        const url = `${this.baseURL}${endpoint}`;
        let lastError = null;

        for (let attempt = 1; attempt <= this.retryAttempts; attempt++) {
            try {
                const controller = new AbortController();
                const timeoutId = setTimeout(() => controller.abort(), this.timeout);

                const response = await fetch(url, {
                    method,
                    headers: {
                        'Content-Type': 'application/json',
                        ...options.headers
                    },
                    body: data ? JSON.stringify(data) : null,
                    signal: controller.signal,
                    ...options
                });

                clearTimeout(timeoutId);

                if (!response.ok) {
                    throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                }

                const result = await response.json();
                return result;

            } catch (error) {
                lastError = error;
                console.warn(`Request attempt ${attempt} failed:`, error.message);

                if (attempt < this.retryAttempts) {
                    await new Promise(resolve => setTimeout(resolve, this.retryDelay * attempt));
                }
            }
        }

        throw lastError;
    }

    // Load environment file
    async loadEnvFile(filePath = '.env') {
        try {
            const response = await this.request('GET', `/env/load?file=${encodeURIComponent(filePath)}`);
            
            if (response.encrypted) {
                // Decrypt sensitive values
                const decrypted = {};
                for (const [key, value] of Object.entries(response.variables)) {
                    decrypted[key] = await this.decryptValue(value);
                }
                response.variables = decrypted;
            }
            
            this.emit('env-loaded', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'load-failed', error: error.message });
            throw error;
        }
    }

    // Save environment file
    async saveEnvFile(content, options = {}) {
        try {
            // Parse content and encrypt sensitive values
            const variables = this.parseEnvContent(content);
            const encrypted = {};
            
            for (const [key, value] of Object.entries(variables)) {
                if (this.isSensitiveVariable(key)) {
                    encrypted[key] = await this.encryptValue(value);
                } else {
                    encrypted[key] = value;
                }
            }

            const data = {
                content,
                variables: encrypted,
                backup: options.backup !== false,
                validate: options.validate !== false,
                deploy: options.deploy === true,
                filePath: options.filePath || '.env'
            };

            const response = await this.request('POST', '/env/save', data);
            this.emit('env-saved', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'save-failed', error: error.message });
            throw error;
        }
    }

    // Validate environment configuration
    async validateConfiguration(content) {
        try {
            const response = await this.request('POST', '/env/validate', { content });
            this.emit('validation-complete', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'validation-failed', error: error.message });
            throw error;
        }
    }

    // Get service status
    async getServiceStatus() {
        try {
            const response = await this.request('GET', '/services/status');
            this.emit('service-status', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'service-status-failed', error: error.message });
            throw error;
        }
    }

    // Control services
    async controlService(serviceName, action) {
        try {
            const response = await this.request('POST', `/services/${serviceName}/${action}`);
            this.emit('service-action', { service: serviceName, action, result: response });
            return response;
        } catch (error) {
            this.emit('error', { type: 'service-control-failed', error: error.message });
            throw error;
        }
    }

    // Generate secure keys
    async generateSecureKey(type, options = {}) {
        try {
            const response = await this.request('POST', '/security/generate-key', { type, options });
            this.emit('key-generated', { type, key: response.key });
            return response;
        } catch (error) {
            this.emit('error', { type: 'key-generation-failed', error: error.message });
            throw error;
        }
    }

    // Get configuration templates
    async getTemplates() {
        try {
            const response = await this.request('GET', '/templates');
            this.emit('templates-loaded', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'templates-failed', error: error.message });
            throw error;
        }
    }

    // Load specific template
    async loadTemplate(templateName) {
        try {
            const response = await this.request('GET', `/templates/${templateName}`);
            this.emit('template-loaded', { name: templateName, content: response.content });
            return response;
        } catch (error) {
            this.emit('error', { type: 'template-load-failed', error: error.message });
            throw error;
        }
    }

    // Backup configuration
    async backupConfiguration(options = {}) {
        try {
            const response = await this.request('POST', '/env/backup', options);
            this.emit('backup-created', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'backup-failed', error: error.message });
            throw error;
        }
    }

    // Get backup history
    async getBackupHistory() {
        try {
            const response = await this.request('GET', '/env/backups');
            this.emit('backup-history', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'backup-history-failed', error: error.message });
            throw error;
        }
    }

    // Restore from backup
    async restoreFromBackup(backupId) {
        try {
            const response = await this.request('POST', `/env/restore/${backupId}`);
            this.emit('backup-restored', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'restore-failed', error: error.message });
            throw error;
        }
    }

    // Deploy configuration
    async deployConfiguration(options = {}) {
        try {
            const response = await this.request('POST', '/env/deploy', options);
            this.emit('deployment-started', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'deployment-failed', error: error.message });
            throw error;
        }
    }

    // Get deployment status
    async getDeploymentStatus(deploymentId) {
        try {
            const response = await this.request('GET', `/env/deploy/${deploymentId}`);
            this.emit('deployment-status', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'deployment-status-failed', error: error.message });
            throw error;
        }
    }

    // Security analysis
    async performSecurityAnalysis(content) {
        try {
            const response = await this.request('POST', '/security/analyze', { content });
            this.emit('security-analysis', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'security-analysis-failed', error: error.message });
            throw error;
        }
    }

    // Get AI suggestions
    async getAISuggestions(content, category = 'all') {
        try {
            const response = await this.request('POST', '/ai/suggestions', { content, category });
            this.emit('ai-suggestions', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'ai-suggestions-failed', error: error.message });
            throw error;
        }
    }

    // Check port availability
    async checkPortAvailability(port) {
        try {
            const response = await this.request('GET', `/network/port-check/${port}`);
            this.emit('port-check', { port, ...response });
            return response;
        } catch (error) {
            this.emit('error', { type: 'port-check-failed', error: error.message });
            throw error;
        }
    }

    // Utility methods
    parseEnvContent(content) {
        const variables = {};
        const lines = content.split('\n');
        
        lines.forEach(line => {
            const trimmed = line.trim();
            if (trimmed && !trimmed.startsWith('#') && trimmed.includes('=')) {
                const [key, ...valueParts] = trimmed.split('=');
                variables[key.trim()] = valueParts.join('=').trim();
            }
        });
        
        return variables;
    }

    isSensitiveVariable(key) {
        const sensitivePatterns = [
            /password/i,
            /secret/i,
            /key/i,
            /token/i,
            /auth/i,
            /credential/i,
            /private/i
        ];
        
        return sensitivePatterns.some(pattern => pattern.test(key));
    }

    // Health check
    async healthCheck() {
        try {
            const response = await this.request('GET', '/health', null, { timeout: 5000 });
            this.emit('health-check', response);
            return response;
        } catch (error) {
            this.emit('error', { type: 'health-check-failed', error: error.message });
            throw error;
        }
    }

    // Cleanup resources
    destroy() {
        if (this.ws) {
            this.ws.close();
            this.ws = null;
        }
        this.listeners = {};
        this.encryptionKey = null;
    }
}

// Factory function for easy initialization
function createEnvManagerAPI(options = {}) {
    return new EnvManagerAPI(options);
}

// Export for both browser and Node.js environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { EnvManagerAPI, createEnvManagerAPI };
} else if (typeof window !== 'undefined') {
    window.EnvManagerAPI = EnvManagerAPI;
    window.createEnvManagerAPI = createEnvManagerAPI;
}