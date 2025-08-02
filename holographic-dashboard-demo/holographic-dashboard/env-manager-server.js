#!/usr/bin/env node

/**
 * Advanced Environment Manager Server
 * Provides backend functionality for the .env management interface
 * Supports real-time updates, security analysis, and service management
 */

const express = require('express');
const cors = require('cors');
const fs = require('fs').promises;
const path = require('path');
const crypto = require('crypto');
const { WebSocketServer } = require('ws');
const { spawn, exec } = require('child_process');
const chokidar = require('chokidar');

class EnvManagerServer {
    constructor(options = {}) {
        this.port = options.port || 3001;
        this.envFilePath = options.envFilePath || '.env';
        this.backupDir = options.backupDir || './backups';
        this.configDir = options.configDir || './config';
        
        this.app = express();
        this.clients = new Set();
        this.fileWatcher = null;
        
        this.setupMiddleware();
        this.setupRoutes();
        this.setupWebSocket();
    }

    setupMiddleware() {
        this.app.use(cors({
            origin: true,
            credentials: true
        }));
        
        this.app.use(express.json({ limit: '10mb' }));
        this.app.use(express.static(path.join(__dirname)));
        
        // Security headers
        this.app.use((req, res, next) => {
            res.setHeader('X-Content-Type-Options', 'nosniff');
            res.setHeader('X-Frame-Options', 'DENY');
            res.setHeader('X-XSS-Protection', '1; mode=block');
            next();
        });

        // Request logging
        this.app.use((req, res, next) => {
            console.log(`${new Date().toISOString()} ${req.method} ${req.path}`);
            next();
        });
    }

    setupRoutes() {
        // Health check
        this.app.get('/api/health', (req, res) => {
            res.json({
                status: 'healthy',
                timestamp: new Date().toISOString(),
                version: '2.0.0',
                features: {
                    encryption: true,
                    realtime: true,
                    backups: true,
                    validation: true,
                    ai_suggestions: true
                }
            });
        });

        // Environment file operations
        this.app.get('/api/env/load', this.handleLoadEnv.bind(this));
        this.app.post('/api/env/save', this.handleSaveEnv.bind(this));
        this.app.post('/api/env/validate', this.handleValidateEnv.bind(this));
        this.app.post('/api/env/backup', this.handleBackupEnv.bind(this));
        this.app.get('/api/env/backups', this.handleGetBackups.bind(this));
        this.app.post('/api/env/restore/:id', this.handleRestoreBackup.bind(this));
        this.app.post('/api/env/deploy', this.handleDeployEnv.bind(this));
        this.app.get('/api/env/deploy/:id', this.handleGetDeploymentStatus.bind(this));

        // Service management
        this.app.get('/api/services/status', this.handleGetServiceStatus.bind(this));
        this.app.post('/api/services/:service/:action', this.handleServiceAction.bind(this));

        // Security
        this.app.post('/api/security/generate-key', this.handleGenerateKey.bind(this));
        this.app.post('/api/security/analyze', this.handleSecurityAnalysis.bind(this));

        // Templates
        this.app.get('/api/templates', this.handleGetTemplates.bind(this));
        this.app.get('/api/templates/:name', this.handleGetTemplate.bind(this));

        // AI features
        this.app.post('/api/ai/suggestions', this.handleAISuggestions.bind(this));

        // Network utilities
        this.app.get('/api/network/port-check/:port', this.handlePortCheck.bind(this));

        // Error handling
        this.app.use((error, req, res, next) => {
            console.error('API Error:', error);
            res.status(500).json({
                error: 'Internal server error',
                message: error.message,
                timestamp: new Date().toISOString()
            });
        });
    }

    setupWebSocket() {
        this.server = this.app.listen(this.port, () => {
            console.log(`🚀 Environment Manager Server running on port ${this.port}`);
            console.log(`📊 Dashboard: http://localhost:${this.port}/advanced-env-manager.html`);
        });

        this.wss = new WebSocketServer({ server: this.server, path: '/api/env-updates' });
        
        this.wss.on('connection', (ws) => {
            console.log('📡 WebSocket client connected');
            this.clients.add(ws);

            ws.on('close', () => {
                console.log('📡 WebSocket client disconnected');
                this.clients.delete(ws);
            });

            ws.on('error', (error) => {
                console.error('WebSocket error:', error);
                this.clients.delete(ws);
            });
        });

        this.setupFileWatcher();
    }

    setupFileWatcher() {
        this.fileWatcher = chokidar.watch(this.envFilePath, {
            persistent: true,
            ignoreInitial: true
        });

        this.fileWatcher.on('change', async () => {
            try {
                const content = await fs.readFile(this.envFilePath, 'utf8');
                this.broadcast({
                    type: 'env-file-changed',
                    content,
                    timestamp: new Date().toISOString()
                });
            } catch (error) {
                console.error('File watch error:', error);
            }
        });
    }

    broadcast(message) {
        const data = JSON.stringify(message);
        this.clients.forEach(client => {
            if (client.readyState === client.OPEN) {
                client.send(data);
            }
        });
    }

    // Route handlers
    async handleLoadEnv(req, res) {
        try {
            const filePath = req.query.file || this.envFilePath;
            const content = await fs.readFile(filePath, 'utf8');
            const variables = this.parseEnvContent(content);
            
            res.json({
                content,
                variables,
                filePath,
                lastModified: (await fs.stat(filePath)).mtime,
                size: content.length
            });
        } catch (error) {
            if (error.code === 'ENOENT') {
                // File doesn't exist, return empty content
                res.json({
                    content: '',
                    variables: {},
                    filePath: req.query.file || this.envFilePath,
                    lastModified: null,
                    size: 0
                });
            } else {
                throw error;
            }
        }
    }

    async handleSaveEnv(req, res) {
        try {
            const { content, filePath = this.envFilePath, backup = true, validate = true, deploy = false } = req.body;
            
            // Create backup if requested
            let backupId = null;
            if (backup) {
                backupId = await this.createBackup(filePath);
            }

            // Validate if requested
            let validationResult = null;
            if (validate) {
                validationResult = await this.validateEnvContent(content);
                if (validationResult.errors > 0) {
                    return res.status(400).json({
                        error: 'Validation failed',
                        validation: validationResult
                    });
                }
            }

            // Save file
            await this.ensureDirectoryExists(path.dirname(filePath));
            await fs.writeFile(filePath, content, 'utf8');

            // Deploy if requested
            let deploymentId = null;
            if (deploy) {
                deploymentId = await this.deployConfiguration(content);
            }

            const response = {
                success: true,
                filePath,
                size: content.length,
                backupId,
                validationResult,
                deploymentId,
                timestamp: new Date().toISOString()
            };

            // Broadcast update
            this.broadcast({
                type: 'env-file-saved',
                ...response
            });

            res.json(response);
        } catch (error) {
            throw error;
        }
    }

    async handleValidateEnv(req, res) {
        try {
            const { content } = req.body;
            const result = await this.validateEnvContent(content);
            
            this.broadcast({
                type: 'validation-result',
                result
            });

            res.json(result);
        } catch (error) {
            throw error;
        }
    }

    async handleBackupEnv(req, res) {
        try {
            const { filePath = this.envFilePath, description = '' } = req.body;
            const backupId = await this.createBackup(filePath, description);
            
            res.json({
                success: true,
                backupId,
                timestamp: new Date().toISOString()
            });
        } catch (error) {
            throw error;
        }
    }

    async handleGetBackups(req, res) {
        try {
            await this.ensureDirectoryExists(this.backupDir);
            const files = await fs.readdir(this.backupDir);
            const backups = [];
            
            for (const file of files.filter(f => f.endsWith('.env.backup'))) {
                const filePath = path.join(this.backupDir, file);
                const stats = await fs.stat(filePath);
                const content = await fs.readFile(filePath, 'utf8');
                
                backups.push({
                    id: file.replace('.env.backup', ''),
                    filename: file,
                    created: stats.birthtime,
                    size: stats.size,
                    variables: Object.keys(this.parseEnvContent(content)).length
                });
            }
            
            backups.sort((a, b) => new Date(b.created) - new Date(a.created));
            res.json({ backups });
        } catch (error) {
            throw error;
        }
    }

    async handleRestoreBackup(req, res) {
        try {
            const { id } = req.params;
            const backupPath = path.join(this.backupDir, `${id}.env.backup`);
            const content = await fs.readFile(backupPath, 'utf8');
            
            // Create backup of current file before restore
            const currentBackupId = await this.createBackup(this.envFilePath, 'Before restore');
            
            // Restore content
            await fs.writeFile(this.envFilePath, content, 'utf8');
            
            res.json({
                success: true,
                restoredFrom: id,
                currentBackupId,
                timestamp: new Date().toISOString()
            });
            
            this.broadcast({
                type: 'backup-restored',
                backupId: id
            });
        } catch (error) {
            throw error;
        }
    }

    async handleDeployEnv(req, res) {
        try {
            const deploymentId = crypto.randomUUID();
            const { content, services = [], restartServices = true } = req.body;
            
            // Save deployment info
            const deployment = {
                id: deploymentId,
                status: 'starting',
                content,
                services,
                restartServices,
                startTime: new Date().toISOString()
            };
            
            // Store deployment info (in production, use a database)
            this.deployments = this.deployments || {};
            this.deployments[deploymentId] = deployment;
            
            // Start deployment process
            this.processDeployment(deploymentId);
            
            res.json({
                deploymentId,
                status: 'started',
                estimatedDuration: '30-60 seconds'
            });
        } catch (error) {
            throw error;
        }
    }

    async handleGetDeploymentStatus(req, res) {
        try {
            const { id } = req.params;
            const deployment = (this.deployments || {})[id];
            
            if (!deployment) {
                return res.status(404).json({ error: 'Deployment not found' });
            }
            
            res.json(deployment);
        } catch (error) {
            throw error;
        }
    }

    async handleGetServiceStatus(req, res) {
        try {
            const services = await this.getDockerServices();
            
            this.broadcast({
                type: 'service-status-updated',
                services
            });
            
            res.json({ services });
        } catch (error) {
            throw error;
        }
    }

    async handleServiceAction(req, res) {
        try {
            const { service, action } = req.params;
            const result = await this.executeServiceAction(service, action);
            
            res.json({
                success: true,
                service,
                action,
                result,
                timestamp: new Date().toISOString()
            });
            
            // Refresh service status after action
            setTimeout(() => this.handleGetServiceStatus({ query: {} }, { json: () => {} }), 2000);
        } catch (error) {
            throw error;
        }
    }

    async handleGenerateKey(req, res) {
        try {
            const { type, options = {} } = req.body;
            const key = this.generateSecureKey(type, options);
            
            res.json({
                key,
                type,
                length: key.length,
                generated: new Date().toISOString()
            });
        } catch (error) {
            throw error;
        }
    }

    async handleSecurityAnalysis(req, res) {
        try {
            const { content } = req.body;
            const analysis = await this.performSecurityAnalysis(content);
            
            res.json(analysis);
        } catch (error) {
            throw error;
        }
    }

    async handleGetTemplates(req, res) {
        try {
            const templates = {
                development: 'Development environment with debug features',
                production: 'Production-ready configuration',
                docker: 'Docker containerized setup',
                kubernetes: 'Kubernetes deployment configuration',
                minimal: 'Minimal configuration for simple apps',
                'full-stack': 'Complete full-stack application setup'
            };
            
            res.json({ templates });
        } catch (error) {
            throw error;
        }
    }

    async handleGetTemplate(req, res) {
        try {
            const { name } = req.params;
            const template = this.getTemplate(name);
            
            if (!template) {
                return res.status(404).json({ error: 'Template not found' });
            }
            
            res.json({
                name,
                content: template,
                variables: Object.keys(this.parseEnvContent(template)).length
            });
        } catch (error) {
            throw error;
        }
    }

    async handleAISuggestions(req, res) {
        try {
            const { content, category = 'all' } = req.body;
            const suggestions = await this.generateAISuggestions(content, category);
            
            res.json({ suggestions, category });
        } catch (error) {
            throw error;
        }
    }

    async handlePortCheck(req, res) {
        try {
            const { port } = req.params;
            const portNum = parseInt(port);
            
            if (isNaN(portNum) || portNum < 1 || portNum > 65535) {
                return res.status(400).json({ error: 'Invalid port number' });
            }
            
            const isAvailable = await this.checkPortAvailability(portNum);
            
            res.json({
                port: portNum,
                available: isAvailable,
                checked: new Date().toISOString()
            });
        } catch (error) {
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

    async validateEnvContent(content) {
        const variables = this.parseEnvContent(content);
        const issues = { errors: 0, warnings: 0, suggestions: 0 };
        const messages = [];
        
        // Basic validation rules
        const requiredVars = ['NODE_ENV'];
        requiredVars.forEach(varName => {
            if (!variables[varName]) {
                messages.push({
                    type: 'error',
                    message: `Missing required variable: ${varName}`,
                    variable: varName
                });
                issues.errors++;
            }
        });
        
        // Port validation
        Object.entries(variables).forEach(([key, value]) => {
            if (key === 'PORT' && value) {
                const port = parseInt(value);
                if (isNaN(port) || port < 1 || port > 65535) {
                    messages.push({
                        type: 'error',
                        message: `Invalid port number: ${value}`,
                        variable: key
                    });
                    issues.errors++;
                }
            }
            
            // Security checks
            if (key.includes('SECRET') || key.includes('PASSWORD')) {
                if (!value || value.length < 8) {
                    messages.push({
                        type: 'warning',
                        message: `${key} should be stronger for security`,
                        variable: key
                    });
                    issues.warnings++;
                }
            }
        });
        
        return {
            valid: issues.errors === 0,
            issues,
            messages,
            variableCount: Object.keys(variables).length,
            analyzed: new Date().toISOString()
        };
    }

    async createBackup(filePath, description = '') {
        const backupId = `${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
        const backupPath = path.join(this.backupDir, `${backupId}.env.backup`);
        
        await this.ensureDirectoryExists(this.backupDir);
        
        try {
            const content = await fs.readFile(filePath, 'utf8');
            const metadata = {
                originalPath: filePath,
                created: new Date().toISOString(),
                description,
                size: content.length
            };
            
            const backupContent = `# Backup Metadata: ${JSON.stringify(metadata)}\n${content}`;
            await fs.writeFile(backupPath, backupContent, 'utf8');
            
            return backupId;
        } catch (error) {
            if (error.code === 'ENOENT') {
                // Original file doesn't exist, create empty backup
                await fs.writeFile(backupPath, `# Empty backup - original file did not exist\n`, 'utf8');
                return backupId;
            }
            throw error;
        }
    }

    async processDeployment(deploymentId) {
        const deployment = this.deployments[deploymentId];
        
        try {
            deployment.status = 'validating';
            this.broadcast({ type: 'deployment-status', deployment });
            
            // Validate configuration
            const validation = await this.validateEnvContent(deployment.content);
            if (!validation.valid) {
                deployment.status = 'failed';
                deployment.error = 'Validation failed';
                deployment.validation = validation;
                return;
            }
            
            deployment.status = 'deploying';
            this.broadcast({ type: 'deployment-status', deployment });
            
            // Save environment file
            await fs.writeFile(this.envFilePath, deployment.content, 'utf8');
            
            // Restart services if requested
            if (deployment.restartServices) {
                deployment.status = 'restarting-services';
                this.broadcast({ type: 'deployment-status', deployment });
                
                await this.restartServices(deployment.services);
            }
            
            deployment.status = 'completed';
            deployment.endTime = new Date().toISOString();
            this.broadcast({ type: 'deployment-status', deployment });
            
        } catch (error) {
            deployment.status = 'failed';
            deployment.error = error.message;
            deployment.endTime = new Date().toISOString();
            this.broadcast({ type: 'deployment-status', deployment });
        }
    }

    async getDockerServices() {
        return new Promise((resolve) => {
            exec('docker ps --format "table {{.Names}}\\t{{.Status}}\\t{{.Ports}}"', (error, stdout) => {
                if (error) {
                    console.warn('Docker not available or no containers running');
                    resolve([]);
                    return;
                }
                
                const services = [];
                const lines = stdout.split('\n').slice(1); // Skip header
                
                lines.forEach(line => {
                    const parts = line.trim().split(/\s+/);
                    if (parts.length >= 2) {
                        const name = parts[0];
                        const status = parts[1];
                        const ports = parts.slice(2).join(' ');
                        
                        services.push({
                            name: name,
                            status: status.includes('Up') ? 'running' : 'stopped',
                            ports: ports,
                            healthStatus: status.includes('healthy') ? 'healthy' : 
                                         status.includes('unhealthy') ? 'unhealthy' : 'unknown'
                        });
                    }
                });
                
                resolve(services);
            });
        });
    }

    async executeServiceAction(serviceName, action) {
        return new Promise((resolve, reject) => {
            const commands = {
                start: `docker start ${serviceName}`,
                stop: `docker stop ${serviceName}`,
                restart: `docker restart ${serviceName}`
            };
            
            const command = commands[action];
            if (!command) {
                reject(new Error(`Unknown action: ${action}`));
                return;
            }
            
            exec(command, (error, stdout, stderr) => {
                if (error) {
                    reject(new Error(`Service action failed: ${stderr || error.message}`));
                } else {
                    resolve({ output: stdout.trim() });
                }
            });
        });
    }

    async restartServices(services = []) {
        if (services.length === 0) {
            // Restart all services
            return new Promise((resolve, reject) => {
                exec('docker-compose restart', (error, stdout, stderr) => {
                    if (error) {
                        reject(new Error(`Failed to restart services: ${stderr || error.message}`));
                    } else {
                        resolve({ output: stdout.trim() });
                    }
                });
            });
        } else {
            // Restart specific services
            const promises = services.map(service => this.executeServiceAction(service, 'restart'));
            return Promise.all(promises);
        }
    }

    generateSecureKey(type, options = {}) {
        switch (type) {
            case 'jwt':
                return crypto.randomBytes(32).toString('hex');
            case 'api':
                return `sk_${crypto.randomBytes(16).toString('hex')}`;
            case 'password':
                return this.generateStrongPassword(options.length || 16);
            case 'uuid':
                return crypto.randomUUID();
            case 'session':
                return crypto.randomBytes(24).toString('hex');
            case 'encryption':
                return crypto.randomBytes(32).toString('hex');
            default:
                return crypto.randomBytes(16).toString('hex');
        }
    }

    generateStrongPassword(length = 16) {
        const uppercase = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
        const lowercase = 'abcdefghijklmnopqrstuvwxyz';
        const numbers = '0123456789';
        const symbols = '!@#$%^&*()_+-=[]{}|;:,.<>?';
        const allChars = uppercase + lowercase + numbers + symbols;
        
        let password = '';
        // Ensure at least one character from each set
        password += uppercase[Math.floor(Math.random() * uppercase.length)];
        password += lowercase[Math.floor(Math.random() * lowercase.length)];
        password += numbers[Math.floor(Math.random() * numbers.length)];
        password += symbols[Math.floor(Math.random() * symbols.length)];
        
        // Fill the rest randomly
        for (let i = 4; i < length; i++) {
            password += allChars[Math.floor(Math.random() * allChars.length)];
        }
        
        // Shuffle the password
        return password.split('').sort(() => Math.random() - 0.5).join('');
    }

    async performSecurityAnalysis(content) {
        const variables = this.parseEnvContent(content);
        const issues = { critical: 0, high: 0, medium: 0, low: 0 };
        const findings = [];
        
        // Check for common security issues
        Object.entries(variables).forEach(([key, value]) => {
            // Default/weak passwords
            if ((key.includes('PASSWORD') || key.includes('SECRET')) && 
                ['admin', 'password', '123456', 'change-me'].includes(value.toLowerCase())) {
                findings.push({
                    severity: 'critical',
                    type: 'weak-credential',
                    variable: key,
                    message: 'Using default or weak credential'
                });
                issues.critical++;
            }
            
            // Empty secrets
            if ((key.includes('SECRET') || key.includes('KEY')) && !value) {
                findings.push({
                    severity: 'high',
                    type: 'empty-secret',
                    variable: key,
                    message: 'Secret variable is empty'
                });
                issues.high++;
            }
            
            // Debug mode in production
            if (key === 'DEBUG' && value === 'true' && variables.NODE_ENV === 'production') {
                findings.push({
                    severity: 'medium',
                    type: 'debug-in-production',
                    variable: key,
                    message: 'Debug mode enabled in production'
                });
                issues.medium++;
            }
        });
        
        const totalIssues = Object.values(issues).reduce((sum, count) => sum + count, 0);
        let score = Math.max(0, 100 - (issues.critical * 30 + issues.high * 20 + issues.medium * 10 + issues.low * 5));
        
        return {
            score,
            level: score >= 90 ? 'excellent' : score >= 70 ? 'good' : score >= 50 ? 'fair' : 'poor',
            issues,
            findings,
            totalIssues,
            analyzed: new Date().toISOString()
        };
    }

    async generateAISuggestions(content, category) {
        const variables = this.parseEnvContent(content);
        const suggestions = [];
        
        // Basic suggestions based on missing common variables
        if (!variables.NODE_ENV) {
            suggestions.push({
                title: 'Set Environment Mode',
                description: 'Define NODE_ENV to specify development/production mode',
                priority: 'high',
                code: 'NODE_ENV=development'
            });
        }
        
        if (!variables.PORT) {
            suggestions.push({
                title: 'Configure Application Port',
                description: 'Set PORT variable for your application server',
                priority: 'medium',
                code: 'PORT=3000'
            });
        }
        
        // Security suggestions
        const secretVars = Object.keys(variables).filter(key => 
            key.includes('SECRET') || key.includes('PASSWORD') || key.includes('KEY')
        );
        
        if (secretVars.length === 0) {
            suggestions.push({
                title: 'Add Security Configuration',
                description: 'Configure authentication secrets for better security',
                priority: 'high',
                code: 'JWT_SECRET=your-secure-jwt-secret-here\nSESSION_SECRET=your-session-secret-here'
            });
        }
        
        // Database suggestions
        if (category === 'database' || category === 'all') {
            if (!variables.DATABASE_URL) {
                suggestions.push({
                    title: 'Configure Database Connection',
                    description: 'Add database connection string',
                    priority: 'high',
                    code: 'DATABASE_URL=postgresql://user:password@localhost:5432/dbname'
                });
            }
        }
        
        return suggestions;
    }

    async checkPortAvailability(port) {
        return new Promise((resolve) => {
            const net = require('net');
            const server = net.createServer();
            
            server.listen(port, () => {
                server.once('close', () => resolve(true));
                server.close();
            });
            
            server.on('error', () => resolve(false));
        });
    }

    getTemplate(name) {
        const templates = {
            development: `# Development Environment
NODE_ENV=development
PORT=3000
DEBUG=true
LOG_LEVEL=debug

# Database
DATABASE_URL=postgresql://dev:dev@localhost:5432/app_dev
DATABASE_POOL_MIN=2
DATABASE_POOL_MAX=10

# Cache
REDIS_URL=redis://localhost:6379

# Security (Development)
JWT_SECRET=dev-jwt-secret-change-in-production
SESSION_SECRET=dev-session-secret`,

            production: `# Production Environment
NODE_ENV=production
PORT=443
DEBUG=false
LOG_LEVEL=error

# Database
DATABASE_URL=\${DATABASE_URL}
DATABASE_SSL=true
DATABASE_POOL_MIN=10
DATABASE_POOL_MAX=100

# Cache
REDIS_URL=\${REDIS_URL}
REDIS_TLS=true

# Security
JWT_SECRET=\${JWT_SECRET}
SESSION_SECRET=\${SESSION_SECRET}
BCRYPT_ROUNDS=12

# Performance
ENABLE_COMPRESSION=true
ENABLE_CACHING=true`,

            docker: `# Docker Configuration
COMPOSE_PROJECT_NAME=app
DOCKER_BUILDKIT=1

# User/Group
PUID=1000
PGID=1000
TZ=UTC

# Volumes
CONFIG_ROOT=./config
DATA_ROOT=./data

# Network
NETWORK_NAME=app_network`,

            minimal: `# Minimal Configuration
NODE_ENV=development
PORT=3000
DATABASE_URL=sqlite:./database.db
JWT_SECRET=change-me-in-production`
        };
        
        return templates[name];
    }

    async ensureDirectoryExists(dirPath) {
        try {
            await fs.access(dirPath);
        } catch {
            await fs.mkdir(dirPath, { recursive: true });
        }
    }

    async shutdown() {
        console.log('🛑 Shutting down Environment Manager Server...');
        
        if (this.fileWatcher) {
            await this.fileWatcher.close();
        }
        
        if (this.wss) {
            this.wss.close();
        }
        
        if (this.server) {
            this.server.close();
        }
        
        console.log('✅ Server shutdown completed');
    }
}

// Start server if this file is run directly
if (require.main === module) {
    const server = new EnvManagerServer({
        port: process.env.PORT || 3001,
        envFilePath: process.env.ENV_FILE_PATH || '.env'
    });
    
    // Graceful shutdown
    process.on('SIGINT', () => server.shutdown().then(() => process.exit(0)));
    process.on('SIGTERM', () => server.shutdown().then(() => process.exit(0)));
}

module.exports = EnvManagerServer;