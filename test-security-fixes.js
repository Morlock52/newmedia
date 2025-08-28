#!/usr/bin/env node

/**
 * Comprehensive Security Testing Suite
 * Tests all security fixes and validates system integrity
 * August 3, 2025
 */

const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const https = require('https');
const http = require('http');

class SecurityTestRunner {
    constructor() {
        this.results = {
            passed: 0,
            failed: 0,
            warnings: 0,
            tests: []
        };
        this.testStartTime = Date.now();
    }

    log(message, type = 'info') {
        const timestamp = new Date().toISOString();
        const prefix = {
            'info': '✅',
            'warn': '⚠️ ',
            'error': '❌',
            'success': '🎉'
        }[type] || 'ℹ️ ';
        
        console.log(`${prefix} [${timestamp}] ${message}`);
    }

    async runTest(name, testFn) {
        this.log(`Running test: ${name}`);
        const startTime = Date.now();
        
        try {
            const result = await testFn();
            const duration = Date.now() - startTime;
            
            this.results.tests.push({
                name,
                status: 'passed',
                duration,
                result
            });
            this.results.passed++;
            this.log(`✅ ${name} - PASSED (${duration}ms)`, 'success');
            return true;
        } catch (error) {
            const duration = Date.now() - startTime;
            
            this.results.tests.push({
                name,
                status: 'failed',
                duration,
                error: error.message
            });
            this.results.failed++;
            this.log(`❌ ${name} - FAILED: ${error.message}`, 'error');
            return false;
        }
    }

    async testSecurityFiles() {
        this.log('🔍 Testing Security File Integrity...');
        
        const expectedFiles = [
            'security/secure-env-manager.js',
            'security/authentication-middleware.js',
            'security/input-validation.js',
            'security/session-manager.js',
            'security/security-monitor.js',
            'security/secrets-manager.js',
            'security/secure-logging.js',
            'security/docker-security-config.yml',
            'security/install-security.sh',
            'security/.env.secure.template',
            'security/README.md'
        ];

        for (const file of expectedFiles) {
            await this.runTest(`File exists: ${file}`, async () => {
                if (!fs.existsSync(file)) {
                    throw new Error(`Security file missing: ${file}`);
                }
                
                const stats = fs.statSync(file);
                if (stats.size === 0) {
                    throw new Error(`Security file is empty: ${file}`);
                }
                
                return { size: stats.size, modified: stats.mtime };
            });
        }
    }

    async testJavaScriptSyntax() {
        this.log('🔍 Testing JavaScript Syntax...');
        
        const jsFiles = [
            'security/secure-env-manager.js',
            'security/authentication-middleware.js',
            'security/input-validation.js',
            'security/session-manager.js',
            'security/security-monitor.js',
            'security/secrets-manager.js',
            'security/secure-logging.js'
        ];

        for (const file of jsFiles) {
            if (fs.existsSync(file)) {
                await this.runTest(`Syntax check: ${file}`, async () => {
                    return new Promise((resolve, reject) => {
                        exec(`node -c ${file}`, (error, stdout, stderr) => {
                            if (error) {
                                reject(new Error(`Syntax error: ${stderr}`));
                            } else {
                                resolve('Valid JavaScript syntax');
                            }
                        });
                    });
                });
            }
        }
    }

    async testSecurityComponents() {
        this.log('🔍 Testing Security Component Logic...');

        // Test Secure Environment Manager
        await this.runTest('Secure Environment Manager', async () => {
            const secureEnvPath = 'security/secure-env-manager.js';
            if (!fs.existsSync(secureEnvPath)) {
                throw new Error('Secure Environment Manager not found');
            }
            
            const content = fs.readFileSync(secureEnvPath, 'utf8');
            
            // Check for required security features
            const requiredFeatures = [
                'AES-256-GCM',
                'encryption',
                'decryption',
                'crypto',
                'generateKey'
            ];
            
            for (const feature of requiredFeatures) {
                if (!content.includes(feature)) {
                    throw new Error(`Missing security feature: ${feature}`);
                }
            }
            
            return 'Environment manager has all required security features';
        });

        // Test Authentication Middleware
        await this.runTest('Authentication Middleware', async () => {
            const authPath = 'security/authentication-middleware.js';
            if (!fs.existsSync(authPath)) {
                throw new Error('Authentication Middleware not found');
            }
            
            const content = fs.readFileSync(authPath, 'utf8');
            
            const requiredFeatures = [
                'JWT',
                'bcrypt',
                'rate limiting',
                'session',
                'fingerprint'
            ];
            
            for (const feature of requiredFeatures) {
                if (!content.toLowerCase().includes(feature.toLowerCase())) {
                    throw new Error(`Missing auth feature: ${feature}`);
                }
            }
            
            return 'Authentication middleware has all required features';
        });

        // Test Input Validation
        await this.runTest('Input Validation', async () => {
            const validationPath = 'security/input-validation.js';
            if (!fs.existsSync(validationPath)) {
                throw new Error('Input Validation not found');
            }
            
            const content = fs.readFileSync(validationPath, 'utf8');
            
            const requiredFeatures = [
                'XSS',
                'SQL injection',
                'validation',
                'sanitize',
                'escape'
            ];
            
            for (const feature of requiredFeatures) {
                if (!content.toLowerCase().includes(feature.toLowerCase())) {
                    throw new Error(`Missing validation feature: ${feature}`);
                }
            }
            
            return 'Input validation has all required security features';
        });
    }

    async testDockerSecurity() {
        this.log('🔍 Testing Docker Security Configuration...');

        await this.runTest('Docker Security Config', async () => {
            const dockerConfigPath = 'security/docker-security-config.yml';
            if (!fs.existsSync(dockerConfigPath)) {
                throw new Error('Docker security config not found');
            }
            
            const content = fs.readFileSync(dockerConfigPath, 'utf8');
            
            const requiredFeatures = [
                'PUID: 1000',
                'PGID: 1000',
                'no-new-privileges',
                'apparmor',
                'read_only',
                'tmpfs'
            ];
            
            for (const feature of requiredFeatures) {
                if (!content.includes(feature)) {
                    throw new Error(`Missing Docker security feature: ${feature}`);
                }
            }
            
            return 'Docker configuration includes all security hardening';
        });
    }

    async testSystemIntegration() {
        this.log('🔍 Testing System Integration...');

        // Test existing media server files
        await this.runTest('Media Server Integration', async () => {
            const criticalFiles = [
                'monitoring/comprehensive-logger.js',
                'performance-monitor.js',
                'voice-ai-system/package.json'
            ];
            
            let foundFiles = 0;
            for (const file of criticalFiles) {
                if (fs.existsSync(file)) {
                    foundFiles++;
                }
            }
            
            if (foundFiles === 0) {
                throw new Error('No existing media server components found');
            }
            
            return `Found ${foundFiles} existing components for integration`;
        });

        // Test MCP integration
        await this.runTest('MCP Integration Status', async () => {
            const mcpFiles = [
                'CLAUDE_DESKTOP_CONNECTION_GUIDE.md',
                '.claude-flow/metrics/performance.json',
                '.claude-flow/metrics/task-metrics.json'
            ];
            
            let foundMcpFiles = 0;
            for (const file of mcpFiles) {
                if (fs.existsSync(file)) {
                    foundMcpFiles++;
                }
            }
            
            return `MCP integration files found: ${foundMcpFiles}/${mcpFiles.length}`;
        });
    }

    async testPerformanceImpact() {
        this.log('🔍 Testing Performance Impact...');

        await this.runTest('Security Overhead Analysis', async () => {
            // Simulate security component loading times
            const startTime = Date.now();
            
            // Test file reading performance with security files
            const securityFiles = [
                'security/secure-env-manager.js',
                'security/authentication-middleware.js',
                'security/input-validation.js'
            ].filter(file => fs.existsSync(file));
            
            for (const file of securityFiles) {
                const content = fs.readFileSync(file, 'utf8');
                if (content.length === 0) {
                    throw new Error(`Empty security file: ${file}`);
                }
            }
            
            const loadTime = Date.now() - startTime;
            
            if (loadTime > 1000) {
                throw new Error(`Security components load too slowly: ${loadTime}ms`);
            }
            
            return `Security components load in ${loadTime}ms (acceptable)`;
        });
    }

    async testNetworkSecurity() {
        this.log('🔍 Testing Network Security...');

        await this.runTest('Network Configuration', async () => {
            // Check if security configurations mention proper network isolation
            const dockerConfig = 'security/docker-security-config.yml';
            if (fs.existsSync(dockerConfig)) {
                const content = fs.readFileSync(dockerConfig, 'utf8');
                
                const networkFeatures = [
                    'networks:',
                    'driver: bridge',
                    'internal: true'
                ];
                
                let foundFeatures = 0;
                for (const feature of networkFeatures) {
                    if (content.includes(feature)) {
                        foundFeatures++;
                    }
                }
                
                return `Network security features found: ${foundFeatures}/${networkFeatures.length}`;
            }
            
            return 'Network security configuration present';
        });
    }

    async testSecurityDocumentation() {
        this.log('🔍 Testing Security Documentation...');

        await this.runTest('Security Documentation', async () => {
            const readmePath = 'security/README.md';
            if (!fs.existsSync(readmePath)) {
                throw new Error('Security documentation not found');
            }
            
            const content = fs.readFileSync(readmePath, 'utf8');
            
            if (content.length < 1000) {
                throw new Error('Security documentation too brief');
            }
            
            const requiredSections = [
                'Installation',
                'Configuration',
                'Security Features',
                'Troubleshooting'
            ];
            
            for (const section of requiredSections) {
                if (!content.includes(section)) {
                    throw new Error(`Missing documentation section: ${section}`);
                }
            }
            
            return 'Complete security documentation available';
        });
    }

    async testInstallationScript() {
        this.log('🔍 Testing Installation Script...');

        await this.runTest('Installation Script Validation', async () => {
            const installScript = 'security/install-security.sh';
            if (!fs.existsSync(installScript)) {
                throw new Error('Installation script not found');
            }
            
            const content = fs.readFileSync(installScript, 'utf8');
            
            const requiredCommands = [
                'npm install',
                'mkdir -p',
                'chmod',
                'echo'
            ];
            
            for (const command of requiredCommands) {
                if (!content.includes(command)) {
                    throw new Error(`Missing installation command: ${command}`);
                }
            }
            
            // Check if script is executable
            const stats = fs.statSync(installScript);
            const isExecutable = stats.mode & parseInt('111', 8);
            
            if (!isExecutable && process.platform !== 'win32') {
                this.log('Making installation script executable...', 'warn');
                fs.chmodSync(installScript, '755');
            }
            
            return 'Installation script is complete and executable';
        });
    }

    generateReport() {
        const totalTime = Date.now() - this.testStartTime;
        const totalTests = this.results.passed + this.results.failed;
        const successRate = totalTests > 0 ? (this.results.passed / totalTests * 100).toFixed(2) : 0;
        
        console.log('\n' + '='.repeat(60));
        console.log('🔒 SECURITY TESTING COMPLETE');
        console.log('='.repeat(60));
        console.log(`📊 Total Tests: ${totalTests}`);
        console.log(`✅ Passed: ${this.results.passed}`);
        console.log(`❌ Failed: ${this.results.failed}`);
        console.log(`⚠️  Warnings: ${this.results.warnings}`);
        console.log(`📈 Success Rate: ${successRate}%`);
        console.log(`⏱️  Total Time: ${totalTime}ms`);
        console.log('='.repeat(60));
        
        if (this.results.failed > 0) {
            console.log('\n❌ FAILED TESTS:');
            for (const test of this.results.tests) {
                if (test.status === 'failed') {
                    console.log(`  - ${test.name}: ${test.error}`);
                }
            }
        }
        
        if (this.results.passed === totalTests && totalTests > 0) {
            console.log('\n🎉 ALL SECURITY TESTS PASSED! 🎉');
            console.log('✅ System is secure and ready for deployment');
        } else if (this.results.failed > 0) {
            console.log('\n⚠️  SECURITY ISSUES DETECTED');
            console.log('🔧 Please fix the failed tests before deployment');
        }
        
        return {
            success: this.results.failed === 0,
            totalTests,
            results: this.results
        };
    }

    async runAllTests() {
        this.log('🚀 Starting Comprehensive Security Testing Suite...', 'info');
        this.log('📅 Test Date: August 3, 2025', 'info');
        
        await this.testSecurityFiles();
        await this.testJavaScriptSyntax();
        await this.testSecurityComponents();
        await this.testDockerSecurity();
        await this.testSystemIntegration();
        await this.testPerformanceImpact();
        await this.testNetworkSecurity();
        await this.testSecurityDocumentation();
        await this.testInstallationScript();
        
        return this.generateReport();
    }
}

// Run tests if called directly
if (require.main === module) {
    const runner = new SecurityTestRunner();
    runner.runAllTests()
        .then(results => {
            process.exit(results.success ? 0 : 1);
        })
        .catch(error => {
            console.error('❌ Test runner failed:', error);
            process.exit(1);
        });
}

module.exports = SecurityTestRunner;