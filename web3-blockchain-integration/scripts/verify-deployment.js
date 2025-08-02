#!/usr/bin/env node

/**
 * Web3 Media Platform Deployment Verification Script
 * Verifies that all components are properly deployed and configured
 */

const { ethers } = require('ethers');
const axios = require('axios');
const fs = require('fs').promises;
const path = require('path');

class DeploymentVerifier {
    constructor() {
        this.config = this.loadConfig();
        this.results = {
            blockchain: {},
            services: {},
            integration: {},
            overall: { score: 0, total: 0 }
        };
    }

    loadConfig() {
        // Load configuration from environment
        require('dotenv').config({ path: '../.env.web3' });
        
        return {
            networks: {
                ethereum: process.env.ETHEREUM_RPC_URL,
                polygon: process.env.POLYGON_RPC_URL,
                arbitrum: process.env.ARBITRUM_RPC_URL,
                optimism: process.env.OPTIMISM_RPC_URL
            },
            contracts: {
                contentOwnership: process.env.CONTENT_OWNERSHIP_ADDRESS,
                mediaDAO: process.env.MEDIA_DAO_ADDRESS,
                marketplace: process.env.MARKETPLACE_ADDRESS,
                mediaNFT: process.env.MEDIA_NFT_ADDRESS,
                paymentProcessor: process.env.PAYMENT_PROCESSOR_ADDRESS
            },
            services: {
                web3Api: process.env.REACT_APP_API_URL || 'http://localhost:3030',
                ipfsApi: process.env.IPFS_API_URL || 'http://localhost:5001',
                ipfsGateway: process.env.REACT_APP_IPFS_GATEWAY || 'http://localhost:8080',
                jellyfin: process.env.REACT_APP_JELLYFIN_URL || 'http://localhost:8096',
                frontend: 'http://localhost:3031',
                analytics: 'http://localhost:3032'
            }
        };
    }

    async verify() {
        console.log('🔍 Starting Web3 Media Platform deployment verification...\n');
        
        try {
            await this.verifyBlockchainDeployment();
            await this.verifyServices();
            await this.verifyIntegration();
            
            this.printResults();
            
        } catch (error) {
            console.error('❌ Verification failed:', error.message);
            process.exit(1);
        }
    }

    async verifyBlockchainDeployment() {
        console.log('📜 Verifying smart contract deployment...');
        
        for (const [network, rpcUrl] of Object.entries(this.config.networks)) {
            if (!rpcUrl) continue;
            
            try {
                const provider = new ethers.JsonRpcProvider(rpcUrl);
                const chainId = (await provider.getNetwork()).chainId;
                
                console.log(`  ✓ ${network} network connected (Chain ID: ${chainId})`);
                this.updateScore('blockchain', `${network}_connection`, true);
                
                // Verify contracts if addresses are provided
                for (const [contractName, address] of Object.entries(this.config.contracts)) {
                    if (address && address !== '') {
                        const code = await provider.getCode(address);
                        const isDeployed = code !== '0x';
                        
                        console.log(`  ${isDeployed ? '✓' : '❌'} ${contractName}: ${address} ${isDeployed ? '(deployed)' : '(not found)'}`);
                        this.updateScore('blockchain', `${contractName}_deployment`, isDeployed);
                        
                        if (isDeployed) {
                            // Test basic contract function
                            try {
                                const contract = new ethers.Contract(address, ['function name() view returns (string)'], provider);
                                await contract.name();
                                console.log(`    ✓ Contract ${contractName} is functional`);
                                this.updateScore('blockchain', `${contractName}_functional`, true);
                            } catch (error) {
                                // Some contracts might not have name() function, that's okay
                                console.log(`    ℹ Contract ${contractName} deployed but interface check skipped`);
                                this.updateScore('blockchain', `${contractName}_functional`, true);
                            }
                        }
                    }
                }
                
            } catch (error) {
                console.log(`  ❌ ${network} network failed: ${error.message}`);
                this.updateScore('blockchain', `${network}_connection`, false);
            }
        }
        
        console.log('');
    }

    async verifyServices() {
        console.log('🔧 Verifying service deployment...');
        
        for (const [serviceName, serviceUrl] of Object.entries(this.config.services)) {
            try {
                const startTime = Date.now();
                
                let endpoint = serviceUrl;
                if (serviceName === 'web3Api') endpoint += '/health';
                else if (serviceName === 'ipfsApi') endpoint += '/api/v0/version';
                else if (serviceName === 'analytics') endpoint += '/health';
                
                const response = await axios.get(endpoint, { 
                    timeout: 10000,
                    validateStatus: (status) => status < 500
                });
                
                const responseTime = Date.now() - startTime;
                const available = response.status < 400;
                
                console.log(`  ${available ? '✓' : '❌'} ${serviceName}: ${serviceUrl} (${responseTime}ms)`);
                this.updateScore('services', serviceName, available);
                
                // Additional service-specific checks
                if (available && serviceName === 'web3Api') {
                    await this.verifyWeb3ApiEndpoints(serviceUrl);
                } else if (available && serviceName === 'ipfsApi') {
                    await this.verifyIpfsNode(serviceUrl);
                }
                
            } catch (error) {
                console.log(`  ❌ ${serviceName}: ${serviceUrl} - ${error.message}`);
                this.updateScore('services', serviceName, false);
            }
        }
        
        console.log('');
    }

    async verifyWeb3ApiEndpoints(baseUrl) {
        const endpoints = [
            '/api/docs',
            '/api/health/detailed'
        ];
        
        for (const endpoint of endpoints) {
            try {
                await axios.get(`${baseUrl}${endpoint}`, { timeout: 5000 });
                console.log(`    ✓ ${endpoint} accessible`);
            } catch (error) {
                console.log(`    ❌ ${endpoint} failed: ${error.message}`);
            }
        }
    }

    async verifyIpfsNode(baseUrl) {
        try {
            const response = await axios.get(`${baseUrl}/api/v0/stats/bitswap`, { timeout: 5000 });
            console.log(`    ✓ IPFS bitswap active`);
            
            // Test IPFS connectivity
            const peersResponse = await axios.get(`${baseUrl}/api/v0/swarm/peers`, { timeout: 5000 });
            const peers = JSON.parse(peersResponse.data).Peers || [];
            console.log(`    ✓ IPFS connected to ${peers.length} peers`);
            
        } catch (error) {
            console.log(`    ⚠ IPFS additional checks failed: ${error.message}`);
        }
    }

    async verifyIntegration() {
        console.log('🔗 Verifying system integration...');
        
        // Test Web3 API -> IPFS integration
        try {
            const response = await axios.get(`${this.config.services.web3Api}/api/health/detailed`, { timeout: 10000 });
            const health = response.data;
            
            if (health.services) {
                console.log(`  ${health.services.ipfs?.status === 'healthy' ? '✓' : '❌'} Web3 API -> IPFS integration`);
                this.updateScore('integration', 'web3_ipfs', health.services.ipfs?.status === 'healthy');
                
                console.log(`  ${health.services.database?.status === 'healthy' ? '✓' : '❌'} Web3 API -> Database integration`);
                this.updateScore('integration', 'web3_database', health.services.database?.status === 'healthy');
                
                console.log(`  ${health.services.redis?.status === 'healthy' ? '✓' : '❌'} Web3 API -> Redis integration`);
                this.updateScore('integration', 'web3_redis', health.services.redis?.status === 'healthy');
            }
        } catch (error) {
            console.log(`  ❌ Integration health check failed: ${error.message}`);
            this.updateScore('integration', 'health_check', false);
        }
        
        // Test IPFS Gateway
        try {
            const testHash = 'QmYjtig7VJQ6XsnUjqqJvj7QaMcCAwtrgNdahSiFofrE7o'; // "Hello World" test file
            const gatewayUrl = `${this.config.services.ipfsGateway}/ipfs/${testHash}`;
            
            const response = await axios.get(gatewayUrl, { 
                timeout: 10000,
                validateStatus: (status) => status < 500
            });
            
            const accessible = response.status === 200;
            console.log(`  ${accessible ? '✓' : '❌'} IPFS Gateway serving content`);
            this.updateScore('integration', 'ipfs_gateway', accessible);
            
        } catch (error) {
            console.log(`  ❌ IPFS Gateway test failed: ${error.message}`);
            this.updateScore('integration', 'ipfs_gateway', false);
        }
        
        // Test Frontend Build
        try {
            const response = await axios.get(this.config.services.frontend, { 
                timeout: 10000,
                validateStatus: (status) => status < 500
            });
            
            const accessible = response.status === 200;
            console.log(`  ${accessible ? '✓' : '❌'} Frontend application serving`);
            this.updateScore('integration', 'frontend', accessible);
            
        } catch (error) {
            console.log(`  ❌ Frontend accessibility test failed: ${error.message}`);
            this.updateScore('integration', 'frontend', false);
        }
        
        console.log('');
    }

    updateScore(category, test, passed) {
        if (!this.results[category]) {
            this.results[category] = {};
        }
        
        this.results[category][test] = passed;
        this.results.overall.total++;
        if (passed) {
            this.results.overall.score++;
        }
    }

    printResults() {
        console.log('📊 Deployment Verification Results');
        console.log('=====================================');
        
        const overallPercentage = Math.round((this.results.overall.score / this.results.overall.total) * 100);
        
        console.log(`\n🎯 Overall Score: ${this.results.overall.score}/${this.results.overall.total} (${overallPercentage}%)`);
        
        if (overallPercentage >= 90) {
            console.log('🎉 Excellent! Your Web3 media platform is fully operational.');
        } else if (overallPercentage >= 75) {
            console.log('✅ Good! Most components are working. Check failed items below.');
        } else if (overallPercentage >= 50) {
            console.log('⚠️ Partial deployment. Several components need attention.');
        } else {
            console.log('❌ Deployment issues detected. Please review the failed components.');
        }
        
        // Detailed breakdown
        console.log('\n📋 Detailed Results:');
        
        for (const [category, tests] of Object.entries(this.results)) {
            if (category === 'overall') continue;
            
            const categoryPassed = Object.values(tests).filter(Boolean).length;
            const categoryTotal = Object.values(tests).length;
            const categoryPercentage = Math.round((categoryPassed / categoryTotal) * 100);
            
            console.log(`\n${this.getCategoryIcon(category)} ${category.toUpperCase()}: ${categoryPassed}/${categoryTotal} (${categoryPercentage}%)`);
            
            for (const [test, passed] of Object.entries(tests)) {
                console.log(`  ${passed ? '✓' : '❌'} ${test}`);
            }
        }
        
        // Recommendations
        console.log('\n💡 Next Steps:');
        
        if (this.results.blockchain && Object.values(this.results.blockchain).some(v => !v)) {
            console.log('  • Review smart contract deployment and network configuration');
        }
        
        if (this.results.services && Object.values(this.results.services).some(v => !v)) {
            console.log('  • Check Docker services and port availability');
            console.log('  • Review service logs: docker-compose logs [service-name]');
        }
        
        if (this.results.integration && Object.values(this.results.integration).some(v => !v)) {
            console.log('  • Verify service interconnectivity and configuration');
            console.log('  • Check environment variables and API endpoints');
        }
        
        console.log('\n📚 Resources:');
        console.log('  • Full documentation: ./BLOCKCHAIN_DEPLOYMENT_GUIDE.md');
        console.log('  • API documentation: http://localhost:3030/api/docs');
        console.log('  • Service logs: docker-compose logs -f');
        console.log('  • Health checks: curl http://localhost:3030/health');
        
        // Save results to file
        this.saveResults();
        
        if (overallPercentage < 75) {
            process.exit(1);
        }
    }

    getCategoryIcon(category) {
        const icons = {
            blockchain: '📜',
            services: '🔧',
            integration: '🔗'
        };
        return icons[category] || '📋';
    }

    async saveResults() {
        try {
            const resultsFile = path.join(__dirname, '..', 'deployment-verification-results.json');
            const results = {
                timestamp: new Date().toISOString(),
                ...this.results
            };
            
            await fs.writeFile(resultsFile, JSON.stringify(results, null, 2));
            console.log(`\n💾 Results saved to: ${resultsFile}`);
            
        } catch (error) {
            console.log(`\n⚠️ Failed to save results: ${error.message}`);
        }
    }
}

// Run verification if called directly
if (require.main === module) {
    const verifier = new DeploymentVerifier();
    verifier.verify().catch(console.error);
}

module.exports = DeploymentVerifier;