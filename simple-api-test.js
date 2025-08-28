#!/usr/bin/env node

/**
 * Simple API Connection Test
 * Quick test to verify services are accessible
 */

const axios = require('axios');

async function testService(name, url, timeout = 10000) {
    console.log(`\n🔍 Testing ${name}...`);
    
    try {
        const startTime = Date.now();
        const response = await axios.get(url, {
            timeout,
            validateStatus: (status) => status < 500,
            headers: {
                'User-Agent': 'API-Integration-Tester/1.0'
            }
        });
        
        const responseTime = Date.now() - startTime;
        
        console.log(`✅ ${name} - HTTP ${response.status} (${responseTime}ms)`);
        
        if (response.headers['content-type']) {
            console.log(`   Content-Type: ${response.headers['content-type']}`);
        }
        
        if (response.headers['server']) {
            console.log(`   Server: ${response.headers['server']}`);
        }
        
        return true;
        
    } catch (error) {
        if (error.code === 'ECONNREFUSED') {
            console.log(`❌ ${name} - Connection refused (service not running)`);
        } else if (error.code === 'ETIMEDOUT') {
            console.log(`❌ ${name} - Timeout after ${timeout}ms`);
        } else if (error.response) {
            console.log(`⚠️  ${name} - HTTP ${error.response.status} (${error.response.statusText})`);
            return true; // Service is running, just returned an error
        } else {
            console.log(`❌ ${name} - Error: ${error.message}`);
        }
        return false;
    }
}

async function testServices() {
    console.log('🚀 Simple API Connection Test\n');
    console.log('Testing services in ultimate-media-server-simple container...\n');
    
    const services = [
        { name: 'Container Root', url: 'http://localhost:80' },
        { name: 'Jellyfin Media Server', url: 'http://localhost:8096' },
        { name: 'Sonarr TV', url: 'http://localhost:8989' },
        { name: 'Radarr Movies', url: 'http://localhost:7878' },
        { name: 'Lidarr Music', url: 'http://localhost:8686' },
        { name: 'Prowlarr Indexer', url: 'http://localhost:9696' },
        { name: 'Bazarr Subtitles', url: 'http://localhost:6767' },
        { name: 'qBittorrent', url: 'http://localhost:8080' }
    ];
    
    let accessible = 0;
    let total = services.length;
    
    for (const service of services) {
        const isAccessible = await testService(service.name, service.url);
        if (isAccessible) accessible++;
        
        // Small delay between tests
        await new Promise(resolve => setTimeout(resolve, 1000));
    }
    
    console.log('\n' + '='.repeat(50));
    console.log(`📊 Results: ${accessible}/${total} services accessible`);
    console.log(`Success rate: ${((accessible/total)*100).toFixed(1)}%`);
    
    if (accessible === 0) {
        console.log('\n🔧 Troubleshooting:');
        console.log('   1. Check if container is fully started: docker logs ultimate-media-server-simple');
        console.log('   2. Verify ports are bound: docker port ultimate-media-server-simple');
        console.log('   3. Check service status inside container: docker exec ultimate-media-server-simple ps aux');
    } else if (accessible < total) {
        console.log('\n⚠️  Some services may still be starting up.');
        console.log('   Services in containers can take 1-2 minutes to fully initialize.');
        console.log('   Consider waiting and running the test again.');
    } else {
        console.log('\n✅ All services are accessible! Ready for integration testing.');
    }
}

// Also test inside the container
async function testContainerInternal() {
    console.log('\n🔍 Testing internal container connectivity...');
    
    try {
        const { execSync } = require('child_process');
        
        // Test if services respond internally
        const internalTests = [
            'curl -s -o /dev/null -w "%{http_code}" http://localhost:8096 || echo "FAIL"',
            'curl -s -o /dev/null -w "%{http_code}" http://localhost:8989 || echo "FAIL"',
            'curl -s -o /dev/null -w "%{http_code}" http://localhost:7878 || echo "FAIL"'
        ];
        
        for (const test of internalTests) {
            try {
                const result = execSync(`docker exec ultimate-media-server-simple ${test}`, { 
                    encoding: 'utf8', 
                    timeout: 5000 
                }).trim();
                
                if (result === 'FAIL') {
                    console.log(`❌ Internal test failed`);
                } else {
                    console.log(`✅ Internal response: ${result}`);
                }
            } catch (error) {
                console.log(`❌ Internal test error: ${error.message}`);
            }
        }
        
    } catch (error) {
        console.log(`⚠️  Could not run internal tests: ${error.message}`);
    }
}

async function main() {
    await testServices();
    await testContainerInternal();
}

if (require.main === module) {
    main().catch(console.error);
}

module.exports = { testService, testServices };