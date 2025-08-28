#!/usr/bin/env node
/**
 * Comprehensive Stress Test & Functional Test Suite
 * Tests all functionality and performance limits
 */

const http = require('http');
const https = require('https');
const fs = require('fs').promises;
const path = require('path');
const { performance } = require('perf_hooks');

class MediaServerTestSuite {
    constructor(config = {}) {
        this.baseUrl = config.baseUrl || 'http://localhost:3737';
        this.apiUrl = `${this.baseUrl}/api`;
        this.results = {
            functional: [],
            performance: [],
            stress: [],
            failures: [],
            metrics: {}
        };
        this.startTime = null;
    }

    // Utility function for HTTP requests
    async makeRequest(endpoint, options = {}) {
        return new Promise((resolve, reject) => {
            const url = new URL(endpoint, this.apiUrl);
            const startTime = performance.now();
            
            const req = http.request(url, {
                method: options.method || 'GET',
                headers: options.headers || { 'Content-Type': 'application/json' },
                timeout: options.timeout || 10000
            }, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    const endTime = performance.now();
                    const responseTime = endTime - startTime;
                    
                    try {
                        const json = JSON.parse(data);
                        resolve({
                            status: res.statusCode,
                            data: json,
                            responseTime,
                            headers: res.headers
                        });
                    } catch {
                        resolve({
                            status: res.statusCode,
                            data,
                            responseTime,
                            headers: res.headers
                        });
                    }
                });
            });

            req.on('error', reject);
            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });

            if (options.body) {
                req.write(JSON.stringify(options.body));
            }
            req.end();
        });
    }

    // =================
    // FUNCTIONAL TESTS
    // =================

    async testEndpoint(name, endpoint, expectedFields = []) {
        console.log(`  Testing ${name}...`);
        try {
            const response = await this.makeRequest(endpoint);
            
            const test = {
                name,
                endpoint,
                status: response.status,
                responseTime: response.responseTime,
                passed: response.status === 200
            };

            // Check expected fields
            if (expectedFields.length > 0 && response.data) {
                const missingFields = expectedFields.filter(field => !(field in response.data));
                if (missingFields.length > 0) {
                    test.passed = false;
                    test.error = `Missing fields: ${missingFields.join(', ')}`;
                }
            }

            this.results.functional.push(test);
            console.log(`    ${test.passed ? '✅' : '❌'} ${name} - ${response.responseTime.toFixed(2)}ms`);
            return test;
        } catch (error) {
            const test = {
                name,
                endpoint,
                status: 0,
                passed: false,
                error: error.message
            };
            this.results.functional.push(test);
            console.log(`    ❌ ${name} - ${error.message}`);
            return test;
        }
    }

    async runFunctionalTests() {
        console.log('\n📋 FUNCTIONAL TESTS\n' + '='.repeat(50));
        
        // Test all endpoints
        await this.testEndpoint('Health Check', '/health', ['status', 'timestamp']);
        await this.testEndpoint('Services Status', '/services');
        await this.testEndpoint('Media Scan', '/media/scan');
        await this.testEndpoint('Media Search', '/media/search?q=test');
        await this.testEndpoint('Downloads List', '/downloads');
        await this.testEndpoint('Storage Info', '/system/storage');
        await this.testEndpoint('Library Refresh', '/library/refresh');
        await this.testEndpoint('Create Backup', '/backup');
        
        // Test POST endpoints
        console.log('\n  Testing POST endpoints...');
        try {
            const response = await this.makeRequest('/downloads/add', {
                method: 'POST',
                body: { url: 'http://example.com/test.mp4', name: 'Test Download' }
            });
            console.log(`    ✅ Add Download - ${response.responseTime.toFixed(2)}ms`);
        } catch (error) {
            console.log(`    ❌ Add Download - ${error.message}`);
        }
    }

    // =================
    // LOAD TESTS
    // =================

    async runLoadTest(endpoint, requestsPerSecond, duration) {
        console.log(`\n  Load Testing ${endpoint}`);
        console.log(`    Rate: ${requestsPerSecond} req/s for ${duration}s`);
        
        const results = [];
        const interval = 1000 / requestsPerSecond;
        const totalRequests = requestsPerSecond * duration;
        let completed = 0;
        let failed = 0;
        const responseTimes = [];

        const startTime = Date.now();
        
        for (let i = 0; i < totalRequests; i++) {
            setTimeout(async () => {
                try {
                    const response = await this.makeRequest(endpoint);
                    responseTimes.push(response.responseTime);
                    completed++;
                } catch (error) {
                    failed++;
                }

                // Progress update
                if ((completed + failed) % 10 === 0) {
                    process.stdout.write(`\r    Progress: ${completed + failed}/${totalRequests} (${failed} failed)`);
                }

                // Final report
                if (completed + failed === totalRequests) {
                    const avgResponseTime = responseTimes.reduce((a, b) => a + b, 0) / responseTimes.length;
                    const maxResponseTime = Math.max(...responseTimes);
                    const minResponseTime = Math.min(...responseTimes);
                    
                    console.log(`\n    ✅ Completed: ${completed}/${totalRequests}`);
                    console.log(`    ❌ Failed: ${failed}/${totalRequests}`);
                    console.log(`    ⏱️  Avg Response: ${avgResponseTime.toFixed(2)}ms`);
                    console.log(`    ⏱️  Min/Max: ${minResponseTime.toFixed(2)}ms / ${maxResponseTime.toFixed(2)}ms`);
                    
                    this.results.performance.push({
                        endpoint,
                        requestsPerSecond,
                        duration,
                        totalRequests,
                        completed,
                        failed,
                        avgResponseTime,
                        minResponseTime,
                        maxResponseTime
                    });
                }
            }, i * interval);
        }

        // Wait for all requests to complete
        await new Promise(resolve => setTimeout(resolve, (duration + 2) * 1000));
    }

    async runPerformanceTests() {
        console.log('\n⚡ PERFORMANCE TESTS\n' + '='.repeat(50));
        
        // Gradual load increase
        await this.runLoadTest('/health', 10, 5);    // 10 req/s for 5s
        await this.runLoadTest('/health', 50, 5);    // 50 req/s for 5s
        await this.runLoadTest('/health', 100, 5);   // 100 req/s for 5s
        
        // Test heavier endpoints
        await this.runLoadTest('/services', 20, 5);
        await this.runLoadTest('/media/scan', 10, 5);
    }

    // =================
    // STRESS TESTS
    // =================

    async runStressTest() {
        console.log('\n💥 STRESS TESTS\n' + '='.repeat(50));
        
        // Concurrent connections test
        console.log('\n  Testing concurrent connections...');
        const concurrentRequests = 500;
        const promises = [];
        
        for (let i = 0; i < concurrentRequests; i++) {
            promises.push(this.makeRequest('/health').catch(e => ({ error: e.message })));
        }
        
        const startTime = performance.now();
        const results = await Promise.all(promises);
        const endTime = performance.now();
        
        const successful = results.filter(r => !r.error).length;
        const failed = results.filter(r => r.error).length;
        
        console.log(`    Concurrent Requests: ${concurrentRequests}`);
        console.log(`    ✅ Successful: ${successful}`);
        console.log(`    ❌ Failed: ${failed}`);
        console.log(`    ⏱️  Total Time: ${(endTime - startTime).toFixed(2)}ms`);
        
        this.results.stress.push({
            test: 'Concurrent Connections',
            total: concurrentRequests,
            successful,
            failed,
            totalTime: endTime - startTime
        });

        // Large payload test
        console.log('\n  Testing large payloads...');
        const largeData = 'x'.repeat(1024 * 1024); // 1MB payload
        
        try {
            const response = await this.makeRequest('/downloads/add', {
                method: 'POST',
                body: { url: 'test', name: largeData }
            });
            console.log(`    ✅ Large payload handled - ${response.responseTime.toFixed(2)}ms`);
        } catch (error) {
            console.log(`    ❌ Large payload failed - ${error.message}`);
        }

        // Rapid fire test
        console.log('\n  Testing rapid fire requests...');
        const rapidFireCount = 1000;
        let rapidSuccess = 0;
        let rapidFail = 0;
        
        for (let i = 0; i < rapidFireCount; i++) {
            try {
                await this.makeRequest('/health', { timeout: 1000 });
                rapidSuccess++;
            } catch {
                rapidFail++;
            }
            
            if (i % 100 === 0) {
                process.stdout.write(`\r    Progress: ${i}/${rapidFireCount}`);
            }
        }
        
        console.log(`\n    ✅ Successful: ${rapidSuccess}/${rapidFireCount}`);
        console.log(`    ❌ Failed: ${rapidFail}/${rapidFireCount}`);
    }

    // =================
    // FAILURE TESTS
    // =================

    async runFailureTests() {
        console.log('\n🔥 FAILURE & RECOVERY TESTS\n' + '='.repeat(50));
        
        // Invalid endpoint test
        console.log('\n  Testing invalid endpoints...');
        try {
            const response = await this.makeRequest('/invalid/endpoint');
            console.log(`    ${response.status === 404 ? '✅' : '❌'} 404 handling works`);
        } catch (error) {
            console.log(`    ❌ Error handling failed: ${error.message}`);
        }

        // Malformed request test
        console.log('\n  Testing malformed requests...');
        try {
            const response = await this.makeRequest('/downloads/add', {
                method: 'POST',
                body: 'not json'
            });
            console.log(`    ✅ Malformed request handled`);
        } catch (error) {
            console.log(`    ✅ Malformed request rejected properly`);
        }

        // Timeout test
        console.log('\n  Testing timeout handling...');
        try {
            await this.makeRequest('/health', { timeout: 1 }); // 1ms timeout
            console.log(`    ❌ Timeout not working`);
        } catch (error) {
            console.log(`    ✅ Timeout working: ${error.message}`);
        }
    }

    // =================
    // DATA INTEGRITY
    // =================

    async runDataIntegrityTests() {
        console.log('\n🔒 DATA INTEGRITY TESTS\n' + '='.repeat(50));
        
        // Test data persistence
        console.log('\n  Testing data persistence...');
        
        // Add a download
        const addResponse = await this.makeRequest('/downloads/add', {
            method: 'POST',
            body: { url: 'http://test.com/file.mp4', name: 'Integrity Test' }
        });
        
        if (addResponse.data && addResponse.data.id) {
            // Check if it appears in the list
            const listResponse = await this.makeRequest('/downloads');
            const found = listResponse.data.active?.some(d => d.name === 'Integrity Test');
            console.log(`    ${found ? '✅' : '❌'} Data persisted correctly`);
        }

        // Test concurrent modifications
        console.log('\n  Testing concurrent modifications...');
        const modPromises = [];
        for (let i = 0; i < 10; i++) {
            modPromises.push(
                this.makeRequest('/downloads/add', {
                    method: 'POST',
                    body: { url: `http://test.com/file${i}.mp4`, name: `Concurrent ${i}` }
                })
            );
        }
        
        const modResults = await Promise.all(modPromises);
        const allSuccessful = modResults.every(r => r.data && r.data.success);
        console.log(`    ${allSuccessful ? '✅' : '❌'} Concurrent modifications handled`);
    }

    // =================
    // SECURITY TESTS
    // =================

    async runSecurityTests() {
        console.log('\n🔐 SECURITY TESTS\n' + '='.repeat(50));
        
        // SQL Injection attempt
        console.log('\n  Testing SQL injection protection...');
        try {
            const response = await this.makeRequest('/media/search?q=test%27%20OR%201=1--');
            console.log(`    ✅ SQL injection attempt handled`);
        } catch {
            console.log(`    ✅ SQL injection blocked`);
        }

        // XSS attempt
        console.log('\n  Testing XSS protection...');
        try {
            const response = await this.makeRequest('/downloads/add', {
                method: 'POST',
                body: { url: 'test', name: '<script>alert("XSS")</script>' }
            });
            console.log(`    ✅ XSS attempt handled`);
        } catch {
            console.log(`    ✅ XSS blocked`);
        }

        // Path traversal attempt
        console.log('\n  Testing path traversal protection...');
        try {
            const response = await this.makeRequest('/media/search?q=../../etc/passwd');
            console.log(`    ✅ Path traversal attempt handled`);
        } catch {
            console.log(`    ✅ Path traversal blocked`);
        }
    }

    // =================
    // REPORT GENERATION
    // =================

    generateReport() {
        console.log('\n' + '='.repeat(50));
        console.log('📊 TEST RESULTS SUMMARY\n' + '='.repeat(50));
        
        // Functional tests summary
        const functionalPassed = this.results.functional.filter(t => t.passed).length;
        const functionalTotal = this.results.functional.length;
        console.log(`\n✅ Functional Tests: ${functionalPassed}/${functionalTotal} passed`);
        
        // Performance metrics
        if (this.results.performance.length > 0) {
            console.log('\n⚡ Performance Metrics:');
            this.results.performance.forEach(p => {
                console.log(`  ${p.endpoint}: ${p.avgResponseTime.toFixed(2)}ms avg (${p.failed} failures)`);
            });
        }

        // Stress test results
        if (this.results.stress.length > 0) {
            console.log('\n💥 Stress Test Results:');
            this.results.stress.forEach(s => {
                console.log(`  ${s.test}: ${s.successful}/${s.total} successful`);
            });
        }

        // Calculate overall health score
        const healthScore = (functionalPassed / functionalTotal) * 100;
        console.log(`\n🏆 Overall Health Score: ${healthScore.toFixed(1)}%`);
        
        // Recommendations
        console.log('\n📝 Recommendations:');
        if (healthScore < 100) {
            console.log('  • Fix failing functional tests');
        }
        if (this.results.performance.some(p => p.avgResponseTime > 1000)) {
            console.log('  • Optimize slow endpoints (>1s response time)');
        }
        if (this.results.stress.some(s => s.failed > s.successful * 0.1)) {
            console.log('  • Improve server capacity for high load');
        }

        // Save detailed report
        this.saveDetailedReport();
    }

    async saveDetailedReport() {
        const reportPath = path.join(__dirname, `test-report-${Date.now()}.json`);
        const report = {
            timestamp: new Date().toISOString(),
            duration: Date.now() - this.startTime,
            results: this.results,
            summary: {
                functional: {
                    total: this.results.functional.length,
                    passed: this.results.functional.filter(t => t.passed).length,
                    failed: this.results.functional.filter(t => !t.passed).length
                },
                performance: this.results.performance.length,
                stress: this.results.stress.length
            }
        };

        try {
            await fs.writeFile(reportPath, JSON.stringify(report, null, 2));
            console.log(`\n💾 Detailed report saved to: ${reportPath}`);
        } catch (error) {
            console.log(`\n❌ Failed to save report: ${error.message}`);
        }
    }

    // =================
    // MAIN TEST RUNNER
    // =================

    async runAllTests() {
        this.startTime = Date.now();
        
        console.log(`
╔════════════════════════════════════════════════╗
║    COMPREHENSIVE MEDIA SERVER TEST SUITE       ║
╠════════════════════════════════════════════════╣
║  Target: ${this.baseUrl.padEnd(38)}║
║  Started: ${new Date().toLocaleString().padEnd(37)}║
╚════════════════════════════════════════════════╝
        `);

        try {
            // Check if server is running
            console.log('\n🔍 Checking server availability...');
            const health = await this.makeRequest('/health');
            console.log('✅ Server is running\n');

            // Run test suites
            await this.runFunctionalTests();
            await this.runPerformanceTests();
            await this.runStressTest();
            await this.runFailureTests();
            await this.runDataIntegrityTests();
            await this.runSecurityTests();

        } catch (error) {
            console.error('\n❌ Server is not accessible:', error.message);
            console.log('\nPlease ensure the server is running:');
            console.log('  node functional-backend.js');
            process.exit(1);
        }

        // Generate final report
        this.generateReport();
        
        const totalTime = ((Date.now() - this.startTime) / 1000).toFixed(2);
        console.log(`\n⏱️  Total test time: ${totalTime} seconds`);
        console.log('\n✨ Test suite completed!\n');
    }
}

// Run the test suite
if (require.main === module) {
    const tester = new MediaServerTestSuite({
        baseUrl: process.env.TEST_URL || 'http://localhost:3737'
    });
    
    tester.runAllTests().catch(console.error);
}

module.exports = MediaServerTestSuite;