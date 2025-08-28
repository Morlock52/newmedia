#!/usr/bin/env node
/**
 * Comprehensive Performance Testing Suite for Media Server Stack
 * Tests container startup, memory usage, CPU, response times, disk I/O, and network throughput
 */

const { spawn, exec } = require('child_process');
const fs = require('fs').promises;
const http = require('http');
const https = require('https');
const util = require('util');
const execAsync = util.promisify(exec);

class PerformanceTestSuite {
    constructor() {
        this.testResults = {
            timestamp: new Date().toISOString(),
            systemInfo: {},
            containerStartup: {},
            memoryUsage: {},
            cpuUtilization: {},
            responseTimeTests: {},
            diskIOTests: {},
            networkTests: {},
            bottlenecks: [],
            recommendations: []
        };
        
        // Service endpoints to test
        this.serviceEndpoints = [
            { name: 'Jellyfin', url: 'http://localhost:8096/health', port: 8096 },
            { name: 'Sonarr', url: 'http://localhost:8989/ping', port: 8989 },
            { name: 'Radarr', url: 'http://localhost:7878/ping', port: 7878 },
            { name: 'Prowlarr', url: 'http://localhost:9696/ping', port: 9696 },
            { name: 'qBittorrent', url: 'http://localhost:8080/api/v2/app/version', port: 8080 },
            { name: 'Plex', url: 'http://localhost:32400/identity', port: 32400 },
            { name: 'Dashboard', url: 'http://localhost:3000', port: 3000 },
            { name: 'API Server', url: 'http://localhost:3002/health', port: 3002 },
            { name: 'Portainer', url: 'http://localhost:9000', port: 9000 },
            { name: 'Grafana', url: 'http://localhost:3000', port: 3000 },
            { name: 'Prometheus', url: 'http://localhost:9090', port: 9090 }
        ];
    }

    async runFullTestSuite() {
        console.log('🚀 Starting Comprehensive Performance Test Suite...\n');
        
        try {
            await this.collectSystemInfo();
            await this.testContainerStartupTimes();
            await this.monitorMemoryUsage();
            await this.checkCPUUtilization();
            await this.testResponseTimes();
            await this.measureDiskIOPerformance();
            await this.testNetworkThroughput();
            await this.identifyBottlenecks();
            await this.generateReport();
            
            console.log('\n✅ Performance test suite completed successfully!');
            console.log(`📊 Results saved to: performance-report-${Date.now()}.json`);
            
        } catch (error) {
            console.error('❌ Error running performance tests:', error);
        }
    }

    async collectSystemInfo() {
        console.log('📊 Collecting system information...');
        
        try {
            // System resources
            const { stdout: memInfo } = await execAsync('free -h 2>/dev/null || vm_stat');
            const { stdout: cpuInfo } = await execAsync('lscpu 2>/dev/null || sysctl -n machdep.cpu.brand_string');
            const { stdout: diskInfo } = await execAsync('df -h');
            
            // Docker info
            const { stdout: dockerInfo } = await execAsync('docker info --format "{{json .}}"');
            const { stdout: dockerVersion } = await execAsync('docker --version');
            
            this.testResults.systemInfo = {
                memory: memInfo.trim(),
                cpu: cpuInfo.trim(),
                disk: diskInfo.trim(),
                docker: {
                    version: dockerVersion.trim(),
                    info: JSON.parse(dockerInfo)
                },
                nodeVersion: process.version,
                platform: process.platform,
                arch: process.arch
            };
            
        } catch (error) {
            console.warn('⚠️  Could not collect all system info:', error.message);
        }
    }

    async testContainerStartupTimes() {
        console.log('⏱️  Testing container startup times...');
        
        try {
            // Get list of containers
            const { stdout: containers } = await execAsync('docker ps --format "{{.Names}}"');
            const containerList = containers.trim().split('\n').filter(name => name);
            
            if (containerList.length === 0) {
                console.log('📋 No running containers found. Starting test containers...');
                
                // Try to start the media server stack
                const startTime = Date.now();
                try {
                    await execAsync('docker-compose up -d --no-recreate', { timeout: 300000 }); // 5 minute timeout
                    const endTime = Date.now();
                    
                    this.testResults.containerStartup = {
                        stackStartupTime: endTime - startTime,
                        message: 'Full stack started successfully'
                    };
                } catch (error) {
                    this.testResults.containerStartup = {
                        error: error.message,
                        message: 'Failed to start container stack'
                    };
                }
            } else {
                // Test individual container restart times
                const startupTimes = {};
                
                for (const container of containerList.slice(0, 5)) { // Test first 5 containers
                    try {
                        const startTime = Date.now();
                        await execAsync(`docker restart ${container}`, { timeout: 60000 });
                        const endTime = Date.now();
                        startupTimes[container] = endTime - startTime;
                        
                        // Wait for container to be healthy
                        await this.waitForContainerHealth(container);
                    } catch (error) {
                        startupTimes[container] = { error: error.message };
                    }
                }
                
                this.testResults.containerStartup = startupTimes;
            }
            
        } catch (error) {
            console.error('❌ Container startup test failed:', error.message);
            this.testResults.containerStartup = { error: error.message };
        }
    }

    async waitForContainerHealth(containerName, timeout = 30000) {
        const startTime = Date.now();
        
        while (Date.now() - startTime < timeout) {
            try {
                const { stdout } = await execAsync(`docker inspect ${containerName} --format "{{.State.Health.Status}}"`);
                if (stdout.trim() === 'healthy') {
                    return true;
                }
            } catch (error) {
                // Container might not have health check
            }
            
            await new Promise(resolve => setTimeout(resolve, 1000));
        }
        
        return false;
    }

    async monitorMemoryUsage() {
        console.log('💾 Monitoring memory usage across containers...');
        
        try {
            const { stdout } = await execAsync('docker stats --no-stream --format "table {{.Name}}\\t{{.MemUsage}}\\t{{.MemPerc}}"');
            const lines = stdout.trim().split('\n').slice(1); // Remove header
            
            const memoryStats = {};
            let totalMemoryUsage = 0;
            
            for (const line of lines) {
                const [name, usage, percentage] = line.split('\t');
                if (name && usage && percentage) {
                    const memUsageBytes = this.parseMemoryUsage(usage);
                    memoryStats[name] = {
                        usage: usage,
                        percentage: percentage,
                        bytes: memUsageBytes
                    };
                    totalMemoryUsage += memUsageBytes;
                }
            }
            
            this.testResults.memoryUsage = {
                containers: memoryStats,
                totalUsage: this.formatBytes(totalMemoryUsage),
                timestamp: new Date().toISOString()
            };
            
        } catch (error) {
            console.error('❌ Memory monitoring failed:', error.message);
            this.testResults.memoryUsage = { error: error.message };
        }
    }

    async checkCPUUtilization() {
        console.log('🔄 Checking CPU utilization...');
        
        try {
            const { stdout } = await execAsync('docker stats --no-stream --format "table {{.Name}}\\t{{.CPUPerc}}"');
            const lines = stdout.trim().split('\n').slice(1);
            
            const cpuStats = {};
            let totalCpuUsage = 0;
            
            for (const line of lines) {
                const [name, cpuPerc] = line.split('\t');
                if (name && cpuPerc) {
                    const percentage = parseFloat(cpuPerc.replace('%', ''));
                    cpuStats[name] = {
                        percentage: cpuPerc,
                        value: percentage
                    };
                    totalCpuUsage += percentage;
                }
            }
            
            this.testResults.cpuUtilization = {
                containers: cpuStats,
                totalUsage: `${totalCpuUsage.toFixed(2)}%`,
                timestamp: new Date().toISOString()
            };
            
        } catch (error) {
            console.error('❌ CPU monitoring failed:', error.message);
            this.testResults.cpuUtilization = { error: error.message };
        }
    }

    async testResponseTimes() {
        console.log('🌐 Testing response times for web interfaces...');
        
        const responseTests = {};
        
        for (const service of this.serviceEndpoints) {
            try {
                const startTime = Date.now();
                const result = await this.httpRequest(service.url);
                const endTime = Date.now();
                
                responseTests[service.name] = {
                    responseTime: endTime - startTime,
                    status: result.status,
                    available: result.status < 400,
                    url: service.url
                };
                
            } catch (error) {
                responseTests[service.name] = {
                    responseTime: -1,
                    error: error.message,
                    available: false,
                    url: service.url
                };
            }
        }
        
        this.testResults.responseTimeTests = responseTests;
    }

    async httpRequest(url, timeout = 10000) {
        return new Promise((resolve, reject) => {
            const request = url.startsWith('https') ? https : http;
            const req = request.get(url, { timeout }, (res) => {
                resolve({
                    status: res.statusCode,
                    headers: res.headers
                });
            });
            
            req.on('error', reject);
            req.on('timeout', () => {
                req.destroy();
                reject(new Error('Request timeout'));
            });
        });
    }

    async measureDiskIOPerformance() {
        console.log('💾 Measuring disk I/O performance...');
        
        try {
            // Test write performance
            const writeTestFile = '/tmp/perf-write-test';
            const writeStartTime = Date.now();
            await execAsync(`dd if=/dev/zero of=${writeTestFile} bs=1M count=100 2>/dev/null || head -c 100M /dev/zero > ${writeTestFile}`);
            const writeEndTime = Date.now();
            
            // Test read performance
            const readStartTime = Date.now();
            await execAsync(`cat ${writeTestFile} > /dev/null`);
            const readEndTime = Date.now();
            
            // Clean up
            await execAsync(`rm -f ${writeTestFile}`);
            
            // Get disk usage stats
            const { stdout: diskStats } = await execAsync('df -h .');
            
            this.testResults.diskIOTests = {
                writeTime: writeEndTime - writeStartTime,
                readTime: readEndTime - readStartTime,
                testSize: '100MB',
                diskStats: diskStats.trim(),
                writeSpeed: `${(100 * 1000 / (writeEndTime - writeStartTime)).toFixed(2)} MB/s`,
                readSpeed: `${(100 * 1000 / (readEndTime - readStartTime)).toFixed(2)} MB/s`
            };
            
        } catch (error) {
            console.error('❌ Disk I/O test failed:', error.message);
            this.testResults.diskIOTests = { error: error.message };
        }
    }

    async testNetworkThroughput() {
        console.log('🔗 Testing network throughput between services...');
        
        try {
            // Test internal Docker network performance
            const { stdout: networkInfo } = await execAsync('docker network ls --format "table {{.Name}}\\t{{.Driver}}\\t{{.Scope}}"');
            
            // Test container-to-container connectivity
            const connectivityTests = {};
            
            // Get list of running containers
            const { stdout: containers } = await execAsync('docker ps --format "{{.Names}}"');
            const containerList = containers.trim().split('\n').filter(name => name).slice(0, 3); // Test first 3
            
            for (const container of containerList) {
                try {
                    const startTime = Date.now();
                    await execAsync(`docker exec ${container} ping -c 3 google.com`, { timeout: 10000 });
                    const endTime = Date.now();
                    
                    connectivityTests[container] = {
                        pingTime: endTime - startTime,
                        status: 'success'
                    };
                } catch (error) {
                    connectivityTests[container] = {
                        error: error.message,
                        status: 'failed'
                    };
                }
            }
            
            this.testResults.networkTests = {
                networkInfo: networkInfo.trim(),
                connectivityTests,
                timestamp: new Date().toISOString()
            };
            
        } catch (error) {
            console.error('❌ Network test failed:', error.message);
            this.testResults.networkTests = { error: error.message };
        }
    }

    async identifyBottlenecks() {
        console.log('🔍 Identifying performance bottlenecks...');
        
        const bottlenecks = [];
        const recommendations = [];
        
        // CPU bottlenecks
        if (this.testResults.cpuUtilization.containers) {
            for (const [container, stats] of Object.entries(this.testResults.cpuUtilization.containers)) {
                if (stats.value > 80) {
                    bottlenecks.push({
                        type: 'CPU',
                        container: container,
                        severity: 'HIGH',
                        value: stats.percentage,
                        description: `Container ${container} is using ${stats.percentage} CPU`
                    });
                    
                    recommendations.push({
                        category: 'CPU Optimization',
                        priority: 'HIGH',
                        suggestion: `Consider CPU limits or resource optimization for ${container}`
                    });
                }
            }
        }
        
        // Memory bottlenecks
        if (this.testResults.memoryUsage.containers) {
            for (const [container, stats] of Object.entries(this.testResults.memoryUsage.containers)) {
                const memPerc = parseFloat(stats.percentage.replace('%', ''));
                if (memPerc > 85) {
                    bottlenecks.push({
                        type: 'MEMORY',
                        container: container,
                        severity: 'HIGH',
                        value: stats.percentage,
                        description: `Container ${container} is using ${stats.percentage} memory`
                    });
                    
                    recommendations.push({
                        category: 'Memory Optimization',
                        priority: 'HIGH',
                        suggestion: `Increase memory limits or optimize ${container} configuration`
                    });
                }
            }
        }
        
        // Response time bottlenecks
        if (this.testResults.responseTimeTests) {
            for (const [service, stats] of Object.entries(this.testResults.responseTimeTests)) {
                if (stats.responseTime > 5000 && stats.available) {
                    bottlenecks.push({
                        type: 'RESPONSE_TIME',
                        service: service,
                        severity: 'MEDIUM',
                        value: `${stats.responseTime}ms`,
                        description: `Service ${service} has slow response time: ${stats.responseTime}ms`
                    });
                    
                    recommendations.push({
                        category: 'Performance Optimization',
                        priority: 'MEDIUM',
                        suggestion: `Investigate ${service} performance and consider caching or resource allocation`
                    });
                }
                
                if (!stats.available) {
                    bottlenecks.push({
                        type: 'AVAILABILITY',
                        service: service,
                        severity: 'CRITICAL',
                        description: `Service ${service} is not responding`
                    });
                    
                    recommendations.push({
                        category: 'Service Health',
                        priority: 'CRITICAL',
                        suggestion: `Check ${service} container health and logs`
                    });
                }
            }
        }
        
        // Container startup bottlenecks
        if (this.testResults.containerStartup) {
            for (const [container, time] of Object.entries(this.testResults.containerStartup)) {
                if (typeof time === 'number' && time > 30000) {
                    bottlenecks.push({
                        type: 'STARTUP_TIME',
                        container: container,
                        severity: 'MEDIUM',
                        value: `${time}ms`,
                        description: `Container ${container} takes ${time}ms to start`
                    });
                    
                    recommendations.push({
                        category: 'Startup Optimization',
                        priority: 'MEDIUM',
                        suggestion: `Optimize ${container} image size and startup configuration`
                    });
                }
            }
        }
        
        this.testResults.bottlenecks = bottlenecks;
        this.testResults.recommendations = recommendations;
    }

    async generateReport() {
        console.log('📋 Generating comprehensive performance report...');
        
        const reportFilename = `performance-report-${Date.now()}.json`;
        const htmlReportFilename = `performance-report-${Date.now()}.html`;
        
        // Save JSON report
        await fs.writeFile(reportFilename, JSON.stringify(this.testResults, null, 2));
        
        // Generate HTML report
        const htmlReport = this.generateHTMLReport();
        await fs.writeFile(htmlReportFilename, htmlReport);
        
        // Print summary to console
        this.printReportSummary();
        
        console.log(`\n📊 Detailed reports saved:`);
        console.log(`   • JSON: ${reportFilename}`);
        console.log(`   • HTML: ${htmlReportFilename}`);
    }

    generateHTMLReport() {
        const bottlenecksHtml = this.testResults.bottlenecks.map(b => 
            `<tr class="${b.severity.toLowerCase()}">
                <td>${b.type}</td>
                <td>${b.container || b.service || '-'}</td>
                <td>${b.severity}</td>
                <td>${b.description}</td>
            </tr>`
        ).join('');
        
        const recommendationsHtml = this.testResults.recommendations.map(r =>
            `<tr class="${r.priority.toLowerCase()}">
                <td>${r.category}</td>
                <td>${r.priority}</td>
                <td>${r.suggestion}</td>
            </tr>`
        ).join('');

        return `
<!DOCTYPE html>
<html>
<head>
    <title>Media Server Performance Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }
        h1, h2 { color: #333; }
        .summary { background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 20px 0; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 10px; text-align: left; border: 1px solid #ddd; }
        th { background-color: #f2f2f2; }
        .critical { background-color: #ffebee; }
        .high { background-color: #fff3e0; }
        .medium { background-color: #f3e5f5; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background: #f8f9fa; border-radius: 5px; }
        .good { color: #4caf50; }
        .warning { color: #ff9800; }
        .error { color: #f44336; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 Media Server Performance Report</h1>
        <div class="summary">
            <strong>Test Date:</strong> ${this.testResults.timestamp}<br>
            <strong>Platform:</strong> ${this.testResults.systemInfo.platform || 'Unknown'}<br>
            <strong>Node Version:</strong> ${this.testResults.systemInfo.nodeVersion || 'Unknown'}
        </div>

        <h2>📊 Performance Overview</h2>
        <div class="metrics">
            <div class="metric">
                <strong>Total Bottlenecks:</strong> ${this.testResults.bottlenecks.length}
            </div>
            <div class="metric">
                <strong>Critical Issues:</strong> ${this.testResults.bottlenecks.filter(b => b.severity === 'CRITICAL').length}
            </div>
            <div class="metric">
                <strong>Services Tested:</strong> ${Object.keys(this.testResults.responseTimeTests || {}).length}
            </div>
        </div>

        <h2>🔍 Identified Bottlenecks</h2>
        <table>
            <thead>
                <tr>
                    <th>Type</th>
                    <th>Component</th>
                    <th>Severity</th>
                    <th>Description</th>
                </tr>
            </thead>
            <tbody>
                ${bottlenecksHtml || '<tr><td colspan="4">No bottlenecks identified</td></tr>'}
            </tbody>
        </table>

        <h2>💡 Recommendations</h2>
        <table>
            <thead>
                <tr>
                    <th>Category</th>
                    <th>Priority</th>
                    <th>Suggestion</th>
                </tr>
            </thead>
            <tbody>
                ${recommendationsHtml || '<tr><td colspan="3">No recommendations at this time</td></tr>'}
            </tbody>
        </table>

        <h2>📈 Detailed Metrics</h2>
        <h3>Container Startup Times</h3>
        <pre>${JSON.stringify(this.testResults.containerStartup, null, 2)}</pre>

        <h3>Memory Usage</h3>
        <pre>${JSON.stringify(this.testResults.memoryUsage, null, 2)}</pre>

        <h3>CPU Utilization</h3>
        <pre>${JSON.stringify(this.testResults.cpuUtilization, null, 2)}</pre>

        <h3>Response Time Tests</h3>
        <pre>${JSON.stringify(this.testResults.responseTimeTests, null, 2)}</pre>
    </div>
</body>
</html>
        `;
    }

    printReportSummary() {
        console.log('\n' + '='.repeat(80));
        console.log('📋 PERFORMANCE TEST SUMMARY');
        console.log('='.repeat(80));
        
        console.log(`\n🕐 Test completed: ${this.testResults.timestamp}`);
        console.log(`🖥️  Platform: ${this.testResults.systemInfo.platform} (${this.testResults.systemInfo.arch})`);
        
        // Bottlenecks summary
        if (this.testResults.bottlenecks.length > 0) {
            console.log(`\n🚨 BOTTLENECKS IDENTIFIED: ${this.testResults.bottlenecks.length}`);
            
            const critical = this.testResults.bottlenecks.filter(b => b.severity === 'CRITICAL');
            const high = this.testResults.bottlenecks.filter(b => b.severity === 'HIGH');
            const medium = this.testResults.bottlenecks.filter(b => b.severity === 'MEDIUM');
            
            if (critical.length > 0) console.log(`   🔴 Critical: ${critical.length}`);
            if (high.length > 0) console.log(`   🟠 High: ${high.length}`);
            if (medium.length > 0) console.log(`   🟡 Medium: ${medium.length}`);
            
            console.log('\n   Top Issues:');
            this.testResults.bottlenecks.slice(0, 3).forEach(b => {
                console.log(`   • ${b.type}: ${b.description}`);
            });
        } else {
            console.log('\n✅ No critical bottlenecks identified!');
        }
        
        // Quick stats
        if (this.testResults.responseTimeTests) {
            const avgResponseTime = Object.values(this.testResults.responseTimeTests)
                .filter(test => test.available && test.responseTime > 0)
                .reduce((sum, test) => sum + test.responseTime, 0) / 
                Object.values(this.testResults.responseTimeTests).filter(test => test.available).length;
                
            if (!isNaN(avgResponseTime)) {
                console.log(`\n⚡ Average Response Time: ${avgResponseTime.toFixed(2)}ms`);
            }
        }
        
        console.log('\n💡 TOP RECOMMENDATIONS:');
        this.testResults.recommendations.slice(0, 3).forEach((rec, index) => {
            console.log(`   ${index + 1}. [${rec.priority}] ${rec.suggestion}`);
        });
        
        console.log('\n' + '='.repeat(80));
    }

    parseMemoryUsage(usage) {
        const match = usage.match(/([0-9.]+)([KMGT]i?B)/);
        if (!match) return 0;
        
        const value = parseFloat(match[1]);
        const unit = match[2];
        
        const multipliers = {
            'B': 1,
            'KB': 1024, 'KiB': 1024,
            'MB': 1024 * 1024, 'MiB': 1024 * 1024,
            'GB': 1024 * 1024 * 1024, 'GiB': 1024 * 1024 * 1024,
            'TB': 1024 * 1024 * 1024 * 1024, 'TiB': 1024 * 1024 * 1024 * 1024
        };
        
        return value * (multipliers[unit] || 1);
    }

    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
    }
}

// Run the test suite if this script is called directly
if (require.main === module) {
    const testSuite = new PerformanceTestSuite();
    testSuite.runFullTestSuite().catch(console.error);
}

module.exports = PerformanceTestSuite;