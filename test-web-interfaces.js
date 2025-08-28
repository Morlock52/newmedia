#!/usr/bin/env node

const http = require('http');
const https = require('https');
const fs = require('fs');
const path = require('path');

// Service configuration
const services = [
  { name: 'Jellyfin', url: 'http://localhost:8096', path: '/' },
  { name: 'Sonarr', url: 'http://localhost:8989', path: '/' },
  { name: 'Radarr', url: 'http://localhost:7878', path: '/' },
  { name: 'Prowlarr', url: 'http://localhost:9696', path: '/' },
  { name: 'qBittorrent', url: 'http://localhost:8080', path: '/' },
  { name: 'Bazarr', url: 'http://localhost:6767', path: '/' },
  { name: 'Lidarr', url: 'http://localhost:8686', path: '/' },
  { name: 'SABnzbd', url: 'http://localhost:8085', path: '/' },
  { name: 'Transmission', url: 'http://localhost:9091', path: '/' },
  { name: 'Jellyseerr', url: 'http://localhost:5055', path: '/' },
  { name: 'Tautulli', url: 'http://localhost:8181', path: '/' },
  { name: 'Portainer', url: 'http://localhost:9000', path: '/' },
  { name: 'Uptime Kuma', url: 'http://localhost:3001', path: '/' },
  { name: 'Nginx Proxy Manager', url: 'http://localhost:81', path: '/' },
  { name: 'Dashboard', url: 'http://localhost:3000', path: '/' }
];

const testResults = [];

// Test a single service
function testService(service) {
  return new Promise((resolve) => {
    const url = new URL(service.url + service.path);
    const client = url.protocol === 'https:' ? https : http;
    
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'GET',
      timeout: 10000,
      headers: {
        'User-Agent': 'Media-Server-Tester/1.0'
      }
    };

    console.log(`Testing ${service.name} at ${service.url}${service.path}...`);

    const req = client.request(options, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        const result = {
          service: service.name,
          url: service.url,
          status: res.statusCode,
          headers: res.headers,
          accessible: res.statusCode < 400,
          responseTime: Date.now() - startTime,
          contentLength: data.length,
          hasHtml: data.toLowerCase().includes('<html'),
          hasTitle: /<title[^>]*>([^<]+)<\/title>/i.test(data),
          title: (data.match(/<title[^>]*>([^<]+)<\/title>/i) || [])[1] || '',
          hasJavaScriptErrors: checkForJSErrors(data),
          hasLoginForm: hasLoginForm(data),
          redirects: res.statusCode >= 300 && res.statusCode < 400 ? res.headers.location : null
        };

        console.log(`✅ ${service.name}: ${res.statusCode} (${result.responseTime}ms)`);
        resolve(result);
      });
    });

    req.on('error', (err) => {
      const result = {
        service: service.name,
        url: service.url,
        status: 'ERROR',
        error: err.message,
        accessible: false,
        responseTime: Date.now() - startTime
      };

      console.log(`❌ ${service.name}: ${err.message}`);
      resolve(result);
    });

    req.on('timeout', () => {
      req.destroy();
      const result = {
        service: service.name,
        url: service.url,
        status: 'TIMEOUT',
        error: 'Request timeout',
        accessible: false,
        responseTime: Date.now() - startTime
      };

      console.log(`⏱️ ${service.name}: Request timeout`);
      resolve(result);
    });

    const startTime = Date.now();
    req.end();
  });
}

// Check for common JavaScript errors in HTML
function checkForJSErrors(html) {
  const errorPatterns = [
    /script.*error/i,
    /javascript.*error/i,
    /uncaught/i,
    /undefined.*function/i,
    /failed.*to.*load/i,
    /404.*\.js/i,
    /500.*\.js/i
  ];

  return errorPatterns.some(pattern => pattern.test(html));
}

// Check for login forms
function hasLoginForm(html) {
  const loginPatterns = [
    /<form[^>]*login/i,
    /<input[^>]*password/i,
    /<input[^>]*username/i,
    /type=['"](password|email)['"]|name=['"](password|username|email)['"]/i,
    /login|sign.*in|authentication/i
  ];

  return loginPatterns.some(pattern => pattern.test(html));
}

// Generate detailed report
function generateReport(results) {
  const timestamp = new Date().toISOString();
  const accessibleCount = results.filter(r => r.accessible).length;
  const totalCount = results.length;

  let report = `# Media Server Web Interface Test Report
Generated: ${timestamp}
Total Services: ${totalCount}
Accessible: ${accessibleCount}/${totalCount} (${Math.round((accessibleCount/totalCount)*100)}%)

## Summary

### ✅ Accessible Services (${accessibleCount})
${results.filter(r => r.accessible).map(r => 
  `- **${r.service}** (${r.url}) - Status: ${r.status} - Response: ${r.responseTime}ms${r.title ? ` - Title: "${r.title}"` : ''}`
).join('\n')}

### ❌ Inaccessible Services (${totalCount - accessibleCount})
${results.filter(r => !r.accessible).map(r => 
  `- **${r.service}** (${r.url}) - Error: ${r.error || r.status}`
).join('\n')}

## Detailed Results

`;

  results.forEach(result => {
    report += `### ${result.service}
- **URL**: ${result.url}
- **Status**: ${result.status}
- **Accessible**: ${result.accessible ? '✅ Yes' : '❌ No'}
- **Response Time**: ${result.responseTime}ms
`;

    if (result.accessible) {
      report += `- **Content Length**: ${result.contentLength} bytes
- **Has HTML**: ${result.hasHtml ? '✅ Yes' : '❌ No'}
- **Page Title**: ${result.title || 'Not found'}
- **Has Login Form**: ${result.hasLoginForm ? '✅ Yes' : '❌ No'}
- **JavaScript Errors Detected**: ${result.hasJavaScriptErrors ? '⚠️ Yes' : '✅ No'}
`;

      if (result.redirects) {
        report += `- **Redirects To**: ${result.redirects}
`;
      }
    } else {
      report += `- **Error**: ${result.error || 'Unknown error'}
`;
    }

    report += '\n';
  });

  return report;
}

// Main test function
async function runTests() {
  console.log('🚀 Starting Media Server Web Interface Tests...\n');
  
  const startTime = Date.now();
  
  // Test all services in parallel with some concurrency control
  const concurrency = 5;
  const results = [];
  
  for (let i = 0; i < services.length; i += concurrency) {
    const batch = services.slice(i, i + concurrency);
    const batchResults = await Promise.all(batch.map(testService));
    results.push(...batchResults);
    
    // Small delay between batches to avoid overwhelming the system
    if (i + concurrency < services.length) {
      await new Promise(resolve => setTimeout(resolve, 1000));
    }
  }
  
  const totalTime = Date.now() - startTime;
  
  console.log(`\n🏁 Tests completed in ${totalTime}ms\n`);
  
  // Generate and save report
  const report = generateReport(results);
  const reportPath = path.join(__dirname, 'web-interface-test-report.md');
  
  fs.writeFileSync(reportPath, report);
  console.log(`📄 Report saved to: ${reportPath}`);
  
  // Print summary to console
  const accessible = results.filter(r => r.accessible);
  const inaccessible = results.filter(r => !r.accessible);
  
  console.log('\n📊 SUMMARY:');
  console.log(`Total Services Tested: ${results.length}`);
  console.log(`✅ Accessible: ${accessible.length}`);
  console.log(`❌ Inaccessible: ${inaccessible.length}`);
  console.log(`📈 Success Rate: ${Math.round((accessible.length/results.length)*100)}%`);
  
  if (accessible.length > 0) {
    console.log('\n✅ WORKING SERVICES:');
    accessible.forEach(r => {
      console.log(`  - ${r.service} (${r.url}) - ${r.status} - ${r.responseTime}ms`);
    });
  }
  
  if (inaccessible.length > 0) {
    console.log('\n❌ FAILED SERVICES:');
    inaccessible.forEach(r => {
      console.log(`  - ${r.service} (${r.url}) - ${r.error || r.status}`);
    });
  }
  
  return results;
}

// Run the tests
if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = { runTests, testService };