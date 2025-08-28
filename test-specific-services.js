#!/usr/bin/env node

const http = require('http');

// Test specific services that are running
const runningServices = [
  { name: 'Jellyfin', url: 'http://localhost:8096', path: '/' },
  { name: 'Sonarr', url: 'http://localhost:8989', path: '/system/status' },
  { name: 'Radarr', url: 'http://localhost:7878', path: '/system/status' },
  { name: 'Prowlarr', url: 'http://localhost:9696', path: '/' },
  { name: 'qBittorrent', url: 'http://localhost:8080', path: '/' },
  { name: 'Portainer', url: 'http://localhost:9000', path: '/' },
  { name: 'Uptime Kuma', url: 'http://localhost:3001', path: '/' }
];

async function testService(service) {
  return new Promise((resolve) => {
    const url = new URL(service.url + service.path);
    
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname,
      method: 'GET',
      timeout: 10000,
      headers: {
        'User-Agent': 'Media-Server-Frontend-Tester/1.0'
      }
    };

    console.log(`🔍 Testing ${service.name} at ${service.url}${service.path}...`);

    const startTime = Date.now();
    const req = http.request(options, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        const responseTime = Date.now() - startTime;
        const result = {
          service: service.name,
          url: service.url,
          status: res.statusCode,
          responseTime: responseTime,
          accessible: res.statusCode < 500, // Accept redirects and auth errors as "accessible"
          hasHtml: data.toLowerCase().includes('<html') || data.toLowerCase().includes('<!doctype'),
          title: (data.match(/<title[^>]*>([^<]+)<\/title>/i) || [])[1] || '',
          isSetupPage: data.includes('setup') || data.includes('wizard') || data.includes('install'),
          isLoginPage: data.includes('login') || data.includes('sign in') || data.includes('authentication'),
          isApi: data.includes('api') || res.headers['content-type']?.includes('json')
        };

        const statusEmoji = result.accessible ? '✅' : '❌';
        const statusText = result.status === 401 ? 'Auth Required' : result.status;
        console.log(`${statusEmoji} ${service.name}: ${statusText} (${responseTime}ms)`);
        
        if (result.title) {
          console.log(`   📄 Title: "${result.title}"`);
        }
        
        if (result.isSetupPage) {
          console.log(`   🔧 Setup page detected`);
        } else if (result.isLoginPage) {
          console.log(`   🔑 Login page detected`);
        } else if (result.isApi) {
          console.log(`   🔌 API endpoint detected`);
        }
        
        resolve(result);
      });
    });

    req.on('error', (err) => {
      const responseTime = Date.now() - startTime;
      console.log(`❌ ${service.name}: ${err.message} (${responseTime}ms)`);
      resolve({
        service: service.name,
        url: service.url,
        error: err.message,
        accessible: false,
        responseTime: responseTime
      });
    });

    req.on('timeout', () => {
      req.destroy();
      console.log(`⏱️ ${service.name}: Request timeout`);
      resolve({
        service: service.name,
        url: service.url,
        error: 'Timeout',
        accessible: false,
        responseTime: Date.now() - startTime
      });
    });

    req.end();
  });
}

async function runTests() {
  console.log('🚀 Testing Running Media Services...\n');
  
  const results = [];
  
  for (const service of runningServices) {
    const result = await testService(service);
    results.push(result);
    console.log(''); // Add spacing
  }
  
  // Summary
  const accessible = results.filter(r => r.accessible);
  const inaccessible = results.filter(r => !r.accessible);
  
  console.log('📊 DETAILED TEST SUMMARY:');
  console.log(`Total Services: ${results.length}`);
  console.log(`✅ Accessible: ${accessible.length}`);
  console.log(`❌ Inaccessible: ${inaccessible.length}`);
  console.log(`📈 Success Rate: ${Math.round((accessible.length/results.length)*100)}%`);
  
  if (accessible.length > 0) {
    console.log('\n✅ ACCESSIBLE SERVICES:');
    accessible.forEach(r => {
      console.log(`   ${r.service} (${r.url}) - ${r.status} - ${r.responseTime}ms`);
    });
  }
  
  if (inaccessible.length > 0) {
    console.log('\n❌ INACCESSIBLE SERVICES:');
    inaccessible.forEach(r => {
      console.log(`   ${r.service} (${r.url}) - ${r.error || r.status}`);
    });
  }

  console.log('\n🔍 INTERFACE ANALYSIS:');
  accessible.forEach(result => {
    console.log(`\n📋 ${result.service}:`);
    console.log(`   URL: ${result.url}`);
    console.log(`   Status: ${result.status}`);
    console.log(`   Response Time: ${result.responseTime}ms`);
    console.log(`   Has HTML UI: ${result.hasHtml ? '✅ Yes' : '❌ No'}`);
    console.log(`   Page Title: ${result.title || 'Not detected'}`);
    console.log(`   Interface Type: ${
      result.isSetupPage ? '🔧 Setup/Configuration' :
      result.isLoginPage ? '🔑 Login Required' :
      result.isApi ? '🔌 API Endpoint' :
      result.hasHtml ? '🖥️ Web Interface' : '❓ Unknown'
    }`);
  });
  
  return results;
}

if (require.main === module) {
  runTests().catch(console.error);
}

module.exports = { runTests, testService };