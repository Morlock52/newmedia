#!/usr/bin/env node

const http = require('http');

// Test actual web UI pages for services
function buildTests() {
  const domain = process.env.DOMAIN;
  if (domain) {
    const https = process.env.PROTOCOL || "https";
    const u = (sub, p = "/") => ({ name: `${sub} Web UI`, url: `${https}://${sub}.${domain}`, path: p });
    return [
      u("jellyfin"),
      u("sonarr"),
      u("radarr"),
      u("prowlarr"),
      u("qbittorrent"),
      u("uptime"),
      u("grafana"),
      u("prometheus"),
      u("traefik")
    ];
  }
  return [
  { name: 'Jellyfin Web UI', url: 'http://localhost:8096', path: '/web/index.html' },
  { name: 'Jellyfin Setup', url: 'http://localhost:8096', path: '/web/index.html#!/wizardstart.html' },
  { name: 'Sonarr Web UI', url: 'http://localhost:8989', path: '/' },
  { name: 'Radarr Web UI', url: 'http://localhost:7878', path: '/' },
  { name: 'Prowlarr Web UI', url: 'http://localhost:9696', path: '/' },
  { name: 'qBittorrent Login', url: 'http://localhost:8080', path: '/' },
  { name: 'Portainer UI', url: 'http://localhost:9000', path: '/' },
  { name: 'Uptime Kuma Setup', url: 'http://localhost:3001', path: '/' }
];

async function testUIPage(test) {
  return new Promise((resolve) => {
    const url = new URL(test.url + test.path);
    
    const options = {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
      method: 'GET',
      timeout: 15000,
      headers: {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive'
      }
    };

    console.log(`🌐 Testing ${test.name}...`);

    const startTime = Date.now();
    const req = http.request(options, (res) => {
      let data = '';
      
      res.on('data', (chunk) => {
        data += chunk;
      });

      res.on('end', () => {
        const responseTime = Date.now() - startTime;
        
        // Analyze the response
        const hasHtml = data.toLowerCase().includes('<html') || data.toLowerCase().includes('<!doctype');
        const hasJavaScript = data.includes('<script') || data.includes('javascript:');
        const hasCss = data.includes('<style') || data.includes('css') || data.includes('<link');
        const hasLoginForm = /type=['"]password['"]|<form[^>]*login|name=['"]username['"]|name=['"]password['"]/.test(data);
        const hasSetupWizard = /setup|wizard|install|configure/.test(data.toLowerCase());
        const hasNavigation = /<nav|<menu|navbar|navigation/.test(data.toLowerCase());
        const hasButtons = /<button|<input[^>]*button|btn-/.test(data);
        const title = (data.match(/<title[^>]*>([^<]+)<\/title>/i) || [])[1] || '';
        
        // Check for JavaScript errors in HTML
        const jsErrorPatterns = [
          /script.*error/i,
          /javascript.*error/i,
          /uncaught/i,
          /undefined.*function/i,
          /failed.*to.*load/i,
          /404.*\.js/i,
          /500.*\.js/i,
          /console\.error/i
        ];
        const hasJsErrors = jsErrorPatterns.some(pattern => pattern.test(data));
        
        // Check for broken assets
        const brokenAssetPatterns = [
          /404.*\.(css|js|png|jpg|ico)/i,
          /failed.*to.*load.*\.(css|js|png|jpg|ico)/i,
          /error.*loading.*\.(css|js|png|jpg|ico)/i
        ];
        const hasBrokenAssets = brokenAssetPatterns.some(pattern => pattern.test(data));
        
        const result = {
          test: test.name,
          url: test.url + test.path,
          status: res.statusCode,
          responseTime: responseTime,
          accessible: res.statusCode < 400,
          title: title,
          hasHtml: hasHtml,
          hasJavaScript: hasJavaScript,
          hasCss: hasCss,
          hasLoginForm: hasLoginForm,
          hasSetupWizard: hasSetupWizard,
          hasNavigation: hasNavigation,
          hasButtons: hasButtons,
          hasJsErrors: hasJsErrors,
          hasBrokenAssets: hasBrokenAssets,
          contentLength: data.length,
          isFullyFunctional: hasHtml && (hasJavaScript || hasButtons) && !hasJsErrors && !hasBrokenAssets
        };

        const statusEmoji = result.accessible ? '✅' : '❌';
        const functionalEmoji = result.isFullyFunctional ? '🟢' : '🟡';
        
        console.log(`${statusEmoji} ${test.name}: ${res.statusCode} (${responseTime}ms) ${functionalEmoji}`);
        
        if (title) {
          console.log(`   📄 Title: "${title}"`);
        }
        
        // UI Analysis
        const features = [];
        if (hasLoginForm) features.push('🔑 Login Form');
        if (hasSetupWizard) features.push('🔧 Setup Wizard');
        if (hasNavigation) features.push('🧭 Navigation');
        if (hasButtons) features.push('🔘 Interactive Elements');
        if (hasJavaScript) features.push('⚡ JavaScript');
        if (hasCss) features.push('🎨 Styling');
        
        if (features.length > 0) {
          console.log(`   Features: ${features.join(', ')}`);
        }
        
        // Issues
        const issues = [];
        if (hasJsErrors) issues.push('❌ JS Errors');
        if (hasBrokenAssets) issues.push('❌ Broken Assets');
        if (!hasHtml && res.statusCode === 200) issues.push('⚠️ No HTML Content');
        
        if (issues.length > 0) {
          console.log(`   Issues: ${issues.join(', ')}`);
        }
        
        resolve(result);
      });
    });

    req.on('error', (err) => {
      const responseTime = Date.now() - startTime;
      console.log(`❌ ${test.name}: ${err.message} (${responseTime}ms)`);
      resolve({
        test: test.name,
        url: test.url + test.path,
        error: err.message,
        accessible: false,
        responseTime: responseTime,
        isFullyFunctional: false
      });
    });

    req.on('timeout', () => {
      req.destroy();
      console.log(`⏱️ ${test.name}: Request timeout`);
      resolve({
        test: test.name,
        url: test.url + test.path,
        error: 'Timeout',
        accessible: false,
        responseTime: Date.now() - startTime,
        isFullyFunctional: false
      });
    });

    req.end();
  });
}

async function runUITests() {
  console.log('🌐 Testing Web Interface Functionality...\n');
  
  const results = [];
  
  const uiTests = buildTests();
  for (const test of uiTests) {
    const result = await testUIPage(test);
    results.push(result);
    console.log(''); // Add spacing
    
    // Small delay between requests to avoid overwhelming services
    await new Promise(resolve => setTimeout(resolve, 500));
  }
  
  // Generate summary
  const accessible = results.filter(r => r.accessible);
  const functional = results.filter(r => r.isFullyFunctional);
  const withIssues = results.filter(r => r.accessible && !r.isFullyFunctional);
  
  console.log('📊 WEB INTERFACE FUNCTIONALITY REPORT:');
  console.log(`Total Interfaces Tested: ${results.length}`);
  console.log(`✅ Accessible: ${accessible.length}/${results.length} (${Math.round((accessible.length/results.length)*100)}%)`);
  console.log(`🟢 Fully Functional: ${functional.length}/${results.length} (${Math.round((functional.length/results.length)*100)}%)`);
  console.log(`🟡 Accessible with Issues: ${withIssues.length}`);
  console.log(`❌ Inaccessible: ${results.length - accessible.length}`);
  
  if (functional.length > 0) {
    console.log('\n🟢 FULLY FUNCTIONAL WEB INTERFACES:');
    functional.forEach(r => {
      console.log(`   ${r.test} - ${r.responseTime}ms - "${r.title}"`);
    });
  }
  
  if (withIssues.length > 0) {
    console.log('\n🟡 ACCESSIBLE BUT WITH POTENTIAL ISSUES:');
    withIssues.forEach(r => {
      console.log(`   ${r.test} - ${r.responseTime}ms - "${r.title}"`);
    });
  }
  
  const failed = results.filter(r => !r.accessible);
  if (failed.length > 0) {
    console.log('\n❌ INACCESSIBLE INTERFACES:');
    failed.forEach(r => {
      console.log(`   ${r.test} - ${r.error || 'Unknown error'}`);
    });
  }

  console.log('\n🔍 DETAILED INTERFACE ANALYSIS:');
  accessible.forEach(result => {
    console.log(`\n📋 ${result.test}:`);
    console.log(`   URL: ${result.url}`);
    console.log(`   Status: ${result.status}`);
    console.log(`   Response Time: ${result.responseTime}ms`);
    console.log(`   Functionality: ${result.isFullyFunctional ? '🟢 Fully Functional' : '🟡 Partial/Issues'}`);
    console.log(`   Content Size: ${result.contentLength} bytes`);
  });
  
  return results;
}

if (require.main === module) {
  runUITests().catch(console.error);
}

module.exports = { runUITests };