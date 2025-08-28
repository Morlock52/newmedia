const puppeteer = require('puppeteer');

async function testUI() {
  console.log('🚀 Starting UI Stress Test...\n');
  
  const browser = await puppeteer.launch({ 
    headless: true,
    args: ['--no-sandbox', '--disable-setuid-sandbox']
  });
  
  try {
    const page = await browser.newPage();
    
    // Capture console logs
    const consoleLogs = [];
    page.on('console', msg => {
      consoleLogs.push({
        type: msg.type(),
        text: msg.text()
      });
    });
    
    // Capture errors
    const pageErrors = [];
    page.on('error', err => {
      pageErrors.push(err.message);
    });
    
    page.on('pageerror', err => {
      pageErrors.push(err.message);
    });
    
    // Navigate to the app
    console.log('📍 Navigating to http://localhost:3001...');
    const response = await page.goto('http://localhost:3001', {
      waitUntil: 'networkidle2',
      timeout: 30000
    });
    
    console.log(`📊 Response Status: ${response.status()}\n`);
    
    // Check for critical elements
    console.log('🔍 Checking for critical elements...');
    
    const elements = {
      'Cyberpunk Dashboard': await page.$eval('body', el => el.textContent.includes('MEDIA NEXUS CONTROL')),
      'Service Orbs': await page.$$eval('canvas', els => els.length),
      'Neon Buttons': await page.$$eval('button', els => els.length),
      'Holographic Cards': await page.$$eval('[class*="holo"]', els => els.length),
    };
    
    console.log('Elements found:');
    Object.entries(elements).forEach(([name, count]) => {
      console.log(`  ✓ ${name}: ${count}`);
    });
    
    // Check performance metrics
    console.log('\n📈 Performance Metrics:');
    const metrics = await page.metrics();
    console.log(`  • JS Heap Size: ${(metrics.JSHeapUsedSize / 1024 / 1024).toFixed(2)} MB`);
    console.log(`  • DOM Nodes: ${metrics.Nodes}`);
    console.log(`  • Layout Count: ${metrics.LayoutCount}`);
    console.log(`  • Recalc Styles: ${metrics.RecalcStyleCount}`);
    
    // Test interactivity
    console.log('\n🎮 Testing Interactivity...');
    
    // Click refresh button if exists
    try {
      await page.click('button:has-text("INITIATE SCAN")', { timeout: 5000 });
      console.log('  ✓ Clicked refresh button');
    } catch (e) {
      console.log('  ⚠ Refresh button not found or clickable');
    }
    
    // Check for memory leaks
    console.log('\n🔬 Checking for memory leaks...');
    await page.evaluate(() => {
      if (window.performance && window.performance.memory) {
        return {
          usedJSHeapSize: window.performance.memory.usedJSHeapSize,
          totalJSHeapSize: window.performance.memory.totalJSHeapSize,
          jsHeapSizeLimit: window.performance.memory.jsHeapSizeLimit
        };
      }
    });
    
    // Report console logs
    if (consoleLogs.length > 0) {
      console.log('\n📝 Console Logs:');
      consoleLogs.forEach(log => {
        console.log(`  [${log.type}] ${log.text}`);
      });
    }
    
    // Report errors
    if (pageErrors.length > 0) {
      console.log('\n❌ Page Errors:');
      pageErrors.forEach(err => {
        console.log(`  • ${err}`);
      });
    } else {
      console.log('\n✅ No page errors detected');
    }
    
    // Check accessibility
    console.log('\n♿ Accessibility Check:');
    const accessibilityTree = await page.accessibility.snapshot();
    console.log(`  • Accessible elements: ${accessibilityTree ? 'Yes' : 'No'}`);
    
    // Take screenshot
    await page.screenshot({ path: 'ui-test-screenshot.png', fullPage: true });
    console.log('\n📸 Screenshot saved as ui-test-screenshot.png');
    
  } catch (error) {
    console.error('\n❌ Test failed:', error.message);
  } finally {
    await browser.close();
    console.log('\n✨ UI Stress Test Complete!');
  }
}

testUI().catch(console.error);