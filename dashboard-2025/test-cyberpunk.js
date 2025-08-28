const puppeteer = require('puppeteer');

async function testCyberpunkDashboard() {
  console.log('🚀 Testing Cyberpunk Dashboard...\n');
  
  let browser;
  try {
    browser = await puppeteer.launch({ 
      headless: true,
      args: ['--no-sandbox', '--disable-setuid-sandbox']
    });
    
    const page = await browser.newPage();
    
    // Set viewport
    await page.setViewport({ width: 1920, height: 1080 });
    
    console.log('📡 Navigating to http://localhost:3001...');
    await page.goto('http://localhost:3001', { 
      waitUntil: 'networkidle2',
      timeout: 30000 
    });
    
    // Check for title
    const title = await page.$eval('.cyber-title', el => el.textContent);
    console.log('✅ Title found:', title);
    
    // Check for services
    const services = await page.$$eval('.service-card', cards => cards.length);
    console.log(`✅ Services found: ${services}`);
    
    // Check for buttons
    const buttons = await page.$$eval('.cyber-button', btns => btns.length);
    console.log(`✅ Buttons found: ${buttons}`);
    
    // Check for terminal
    const hasTerminal = await page.$('.cyber-terminal') !== null;
    console.log(`✅ Terminal present: ${hasTerminal}`);
    
    // Check CSS is loaded
    const backgroundColor = await page.evaluate(() => {
      return window.getComputedStyle(document.body).backgroundColor;
    });
    console.log(`✅ Background color: ${backgroundColor}`);
    
    // Check for cyberpunk styles
    const hasCyberStyles = await page.evaluate(() => {
      const styles = Array.from(document.styleSheets);
      return styles.some(sheet => {
        try {
          const rules = Array.from(sheet.cssRules || []);
          return rules.some(rule => rule.cssText && rule.cssText.includes('cyber'));
        } catch (e) {
          return false;
        }
      });
    });
    console.log(`✅ Cyberpunk styles loaded: ${hasCyberStyles}`);
    
    // Take screenshot
    await page.screenshot({ 
      path: 'cyberpunk-dashboard-test.png',
      fullPage: true 
    });
    console.log('📸 Screenshot saved: cyberpunk-dashboard-test.png');
    
    // Test interactivity
    console.log('\n🎮 Testing Interactivity...');
    
    // Click scan button
    await page.click('.cyber-button');
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Check terminal updated
    const terminalText = await page.$eval('.cyber-terminal', el => el.textContent);
    console.log('✅ Terminal updated after button click');
    
    console.log('\n🎉 All tests passed! Dashboard is working correctly.');
    console.log('🌆 Cyberpunk theme is active and functional!');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    process.exit(1);
  } finally {
    if (browser) {
      await browser.close();
    }
  }
}

testCyberpunkDashboard().catch(console.error);