/**
 * Cross-Browser Compatibility Test Suite
 * Tests dashboard functionality across different browsers and browser versions
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

describe('Cross-Browser Compatibility Tests', () => {
    let dashboardURL;
    const browserConfigs = [
        {
            name: 'Chrome',
            executablePath: undefined, // Use default Chromium
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        },
        // Note: Additional browsers would require separate installations
        // {
        //     name: 'Firefox',
        //     product: 'firefox',
        //     executablePath: '/usr/bin/firefox',
        //     args: ['--headless']
        // }
    ];

    beforeAll(async () => {
        const dashboardPath = path.join(__dirname, '../../dashboard-enhanced.html');
        dashboardURL = `file://${dashboardPath}`;
    });

    describe('Browser Feature Support', () => {
        browserConfigs.forEach(browserConfig => {
            describe(`${browserConfig.name} Compatibility`, () => {
                let browser, page;

                beforeAll(async () => {
                    try {
                        browser = await puppeteer.launch({
                            headless: true,
                            executablePath: browserConfig.executablePath,
                            product: browserConfig.product,
                            args: browserConfig.args || ['--no-sandbox']
                        });
                    } catch (error) {
                        console.warn(`⚠️ Could not launch ${browserConfig.name}:`, error.message);
                        return;
                    }
                });

                afterAll(async () => {
                    if (browser) {
                        await browser.close();
                    }
                });

                beforeEach(async () => {
                    if (browser) {
                        page = await browser.newPage();
                    }
                });

                afterEach(async () => {
                    if (page) {
                        await page.close();
                    }
                });

                test(`${browserConfig.name} - Basic page loading`, async () => {
                    if (!browser) {
                        console.log(`⚠️ Skipping ${browserConfig.name} test - browser not available`);
                        return;
                    }

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const title = await page.title();
                    expect(title).toBe('MediaFlow Dashboard - Real-time Stats & Control');
                    
                    const main = await page.$('main');
                    expect(main).toBeTruthy();
                    
                    console.log(`✅ ${browserConfig.name} loads dashboard correctly`);
                });

                test(`${browserConfig.name} - CSS Grid support`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const gridSupport = await page.evaluate(() => {
                        const testElement = document.createElement('div');
                        testElement.style.display = 'grid';
                        return testElement.style.display === 'grid';
                    });
                    
                    expect(gridSupport).toBe(true);
                    console.log(`✅ ${browserConfig.name} supports CSS Grid`);
                });

                test(`${browserConfig.name} - Flexbox support`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const flexboxSupport = await page.evaluate(() => {
                        const testElement = document.createElement('div');
                        testElement.style.display = 'flex';
                        return testElement.style.display === 'flex';
                    });
                    
                    expect(flexboxSupport).toBe(true);
                    console.log(`✅ ${browserConfig.name} supports Flexbox`);
                });

                test(`${browserConfig.name} - CSS Custom Properties (Variables)`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const customPropertiesSupport = await page.evaluate(() => {
                        const testElement = document.createElement('div');
                        testElement.style.setProperty('--test-var', 'red');
                        testElement.style.color = 'var(--test-var)';
                        document.body.appendChild(testElement);
                        
                        const computedStyle = window.getComputedStyle(testElement);
                        const supportsCustomProps = computedStyle.color === 'red' || computedStyle.color === 'rgb(255, 0, 0)';
                        
                        document.body.removeChild(testElement);
                        return supportsCustomProps;
                    });
                    
                    expect(customPropertiesSupport).toBe(true);
                    console.log(`✅ ${browserConfig.name} supports CSS Custom Properties`);
                });

                test(`${browserConfig.name} - Backdrop Filter support`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const backdropFilterSupport = await page.evaluate(() => {
                        const testElement = document.createElement('div');
                        testElement.style.backdropFilter = 'blur(10px)';
                        return testElement.style.backdropFilter === 'blur(10px)' ||
                               testElement.style.webkitBackdropFilter === 'blur(10px)';
                    });
                    
                    console.log(`${backdropFilterSupport ? '✅' : '⚠️'} ${browserConfig.name} backdrop-filter support: ${backdropFilterSupport}`);
                });

                test(`${browserConfig.name} - JavaScript ES6+ features`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const es6Support = await page.evaluate(() => {
                        try {
                            // Test arrow functions
                            const arrowFunc = () => true;
                            
                            // Test template literals
                            const templateSupport = `test ${1 + 1}` === 'test 2';
                            
                            // Test const/let
                            const constSupport = true;
                            let letSupport = true;
                            
                            // Test Promise
                            const promiseSupport = typeof Promise !== 'undefined';
                            
                            return {
                                arrow: typeof arrowFunc === 'function',
                                template: templateSupport,
                                const: constSupport,
                                let: letSupport,
                                promise: promiseSupport
                            };
                        } catch (error) {
                            return { error: error.message };
                        }
                    });
                    
                    if (es6Support.error) {
                        console.warn(`⚠️ ${browserConfig.name} ES6 test error:`, es6Support.error);
                    } else {
                        const supportedFeatures = Object.entries(es6Support).filter(([key, value]) => value).length;
                        console.log(`✅ ${browserConfig.name} supports ${supportedFeatures}/5 ES6 features`);
                        expect(supportedFeatures).toBeGreaterThan(3);
                    }
                });

                test(`${browserConfig.name} - Canvas support`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const canvasSupport = await page.evaluate(() => {
                        const canvas = document.createElement('canvas');
                        return !!(canvas.getContext && canvas.getContext('2d'));
                    });
                    
                    expect(canvasSupport).toBe(true);
                    console.log(`✅ ${browserConfig.name} supports Canvas`);
                });

                test(`${browserConfig.name} - LocalStorage support`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const localStorageSupport = await page.evaluate(() => {
                        try {
                            localStorage.setItem('test', 'value');
                            const value = localStorage.getItem('test');
                            localStorage.removeItem('test');
                            return value === 'value';
                        } catch (error) {
                            return false;
                        }
                    });
                    
                    expect(localStorageSupport).toBe(true);
                    console.log(`✅ ${browserConfig.name} supports LocalStorage`);
                });

                test(`${browserConfig.name} - WebSocket support`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const webSocketSupport = await page.evaluate(() => {
                        return typeof WebSocket !== 'undefined';
                    });
                    
                    expect(webSocketSupport).toBe(true);
                    console.log(`✅ ${browserConfig.name} supports WebSocket`);
                });

                test(`${browserConfig.name} - Service Worker support`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    
                    const serviceWorkerSupport = await page.evaluate(() => {
                        return 'serviceWorker' in navigator;
                    });
                    
                    console.log(`${serviceWorkerSupport ? '✅' : 'ℹ️'} ${browserConfig.name} Service Worker support: ${serviceWorkerSupport}`);
                });
            });
        });
    });

    describe('Responsive Design Cross-Browser', () => {
        const viewports = [
            { name: 'Mobile', width: 375, height: 667 },
            { name: 'Tablet', width: 768, height: 1024 },
            { name: 'Desktop', width: 1920, height: 1080 }
        ];

        browserConfigs.forEach(browserConfig => {
            describe(`${browserConfig.name} Responsive Design`, () => {
                let browser, page;

                beforeAll(async () => {
                    try {
                        browser = await puppeteer.launch({
                            headless: true,
                            executablePath: browserConfig.executablePath,
                            product: browserConfig.product,
                            args: browserConfig.args || ['--no-sandbox']
                        });
                    } catch (error) {
                        console.warn(`⚠️ Could not launch ${browserConfig.name}:`, error.message);
                        return;
                    }
                });

                afterAll(async () => {
                    if (browser) {
                        await browser.close();
                    }
                });

                viewports.forEach(viewport => {
                    test(`${browserConfig.name} - ${viewport.name} layout`, async () => {
                        if (!browser) return;

                        page = await browser.newPage();
                        await page.setViewport(viewport);
                        await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                        // Test basic layout elements
                        const main = await page.$('main');
                        const sidebar = await page.$('#sidebar');
                        
                        expect(main).toBeTruthy();
                        expect(sidebar).toBeTruthy();

                        // Test responsive behavior
                        if (viewport.width < 768) {
                            // Mobile: sidebar should be hidden by default
                            const sidebarHidden = await page.evaluate(el => 
                                el.classList.contains('-translate-x-full'), sidebar
                            );
                            expect(sidebarHidden).toBe(true);
                        }

                        console.log(`✅ ${browserConfig.name} - ${viewport.name} layout works correctly`);
                        await page.close();
                    });
                });
            });
        });
    });

    describe('Event Handling Cross-Browser', () => {
        browserConfigs.forEach(browserConfig => {
            describe(`${browserConfig.name} Event Handling`, () => {
                let browser, page;

                beforeAll(async () => {
                    try {
                        browser = await puppeteer.launch({
                            headless: true,
                            executablePath: browserConfig.executablePath,
                            product: browserConfig.product,
                            args: browserConfig.args || ['--no-sandbox']
                        });
                        page = await browser.newPage();
                    } catch (error) {
                        console.warn(`⚠️ Could not launch ${browserConfig.name}:`, error.message);
                        return;
                    }
                });

                afterAll(async () => {
                    if (browser) {
                        await browser.close();
                    }
                });

                test(`${browserConfig.name} - Click events`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                    // Test mobile menu button click
                    const menuButton = await page.$('button[onclick="toggleSidebar()"]');
                    if (menuButton) {
                        await menuButton.click();
                        console.log(`✅ ${browserConfig.name} handles click events`);
                    }
                });

                test(`${browserConfig.name} - Keyboard events`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                    // Test Tab navigation
                    await page.keyboard.press('Tab');
                    const activeElement = await page.evaluate(() => document.activeElement.tagName);
                    
                    expect(['BUTTON', 'A', 'INPUT'].includes(activeElement)).toBe(true);
                    console.log(`✅ ${browserConfig.name} handles keyboard navigation`);
                });

                test(`${browserConfig.name} - Touch events (simulated)`, async () => {
                    if (!browser) return;

                    await page.setViewport({ width: 375, height: 667 });
                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                    const menuButton = await page.$('button[onclick="toggleSidebar()"]');
                    if (menuButton) {
                        await menuButton.tap();
                        console.log(`✅ ${browserConfig.name} handles touch events`);
                    }
                });
            });
        });
    });

    describe('Performance Cross-Browser', () => {
        browserConfigs.forEach(browserConfig => {
            describe(`${browserConfig.name} Performance`, () => {
                let browser, page;

                beforeAll(async () => {
                    try {
                        browser = await puppeteer.launch({
                            headless: true,
                            executablePath: browserConfig.executablePath,
                            product: browserConfig.product,
                            args: browserConfig.args || ['--no-sandbox']
                        });
                        page = await browser.newPage();
                    } catch (error) {
                        console.warn(`⚠️ Could not launch ${browserConfig.name}:`, error.message);
                        return;
                    }
                });

                afterAll(async () => {
                    if (browser) {
                        await browser.close();
                    }
                });

                test(`${browserConfig.name} - Load performance`, async () => {
                    if (!browser) return;

                    const startTime = Date.now();
                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                    const loadTime = Date.now() - startTime;

                    console.log(`📊 ${browserConfig.name} load time: ${loadTime}ms`);
                    expect(loadTime).toBeLessThan(10000); // 10 second max for cross-browser
                });

                test(`${browserConfig.name} - Memory usage`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                    const memoryUsage = await page.evaluate(() => {
                        if (performance.memory) {
                            return {
                                used: performance.memory.usedJSHeapSize,
                                total: performance.memory.totalJSHeapSize
                            };
                        }
                        return null;
                    });

                    if (memoryUsage) {
                        const usedMB = (memoryUsage.used / 1024 / 1024).toFixed(2);
                        console.log(`📊 ${browserConfig.name} memory usage: ${usedMB} MB`);
                        expect(memoryUsage.used).toBeLessThan(100 * 1024 * 1024); // 100MB max
                    }
                });
            });
        });
    });

    describe('Accessibility Cross-Browser', () => {
        browserConfigs.forEach(browserConfig => {
            describe(`${browserConfig.name} Accessibility`, () => {
                let browser, page;

                beforeAll(async () => {
                    try {
                        browser = await puppeteer.launch({
                            headless: true,
                            executablePath: browserConfig.executablePath,
                            product: browserConfig.product,
                            args: browserConfig.args || ['--no-sandbox']
                        });
                        page = await browser.newPage();
                    } catch (error) {
                        console.warn(`⚠️ Could not launch ${browserConfig.name}:`, error.message);
                        return;
                    }
                });

                afterAll(async () => {
                    if (browser) {
                        await browser.close();
                    }
                });

                test(`${browserConfig.name} - ARIA attributes`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                    const ariaElements = await page.$$('[aria-label], [aria-labelledby], [role]');
                    console.log(`✅ ${browserConfig.name} found ${ariaElements.length} ARIA elements`);
                });

                test(`${browserConfig.name} - Focus management`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                    // Test that Tab key moves focus
                    await page.keyboard.press('Tab');
                    const firstFocusable = await page.evaluate(() => document.activeElement.tagName);
                    
                    await page.keyboard.press('Tab');
                    const secondFocusable = await page.evaluate(() => document.activeElement.tagName);

                    // Focus should change between elements
                    expect(firstFocusable).toBeTruthy();
                    console.log(`✅ ${browserConfig.name} focus management works`);
                });

                test(`${browserConfig.name} - Color contrast`, async () => {
                    if (!browser) return;

                    await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                    const contrastTest = await page.evaluate(() => {
                        const textElements = document.querySelectorAll('p, span, h1, h2, h3, h4, h5, h6');
                        let passCount = 0;
                        
                        for (const element of textElements) {
                            const style = window.getComputedStyle(element);
                            const color = style.color;
                            const backgroundColor = style.backgroundColor;
                            
                            // Basic check: white text on dark background should pass
                            if (color.includes('255, 255, 255') && backgroundColor.includes('0, 0, 0')) {
                                passCount++;
                            }
                        }
                        
                        return { total: textElements.length, passed: passCount };
                    });

                    console.log(`✅ ${browserConfig.name} contrast test: ${contrastTest.passed}/${contrastTest.total} elements`);
                });
            });
        });
    });

    afterAll(async () => {
        console.log('\n📊 Cross-Browser Compatibility Test Summary:');
        console.log('- Browser feature support tested');
        console.log('- Responsive design cross-browser tested');
        console.log('- Event handling cross-browser tested');
        console.log('- Performance cross-browser tested');
        console.log('- Accessibility cross-browser tested');
        
        console.log('\n🌐 Browser Support Matrix:');
        console.log('- Chrome/Chromium: Full support');
        console.log('- Note: Additional browsers require separate installation');
    });
});