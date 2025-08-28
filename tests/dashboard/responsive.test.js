/**
 * Responsive Design Test Suite
 * Tests dashboard responsiveness across different screen sizes and devices
 */

const puppeteer = require('puppeteer');
const fs = require('fs').promises;
const path = require('path');

describe('Responsive Design Tests', () => {
    let browser, page;
    let dashboardURL;

    const viewportSizes = [
        { name: 'Mobile Portrait', width: 375, height: 667 },
        { name: 'Mobile Landscape', width: 667, height: 375 },
        { name: 'Tablet Portrait', width: 768, height: 1024 },
        { name: 'Tablet Landscape', width: 1024, height: 768 },
        { name: 'Desktop Small', width: 1280, height: 720 },
        { name: 'Desktop Large', width: 1920, height: 1080 },
        { name: 'Ultrawide', width: 2560, height: 1440 }
    ];

    beforeAll(async () => {
        browser = await puppeteer.launch({
            headless: true,
            args: ['--no-sandbox', '--disable-setuid-sandbox']
        });
        
        // Set up local file URL
        const dashboardPath = path.join(__dirname, '../../dashboard-enhanced.html');
        dashboardURL = `file://${dashboardPath}`;
    });

    afterAll(async () => {
        if (browser) {
            await browser.close();
        }
    });

    beforeEach(async () => {
        page = await browser.newPage();
    });

    afterEach(async () => {
        if (page) {
            await page.close();
        }
    });

    describe('Viewport Responsive Tests', () => {
        viewportSizes.forEach(viewport => {
            test(`Dashboard renders correctly on ${viewport.name} (${viewport.width}x${viewport.height})`, async () => {
                await page.setViewport(viewport);
                await page.goto(dashboardURL, { waitUntil: 'networkidle0' });

                // Wait for the page to load
                await page.waitForSelector('main', { timeout: 5000 });

                // Take screenshot for visual verification
                const screenshotPath = path.join(__dirname, '../reports', `screenshot-${viewport.name.toLowerCase().replace(' ', '-')}.png`);
                await page.screenshot({ 
                    path: screenshotPath, 
                    fullPage: true 
                });

                // Test basic layout elements are visible
                const main = await page.$('main');
                expect(main).toBeTruthy();

                const sidebar = await page.$('#sidebar');
                expect(sidebar).toBeTruthy();

                // Test responsive grid
                const statsGrid = await page.$('.grid');
                expect(statsGrid).toBeTruthy();
                
                console.log(`✅ ${viewport.name} layout verified`);
            });
        });
    });

    describe('Mobile-Specific Tests', () => {
        beforeEach(async () => {
            await page.setViewport({ width: 375, height: 667 });
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
        });

        test('Mobile menu button is visible and functional', async () => {
            // Check if mobile menu button exists
            const menuButton = await page.$('button[onclick="toggleSidebar()"]');
            expect(menuButton).toBeTruthy();

            // Check if button is visible on mobile
            const isVisible = await page.evaluate(el => {
                const style = window.getComputedStyle(el);
                return style.display !== 'none' && style.visibility !== 'hidden';
            }, menuButton);
            
            expect(isVisible).toBe(true);
            console.log('✅ Mobile menu button is visible');
        });

        test('Sidebar is hidden by default on mobile', async () => {
            const sidebar = await page.$('#sidebar');
            const hasHiddenClass = await page.evaluate(el => 
                el.classList.contains('-translate-x-full'), sidebar
            );
            
            expect(hasHiddenClass).toBe(true);
            console.log('✅ Sidebar is hidden on mobile');
        });

        test('Toggle sidebar functionality works on mobile', async () => {
            const menuButton = await page.$('button[onclick="toggleSidebar()"]');
            const sidebar = await page.$('#sidebar');
            
            // Initial state - sidebar should be hidden
            let isHidden = await page.evaluate(el => 
                el.classList.contains('-translate-x-full'), sidebar
            );
            expect(isHidden).toBe(true);
            
            // Click to show sidebar
            await menuButton.click();
            await page.waitForTimeout(300); // Wait for animation
            
            isHidden = await page.evaluate(el => 
                el.classList.contains('-translate-x-full'), sidebar
            );
            expect(isHidden).toBe(false);
            
            console.log('✅ Sidebar toggle works on mobile');
        });

        test('Service cards stack properly on mobile', async () => {
            const serviceCards = await page.$$('.service-card');
            expect(serviceCards.length).toBeGreaterThan(0);
            
            // Check if cards are stacked vertically (mobile-first approach)
            const firstCard = serviceCards[0];
            const firstCardRect = await firstCard.boundingBox();
            
            if (serviceCards.length > 1) {
                const secondCard = serviceCards[1];
                const secondCardRect = await secondCard.boundingBox();
                
                // On mobile, cards should stack vertically
                expect(secondCardRect.y).toBeGreaterThan(firstCardRect.y + firstCardRect.height - 10);
            }
            
            console.log('✅ Service cards stack properly on mobile');
        });
    });

    describe('Tablet-Specific Tests', () => {
        beforeEach(async () => {
            await page.setViewport({ width: 768, height: 1024 });
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
        });

        test('Tablet layout uses appropriate grid columns', async () => {
            const statsGrid = await page.$('.grid');
            const gridClasses = await page.evaluate(el => el.className, statsGrid);
            
            // Should have md: classes for tablet layouts
            expect(gridClasses).toMatch(/md:grid-cols-/);
            console.log('✅ Tablet grid layout configured');
        });

        test('Sidebar is visible on tablet', async () => {
            const sidebar = await page.$('#sidebar');
            const isVisible = await page.evaluate(el => {
                const style = window.getComputedStyle(el);
                return !el.classList.contains('-translate-x-full') || 
                       style.transform === 'translateX(0px)' || 
                       style.transform === 'none';
            }, sidebar);
            
            // On tablet and larger screens, sidebar should be visible
            expect(isVisible).toBe(true);
            console.log('✅ Sidebar is visible on tablet');
        });
    });

    describe('Desktop-Specific Tests', () => {
        beforeEach(async () => {
            await page.setViewport({ width: 1920, height: 1080 });
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
        });

        test('Desktop layout uses full grid columns', async () => {
            const statsGrid = await page.$('.grid');
            const gridClasses = await page.evaluate(el => el.className, statsGrid);
            
            // Should have lg: classes for desktop layouts
            expect(gridClasses).toMatch(/lg:grid-cols-/);
            console.log('✅ Desktop grid layout configured');
        });

        test('Charts are properly sized on desktop', async () => {
            const chartCanvas = await page.$('#activityChart');
            if (chartCanvas) {
                const chartRect = await chartCanvas.boundingBox();
                
                // Chart should have reasonable dimensions on desktop
                expect(chartRect.width).toBeGreaterThan(300);
                expect(chartRect.height).toBeGreaterThan(150);
                console.log(`✅ Chart dimensions: ${chartRect.width}x${chartRect.height}`);
            }
        });

        test('All service cards are visible in grid layout', async () => {
            const serviceCards = await page.$$('.service-card');
            
            for (let i = 0; i < serviceCards.length; i++) {
                const card = serviceCards[i];
                const isVisible = await page.evaluate(el => {
                    const rect = el.getBoundingClientRect();
                    return rect.width > 0 && rect.height > 0;
                }, card);
                
                expect(isVisible).toBe(true);
            }
            
            console.log(`✅ All ${serviceCards.length} service cards are visible`);
        });
    });

    describe('Touch and Interaction Tests', () => {
        test('Touch targets are appropriately sized', async () => {
            await page.setViewport({ width: 375, height: 667 });
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            // Check button sizes
            const buttons = await page.$$('button');
            
            for (const button of buttons) {
                const rect = await button.boundingBox();
                if (rect) {
                    // Touch targets should be at least 44px (Apple) or 48px (Google) in size
                    expect(Math.min(rect.width, rect.height)).toBeGreaterThanOrEqual(40);
                }
            }
            
            console.log(`✅ Touch targets appropriately sized (${buttons.length} buttons checked)`);
        });

        test('Hover effects work on desktop', async () => {
            await page.setViewport({ width: 1920, height: 1080 });
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            const serviceCard = await page.$('.service-card');
            if (serviceCard) {
                // Get initial styles
                const initialClasses = await page.evaluate(el => el.className, serviceCard);
                
                // Hover over the card
                await serviceCard.hover();
                await page.waitForTimeout(100);
                
                // Check if hover classes are applied (this is implicit through CSS)
                // We can't directly test CSS :hover, but we can test the structure is correct
                expect(initialClasses).toMatch(/hover:/);
                console.log('✅ Hover effects structure is present');
            }
        });
    });

    describe('Performance on Different Viewports', () => {
        test('Page load performance across viewports', async () => {
            const performanceResults = [];
            
            for (const viewport of viewportSizes.slice(0, 3)) { // Test first 3 viewports
                await page.setViewport(viewport);
                
                const startTime = Date.now();
                await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
                const loadTime = Date.now() - startTime;
                
                performanceResults.push({
                    viewport: viewport.name,
                    loadTime
                });
                
                expect(loadTime).toBeLessThan(5000); // Should load within 5 seconds
            }
            
            console.log('📊 Performance Results:');
            performanceResults.forEach(result => {
                console.log(`  ${result.viewport}: ${result.loadTime}ms`);
            });
        });
    });

    describe('Content Reflow Tests', () => {
        test('Content reflows properly when resizing', async () => {
            await page.setViewport({ width: 1920, height: 1080 });
            await page.goto(dashboardURL, { waitUntil: 'networkidle0' });
            
            // Get initial layout
            const initialLayout = await page.evaluate(() => {
                const main = document.querySelector('main');
                const sidebar = document.querySelector('#sidebar');
                return {
                    mainRect: main.getBoundingClientRect(),
                    sidebarRect: sidebar.getBoundingClientRect()
                };
            });
            
            // Resize to mobile
            await page.setViewport({ width: 375, height: 667 });
            await page.waitForTimeout(500); // Wait for reflow
            
            // Get new layout
            const mobileLayout = await page.evaluate(() => {
                const main = document.querySelector('main');
                const sidebar = document.querySelector('#sidebar');
                return {
                    mainRect: main.getBoundingClientRect(),
                    sidebarRect: sidebar.getBoundingClientRect()
                };
            });
            
            // Main content should expand when sidebar is hidden
            expect(mobileLayout.mainRect.width).toBeGreaterThan(initialLayout.mainRect.width * 0.8);
            console.log('✅ Content reflows properly on resize');
        });
    });
});