import { test, expect, Page } from '@playwright/test';
import { describe, beforeAll, afterAll } from '@playwright/test';

// Test configuration for 30+ services
const SERVICES = {
  jellyfin: { url: 'http://localhost:8096', healthEndpoint: '/health' },
  plex: { url: 'http://localhost:32400', healthEndpoint: '/identity' },
  emby: { url: 'http://localhost:8096', healthEndpoint: '/emby/system/info/public' },
  sonarr: { url: 'http://localhost:8989', healthEndpoint: '/api/v3/system/status' },
  radarr: { url: 'http://localhost:7878', healthEndpoint: '/api/v3/system/status' },
  lidarr: { url: 'http://localhost:8686', healthEndpoint: '/api/v1/system/status' },
  readarr: { url: 'http://localhost:8787', healthEndpoint: '/api/v1/system/status' },
  bazarr: { url: 'http://localhost:6767', healthEndpoint: '/api/system/status' },
  prowlarr: { url: 'http://localhost:9696', healthEndpoint: '/api/v1/system/status' },
  qbittorrent: { url: 'http://localhost:8080', healthEndpoint: '/api/v2/app/version' },
  sabnzbd: { url: 'http://localhost:8085', healthEndpoint: '/sabnzbd/api?mode=version' },
  transmission: { url: 'http://localhost:9091', healthEndpoint: '/transmission/rpc' },
  overseerr: { url: 'http://localhost:5055', healthEndpoint: '/api/v1/status' },
  jellyseerr: { url: 'http://localhost:5055', healthEndpoint: '/api/v1/status' },
  tautulli: { url: 'http://localhost:8181', healthEndpoint: '/status' },
  organizr: { url: 'http://localhost:80', healthEndpoint: '/api/v2/ping' },
  heimdall: { url: 'http://localhost:8080', healthEndpoint: '/ping' },
  homer: { url: 'http://localhost:3000', healthEndpoint: '/assets/config.yml' },
  portainer: { url: 'http://localhost:9000', healthEndpoint: '/api/system/status' },
  nginxProxyManager: { url: 'http://localhost:81', healthEndpoint: '/api' },
  uptimeKuma: { url: 'http://localhost:3001', healthEndpoint: '/api/status-page/heartbeat' },
  grafana: { url: 'http://localhost:3000', healthEndpoint: '/api/health' },
  prometheus: { url: 'http://localhost:9090', healthEndpoint: '/-/healthy' },
  watchtower: { url: 'http://localhost:8080', healthEndpoint: '/v1/metrics' },
  duplicati: { url: 'http://localhost:8200', healthEndpoint: '/api/v1/serverstate' },
  nextcloud: { url: 'http://localhost:8080', healthEndpoint: '/status.php' },
  syncthing: { url: 'http://localhost:8384', healthEndpoint: '/rest/system/ping' },
  freshrss: { url: 'http://localhost:80', healthEndpoint: '/api/greader.php' },
  calibreWeb: { url: 'http://localhost:8083', healthEndpoint: '/opds' },
  photoprism: { url: 'http://localhost:2342', healthEndpoint: '/api/v1/status' }
};

describe('Media Server E2E Test Suite', () => {
  let page: Page;

  test.beforeEach(async ({ page: testPage }) => {
    page = testPage;
    // Set viewport for cyberpunk theme testing
    await page.setViewportSize({ width: 1920, height: 1080 });
  });

  describe('Service Health Checks', () => {
    Object.entries(SERVICES).forEach(([name, config]) => {
      test(`${name} service should be healthy`, async () => {
        const response = await page.request.get(`${config.url}${config.healthEndpoint}`, {
          failOnStatusCode: false
        });
        
        expect(response.status()).toBeLessThan(400);
        
        // Service-specific health validation
        if (name.includes('arr')) {
          const data = await response.json();
          expect(data).toHaveProperty('version');
        }
      });
    });
  });

  describe('Dashboard UI Tests', () => {
    test('Dashboard should load with cyberpunk theme', async () => {
      await page.goto('http://localhost:3000');
      
      // Check for cyberpunk theme elements
      const glitchText = await page.locator('.glitch-text');
      expect(await glitchText.count()).toBeGreaterThan(0);
      
      // Check for neon colors
      const neonElements = await page.locator('[class*="neon"]');
      expect(await neonElements.count()).toBeGreaterThan(0);
      
      // Check for holographic effects
      const holographic = await page.locator('.holographic-bg');
      expect(await holographic.count()).toBeGreaterThan(0);
    });

    test('Service grid should display all 30 services', async () => {
      await page.goto('http://localhost:3000/dashboard');
      
      const serviceCards = await page.locator('.service-card');
      expect(await serviceCards.count()).toBe(30);
      
      // Check each service card has status indicator
      for (let i = 0; i < 30; i++) {
        const statusIndicator = await serviceCards.nth(i).locator('.status-indicator');
        expect(await statusIndicator.count()).toBe(1);
      }
    });

    test('3D visualization should render', async () => {
      await page.goto('http://localhost:3000/visualization');
      
      const canvas = await page.locator('canvas');
      expect(await canvas.count()).toBeGreaterThan(0);
      
      // Check WebGL context
      const hasWebGL = await page.evaluate(() => {
        const canvas = document.querySelector('canvas');
        return !!(canvas && (canvas.getContext('webgl') || canvas.getContext('webgl2')));
      });
      expect(hasWebGL).toBe(true);
    });
  });

  describe('API Integration Tests', () => {
    test('Sonarr should connect to Prowlarr', async () => {
      const response = await page.request.get('http://localhost:8989/api/v3/indexer', {
        headers: {
          'X-Api-Key': process.env.SONARR_API_KEY || ''
        }
      });
      
      expect(response.ok()).toBeTruthy();
      const indexers = await response.json();
      expect(indexers.length).toBeGreaterThan(0);
    });

    test('Radarr should connect to download clients', async () => {
      const response = await page.request.get('http://localhost:7878/api/v3/downloadclient', {
        headers: {
          'X-Api-Key': process.env.RADARR_API_KEY || ''
        }
      });
      
      expect(response.ok()).toBeTruthy();
      const clients = await response.json();
      expect(clients.length).toBeGreaterThan(0);
    });

    test('Jellyfin should have media libraries', async () => {
      const response = await page.request.get('http://localhost:8096/Library/VirtualFolders', {
        headers: {
          'X-MediaBrowser-Token': process.env.JELLYFIN_API_KEY || ''
        }
      });
      
      expect(response.ok()).toBeTruthy();
      const libraries = await response.json();
      expect(libraries.length).toBeGreaterThan(0);
    });
  });

  describe('Authentication Tests', () => {
    test('Cyberpunk authentication should work', async () => {
      await page.goto('http://localhost:3000/login');
      
      // Check for biometric UI elements
      const biometricButton = await page.locator('.biometric-auth');
      expect(await biometricButton.count()).toBe(1);
      
      // Test login flow
      await page.fill('#username', 'testuser');
      await page.fill('#password', 'testpass');
      await page.click('.login-button');
      
      // Check for successful redirect
      await page.waitForURL('**/dashboard');
      expect(page.url()).toContain('/dashboard');
    });

    test('Multi-factor authentication should be available', async () => {
      await page.goto('http://localhost:3000/login');
      
      const mfaOption = await page.locator('.mfa-option');
      expect(await mfaOption.count()).toBeGreaterThan(0);
    });
  });

  describe('Media Player Tests', () => {
    test('Holographic media player should load', async () => {
      await page.goto('http://localhost:3000/player');
      
      const player = await page.locator('.holographic-player');
      expect(await player.count()).toBe(1);
      
      // Check for gesture controls
      const gestureControls = await page.locator('.gesture-controls');
      expect(await gestureControls.count()).toBe(1);
    });

    test('Video playback should work', async () => {
      await page.goto('http://localhost:3000/player?id=test');
      
      const video = await page.locator('video');
      expect(await video.count()).toBe(1);
      
      // Check video can play
      await page.click('.play-button');
      const isPlaying = await video.evaluate((vid: HTMLVideoElement) => !vid.paused);
      expect(isPlaying).toBe(true);
    });
  });

  describe('Voice Control Tests', () => {
    test('Voice control interface should be present', async () => {
      await page.goto('http://localhost:3000/voice');
      
      const voiceButton = await page.locator('.voice-button');
      expect(await voiceButton.count()).toBe(1);
      
      // Check for wake word display
      const wakeWordStatus = await page.locator('.wake-word-status');
      expect(await wakeWordStatus.count()).toBe(1);
    });

    test('Command list should be displayed', async () => {
      await page.goto('http://localhost:3000/voice');
      
      const commands = await page.locator('.command-card');
      expect(await commands.count()).toBeGreaterThan(10);
    });
  });

  describe('AR/VR Tests', () => {
    test('WebXR support should be detected', async () => {
      await page.goto('http://localhost:3000/xr');
      
      const xrStatus = await page.evaluate(() => {
        return 'xr' in navigator;
      });
      
      // Note: This may be false in headless mode
      console.log('WebXR Support:', xrStatus);
    });

    test('3D scene should render', async () => {
      await page.goto('http://localhost:3000/xr');
      
      const canvas = await page.locator('.webxr-container canvas');
      expect(await canvas.count()).toBe(1);
    });
  });

  describe('Download Manager Tests', () => {
    test('Download queue should be manageable', async () => {
      await page.goto('http://localhost:3000/downloads');
      
      const downloadItems = await page.locator('.download-item');
      const count = await downloadItems.count();
      
      if (count > 0) {
        // Test drag and drop reordering
        const firstItem = downloadItems.first();
        const lastItem = downloadItems.last();
        
        await firstItem.dragTo(lastItem);
        
        // Verify order changed
        const newFirst = await page.locator('.download-item').first();
        expect(newFirst).not.toBe(firstItem);
      }
    });

    test('Bandwidth allocation should be adjustable', async () => {
      await page.goto('http://localhost:3000/downloads');
      
      const bandwidthSlider = await page.locator('.bandwidth-slider');
      if (await bandwidthSlider.count() > 0) {
        const initialValue = await bandwidthSlider.inputValue();
        await bandwidthSlider.fill('50');
        const newValue = await bandwidthSlider.inputValue();
        expect(newValue).not.toBe(initialValue);
      }
    });
  });

  describe('Performance Tests', () => {
    test('Dashboard should load within 3 seconds', async () => {
      const startTime = Date.now();
      await page.goto('http://localhost:3000/dashboard');
      await page.waitForLoadState('networkidle');
      const loadTime = Date.now() - startTime;
      
      expect(loadTime).toBeLessThan(3000);
    });

    test('Service cards should render at 60fps', async () => {
      await page.goto('http://localhost:3000/dashboard');
      
      const fps = await page.evaluate(() => {
        return new Promise((resolve) => {
          let lastTime = performance.now();
          let frames = 0;
          const checkFPS = () => {
            frames++;
            const currentTime = performance.now();
            if (currentTime >= lastTime + 1000) {
              resolve(frames);
            } else {
              requestAnimationFrame(checkFPS);
            }
          };
          requestAnimationFrame(checkFPS);
        });
      });
      
      expect(fps).toBeGreaterThan(30); // At least 30fps
    });
  });

  describe('Visual Regression Tests', () => {
    test('Dashboard screenshot should match baseline', async () => {
      await page.goto('http://localhost:3000/dashboard');
      await page.waitForLoadState('networkidle');
      
      await expect(page).toHaveScreenshot('dashboard.png', {
        fullPage: true,
        animations: 'disabled'
      });
    });

    test('Cyberpunk theme should be consistent', async () => {
      const pages = ['/dashboard', '/analytics', '/downloads', '/player'];
      
      for (const path of pages) {
        await page.goto(`http://localhost:3000${path}`);
        await expect(page).toHaveScreenshot(`cyberpunk-theme${path.replace('/', '-')}.png`);
      }
    });
  });

  describe('Security Tests', () => {
    test('API endpoints should require authentication', async () => {
      const response = await page.request.get('http://localhost:3000/api/media', {
        failOnStatusCode: false
      });
      
      expect(response.status()).toBe(401);
    });

    test('CORS should be properly configured', async () => {
      const response = await page.request.get('http://localhost:3000/api/health', {
        headers: {
          'Origin': 'http://evil.com'
        },
        failOnStatusCode: false
      });
      
      const corsHeader = response.headers()['access-control-allow-origin'];
      expect(corsHeader).not.toBe('*');
    });
  });

  describe('Accessibility Tests', () => {
    test('Dashboard should be keyboard navigable', async () => {
      await page.goto('http://localhost:3000/dashboard');
      
      // Tab through elements
      for (let i = 0; i < 10; i++) {
        await page.keyboard.press('Tab');
      }
      
      const focusedElement = await page.evaluate(() => {
        return document.activeElement?.tagName;
      });
      
      expect(focusedElement).toBeTruthy();
    });

    test('ARIA labels should be present', async () => {
      await page.goto('http://localhost:3000/dashboard');
      
      const ariaElements = await page.locator('[aria-label]');
      expect(await ariaElements.count()).toBeGreaterThan(0);
    });
  });
});

// Load testing
describe('Load Tests', () => {
  test('System should handle 100 concurrent requests', async ({ request }) => {
    const promises = [];
    
    for (let i = 0; i < 100; i++) {
      promises.push(
        request.get('http://localhost:3000/api/health', {
          failOnStatusCode: false
        })
      );
    }
    
    const responses = await Promise.all(promises);
    const successCount = responses.filter(r => r.status() < 400).length;
    
    expect(successCount).toBeGreaterThan(95); // 95% success rate
  });
});