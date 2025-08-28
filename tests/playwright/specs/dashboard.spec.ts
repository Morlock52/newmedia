import { test, expect } from '@playwright/test';
import path from 'path';

function fileUrl(p: string) {
  const full = path.resolve(process.cwd(), p);
  const url = 'file://' + full.replace(/\\/g, '/');
  return url;
}

test.describe('Dashboard page (dashboard.html)', () => {
  test.beforeEach(async ({ page }) => {
    // Stub window.open so we can assert URLs without opening real windows
    await page.addInitScript(() => {
      (window as any).__openCalls = [] as string[];
      const origOpen = window.open;
      window.open = function(url?: string | URL | undefined, target?: string | undefined, features?: string | undefined) {
        (window as any).__openCalls.push(String(url || ''));
        return null as any;
      };
      (window as any).__origOpen = origOpen;
    });
  });

  test('loads and shows key sections', async ({ page }) => {
    await page.goto(fileUrl('dashboard.html'));
    await expect(page.getByRole('heading', { name: 'Ultimate Media Server 2025' })).toBeVisible();
    await expect(page.locator('.subtitle')).toContainText('Single Container');
    await expect(page.locator('.status-bar')).toBeVisible();
    await expect(page.locator('.services')).toBeVisible();
    await expect(page.locator('.info-section')).toBeVisible();
    // Expect at least 8 service cards
    await expect(page.locator('.service')).toHaveCount(8);
  });

  test('hover effects apply styles to service cards', async ({ page }) => {
    await page.goto(fileUrl('dashboard.html'));
    const card = page.locator('.service').first();
    const before = await card.evaluate((el) => getComputedStyle(el).transform);
    await card.hover();
    const after = await card.evaluate((el) => getComputedStyle(el).transform);
    // On hover transform should not be 'none'
    expect(after).not.toBe('none');
  });

  test('clicking service cards triggers window.open with expected URLs', async ({ page }) => {
    await page.goto(fileUrl('dashboard.html'));
    const cards = page.locator('.service');
    const expected = [
      'http://localhost:8096',
      'http://localhost:8989',
      'http://localhost:7878',
      'http://localhost:8686',
      'http://localhost:9696',
      'http://localhost:6767',
      'http://localhost:8080',
      'http://localhost:9091',
    ];
    const count = await cards.count();
    for (let i = 0; i < count; i++) {
      await cards.nth(i).click();
    }
    const openCalls = await page.evaluate(() => (window as any).__openCalls as string[]);
    expect(openCalls.length).toBeGreaterThanOrEqual(expected.length);
    for (const url of expected) {
      expect(openCalls).toContain(url);
    }
  });

  test('clicking service cards triggers window.open with expected URLs (domain mode)', async ({ page }) => {
    const domain = process.env.DOMAIN;
    test.skip(!domain, 'DOMAIN not set for domain-mode dashboard test');
    await page.addInitScript(([d, proto]) => { (window as any).DASHBOARD_DOMAIN = d; (window as any).DASHBOARD_PROTOCOL = proto; }, [domain!, process.env.PROTOCOL || 'https']);
    await page.goto(fileUrl('dashboard.html'));
    const cards = page.locator('.service');
    const subs = ['jellyfin','sonarr','radarr','lidarr','prowlarr','bazarr','qbittorrent','transmission'];
    const count = await cards.count();
    for (let i = 0; i < count; i++) {
      await cards.nth(i).click();
    }
    const openCalls = await page.evaluate(() => (window as any).__openCalls as string[]);
    for (const sub of subs) {
      const url = `${process.env.PROTOCOL || 'https'}://${sub}.${domain}`;
      expect(openCalls).toContain(url);
    }
  });

});
