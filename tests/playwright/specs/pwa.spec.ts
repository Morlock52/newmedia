import { test, expect } from '@playwright/test';
import path from 'path';

function fileUrl(p: string) {
  const full = path.resolve(process.cwd(), p);
  const url = 'file://' + full.replace(/\\/g, '/');
  return url;
}

test.describe('PWA Test Page', () => {
  test('buttons exist and clicking does not cause errors (stubbed)', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });

    await page.goto(fileUrl('pwa-test.html'));

    const buttonSelectors = [
      'text=Test Service Worker',
      'text=Unregister SW',
      'text=Test Manifest',
      'text=Validate',
      'text=Test Install',
      'text=Check Status',
      'text=Test Offline',
      'text=Simulate Offline',
      'text=Test Notifications',
      'text=Request Permission',
      'text=Test Sync',
      'text=Force Sync',
      'text=Test Share',
      'text=Share Target',
      'text=Test Shortcuts',
      'text=Add Shortcut',
      'text=Test Cache',
      'text=Clear Cache',
      'text=Measure Performance',
      'text=Export Metrics',
      'text=Test Storage',
      'text=Request Persistent',
      'text=Test Fullscreen',
      'text=Exit Fullscreen',
      'text=Run All PWA Tests',
      'text=Generate Report'
    ];

    for (const sel of buttonSelectors) {
      const btn = page.locator(sel).first();
      await expect(btn, `button visible: ${sel}`).toBeVisible();
      await btn.click();
    }

    expect(errors, 'no console errors after clicks').toHaveLength(0);
  });
});
