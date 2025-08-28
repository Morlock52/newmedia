import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';

function fileUrl(p: string) {
  const full = path.resolve(process.cwd(), p);
  const url = 'file://' + full.replace(/\\/g, '/');
  return url;
}

const indexPath = 'index.html';

test.describe('Landing page (index.html)', () => {
  test('loads without console errors and shows main sections', async ({ page }) => {
    const errors: string[] = [];
    page.on('pageerror', (err) => errors.push(String(err)));
    page.on('console', (msg) => { if (msg.type() === 'error') errors.push(msg.text()); });

    await page.goto(fileUrl(indexPath));
    await expect(page.locator('header.header')).toBeVisible();
    await expect(page.getByText('ULTIMATE MEDIA SERVER 2025')).toBeVisible();
    await expect(page.locator('.components-section')).toBeVisible();

    expect(errors, 'no console errors').toHaveLength(0);
  });

  test('CTA links exist and are clickable', async ({ page }) => {
    await page.goto(fileUrl(indexPath));
    const ctas = [
      { text: 'ENTER DASHBOARD', href: 'dashboard.html' },
      { text: 'API DOCUMENTATION', href: 'BACKEND_API_INTEGRATION_SUMMARY.md' },
      { text: 'LIVE DEMO', href: 'LIVE_DEMO.html' },
      { text: 'SETUP GUIDE', href: 'SIMPLE-README.md' },
    ];
    for (const { text, href } of ctas) {
      const link = page.getByRole('link', { name: text });
      await expect(link).toHaveAttribute('href', href);
      // Verify target exists on disk
      const targetPath = path.resolve(process.cwd(), href);
      expect(fs.existsSync(targetPath), `target exists: ${href}`).toBeTruthy();
    }
  });

  test('Launch buttons trigger alerts (no crashes)', async ({ page }) => {
    await page.goto(fileUrl(indexPath));
    const dialogs: string[] = [];
    page.on('dialog', async (dialog) => { dialogs.push(dialog.message()); await dialog.dismiss(); });
    const buttons = page.locator('button.launch-btn');
    const count = await buttons.count();
    for (let i = 0; i < count; i++) {
      await buttons.nth(i).click();
    }
    expect(dialogs.length).toBeGreaterThan(0);
  });
});
