import { defineConfig, devices } from '@playwright/test';
import path from 'path';

const root = process.cwd();

export default defineConfig({
  testDir: path.join(root, 'tests/playwright/specs'),
  timeout: 30_000,
  retries: 0,
  fullyParallel: true,
  reporter: [
    ['list'],
    ['html', { open: 'never', outputFolder: 'tests/playwright-report' }],
  ],
  use: {
    headless: true,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 10_000,
    navigationTimeout: 15_000,
    viewport: { width: 1366, height: 768 },
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
});
