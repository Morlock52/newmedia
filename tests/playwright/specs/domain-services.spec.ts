import { test, expect, request } from '@playwright/test';

const services = [
  'jellyfin', 'sonarr', 'radarr', 'prowlarr', 'qbittorrent',
  'uptime', 'grafana', 'prometheus', 'traefik'
];

const protocol = process.env.PROTOCOL || 'https';
const domain = process.env.DOMAIN;

(test as any).skip(!domain, 'DOMAIN env not set; skipping domain-based service UI tests');

test.describe('Service UIs via Traefik domain', () => {
  for (const sub of services) {
    test(`${sub} is accessible`, async ({ request }) => {
      const url = `${protocol}://${sub}.${domain}`;
      const resp = await request.get(url, { ignoreHTTPSErrors: true });
      expect(resp.ok(), `${url} should be OK`).toBeTruthy();
      const contentType = resp.headers()['content-type'] || '';
      expect(contentType.toLowerCase()).toContain('text/html');
    });
  }
});
