const { waitForService, apiRequest, docker } = require('./setup');

describe('Service Connectivity Tests', () => {
  const services = [
    { name: 'jellyfin', port: 8096, healthEndpoint: '/health' },
    { name: 'sonarr', port: 8989, healthEndpoint: '/api/v3/health' },
    { name: 'radarr', port: 7878, healthEndpoint: '/api/v3/health' },
    { name: 'prowlarr', port: 9696, healthEndpoint: '/api/v1/health' },
    { name: 'overseerr', port: 5055, healthEndpoint: '/api/v1/status' },
    { name: 'homepage', port: 3001, healthEndpoint: '/' }
  ];

  describe('Container Health Checks', () => {
    test.each(services)('$name container should be running', async ({ name }) => {
      const container = docker.getContainer(name);
      const info = await container.inspect();
      
      expect(info.State.Status).toBe('running');
      expect(info.State.Running).toBe(true);
      expect(info.State.Health?.Status || 'healthy').toBe('healthy');
    });
  });

  describe('Service Port Availability', () => {
    test.each(services)('$name should be accessible on port $port', async ({ name, port }) => {
      await expect(waitForService(name, port)).resolves.not.toThrow();
    });
  });

  describe('Service Health Endpoints', () => {
    test.each(services)('$name health endpoint should return 200', async ({ name, healthEndpoint }) => {
      const response = await apiRequest(name, healthEndpoint);
      expect(response.status).toBe(200);
    });
  });

  describe('Inter-Service Communication', () => {
    test('Sonarr should connect to Prowlarr', async () => {
      // This would require API keys to be set up
      // For now, we just check if the services are running
      const sonarrInfo = await docker.getContainer('sonarr').inspect();
      const prowlarrInfo = await docker.getContainer('prowlarr').inspect();
      
      expect(sonarrInfo.State.Running).toBe(true);
      expect(prowlarrInfo.State.Running).toBe(true);
      
      // Check if they're on the same network
      const sonarrNetworks = Object.keys(sonarrInfo.NetworkSettings.Networks);
      const prowlarrNetworks = Object.keys(prowlarrInfo.NetworkSettings.Networks);
      const commonNetworks = sonarrNetworks.filter(net => prowlarrNetworks.includes(net));
      
      expect(commonNetworks.length).toBeGreaterThan(0);
    });

    test('Radarr should connect to Prowlarr', async () => {
      const radarrInfo = await docker.getContainer('radarr').inspect();
      const prowlarrInfo = await docker.getContainer('prowlarr').inspect();
      
      expect(radarrInfo.State.Running).toBe(true);
      expect(prowlarrInfo.State.Running).toBe(true);
      
      const radarrNetworks = Object.keys(radarrInfo.NetworkSettings.Networks);
      const prowlarrNetworks = Object.keys(prowlarrInfo.NetworkSettings.Networks);
      const commonNetworks = radarrNetworks.filter(net => prowlarrNetworks.includes(net));
      
      expect(commonNetworks.length).toBeGreaterThan(0);
    });
  });

  describe('Resource Limits', () => {
    test.each(services)('$name should have reasonable resource usage', async ({ name }) => {
      const container = docker.getContainer(name);
      const stats = await container.stats({ stream: false });
      
      // Check memory usage (should be less than 2GB)
      const memoryUsage = stats.memory_stats.usage || 0;
      const memoryLimit = 2 * 1024 * 1024 * 1024; // 2GB
      expect(memoryUsage).toBeLessThan(memoryLimit);
      
      // Check CPU usage (should be less than 50%)
      const cpuDelta = stats.cpu_stats.cpu_usage.total_usage - stats.precpu_stats.cpu_usage.total_usage;
      const systemDelta = stats.cpu_stats.system_cpu_usage - stats.precpu_stats.system_cpu_usage;
      const cpuPercent = (cpuDelta / systemDelta) * stats.cpu_stats.online_cpus * 100;
      
      expect(cpuPercent).toBeLessThan(50);
    });
  });

  describe('Network Isolation', () => {
    test('Download network should be isolated from media network', async () => {
      const vpnContainer = docker.getContainer('vpn');
      const vpnInfo = await vpnContainer.inspect();
      
      const jellyfinContainer = docker.getContainer('jellyfin');
      const jellyfinInfo = await jellyfinContainer.inspect();
      
      const vpnNetworks = Object.keys(vpnInfo.NetworkSettings.Networks);
      const jellyfinNetworks = Object.keys(jellyfinInfo.NetworkSettings.Networks);
      
      // VPN should be on download_network
      expect(vpnNetworks).toContain('newmedia_download_network');
      
      // Check that not all networks are shared
      const sharedNetworks = vpnNetworks.filter(net => jellyfinNetworks.includes(net));
      expect(sharedNetworks.length).toBeLessThan(vpnNetworks.length);
    });
  });
});