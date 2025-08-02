const Docker = require('dockerode');
const waitPort = require('wait-port');
const { execSync } = require('child_process');

// Global test configuration
global.testConfig = {
  baseUrl: process.env.BASE_URL || 'http://localhost',
  services: {
    jellyfin: { port: 8096, healthPath: '/health' },
    sonarr: { port: 8989, healthPath: '/api/v3/health' },
    radarr: { port: 7878, healthPath: '/api/v3/health' },
    prowlarr: { port: 9696, healthPath: '/api/v1/health' },
    bazarr: { port: 6767, healthPath: '/api/v3/health' },
    overseerr: { port: 5055, healthPath: '/api/v1/status' },
    tautulli: { port: 8181, healthPath: '/api/v2' },
    homepage: { port: 3001, healthPath: '/' },
    grafana: { port: 3000, healthPath: '/api/health' }
  },
  timeouts: {
    service: 30000,
    test: 60000
  }
};

// Docker client
global.docker = new Docker();

// Utility functions
global.waitForService = async (serviceName, port, timeout = 30000) => {
  console.log(`Waiting for ${serviceName} on port ${port}...`);
  
  const params = {
    host: 'localhost',
    port: port,
    timeout: timeout,
    waitForDns: true
  };

  try {
    await waitPort(params);
    console.log(`${serviceName} is ready on port ${port}`);
    
    // Additional health check
    if (global.testConfig.services[serviceName]?.healthPath) {
      await waitForHealth(serviceName);
    }
  } catch (err) {
    throw new Error(`${serviceName} failed to start: ${err.message}`);
  }
};

global.waitForHealth = async (serviceName, retries = 10) => {
  const service = global.testConfig.services[serviceName];
  const url = `${global.testConfig.baseUrl}:${service.port}${service.healthPath}`;
  
  for (let i = 0; i < retries; i++) {
    try {
      const response = await fetch(url);
      if (response.ok) {
        console.log(`${serviceName} health check passed`);
        return;
      }
    } catch (err) {
      console.log(`Health check attempt ${i + 1}/${retries} failed for ${serviceName}`);
      await new Promise(resolve => setTimeout(resolve, 3000));
    }
  }
  
  throw new Error(`${serviceName} health check failed after ${retries} attempts`);
};

// Container utilities
global.getContainerLogs = async (containerName, tail = 100) => {
  try {
    const container = docker.getContainer(containerName);
    const stream = await container.logs({
      stdout: true,
      stderr: true,
      tail: tail
    });
    return stream.toString();
  } catch (err) {
    console.error(`Failed to get logs for ${containerName}: ${err.message}`);
    return null;
  }
};

global.execInContainer = async (containerName, command) => {
  try {
    const container = docker.getContainer(containerName);
    const exec = await container.exec({
      Cmd: command.split(' '),
      AttachStdout: true,
      AttachStderr: true
    });
    
    const stream = await exec.start();
    return new Promise((resolve, reject) => {
      let output = '';
      stream.on('data', (chunk) => {
        output += chunk.toString();
      });
      stream.on('end', () => resolve(output));
      stream.on('error', reject);
    });
  } catch (err) {
    throw new Error(`Failed to execute command in ${containerName}: ${err.message}`);
  }
};

// Test data generators
global.generateTestMedia = () => ({
  movie: {
    title: `Test Movie ${Date.now()}`,
    year: 2024,
    imdbId: `tt${Math.floor(Math.random() * 10000000)}`,
    quality: '1080p',
    size: '2GB'
  },
  series: {
    title: `Test Series ${Date.now()}`,
    year: 2024,
    tvdbId: Math.floor(Math.random() * 1000000),
    seasons: 2,
    episodes: 20
  },
  music: {
    artist: `Test Artist ${Date.now()}`,
    album: `Test Album ${Date.now()}`,
    year: 2024,
    tracks: 12
  }
});

// API helpers
global.apiRequest = async (service, endpoint, options = {}) => {
  const serviceConfig = global.testConfig.services[service];
  const url = `${global.testConfig.baseUrl}:${serviceConfig.port}${endpoint}`;
  
  const defaultOptions = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers
    },
    ...options
  };
  
  const response = await fetch(url, defaultOptions);
  const data = await response.json().catch(() => null);
  
  return {
    status: response.status,
    statusText: response.statusText,
    headers: Object.fromEntries(response.headers),
    data
  };
};

// Cleanup helper
global.cleanupTestData = async () => {
  console.log('Cleaning up test data...');
  // Add cleanup logic here
};

// Export for use in tests
module.exports = {
  testConfig: global.testConfig,
  docker: global.docker,
  waitForService: global.waitForService,
  waitForHealth: global.waitForHealth,
  getContainerLogs: global.getContainerLogs,
  execInContainer: global.execInContainer,
  generateTestMedia: global.generateTestMedia,
  apiRequest: global.apiRequest,
  cleanupTestData: global.cleanupTestData
};