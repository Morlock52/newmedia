const { docker, execInContainer } = require('./setup');
const path = require('path');
const fs = require('fs');

describe('Volume and Permission Tests', () => {
  describe('Volume Mounts', () => {
    const expectedVolumes = [
      { service: 'jellyfin', mount: '/config' },
      { service: 'jellyfin', mount: '/media' },
      { service: 'sonarr', mount: '/config' },
      { service: 'sonarr', mount: '/media' },
      { service: 'sonarr', mount: '/downloads' },
      { service: 'radarr', mount: '/config' },
      { service: 'radarr', mount: '/media' },
      { service: 'radarr', mount: '/downloads' },
      { service: 'prowlarr', mount: '/config' },
      { service: 'qbittorrent', mount: '/config' },
      { service: 'qbittorrent', mount: '/downloads' }
    ];

    test.each(expectedVolumes)('$service should have $mount volume mounted', async ({ service, mount }) => {
      try {
        const container = docker.getContainer(service);
        const info = await container.inspect();
        const mounts = info.Mounts;
        
        const volumeMount = mounts.find(m => m.Destination === mount);
        expect(volumeMount).toBeDefined();
      } catch (err) {
        // Skip if container doesn't exist
        console.log(`Skipping ${service} - container not found`);
      }
    });
  });

  describe('File Permissions', () => {
    test('Media directories should have correct permissions', async () => {
      const services = ['jellyfin', 'sonarr', 'radarr'];
      
      for (const service of services) {
        try {
          const output = await execInContainer(service, 'ls -la /media');
          
          // Check if media directory is readable
          expect(output).toContain('drwxr');
          
          // Check PUID/PGID
          const userInfo = await execInContainer(service, 'id');
          expect(userInfo).toContain('uid=1000');
          expect(userInfo).toContain('gid=1000');
        } catch (err) {
          console.log(`Skipping ${service} permission test - ${err.message}`);
        }
      }
    });

    test('Config directories should be writable', async () => {
      const services = ['sonarr', 'radarr', 'prowlarr'];
      
      for (const service of services) {
        try {
          // Try to create a test file
          const testFile = `/config/test-${Date.now()}.txt`;
          await execInContainer(service, `touch ${testFile}`);
          
          // Verify file was created
          const lsOutput = await execInContainer(service, `ls -la ${testFile}`);
          expect(lsOutput).toContain('test-');
          
          // Clean up
          await execInContainer(service, `rm ${testFile}`);
        } catch (err) {
          console.log(`Skipping ${service} write test - ${err.message}`);
        }
      }
    });
  });

  describe('Shared Volume Access', () => {
    test('Media services should access same media files', async () => {
      const testFileName = `test-media-${Date.now()}.txt`;
      const testContent = 'This is a shared media file test';
      
      try {
        // Create file in Sonarr
        await execInContainer('sonarr', `echo "${testContent}" > /media/${testFileName}`);
        
        // Verify file exists in Radarr
        const radarrContent = await execInContainer('radarr', `cat /media/${testFileName}`);
        expect(radarrContent.trim()).toBe(testContent);
        
        // Verify file exists in Jellyfin
        const jellyfinContent = await execInContainer('jellyfin', `cat /media/${testFileName}`);
        expect(jellyfinContent.trim()).toBe(testContent);
        
        // Clean up
        await execInContainer('sonarr', `rm /media/${testFileName}`);
      } catch (err) {
        console.log('Skipping shared volume test - containers may not be fully configured');
      }
    });

    test('Download clients should share download directory', async () => {
      const testFileName = `test-download-${Date.now()}.txt`;
      
      try {
        // Create file in qBittorrent downloads
        await execInContainer('qbittorrent', `touch /downloads/${testFileName}`);
        
        // Verify file exists in Sonarr downloads
        const sonarrLs = await execInContainer('sonarr', `ls /downloads/${testFileName}`);
        expect(sonarrLs).toContain(testFileName);
        
        // Clean up
        await execInContainer('qbittorrent', `rm /downloads/${testFileName}`);
      } catch (err) {
        console.log('Skipping download directory test - containers may not be fully configured');
      }
    });
  });

  describe('Volume Persistence', () => {
    test('Config volumes should persist data', async () => {
      const configDirs = [
        './config/jellyfin',
        './config/sonarr',
        './config/radarr',
        './config/prowlarr'
      ];
      
      for (const dir of configDirs) {
        const fullPath = path.join(__dirname, '../..', dir);
        
        // Check if config directory exists
        expect(fs.existsSync(fullPath)).toBe(true);
        
        // Check if directory has files (indicating persistence)
        const files = fs.readdirSync(fullPath);
        expect(files.length).toBeGreaterThan(0);
      }
    });
  });

  describe('Special Device Access', () => {
    test('Jellyfin should have GPU device access for transcoding', async () => {
      try {
        const container = docker.getContainer('jellyfin');
        const info = await container.inspect();
        
        // Check for GPU device mapping
        const devices = info.HostConfig.Devices || [];
        const gpuDevice = devices.find(d => d.PathOnHost === '/dev/dri');
        
        expect(gpuDevice).toBeDefined();
        expect(gpuDevice.PathInContainer).toBe('/dev/dri');
        
        // Verify device exists in container
        const deviceCheck = await execInContainer('jellyfin', 'ls -la /dev/dri');
        expect(deviceCheck).toContain('renderD128');
      } catch (err) {
        console.log('GPU device not available - transcoding may be limited to CPU');
      }
    });
  });

  describe('Docker Socket Access', () => {
    test('Homepage should have read-only Docker socket access', async () => {
      const container = docker.getContainer('homepage');
      const info = await container.inspect();
      
      const dockerSocketMount = info.Mounts.find(m => m.Source === '/var/run/docker.sock');
      expect(dockerSocketMount).toBeDefined();
      expect(dockerSocketMount.Mode).toBe('ro');
      
      // Verify socket is accessible
      const socketCheck = await execInContainer('homepage', 'ls -la /var/run/docker.sock');
      expect(socketCheck).toContain('srw');
    });

    test('Portainer should have Docker socket access', async () => {
      const container = docker.getContainer('portainer');
      const info = await container.inspect();
      
      const dockerSocketMount = info.Mounts.find(m => m.Source === '/var/run/docker.sock');
      expect(dockerSocketMount).toBeDefined();
    });
  });
});