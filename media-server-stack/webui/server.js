import express from 'express';
import { exec, spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import fs from 'fs/promises';
import os from 'os';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = process.env.PORT || 3000;
const STACK_DIR = process.env.STACK_DIR || '/app/compose';

app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

console.log('🎬 Media Server Stack Web UI starting...');
console.log(`📁 Stack directory: ${STACK_DIR}`);
console.log(`📁 Server directory: ${__dirname}`);

// Utility function to run commands
function runCommand(cmd, options = {}) {
  return new Promise((resolve, reject) => {
    exec(cmd, { cwd: STACK_DIR, ...options }, (err, stdout, stderr) => {
      if (err) {
        reject({ error: err.message, stderr, stdout });
      } else {
        resolve({ stdout, stderr });
      }
    });
  });
}

// API Endpoints
app.get('/api/system-info', async (req, res) => {
  try {
    const systemInfo = {
      puid: process.getuid?.() || 1000,
      pgid: process.getgid?.() || 1000,
      timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'America/New_York',
      platform: os.platform(),
      arch: os.arch(),
      hostname: os.hostname(),
      memory: `${Math.round(os.totalmem() / 1024 / 1024 / 1024)} GB`,
      cpus: os.cpus().length,
      timezoneAbbr: new Date().toLocaleTimeString('en', {timeZoneName:'short'}).split(' ')[2] || 'EST'
    };
    res.json(systemInfo);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/env-status', async (req, res) => {
  try {
    // Check for .env file in the mounted directory
    const envPath = path.join(STACK_DIR, '.env');
    console.log(`🔍 Checking for .env file at: ${envPath}`);
    
    const stat = await fs.stat(envPath);
    const content = await fs.readFile(envPath, 'utf8');
    const lines = content.split('\n').filter(line => line.trim() && !line.startsWith('#'));
    
    res.json({ 
      exists: true, 
      configured: true,
      variables: lines.length,
      message: 'Environment file exists and configured',
      path: envPath
    });
  } catch (error) {
    console.log(`❌ Environment file not found: ${error.message}`);
    res.json({ 
      exists: false, 
      configured: false, 
      message: 'Environment file not found',
      error: error.message 
    });
  }
});

app.get('/api/docker-status', async (req, res) => {
  try {
    const { stdout: version } = await runCommand('docker version --format "{{.Server.Version}}" 2>/dev/null || echo "not available"');
    const { stdout: info } = await runCommand('docker info --format "{{.ServerVersion}}" 2>/dev/null || echo "not running"');
    
    res.json({
      running: !version.includes('not available') && !info.includes('not running'),
      version: version.trim(),
      available: true
    });
  } catch (error) {
    res.json({
      running: false,
      error: error.message,
      available: false
    });
  }
});

app.get('/api/status', async (req, res) => {
  try {
    // Add PATH to include Docker
    const dockerPath = process.env.PATH + ':/usr/local/bin:/usr/bin';
    const { stdout } = await runCommand('docker compose -f docker-compose.yml ps', { env: { ...process.env, PATH: dockerPath } });
    
    res.type('text/plain').send(stdout);
  } catch (error) {
    console.error('Status check error:', error);
    res.status(500).send(`Error checking status: ${error.error || error.message}`);
  }
});

app.post('/api/start', async (req, res) => {
  try {
    const dockerPath = process.env.PATH + ':/usr/local/bin:/usr/bin';
    const { stdout } = await runCommand('docker compose -f docker-compose.yml up -d', { 
      env: { ...process.env, PATH: dockerPath }
    });
    
    res.type('text/plain').send(`✅ Services started successfully!\n\n${stdout}`);
  } catch (error) {
    console.error('Start error:', error);
    res.status(500).send(`❌ Failed to start services: ${error.error || error.message}\n\nSTDERR: ${error.stderr || 'none'}\nSTDOUT: ${error.stdout || 'none'}`);
  }
});

app.post('/api/stop', async (req, res) => {
  try {
    const dockerPath = process.env.PATH + ':/usr/local/bin:/usr/bin';
    const { stdout } = await runCommand('docker compose -f docker-compose.yml down', { 
      env: { ...process.env, PATH: dockerPath }
    });
    
    res.type('text/plain').send(`✅ Services stopped successfully!\n\n${stdout}`);
  } catch (error) {
    console.error('Stop error:', error);
    res.status(500).send(`❌ Failed to stop services: ${error.error || error.message}\n\nSTDERR: ${error.stderr || 'none'}\nSTDOUT: ${error.stdout || 'none'}`);
  }
});

// Fixed deployment endpoint
app.post('/api/deploy', async (req, res) => {
  try {
    // Check if environment is configured
    const envPath = path.join(STACK_DIR, '.env');
    try {
      await fs.access(envPath);
    } catch (error) {
      return res.status(400).send('❌ Environment not configured. Please complete setup first.');
    }

    let output = '🚀 Starting Media Server Stack Deployment\n';
    output += '=========================================\n\n';

    // Add Docker to PATH
    const dockerPath = process.env.PATH + ':/usr/local/bin:/usr/bin';
    const cmdOptions = { env: { ...process.env, PATH: dockerPath } };

    // Clean up any existing containers first
    output += '🧹 Cleaning up existing containers...\n';
    try {
      await runCommand('docker compose -f docker-compose.yml down --remove-orphans 2>/dev/null || true', cmdOptions);
      await runCommand('docker system prune -f 2>/dev/null || true', cmdOptions);
    } catch (error) {
      output += `⚠️ Cleanup warnings: ${error.stderr || error.message || 'Some cleanup operations may have failed'}\n`;
    }

    // Pull latest images
    output += '📥 Pulling latest Docker images...\n';
    try {
      const { stdout: pullOutput } = await runCommand('docker compose -f docker-compose.yml pull', cmdOptions);
      output += pullOutput + '\n';
    } catch (error) {
      output += `⚠️ Warning: Failed to pull some images: ${error.error || error.message}\n`;
    }

    // Deploy the stack
    output += '🚀 Deploying services...\n';
    try {
      const { stdout: deployOutput } = await runCommand('docker compose -f docker-compose.yml up -d', cmdOptions);
      output += deployOutput + '\n';
    } catch (error) {
      console.error('Docker compose deployment error:', error);
      const errorMsg = error.error || error.message || error.stderr || 'Unknown deployment error';
      output += `❌ Failed to deploy: ${errorMsg}\n`;
      if (error.stderr) output += `STDERR: ${error.stderr}\n`;
      if (error.stdout) output += `STDOUT: ${error.stdout}\n`;
      return res.status(500).send(output);
    }

    // Wait for services to start
    output += '⏳ Waiting for services to start...\n';
    await new Promise(resolve => setTimeout(resolve, 10000));

    // Check service status
    output += '🔍 Checking service status...\n';
    try {
      const { stdout: statusOutput } = await runCommand('docker compose -f docker-compose.yml ps', cmdOptions);
      output += statusOutput + '\n';
    } catch (error) {
      output += `Error checking status: ${error.error || error.message}\n`;
    }

    output += '\n✅ Deployment completed!\n';
    output += '🌐 Your media server stack should now be running.\n';
    output += '📝 Check the Monitoring tab for service health status.\n';

    res.type('text/plain').send(output);
  } catch (error) {
    console.error('Deployment error:', error);
    res.status(500).send(`❌ Deployment failed: ${error.message}`);
  }
});

// Environment validation endpoint
app.post('/api/validate-config', async (req, res) => {
  try {
    const config = req.body;
    
    // Basic validation
    const errors = [];
    
    if (!config.domain) {
      errors.push('Domain is required');
    }
    
    if (!config.email) {
      errors.push('Email is required');
    }
    
    if (config.domain && !['localhost', 'local'].includes(config.domain) && !/^[a-zA-Z0-9]([a-zA-Z0-9\-]{0,61}[a-zA-Z0-9])?\.([a-zA-Z]{2,}(\.[a-zA-Z]{2,})*)$/.test(config.domain)) {
      errors.push('Invalid domain format');
    }
    
    if (config.email && !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(config.email)) {
      errors.push('Invalid email format');
    }
    
    if (errors.length > 0) {
      return res.status(400).json({
        success: false,
        message: 'Validation failed',
        errors: errors
      });
    }
    
    res.json({
      success: true,
      message: 'Configuration is valid'
    });
    
  } catch (error) {
    res.status(500).json({
      success: false,
      message: 'Validation error',
      details: error.message
    });
  }
});

// Environment setup endpoint
app.post('/api/setup-environment', async (req, res) => {
  try {
    const config = req.body;
    if (!config) {
      return res.status(400).json({ error: 'Configuration is required' });
    }

    // Generate .env content
    const envContent = generateEnvContent(config);
    
    // Write .env file to the mounted directory
    const envPath = path.join(STACK_DIR, '.env');
    await fs.writeFile(envPath, envContent);
    
    // Create backup
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const backupPath = path.join(STACK_DIR, `.env.backup.${timestamp}`);
    await fs.writeFile(backupPath, envContent);

    let response = `✅ Environment configuration generated successfully!\n\n`;
    response += `📁 Configuration saved to: ${envPath}\n`;
    response += `💾 Backup created: ${backupPath}\n\n`;
    response += `🎬 Media Server Stack Configuration\n`;
    response += `================================\n`;
    response += `Domain: ${config.domain || 'localhost'}\n`;
    response += `Email: ${config.email || 'admin@localhost'}\n`;
    response += `VPN Provider: ${config.vpnProvider || 'pia'}\n`;
    response += `VPN Type: ${config.vpnType || 'wireguard'}\n`;
    response += `Timezone: ${config.timezone || 'UTC'}\n`;
    response += `User/Group ID: ${config.puid || '1000'}:${config.pgid || '1000'}\n\n`;
    response += `🚀 Ready to deploy! Use the Management tab to start services.\n`;

    res.type('text/plain').send(response);
  } catch (error) {
    console.error('Setup error:', error);
    res.status(500).send(`❌ Failed to configure environment: ${error.message}`);
  }
});

// Get environment configuration
app.get('/api/env-config', async (req, res) => {
  try {
    const envPath = path.join(STACK_DIR, '.env');
    const envContent = await fs.readFile(envPath, 'utf8');
    
    // Parse environment variables
    const config = {};
    envContent.split('\n').forEach(line => {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#')) {
        const [key, ...valueParts] = trimmed.split('=');
        if (key && valueParts.length > 0) {
          config[key] = valueParts.join('=');
        }
      }
    });

    res.json({ success: true, config });
  } catch (error) {
    res.status(404).json({ 
      success: false, 
      error: 'Environment file not found',
      details: error.message 
    });
  }
});

// System stats endpoint
app.get('/api/system-stats', async (req, res) => {
  try {
    const stats = {
      cpu: Math.round(Math.random() * 100), // Placeholder
      memory: Math.round((process.memoryUsage().heapUsed / process.memoryUsage().heapTotal) * 100),
      disk: 'N/A',
      network: 'Active'
    };
    
    res.json(stats);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Service logs endpoint
app.get('/api/logs/:service', async (req, res) => {
  try {
    const { service } = req.params;
    const lines = req.query.lines || '100';
    const dockerPath = process.env.PATH + ':/usr/local/bin:/usr/bin';
    
    const { stdout: logs } = await runCommand(`docker compose -f docker-compose.yml logs --tail ${lines} ${service}`, { 
      env: { ...process.env, PATH: dockerPath }
    });
    
    res.type('text/plain').send(logs);
  } catch (error) {
    res.status(500).send(`Error getting logs for ${req.params.service}: ${error.message}`);
  }
});

// Health check endpoint
app.get('/api/health', async (req, res) => {
  try {
    const dockerPath = process.env.PATH + ':/usr/local/bin:/usr/bin';
    
    // Check environment
    let environment = false;
    try {
      await fs.access(path.join(STACK_DIR, '.env'));
      environment = true;
    } catch (error) {
      // Environment not configured
    }
    
    // Check Docker
    let docker = false;
    try {
      await runCommand('docker info', { env: { ...process.env, PATH: dockerPath } });
      docker = true;
    } catch (error) {
      // Docker not running
    }
    
    // Check services
    let services = [];
    try {
      const { stdout } = await runCommand('docker compose -f docker-compose.yml ps --format json', { 
        env: { ...process.env, PATH: dockerPath }
      });
      
      if (stdout.trim()) {
        const lines = stdout.trim().split('\n');
        services = lines.map(line => {
          try {
            const service = JSON.parse(line);
            return {
              name: service.Name || service.Service,
              status: service.State || 'unknown',
              health: service.Health || 'unknown'
            };
          } catch (e) {
            return null;
          }
        }).filter(Boolean);
      }
    } catch (error) {
      // Services not accessible
    }
    
    const health = {
      timestamp: new Date().toISOString(),
      environment: environment,
      docker: docker,
      services: services
    };
    
    res.json(health);
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Helper function to generate .env content
function generateEnvContent(config) {
  const timestamp = new Date().toISOString();
  
  let content = `# Media Server Stack Environment Configuration
# Generated on ${timestamp}
# 
# This file contains environment variables for your media server stack.
# Keep this file secure and do not commit it to version control.
# 
# For security best practices, see: https://12factor.net/config

# Core Configuration
DOMAIN=${config.domain || 'localhost'}
PUID=${config.puid || '1000'}
PGID=${config.pgid || '1000'}
TZ=${config.timezone || 'America/New_York'}
UMASK=002

# VPN Configuration
VPN_PROVIDER=${config.vpnProvider || 'pia'}
VPN_TYPE=${config.vpnType || 'wireguard'}
VPN_PORT_FORWARDING=on
VPN_PORT_FORWARDING_PORT=6881
PIA_REGION=us_east
WIREGUARD_ADDRESSES=10.0.0.0/8

# Cloudflare Configuration
CLOUDFLARE_TUNNEL_TOKEN=eyJhIjoiNmM5NTAxYzY4OWMyZTEzNzE5MGQ1MGZiYTYyN2I3ZmYiLCJzIjoiTUdObU5tVTVNakV0TUdVMFpTMDBNemRsTFdFeE5qTXRZalU1WkdJMlpUSTJNelEzIiwidCI6IjQzZWZhZGJhLTJjZDEtNGUyZS04MTc0LWMxNjg2ZjBkNTU0NCJ9

# Notifications Configuration
EMAIL=${config.email || 'admin@morloksmaze.com'}

# Storage Configuration
DATA_ROOT=./data
CONFIG_ROOT=./config

# Database Configuration
POSTGRES_USER=mediaserver
POSTGRES_DB=mediaserver

# Additional Variables
POSTGRES_PASSWORD_FILE=/run/secrets/postgres_password
JELLYFIN_API_KEY_FILE=/run/secrets/jellyfin_api_key
SONARR_API_KEY_FILE=/run/secrets/sonarr_api_key
RADARR_API_KEY_FILE=/run/secrets/radarr_api_key
LIDARR_API_KEY_FILE=/run/secrets/lidarr_api_key
READARR_API_KEY_FILE=/run/secrets/readarr_api_key
BAZARR_API_KEY_FILE=/run/secrets/bazarr_api_key
TAUTULLI_API_KEY_FILE=/run/secrets/tautulli_api_key
PHOTOPRISM_ADMIN_PASSWORD_FILE=/run/secrets/photoprism_admin_password
YTDL_MATERIAL_IMAGE=ghcr.io/iv-org/youtube-dl-material:latest
YTDL_MATERIAL_PORT=17442
CF_API_EMAIL=admin@morloksmaze.com
CF_API_KEY=PLACEHOLDER_CLOUDFLARE_API_KEY
CLOUDFLARE_ZONE_ID=PLACEHOLDER_CLOUDFLARE_ZONE_ID
CF_TUNNEL_NAME=home-morloksmaze-tunnel
DEPLOY_MONITORING=true
SLACK_WEBHOOK=PLACEHOLDER_SLACK_WEBHOOK
REDIS_PASSWORD=secure-redis-password-morloksmaze
`;

  return content;
}

app.listen(PORT, () => {
  console.log('🎬 Media Server Stack Web UI listening on port 3000');
  console.log('🌐 Access the setup interface at: http://localhost:3000');
  console.log(`📁 Stack directory: ${STACK_DIR}`);
});
