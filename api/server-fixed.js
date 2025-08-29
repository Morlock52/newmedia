const logger = require('../middleware/logger.js');
const express = require('express');
const cors = require('cors');
const axios = require('axios');
const WebSocket = require('ws');
const http = require('http');
const path = require('path');
const fs = require('fs').promises;

const app = express();
const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../dashboard')));

// Service configurations with actual API keys
const services = {
  sonarr: {
    url: 'http://localhost:8989',
    apiKey: null,
    configPath: '../sonarr-config/config.xml'
  },
  radarr: {
    url: 'http://localhost:7878',
    apiKey: null,
    configPath: '../radarr-config/config.xml'
  },
  lidarr: {
    url: 'http://localhost:8686',
    apiKey: null,
    configPath: '../lidarr-config/config.xml'
  },
  prowlarr: {
    url: 'http://localhost:9696',
    apiKey: null,
    configPath: '../prowlarr-config/config.xml'
  },
  jellyfin: {
    url: 'http://localhost:8096',
    apiKey: null
  },
  qbittorrent: {
    url: 'http://localhost:8080',
    username: 'admin',
    password: 'adminadmin'
  }
};

// Extract API keys from config files
async function extractApiKeys() {
  for (const [name, service] of Object.entries(services)) {
    if (service.configPath) {
      try {
        const configPath = path.join(__dirname, service.configPath);
        const content = await fs.readFile(configPath, 'utf8');
        const match = content.match(/<ApiKey>([^<]+)<\/ApiKey>/);
        if (match) {
          service.apiKey = match[1];
          logger.info(`✅ Extracted API key for ${name}`);
        }
      } catch (error) {
        logger.info(`⚠️ Could not extract API key for ${name}:`, error.message);
      }
    }
  }
}

// Get service status
async function getServiceStatus(name, service) {
  try {
    let response;
    
    switch(name) {
      case 'sonarr':
      case 'radarr':
      case 'lidarr':
      case 'prowlarr':
        if (!service.apiKey) return { name, status: 'error', message: 'No API key' };
        response = await axios.get(`${service.url}/api/v3/system/status`, {
          headers: { 'X-Api-Key': service.apiKey },
          timeout: 5000
        });
        return {
          name,
          status: 'online',
          version: response.data.version,
          branch: response.data.branch
        };
        
      case 'jellyfin':
        response = await axios.get(`${service.url}/System/Info/Public`, {
          timeout: 5000
        });
        return {
          name,
          status: 'online',
          version: response.data.Version,
          serverName: response.data.ServerName
        };
        
      case 'qbittorrent':
        // First login to get cookie
        try {
          await axios.post(`${service.url}/api/v2/auth/login`, 
            `username=${service.username}&password=${service.password}`,
            {
              headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
              timeout: 5000
            }
          );
          
          response = await axios.get(`${service.url}/api/v2/app/version`, {
            timeout: 5000
          });
          
          return {
            name,
            status: 'online',
            version: response.data
          };
        } catch (error) {
          return {
            name,
            status: 'online',
            message: 'Running (auth required)'
          };
        }
        
      default:
        return { name, status: 'unknown' };
    }
  } catch (error) {
    return {
      name,
      status: 'offline',
      error: error.message
    };
  }
}

// Routes
app.get('/api/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: new Date() });
});

app.get('/api/services/status', async (req, res) => {
  const statuses = await Promise.all(
    Object.entries(services).map(([name, service]) => 
      getServiceStatus(name, service)
    )
  );
  res.json(statuses);
});

app.get('/api/services/:service/status', async (req, res) => {
  const serviceName = req.params.service;
  const service = services[serviceName];
  
  if (!service) {
    return res.status(404).json({ error: 'Service not found' });
  }
  
  const status = await getServiceStatus(serviceName, service);
  res.json(status);
});

// Media library stats
app.get('/api/media/stats', async (req, res) => {
  const stats = {
    movies: 0,
    series: 0,
    episodes: 0,
    artists: 0,
    albums: 0,
    tracks: 0
  };
  
  try {
    // Get Radarr movies
    if (services.radarr.apiKey) {
      const movies = await axios.get(`${services.radarr.url}/api/v3/movie`, {
        headers: { 'X-Api-Key': services.radarr.apiKey },
        timeout: 5000
      });
      stats.movies = movies.data.length;
    }
    
    // Get Sonarr series
    if (services.sonarr.apiKey) {
      const series = await axios.get(`${services.sonarr.url}/api/v3/series`, {
        headers: { 'X-Api-Key': services.sonarr.apiKey },
        timeout: 5000
      });
      stats.series = series.data.length;
      
      // Count episodes
      for (const show of series.data) {
        stats.episodes += show.statistics?.episodeCount || 0;
      }
    }
    
    // Get Lidarr artists
    if (services.lidarr.apiKey) {
      const artists = await axios.get(`${services.lidarr.url}/api/v1/artist`, {
        headers: { 'X-Api-Key': services.lidarr.apiKey },
        timeout: 5000
      });
      stats.artists = artists.data.length;
    }
  } catch (error) {
    logger.error('Error fetching media stats:', error.message);
  }
  
  res.json(stats);
});

// Download queue
app.get('/api/downloads/queue', async (req, res) => {
  const queue = [];
  
  try {
    // Get Sonarr queue
    if (services.sonarr.apiKey) {
      const sonarrQueue = await axios.get(`${services.sonarr.url}/api/v3/queue`, {
        headers: { 'X-Api-Key': services.sonarr.apiKey },
        timeout: 5000
      });
      
      sonarrQueue.data.records?.forEach(item => {
        queue.push({
          title: item.title,
          status: item.status,
          progress: item.sizeleft ? ((item.size - item.sizeleft) / item.size * 100) : 0,
          size: item.size,
          service: 'sonarr'
        });
      });
    }
    
    // Get Radarr queue
    if (services.radarr.apiKey) {
      const radarrQueue = await axios.get(`${services.radarr.url}/api/v3/queue`, {
        headers: { 'X-Api-Key': services.radarr.apiKey },
        timeout: 5000
      });
      
      radarrQueue.data.records?.forEach(item => {
        queue.push({
          title: item.title,
          status: item.status,
          progress: item.sizeleft ? ((item.size - item.sizeleft) / item.size * 100) : 0,
          size: item.size,
          service: 'radarr'
        });
      });
    }
  } catch (error) {
    logger.error('Error fetching download queue:', error.message);
  }
  
  res.json(queue);
});

// WebSocket for real-time updates
wss.on('connection', (ws) => {
  logger.info('WebSocket client connected');
  
  // Send initial status
  getServiceStatus('sonarr', services.sonarr).then(status => {
    ws.send(JSON.stringify({ type: 'status', data: status }));
  });
  
  // Set up periodic updates
  const interval = setInterval(async () => {
    const statuses = await Promise.all(
      Object.entries(services).map(([name, service]) => 
        getServiceStatus(name, service)
      )
    );
    
    ws.send(JSON.stringify({ type: 'statusUpdate', data: statuses }));
  }, 5000);
  
  ws.on('close', () => {
    logger.info('WebSocket client disconnected');
    clearInterval(interval);
  });
});

// Serve dashboard
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, '../dashboard/index.html'));
});

// Start server
const PORT = process.env.PORT || 3005;

async function start() {
  await extractApiKeys();
  
  server.listen(PORT, () => {
    logger.info(`
╔══════════════════════════════════════════════════════════╗
║  🚀 Media Server API Running                             ║
║  📡 API: http://localhost:${PORT}                            ║
║  🌐 Dashboard: http://localhost:${PORT}                       ║
║  🔄 WebSocket: ws://localhost:${PORT}                         ║
╚══════════════════════════════════════════════════════════╝
    `);
  });
}

start().catch(console.error);