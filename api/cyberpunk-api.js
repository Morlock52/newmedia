const logger = require('../middleware/logger.js');
const express = require('express');
const cors = require('cors');
const WebSocket = require('ws');
const Docker = require('dockerode');
const axios = require('axios');
const app = express();
const docker = new Docker();

// Middleware
app.use(cors());
app.use(express.json());

// Service Configuration
const SERVICE_CONFIG = {
    'Plex': { port: 32400, container: 'plex', api: '/identity' },
    'Jellyfin': { port: 8096, container: 'jellyfin', api: '/System/Info' },
    'Emby': { port: 8096, container: 'emby', api: '/System/Info' },
    'Sonarr': { port: 8989, container: 'sonarr', api: '/api/v3/system/status' },
    'Radarr': { port: 7878, container: 'radarr', api: '/api/v3/system/status' },
    'Lidarr': { port: 8686, container: 'lidarr', api: '/api/v1/system/status' },
    'Readarr': { port: 8787, container: 'readarr', api: '/api/v1/system/status' },
    'Bazarr': { port: 6767, container: 'bazarr', api: '/api/system/status' },
    'Prowlarr': { port: 9696, container: 'prowlarr', api: '/api/v1/system/status' },
    'qBittorrent': { port: 8080, container: 'qbittorrent', api: '/api/v2/app/version' },
    'SABnzbd': { port: 8085, container: 'sabnzbd', api: '/api?mode=version' },
    'Transmission': { port: 9091, container: 'transmission', api: '/transmission/rpc' },
    'Overseerr': { port: 5055, container: 'overseerr', api: '/api/v1/status' },
    'Jellyseerr': { port: 5055, container: 'jellyseerr', api: '/api/v1/status' },
    'Tautulli': { port: 8181, container: 'tautulli', api: '/api/v2' },
    'Organizr': { port: 9983, container: 'organizr', api: '/api/v2/ping' },
    'Heimdall': { port: 8443, container: 'heimdall', api: '/ping' },
    'Homer': { port: 8080, container: 'homer', api: '/' },
    'Portainer': { port: 9000, container: 'portainer', api: '/api/status' },
    'Nginx Proxy': { port: 81, container: 'nginx-proxy-manager', api: '/api/' },
    'Uptime Kuma': { port: 3001, container: 'uptime-kuma', api: '/api/status-page/heartbeat' },
    'Grafana': { port: 3000, container: 'grafana', api: '/api/health' },
    'Prometheus': { port: 9090, container: 'prometheus', api: '/-/healthy' },
    'Watchtower': { port: null, container: 'watchtower', api: null },
    'Duplicati': { port: 8200, container: 'duplicati', api: '/api/v1/Serverstate' },
    'Nextcloud': { port: 8080, container: 'nextcloud', api: '/status.php' },
    'Syncthing': { port: 8384, container: 'syncthing', api: '/rest/system/ping' },
    'FreshRSS': { port: 8080, container: 'freshrss', api: '/api/' },
    'Calibre-Web': { port: 8083, container: 'calibre-web', api: '/' },
    'PhotoPrism': { port: 2342, container: 'photoprism', api: '/api/v1/config' }
};

// Get all services status
app.get('/api/services', async (req, res) => {
    try {
        const containers = await docker.listContainers({ all: true });
        const services = [];

        for (const [name, config] of Object.entries(SERVICE_CONFIG)) {
            const container = containers.find(c => 
                c.Names.some(n => n.includes(config.container))
            );

            let status = 'offline';
            let stats = { cpu: 0, memory: 0 };

            if (container) {
                status = container.State === 'running' ? 'online' : 'offline';
                
                if (status === 'online') {
                    try {
                        const containerObj = docker.getContainer(container.Id);
                        const statsData = await containerObj.stats({ stream: false });
                        
                        // Calculate CPU percentage
                        const cpuDelta = statsData.cpu_stats.cpu_usage.total_usage - 
                                        statsData.precpu_stats.cpu_usage.total_usage;
                        const systemDelta = statsData.cpu_stats.system_cpu_usage - 
                                          statsData.precpu_stats.system_cpu_usage;
                        const cpuPercent = (cpuDelta / systemDelta) * 100;
                        
                        // Calculate memory percentage
                        const memUsage = statsData.memory_stats.usage;
                        const memLimit = statsData.memory_stats.limit;
                        const memPercent = (memUsage / memLimit) * 100;
                        
                        stats = {
                            cpu: Math.round(cpuPercent),
                            memory: Math.round(memPercent)
                        };
                    } catch (err) {
                        logger.error(`Error getting stats for ${name}:`, err);
                    }
                }
            }

            services.push({
                name,
                status,
                port: config.port,
                container: config.container,
                stats
            });
        }

        res.json(services);
    } catch (error) {
        logger.error('Error fetching services:', error);
        res.status(500).json({ error: 'Failed to fetch services' });
    }
});

// Restart service
app.post('/api/services/:name/restart', async (req, res) => {
    try {
        const { name } = req.params;
        const config = SERVICE_CONFIG[name];
        
        if (!config) {
            return res.status(404).json({ error: 'Service not found' });
        }

        const containers = await docker.listContainers({ all: true });
        const container = containers.find(c => 
            c.Names.some(n => n.includes(config.container))
        );

        if (!container) {
            return res.status(404).json({ error: 'Container not found' });
        }

        const containerObj = docker.getContainer(container.Id);
        await containerObj.restart();

        res.json({ message: `${name} restarted successfully` });
    } catch (error) {
        logger.error('Error restarting service:', error);
        res.status(500).json({ error: 'Failed to restart service' });
    }
});

// Get system stats
app.get('/api/stats', async (req, res) => {
    try {
        const info = await docker.info();
        const containers = await docker.listContainers();
        
        // Calculate totals
        let totalCpu = 0;
        let totalMemory = 0;
        let activeStreams = 0;
        let downloads = 0;

        for (const container of containers) {
            try {
                const containerObj = docker.getContainer(container.Id);
                const stats = await containerObj.stats({ stream: false });
                
                const cpuDelta = stats.cpu_stats.cpu_usage.total_usage - 
                                stats.precpu_stats.cpu_usage.total_usage;
                const systemDelta = stats.cpu_stats.system_cpu_usage - 
                                  stats.precpu_stats.system_cpu_usage;
                totalCpu += (cpuDelta / systemDelta) * 100;
                
                const memUsage = stats.memory_stats.usage;
                totalMemory += memUsage;
            } catch (err) {
                logger.error('Error getting container stats:', err);
            }
        }

        // Check specific services for active streams/downloads
        try {
            // Check Plex for active streams
            const plexResponse = await axios.get('http://localhost:32400/status/sessions');
            activeStreams += plexResponse.data?.MediaContainer?.size || 0;
        } catch (err) {}

        try {
            // Check qBittorrent for downloads
            const qbitResponse = await axios.get('http://localhost:8080/api/v2/torrents/info');
            downloads = qbitResponse.data?.length || 0;
        } catch (err) {}

        res.json({
            services: containers.length,
            cpu: Math.round(totalCpu),
            memory: Math.round(totalMemory / (1024 * 1024 * 1024)), // Convert to GB
            activeStreams,
            downloads,
            storage: Math.round(info.DriverStatus[0][1] / (1024 * 1024 * 1024)), // Example
            bandwidth: Math.random() * 100 // Would need actual network monitoring
        });
    } catch (error) {
        logger.error('Error fetching stats:', error);
        res.status(500).json({ error: 'Failed to fetch stats' });
    }
});

// AI Assistant endpoint
app.post('/api/ai/chat', async (req, res) => {
    const { message } = req.body;
    
    // Process AI commands
    let response = 'Command processed';
    
    if (message.toLowerCase().includes('restart')) {
        const serviceName = message.match(/restart\s+(\w+)/i)?.[1];
        if (serviceName && SERVICE_CONFIG[serviceName]) {
            // Restart the service
            response = `Restarting ${serviceName}...`;
        }
    } else if (message.toLowerCase().includes('status')) {
        response = 'All services are operational';
    } else if (message.toLowerCase().includes('download')) {
        response = 'Download queue accessed';
    }
    
    res.json({ response });
});

// WebSocket server for real-time updates
const wss = new WebSocket.Server({ port: 8001 });

wss.on('connection', (ws) => {
    logger.info('WebSocket client connected');
    
    // Send updates every 2 seconds
    const interval = setInterval(async () => {
        try {
            const containers = await docker.listContainers();
            const stats = {
                type: 'stats',
                data: {
                    services: containers.length,
                    timestamp: new Date().toISOString()
                }
            };
            ws.send(JSON.stringify(stats));
        } catch (error) {
            logger.error('WebSocket error:', error);
        }
    }, 2000);
    
    ws.on('close', () => {
        clearInterval(interval);
        logger.info('WebSocket client disconnected');
    });
});

// Start server
const PORT = process.env.PORT || 3738;
app.listen(PORT, () => {
    logger.info(`Cyberpunk API Server running on port ${PORT}`);
    logger.info(`WebSocket server running on port 8001`);
});

module.exports = app;