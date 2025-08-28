#!/usr/bin/env node

/**
 * Ultimate Media Server 2025 - Simple Unified MCP Server
 * Single MCP server for ALL 30 media services with HTTP/JSON approach
 * No SDK dependencies - pure JSON-RPC 2.0 implementation
 */

const http = require('http');
const url = require('url');
const axios = require('axios');

class SimpleUnifiedMCP {
    constructor() {
        this.serverInfo = {
            name: "ultimate-media-server-2025",
            version: "2.0.0",
            description: "Unified MCP server for all 30 media server applications"
        };

        // All 30 services configuration
        this.services = {
            // Media Servers (3)
            jellyfin: { name: "Jellyfin", url: process.env.JELLYFIN_URL || "http://localhost:8096", port: 8096, category: "media-server" },
            plex: { name: "Plex", url: process.env.PLEX_URL || "http://localhost:32400", port: 32400, category: "media-server" },
            emby: { name: "Emby", url: process.env.EMBY_URL || "http://localhost:8097", port: 8097, category: "media-server" },
            
            // Content Management (5)
            sonarr: { name: "Sonarr", url: process.env.SONARR_URL || "http://localhost:8989", port: 8989, category: "content-mgmt" },
            radarr: { name: "Radarr", url: process.env.RADARR_URL || "http://localhost:7878", port: 7878, category: "content-mgmt" },
            lidarr: { name: "Lidarr", url: process.env.LIDARR_URL || "http://localhost:8686", port: 8686, category: "content-mgmt" },
            readarr: { name: "Readarr", url: process.env.READARR_URL || "http://localhost:8787", port: 8787, category: "content-mgmt" },
            bazarr: { name: "Bazarr", url: process.env.BAZARR_URL || "http://localhost:6767", port: 6767, category: "content-mgmt" },
            
            // Indexers & Search (3)
            prowlarr: { name: "Prowlarr", url: process.env.PROWLARR_URL || "http://localhost:9696", port: 9696, category: "indexer" },
            jackett: { name: "Jackett", url: process.env.JACKETT_URL || "http://localhost:9117", port: 9117, category: "indexer" },
            flaresolverr: { name: "FlareSolverr", url: process.env.FLARESOLVERR_URL || "http://localhost:8191", port: 8191, category: "indexer" },
            
            // Download Clients (5)
            qbittorrent: { name: "qBittorrent", url: process.env.QBITTORRENT_URL || "http://localhost:8080", port: 8080, category: "download" },
            transmission: { name: "Transmission", url: process.env.TRANSMISSION_URL || "http://localhost:9091", port: 9091, category: "download" },
            deluge: { name: "Deluge", url: process.env.DELUGE_URL || "http://localhost:8112", port: 8112, category: "download" },
            nzbget: { name: "NZBGet", url: process.env.NZBGET_URL || "http://localhost:6789", port: 6789, category: "download" },
            sabnzbd: { name: "SABnzbd", url: process.env.SABNZBD_URL || "http://localhost:8085", port: 8085, category: "download" },
            
            // Request Management (3)
            overseerr: { name: "Overseerr", url: process.env.OVERSEERR_URL || "http://localhost:5055", port: 5055, category: "requests" },
            requestrr: { name: "Requestrr", url: process.env.REQUESTRR_URL || "http://localhost:4545", port: 4545, category: "requests" },
            ombi: { name: "Ombi", url: process.env.OMBI_URL || "http://localhost:3579", port: 3579, category: "requests" },
            
            // Analytics & Monitoring (2)
            tautulli: { name: "Tautulli", url: process.env.TAUTULLI_URL || "http://localhost:8181", port: 8181, category: "analytics" },
            netdata: { name: "Netdata", url: process.env.NETDATA_URL || "http://localhost:19999", port: 19999, category: "analytics" },
            
            // Dashboards (4)
            homepage: { name: "Homepage", url: process.env.HOMEPAGE_URL || "http://localhost:3000", port: 3000, category: "dashboard" },
            heimdall: { name: "Heimdall", url: process.env.HEIMDALL_URL || "http://localhost:7575", port: 7575, category: "dashboard" },
            organizr: { name: "Organizr", url: process.env.ORGANIZR_URL || "http://localhost:8081", port: 8081, category: "dashboard" },
            homarr: { name: "Homarr", url: process.env.HOMARR_URL || "http://localhost:7576", port: 7576, category: "dashboard" },
            
            // Infrastructure (5)
            nginx_proxy_manager: { name: "Nginx Proxy Manager", url: process.env.NGINX_PROXY_MANAGER_URL || "http://localhost:81", port: 81, category: "infrastructure" },
            portainer: { name: "Portainer", url: process.env.PORTAINER_URL || "http://localhost:9000", port: 9000, category: "infrastructure" },
            watchtower: { name: "Watchtower", url: "internal://watchtower", port: 0, category: "infrastructure" },
            gluetun: { name: "Gluetun VPN", url: "internal://gluetun", port: 0, category: "infrastructure" },
            unpackerr: { name: "Unpackerr", url: "internal://unpackerr", port: 0, category: "infrastructure" }
        };

        this.tools = this.generateTools();
        this.resources = this.generateResources();
        this.prompts = this.generatePrompts();
    }

    generateTools() {
        return [
            {
                name: "get_all_services",
                description: "Get comprehensive overview of all 30 media server services",
                inputSchema: {
                    type: "object",
                    properties: {
                        category: {
                            type: "string",
                            description: "Filter by category: media-server, content-mgmt, indexer, download, requests, analytics, dashboard, infrastructure",
                            enum: ["media-server", "content-mgmt", "indexer", "download", "requests", "analytics", "dashboard", "infrastructure"]
                        },
                        include_health: {
                            type: "boolean",
                            description: "Include health status for each service",
                            default: true
                        }
                    }
                }
            },
            {
                name: "check_service_health",
                description: "Check health and availability of specific services",
                inputSchema: {
                    type: "object",
                    properties: {
                        services: {
                            type: "array",
                            items: { type: "string" },
                            description: "List of service names to check (or 'all' for all services)"
                        },
                        detailed: {
                            type: "boolean",
                            description: "Include detailed response information",
                            default: false
                        }
                    },
                    required: ["services"]
                }
            },
            {
                name: "search_across_services",
                description: "Search for content across all applicable media services",
                inputSchema: {
                    type: "object",
                    properties: {
                        query: {
                            type: "string",
                            description: "Search query"
                        },
                        media_type: {
                            type: "string",
                            description: "Type of media to search for",
                            enum: ["movie", "tv", "music", "book", "all"]
                        },
                        services: {
                            type: "array",
                            items: { type: "string" },
                            description: "Specific services to search (optional)"
                        }
                    },
                    required: ["query"]
                }
            },
            {
                name: "get_download_status",
                description: "Get download status from all download clients",
                inputSchema: {
                    type: "object",
                    properties: {
                        client: {
                            type: "string",
                            description: "Specific download client (optional)",
                            enum: ["qbittorrent", "transmission", "deluge", "nzbget", "sabnzbd", "all"]
                        },
                        status_filter: {
                            type: "string",
                            description: "Filter by download status",
                            enum: ["downloading", "completed", "paused", "error", "all"]
                        }
                    }
                }
            },
            {
                name: "manage_downloads",
                description: "Control downloads across all download clients",
                inputSchema: {
                    type: "object",
                    properties: {
                        action: {
                            type: "string",
                            description: "Action to perform",
                            enum: ["pause", "resume", "delete", "pause_all", "resume_all"]
                        },
                        client: {
                            type: "string",
                            description: "Target download client"
                        },
                        download_id: {
                            type: "string",
                            description: "Specific download ID (for individual actions)"
                        }
                    },
                    required: ["action", "client"]
                }
            },
            {
                name: "get_library_stats",
                description: "Get comprehensive library statistics from all media servers",
                inputSchema: {
                    type: "object",
                    properties: {
                        server: {
                            type: "string",
                            description: "Specific media server",
                            enum: ["jellyfin", "plex", "emby", "all"]
                        },
                        include_recent: {
                            type: "boolean",
                            description: "Include recently added content",
                            default: true
                        }
                    }
                }
            },
            {
                name: "get_requests_overview",
                description: "Get overview of content requests from all request management services",
                inputSchema: {
                    type: "object",
                    properties: {
                        service: {
                            type: "string",
                            description: "Specific request service",
                            enum: ["overseerr", "requestrr", "ombi", "all"]
                        },
                        status: {
                            type: "string",
                            description: "Filter by request status",
                            enum: ["pending", "approved", "available", "denied", "all"]
                        }
                    }
                }
            },
            {
                name: "get_system_overview",
                description: "Get comprehensive system overview including all services, health, and statistics",
                inputSchema: {
                    type: "object",
                    properties: {
                        include_performance: {
                            type: "boolean",
                            description: "Include system performance metrics",
                            default: true
                        }
                    }
                }
            },
            {
                name: "manage_service",
                description: "Manage individual services (restart, enable, disable)",
                inputSchema: {
                    type: "object",
                    properties: {
                        service: {
                            type: "string",
                            description: "Service name to manage"
                        },
                        action: {
                            type: "string",
                            description: "Action to perform",
                            enum: ["restart", "start", "stop", "status"]
                        }
                    },
                    required: ["service", "action"]
                }
            },
            {
                name: "configure_service",
                description: "Get or update service configuration",
                inputSchema: {
                    type: "object",
                    properties: {
                        service: {
                            type: "string",
                            description: "Service name to configure"
                        },
                        action: {
                            type: "string",
                            description: "Configuration action",
                            enum: ["get", "set", "reset"]
                        },
                        config: {
                            type: "object",
                            description: "Configuration parameters (for set action)"
                        }
                    },
                    required: ["service", "action"]
                }
            }
        ];
    }

    generateResources() {
        return [
            {
                uri: "media://services",
                name: "All Media Services",
                description: "Complete list of all 30 media server services with details",
                mimeType: "application/json"
            },
            {
                uri: "media://health",
                name: "System Health",
                description: "Real-time health status of all services",
                mimeType: "application/json"
            },
            {
                uri: "media://stats",
                name: "System Statistics",
                description: "Comprehensive statistics from all services",
                mimeType: "application/json"
            },
            {
                uri: "media://downloads",
                name: "Download Overview",
                description: "Current download status across all clients",
                mimeType: "application/json"
            },
            {
                uri: "media://requests",
                name: "Content Requests",
                description: "Pending and completed content requests",
                mimeType: "application/json"
            }
        ];
    }

    generatePrompts() {
        return [
            {
                name: "media_dashboard_assistant",
                description: "AI assistant for comprehensive media server management across all 30 services",
                arguments: [
                    {
                        name: "query",
                        description: "User query about media server management",
                        required: true
                    },
                    {
                        name: "context",
                        description: "Current system context",
                        required: false
                    }
                ]
            },
            {
                name: "content_curator",
                description: "AI assistant for content discovery and recommendation across all media types",
                arguments: [
                    {
                        name: "preferences",
                        description: "User content preferences",
                        required: true
                    },
                    {
                        name: "media_type",
                        description: "Type of media for recommendations",
                        required: false
                    }
                ]
            },
            {
                name: "system_optimizer",
                description: "AI assistant for optimizing system performance and configuration",
                arguments: [
                    {
                        name: "focus_area",
                        description: "Specific area to optimize",
                        required: false
                    }
                ]
            }
        ];
    }

    async checkServiceHealth(serviceKey) {
        const service = this.services[serviceKey];
        if (!service) return { status: 'unknown', error: 'Service not found' };

        try {
            const response = await axios.get(service.url, { 
                timeout: 5000,
                validateStatus: () => true // Accept all status codes
            });
            
            const isHealthy = response.status >= 200 && response.status < 400;
            return {
                status: isHealthy ? 'healthy' : 'unhealthy',
                statusCode: response.status,
                responseTime: response.headers['x-response-time'] || 'unknown',
                lastChecked: new Date().toISOString()
            };
        } catch (error) {
            return {
                status: 'offline',
                error: error.message,
                lastChecked: new Date().toISOString()
            };
        }
    }

    async handleToolCall(toolName, parameters) {
        switch (toolName) {
            case 'get_all_services':
                return await this.getAllServices(parameters);
            
            case 'check_service_health':
                return await this.checkServicesHealth(parameters);
            
            case 'search_across_services':
                return await this.searchAcrossServices(parameters);
            
            case 'get_download_status':
                return await this.getDownloadStatus(parameters);
            
            case 'manage_downloads':
                return await this.manageDownloads(parameters);
            
            case 'get_library_stats':
                return await this.getLibraryStats(parameters);
            
            case 'get_requests_overview':
                return await this.getRequestsOverview(parameters);
            
            case 'get_system_overview':
                return await this.getSystemOverview(parameters);
            
            case 'manage_service':
                return await this.manageService(parameters);
            
            case 'configure_service':
                return await this.configureService(parameters);
            
            default:
                throw new Error(`Unknown tool: ${toolName}`);
        }
    }

    async getAllServices(params = {}) {
        const { category, include_health } = params;
        
        let services = Object.entries(this.services);
        
        if (category) {
            services = services.filter(([key, service]) => service.category === category);
        }

        const result = {};
        
        for (const [key, service] of services) {
            result[key] = {
                ...service,
                health: include_health ? await this.checkServiceHealth(key) : undefined
            };
        }

        return {
            total_services: Object.keys(this.services).length,
            filtered_services: services.length,
            categories: [...new Set(Object.values(this.services).map(s => s.category))],
            services: result,
            timestamp: new Date().toISOString()
        };
    }

    async checkServicesHealth(params) {
        const { services, detailed } = params;
        const servicesToCheck = services.includes('all') ? Object.keys(this.services) : services;
        
        const healthResults = {};
        
        for (const serviceKey of servicesToCheck) {
            if (this.services[serviceKey]) {
                healthResults[serviceKey] = await this.checkServiceHealth(serviceKey);
            }
        }

        const summary = {
            total_checked: servicesToCheck.length,
            healthy: Object.values(healthResults).filter(h => h.status === 'healthy').length,
            unhealthy: Object.values(healthResults).filter(h => h.status === 'unhealthy').length,
            offline: Object.values(healthResults).filter(h => h.status === 'offline').length
        };

        return {
            summary,
            services: detailed ? healthResults : Object.fromEntries(
                Object.entries(healthResults).map(([key, health]) => [key, health.status])
            ),
            timestamp: new Date().toISOString()
        };
    }

    async searchAcrossServices(params) {
        const { query, media_type, services } = params;
        
        // Mock search results - in real implementation, this would query actual services
        return {
            query,
            media_type: media_type || 'all',
            results: {
                movies: [
                    { title: `${query} Movie Result`, year: 2024, source: 'radarr', available: false },
                    { title: `${query} Film`, year: 2023, source: 'jellyfin', available: true }
                ],
                tv: [
                    { title: `${query} Series`, seasons: 3, source: 'sonarr', available: false },
                    { title: `${query} Show`, seasons: 1, source: 'jellyfin', available: true }
                ],
                music: [
                    { artist: `${query} Artist`, album: `${query} Album`, source: 'lidarr', available: false }
                ],
                books: [
                    { title: `${query} Book`, author: 'Author Name', source: 'readarr', available: false }
                ]
            },
            total_results: 5,
            timestamp: new Date().toISOString()
        };
    }

    async getDownloadStatus(params = {}) {
        const { client, status_filter } = params;
        
        // Mock download status - in real implementation, this would query actual download clients
        return {
            summary: {
                total_downloads: 12,
                active: 3,
                completed: 8,
                paused: 1,
                errors: 0
            },
            clients: {
                qbittorrent: {
                    status: 'online',
                    downloads: 5,
                    active: 2,
                    upload_speed: '1.2 MB/s',
                    download_speed: '8.5 MB/s'
                },
                transmission: {
                    status: 'online',
                    downloads: 3,
                    active: 1,
                    upload_speed: '0.8 MB/s',
                    download_speed: '4.2 MB/s'
                },
                deluge: {
                    status: 'offline',
                    downloads: 0,
                    active: 0
                },
                nzbget: {
                    status: 'online',
                    downloads: 2,
                    active: 0,
                    queue_size: '1.2 GB'
                },
                sabnzbd: {
                    status: 'online',
                    downloads: 2,
                    active: 0,
                    queue_size: '0.8 GB'
                }
            },
            timestamp: new Date().toISOString()
        };
    }

    async manageDownloads(params) {
        const { action, client, download_id } = params;
        
        // Mock management response
        return {
            action,
            client,
            download_id,
            success: true,
            message: `Successfully executed ${action} on ${client}${download_id ? ` for download ${download_id}` : ''}`,
            timestamp: new Date().toISOString()
        };
    }

    async getLibraryStats(params = {}) {
        const { server, include_recent } = params;
        
        // Mock library statistics
        return {
            servers: {
                jellyfin: {
                    status: 'online',
                    libraries: {
                        movies: { count: 1247, size: '2.3 TB' },
                        tv: { count: 89, episodes: 2156, size: '1.8 TB' },
                        music: { count: 456, albums: 123, size: '89 GB' }
                    },
                    recent_activity: include_recent ? [
                        { title: 'New Movie Added', time: '2 hours ago' },
                        { title: 'TV Episode Added', time: '1 day ago' }
                    ] : undefined
                },
                plex: {
                    status: 'online',
                    libraries: {
                        movies: { count: 1189, size: '2.1 TB' },
                        tv: { count: 78, episodes: 1987, size: '1.6 TB' }
                    }
                },
                emby: {
                    status: 'offline',
                    libraries: {}
                }
            },
            totals: {
                movies: 2436,
                tv_shows: 167,
                tv_episodes: 4143,
                music_albums: 123,
                total_size: '6.8 TB'
            },
            timestamp: new Date().toISOString()
        };
    }

    async getRequestsOverview(params = {}) {
        const { service, status } = params;
        
        // Mock requests data
        return {
            summary: {
                total_requests: 45,
                pending: 12,
                approved: 28,
                available: 15,
                denied: 5
            },
            services: {
                overseerr: {
                    status: 'online',
                    requests: 32,
                    pending: 8,
                    recent: [
                        { title: 'Requested Movie', user: 'user1', status: 'pending' },
                        { title: 'TV Series', user: 'user2', status: 'approved' }
                    ]
                },
                requestrr: {
                    status: 'online',
                    requests: 8,
                    pending: 3
                },
                ombi: {
                    status: 'offline',
                    requests: 5,
                    pending: 1
                }
            },
            timestamp: new Date().toISOString()
        };
    }

    async getSystemOverview(params = {}) {
        const { include_performance } = params;
        
        const health = await this.checkServicesHealth({ services: ['all'], detailed: false });
        
        return {
            system: {
                total_services: 30,
                online_services: health.summary.healthy,
                offline_services: health.summary.offline + health.summary.unhealthy,
                uptime: process.uptime(),
                memory_usage: process.memoryUsage(),
                cpu_usage: include_performance ? '23%' : undefined,
                disk_usage: include_performance ? {
                    total: '10 TB',
                    used: '6.8 TB',
                    free: '3.2 TB',
                    percentage: 68
                } : undefined
            },
            services_by_category: {
                'media-server': 3,
                'content-mgmt': 5,
                'indexer': 3,
                'download': 5,
                'requests': 3,
                'analytics': 2,
                'dashboard': 4,
                'infrastructure': 5
            },
            quick_stats: {
                total_media_items: 6700,
                active_downloads: 3,
                pending_requests: 12,
                recent_activity: '15 items added today'
            },
            timestamp: new Date().toISOString()
        };
    }

    async manageService(params) {
        const { service, action } = params;
        
        if (!this.services[service]) {
            throw new Error(`Service '${service}' not found`);
        }

        // Mock service management
        return {
            service,
            action,
            success: true,
            message: `Successfully executed ${action} on ${service}`,
            timestamp: new Date().toISOString()
        };
    }

    async configureService(params) {
        const { service, action, config } = params;
        
        if (!this.services[service]) {
            throw new Error(`Service '${service}' not found`);
        }

        // Mock configuration management
        return {
            service,
            action,
            config: action === 'get' ? { 
                example_setting: 'value',
                another_setting: true 
            } : config,
            success: true,
            message: `Successfully ${action} configuration for ${service}`,
            timestamp: new Date().toISOString()
        };
    }

    async handleResourceRead(uri) {
        const resourceMap = {
            'media://services': () => this.getAllServices({ include_health: true }),
            'media://health': () => this.checkServicesHealth({ services: ['all'], detailed: true }),
            'media://stats': () => this.getLibraryStats({ server: 'all', include_recent: true }),
            'media://downloads': () => this.getDownloadStatus(),
            'media://requests': () => this.getRequestsOverview()
        };

        const handler = resourceMap[uri];
        if (!handler) {
            throw new Error(`Resource not found: ${uri}`);
        }

        return await handler();
    }

    async handlePrompt(name, args) {
        switch (name) {
            case 'media_dashboard_assistant':
                return `You are an AI assistant for the Ultimate Media Server 2025 system managing 30 different services. 
                
Current system status: ${JSON.stringify(await this.getSystemOverview(), null, 2)}

User query: ${args.query}
Context: ${args.context || 'None provided'}

Please provide helpful guidance for managing the media server ecosystem.`;

            case 'content_curator':
                return `You are a content curation assistant for a comprehensive media server with movies, TV shows, music, and books.

User preferences: ${args.preferences}
Media type focus: ${args.media_type || 'all types'}

Available services for content discovery:
- Movies: Radarr, Jellyfin, Plex, Emby, Overseerr
- TV Shows: Sonarr, Jellyfin, Plex, Emby, Overseerr  
- Music: Lidarr, Jellyfin, Plex
- Books: Readarr

Please provide personalized content recommendations.`;

            case 'system_optimizer':
                const systemStats = await this.getSystemOverview({ include_performance: true });
                return `You are a system optimization assistant for the Ultimate Media Server 2025.

Current system metrics: ${JSON.stringify(systemStats, null, 2)}

Focus area: ${args.focus_area || 'general optimization'}

Please analyze the system and provide optimization recommendations.`;

            default:
                throw new Error(`Unknown prompt: ${name}`);
        }
    }

    createJsonRpcResponse(id, result, error = null) {
        const response = {
            jsonrpc: "2.0",
            id
        };

        if (error) {
            response.error = {
                code: error.code || -32603,
                message: error.message || "Internal error",
                data: error.data
            };
        } else {
            response.result = result;
        }

        return response;
    }

    async handleRequest(method, params, id) {
        try {
            switch (method) {
                case 'initialize':
                    return this.createJsonRpcResponse(id, {
                        protocolVersion: "2024-11-05",
                        capabilities: {
                            tools: {},
                            resources: {},
                            prompts: {}
                        },
                        serverInfo: this.serverInfo
                    });

                case 'tools/list':
                    return this.createJsonRpcResponse(id, { tools: this.tools });

                case 'tools/call':
                    const toolResult = await this.handleToolCall(params.name, params.arguments || {});
                    return this.createJsonRpcResponse(id, { content: [{ type: "text", text: JSON.stringify(toolResult, null, 2) }] });

                case 'resources/list':
                    return this.createJsonRpcResponse(id, { resources: this.resources });

                case 'resources/read':
                    const resourceResult = await this.handleResourceRead(params.uri);
                    return this.createJsonRpcResponse(id, { 
                        contents: [{ 
                            uri: params.uri, 
                            mimeType: "application/json", 
                            text: JSON.stringify(resourceResult, null, 2) 
                        }] 
                    });

                case 'prompts/list':
                    return this.createJsonRpcResponse(id, { prompts: this.prompts });

                case 'prompts/get':
                    const promptResult = await this.handlePrompt(params.name, params.arguments || {});
                    return this.createJsonRpcResponse(id, { 
                        description: `Prompt for ${params.name}`,
                        messages: [{ 
                            role: "user", 
                            content: { 
                                type: "text", 
                                text: promptResult 
                            } 
                        }] 
                    });

                default:
                    throw new Error(`Unknown method: ${method}`);
            }
        } catch (error) {
            console.error('Error handling request:', error);
            return this.createJsonRpcResponse(id, null, {
                code: -32603,
                message: error.message,
                data: error.stack
            });
        }
    }

    start(port = 3001) {
        const server = http.createServer(async (req, res) => {
            // Enable CORS
            res.setHeader('Access-Control-Allow-Origin', '*');
            res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
            res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
            res.setHeader('Content-Type', 'application/json');

            if (req.method === 'OPTIONS') {
                res.writeHead(200);
                res.end();
                return;
            }

            if (req.method === 'GET') {
                // Health check endpoint
                if (req.url === '/health') {
                    res.writeHead(200);
                    res.end(JSON.stringify({ status: 'healthy', services: 30, timestamp: new Date().toISOString() }));
                    return;
                }

                // Service listing endpoint
                if (req.url === '/services') {
                    const services = await this.getAllServices({ include_health: true });
                    res.writeHead(200);
                    res.end(JSON.stringify(services));
                    return;
                }

                // Default response
                res.writeHead(200);
                res.end(JSON.stringify({
                    name: this.serverInfo.name,
                    version: this.serverInfo.version,
                    description: this.serverInfo.description,
                    services: Object.keys(this.services).length,
                    endpoints: {
                        health: '/health',
                        services: '/services',
                        mcp: '/ (POST with JSON-RPC 2.0)'
                    }
                }));
                return;
            }

            if (req.method === 'POST') {
                let body = '';
                req.on('data', chunk => body += chunk);
                req.on('end', async () => {
                    try {
                        const request = JSON.parse(body);
                        const response = await this.handleRequest(request.method, request.params, request.id);
                        res.writeHead(200);
                        res.end(JSON.stringify(response));
                    } catch (error) {
                        console.error('Error processing request:', error);
                        res.writeHead(400);
                        res.end(JSON.stringify({
                            jsonrpc: "2.0",
                            id: null,
                            error: {
                                code: -32700,
                                message: "Parse error"
                            }
                        }));
                    }
                });
                return;
            }

            res.writeHead(405);
            res.end(JSON.stringify({ error: 'Method not allowed' }));
        });

        server.listen(port, () => {
            console.log(`🚀 Ultimate Media Server 2025 MCP Server running on port ${port}`);
            console.log(`📊 Managing ${Object.keys(this.services).length} services across ${[...new Set(Object.values(this.services).map(s => s.category))].length} categories`);
            console.log(`🌐 Health check: http://localhost:${port}/health`);
            console.log(`📋 Services list: http://localhost:${port}/services`);
            console.log(`🔧 MCP endpoint: http://localhost:${port}/ (POST)`);
        });

        return server;
    }
}

// Start the server if run directly
if (require.main === module) {
    const server = new SimpleUnifiedMCP();
    server.start(process.env.MCP_PORT || 3001);
}

module.exports = SimpleUnifiedMCP;