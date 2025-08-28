/**
 * NetFlow Integration Wrapper
 * Simplified interface for NetFlow network monitoring
 */

const NetflowIntegration = require('./NetflowIntegration');
const EventEmitter = require('events');

/**
 * Factory function to create a NetFlow integration instance
 * @param {Object} config - Configuration options
 * @returns {NetflowIntegration} Configured NetFlow integration instance
 */
function createNetflowIntegration(config = {}) {
    return new NetflowIntegration(config);
}

/**
 * Default configuration for NetFlow integration
 */
const defaultConfig = {
    collectorURL: process.env.NETFLOW_COLLECTOR_URL || 'http://localhost:9995',
    collectorPort: process.env.NETFLOW_COLLECTOR_PORT || 2055,
    apiKey: process.env.NETFLOW_API_KEY,
    timeout: 30000,
    retries: 3,
    webhookEnabled: true,
    samplingRate: 1000,
    aggregationInterval: 300 // 5 minutes
};

/**
 * Quick setup function for common use cases
 * @param {Object} options - Setup options
 * @returns {Promise<NetflowIntegration>} Configured and authenticated integration
 */
async function quickSetup(options = {}) {
    const config = { ...defaultConfig, ...options };
    const netflow = new NetflowIntegration(config);
    
    try {
        // Test connection
        const connectionResult = await netflow.testConnection();
        if (connectionResult.success) {
            console.log('✅ NetFlow integration setup successfully');
        } else {
            console.warn('⚠️ NetFlow connection test failed:', connectionResult.error);
        }
        
        return netflow;
    } catch (error) {
        console.error('❌ NetFlow quick setup failed:', error.message);
        throw error;
    }
}

/**
 * Utility functions for common NetFlow operations
 */
const utils = {
    /**
     * Format flow record for display
     * @param {Object} flow - Flow record from NetFlow
     * @returns {Object} Formatted flow record
     */
    formatFlowRecord(flow) {
        return {
            timestamp: new Date(flow.timestamp || Date.now()),
            sourceIP: flow.srcaddr || flow.source_ip,
            destinationIP: flow.dstaddr || flow.dest_ip,
            sourcePort: flow.srcport || flow.source_port,
            destinationPort: flow.dstport || flow.dest_port,
            protocol: utils.getProtocolName(flow.protocol || flow.prot),
            packets: flow.packets || flow.packet_count || 0,
            bytes: flow.bytes || flow.byte_count || 0,
            duration: flow.duration || 0,
            tcpFlags: flow.tcp_flags || 0,
            tos: flow.tos || 0,
            inputInterface: flow.input_snmp || flow.input_interface,
            outputInterface: flow.output_snmp || flow.output_interface,
            nextHop: flow.nexthop || flow.next_hop,
            sourceAS: flow.src_as || flow.source_as,
            destinationAS: flow.dst_as || flow.dest_as,
            sourceMask: flow.src_mask || flow.source_mask,
            destinationMask: flow.dst_mask || flow.dest_mask,
            vlan: flow.vlan_id || flow.vlan,
            engineType: flow.engine_type,
            engineID: flow.engine_id,
            flowSeq: flow.flow_seq || flow.sequence,
            samplingRate: flow.sampling_rate || config.samplingRate
        };
    },

    /**
     * Get protocol name from number
     * @param {number} protocolNumber - Protocol number
     * @returns {string} Protocol name
     */
    getProtocolName(protocolNumber) {
        const protocols = {
            1: 'ICMP',
            6: 'TCP',
            17: 'UDP',
            47: 'GRE',
            50: 'ESP',
            51: 'AH',
            89: 'OSPF',
            103: 'PIM',
            115: 'L2TP'
        };
        return protocols[protocolNumber] || `Protocol-${protocolNumber}`;
    },

    /**
     * Format bytes to human readable
     * @param {number} bytes - Size in bytes
     * @param {number} decimals - Number of decimal places
     * @returns {string} Formatted size
     */
    formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    },

    /**
     * Format packets per second
     * @param {number} packets - Number of packets
     * @param {number} duration - Duration in seconds
     * @returns {string} Formatted PPS
     */
    formatPPS(packets, duration) {
        if (!duration || duration === 0) return '0 PPS';
        const pps = Math.round(packets / duration);
        if (pps >= 1000000) {
            return `${(pps / 1000000).toFixed(2)}M PPS`;
        } else if (pps >= 1000) {
            return `${(pps / 1000).toFixed(2)}K PPS`;
        }
        return `${pps} PPS`;
    },

    /**
     * Format bits per second
     * @param {number} bytes - Number of bytes
     * @param {number} duration - Duration in seconds
     * @returns {string} Formatted BPS
     */
    formatBPS(bytes, duration) {
        if (!duration || duration === 0) return '0 bps';
        const bits = bytes * 8;
        const bps = Math.round(bits / duration);
        
        if (bps >= 1000000000) {
            return `${(bps / 1000000000).toFixed(2)} Gbps`;
        } else if (bps >= 1000000) {
            return `${(bps / 1000000).toFixed(2)} Mbps`;
        } else if (bps >= 1000) {
            return `${(bps / 1000).toFixed(2)} Kbps`;
        }
        return `${bps} bps`;
    },

    /**
     * Classify flow by service/application
     * @param {Object} flow - Flow record
     * @returns {string} Service classification
     */
    classifyFlow(flow) {
        const srcPort = flow.sourcePort || 0;
        const dstPort = flow.destinationPort || 0;
        const protocol = flow.protocol || '';
        
        // Well-known ports classification
        const services = {
            80: 'HTTP',
            443: 'HTTPS',
            21: 'FTP',
            22: 'SSH',
            23: 'Telnet',
            25: 'SMTP',
            53: 'DNS',
            110: 'POP3',
            143: 'IMAP',
            993: 'IMAPS',
            995: 'POP3S',
            1883: 'MQTT',
            3389: 'RDP',
            5432: 'PostgreSQL',
            3306: 'MySQL',
            6379: 'Redis',
            8080: 'HTTP-Alt',
            8443: 'HTTPS-Alt',
            8096: 'Jellyfin',
            32400: 'Plex',
            9696: 'Prowlarr',
            8989: 'Sonarr',
            7878: 'Radarr'
        };
        
        // Check destination port first, then source port
        if (services[dstPort]) {
            return services[dstPort];
        } else if (services[srcPort]) {
            return services[srcPort];
        }
        
        // Protocol-based classification
        if (protocol === 'TCP') {
            if (dstPort >= 1024 && srcPort >= 1024) {
                return 'TCP-Dynamic';
            } else if (dstPort < 1024) {
                return `TCP-${dstPort}`;
            } else {
                return `TCP-${srcPort}`;
            }
        } else if (protocol === 'UDP') {
            if (dstPort >= 1024 && srcPort >= 1024) {
                return 'UDP-Dynamic';
            } else if (dstPort < 1024) {
                return `UDP-${dstPort}`;
            } else {
                return `UDP-${srcPort}`;
            }
        }
        
        return protocol || 'Unknown';
    },

    /**
     * Detect potential security issues
     * @param {Object} flow - Flow record
     * @returns {Array} Array of security alerts
     */
    detectSecurityIssues(flow) {
        const alerts = [];
        const srcPort = flow.sourcePort || 0;
        const dstPort = flow.destinationPort || 0;
        const bytes = flow.bytes || 0;
        const packets = flow.packets || 0;
        const duration = flow.duration || 0;
        
        // Port scanning detection
        if (packets < 3 && duration < 1) {
            alerts.push({
                type: 'port_scan',
                severity: 'medium',
                description: 'Potential port scanning activity detected'
            });
        }
        
        // Large data transfer detection
        if (bytes > 100 * 1024 * 1024) { // > 100MB
            alerts.push({
                type: 'large_transfer',
                severity: 'low',
                description: `Large data transfer detected: ${utils.formatBytes(bytes)}`
            });
        }
        
        // Suspicious ports
        const suspiciousPorts = [1337, 31337, 4444, 5555, 6666, 12345, 54321];
        if (suspiciousPorts.includes(srcPort) || suspiciousPorts.includes(dstPort)) {
            alerts.push({
                type: 'suspicious_port',
                severity: 'high',
                description: `Traffic on suspicious port: ${srcPort} -> ${dstPort}`
            });
        }
        
        // High packet rate (potential DDoS)
        if (duration > 0 && (packets / duration) > 1000) {
            alerts.push({
                type: 'high_packet_rate',
                severity: 'high',
                description: `High packet rate detected: ${utils.formatPPS(packets, duration)}`
            });
        }
        
        return alerts;
    },

    /**
     * Parse NetFlow webhook payload
     * @param {Object} payload - Webhook payload
     * @returns {Object} Parsed webhook data
     */
    parseWebhookPayload(payload) {
        return {
            event: payload.event_type || payload.type,
            timestamp: new Date(payload.timestamp || Date.now()),
            collector: {
                id: payload.collector_id,
                name: payload.collector_name,
                version: payload.collector_version
            },
            alert: payload.alert ? {
                id: payload.alert.id,
                type: payload.alert.type,
                severity: payload.alert.severity,
                description: payload.alert.description,
                threshold: payload.alert.threshold,
                value: payload.alert.value
            } : null,
            flow: payload.flow ? utils.formatFlowRecord(payload.flow) : null,
            statistics: payload.statistics ? {
                totalFlows: payload.statistics.total_flows,
                totalBytes: payload.statistics.total_bytes,
                totalPackets: payload.statistics.total_packets,
                uniqueHosts: payload.statistics.unique_hosts,
                topProtocols: payload.statistics.top_protocols
            } : null
        };
    }
};

/**
 * Health check function
 * @param {Object} config - NetFlow configuration
 * @returns {Promise<Object>} Health check result
 */
async function healthCheck(config = {}) {
    try {
        const netflow = createNetflowIntegration(config);
        const result = await netflow.testConnection();
        
        return {
            service: 'netflow',
            healthy: result.success,
            timestamp: new Date(),
            response_time: result.responseTime,
            collector_status: result.collectorStatus,
            error: result.success ? null : result.error
        };
    } catch (error) {
        return {
            service: 'netflow',
            healthy: false,
            timestamp: new Date(),
            error: error.message
        };
    }
}

/**
 * Network monitoring and analysis utilities
 */
const monitor = {
    /**
     * Get network statistics
     * @param {NetflowIntegration} netflow - NetFlow integration instance
     * @param {Object} options - Statistics options
     * @returns {Promise<Object>} Network statistics
     */
    async getNetworkStats(netflow, options = {}) {
        try {
            const timeRange = options.timeRange || '1h';
            const stats = await netflow.getFlowStatistics({ time_range: timeRange });
            
            const totalFlows = stats.total_flows || 0;
            const totalBytes = stats.total_bytes || 0;
            const totalPackets = stats.total_packets || 0;
            const uniqueSrcIPs = stats.unique_source_ips || 0;
            const uniqueDstIPs = stats.unique_dest_ips || 0;
            
            const topProtocols = (stats.protocol_breakdown || []).map(p => ({
                protocol: utils.getProtocolName(p.protocol),
                flows: p.flow_count,
                bytes: utils.formatBytes(p.byte_count),
                packets: p.packet_count,
                percentage: totalFlows > 0 ? Math.round((p.flow_count / totalFlows) * 100) : 0
            }));
            
            const topServices = (stats.service_breakdown || []).map(s => ({
                service: s.service_name,
                flows: s.flow_count,
                bytes: utils.formatBytes(s.byte_count),
                packets: s.packet_count,
                percentage: totalFlows > 0 ? Math.round((s.flow_count / totalFlows) * 100) : 0
            }));
            
            return {
                timeRange,
                summary: {
                    totalFlows,
                    totalBytes: utils.formatBytes(totalBytes),
                    totalPackets,
                    uniqueSourceIPs: uniqueSrcIPs,
                    uniqueDestIPs: uniqueDstIPs,
                    avgFlowSize: totalFlows > 0 ? utils.formatBytes(Math.round(totalBytes / totalFlows)) : '0 Bytes',
                    avgPacketsPerFlow: totalFlows > 0 ? Math.round(totalPackets / totalFlows) : 0
                },
                breakdowns: {
                    protocols: topProtocols,
                    services: topServices
                }
            };
        } catch (error) {
            return {
                error: error.message,
                timeRange: options.timeRange || '1h'
            };
        }
    },

    /**
     * Get top talkers (most active hosts)
     * @param {NetflowIntegration} netflow - NetFlow integration instance
     * @param {Object} options - Query options
     * @returns {Promise<Object>} Top talkers data
     */
    async getTopTalkers(netflow, options = {}) {
        try {
            const limit = options.limit || 10;
            const timeRange = options.timeRange || '1h';
            const metric = options.metric || 'bytes'; // bytes, packets, flows
            
            const data = await netflow.getTopTalkers({ 
                limit, 
                time_range: timeRange, 
                metric 
            });
            
            const topTalkers = (data.top_talkers || []).map(host => ({
                ipAddress: host.ip_address,
                hostname: host.hostname || host.ip_address,
                flows: host.flow_count || 0,
                bytes: utils.formatBytes(host.byte_count || 0),
                packets: host.packet_count || 0,
                bytesRaw: host.byte_count || 0,
                percentage: host.percentage || 0,
                firstSeen: host.first_seen ? new Date(host.first_seen) : null,
                lastSeen: host.last_seen ? new Date(host.last_seen) : null,
                portCount: host.unique_ports || 0,
                protocols: host.protocols || []
            }));
            
            return {
                timeRange,
                metric,
                totalHosts: data.total_hosts || 0,
                topTalkers
            };
        } catch (error) {
            return {
                error: error.message,
                timeRange: options.timeRange || '1h',
                topTalkers: []
            };
        }
    },

    /**
     * Monitor bandwidth usage
     * @param {NetflowIntegration} netflow - NetFlow integration instance
     * @param {Object} options - Monitoring options
     * @returns {Promise<Object>} Bandwidth usage data
     */
    async getBandwidthUsage(netflow, options = {}) {
        try {
            const timeRange = options.timeRange || '24h';
            const interval = options.interval || '1h';
            
            const data = await netflow.getBandwidthUsage({ 
                time_range: timeRange, 
                interval 
            });
            
            const timeline = (data.timeline || []).map(point => ({
                timestamp: new Date(point.timestamp),
                bytesIn: point.bytes_in || 0,
                bytesOut: point.bytes_out || 0,
                packetsIn: point.packets_in || 0,
                packetsOut: point.packets_out || 0,
                totalBytes: (point.bytes_in || 0) + (point.bytes_out || 0),
                totalPackets: (point.packets_in || 0) + (point.packets_out || 0),
                formattedBytesIn: utils.formatBytes(point.bytes_in || 0),
                formattedBytesOut: utils.formatBytes(point.bytes_out || 0),
                formattedTotal: utils.formatBytes((point.bytes_in || 0) + (point.bytes_out || 0))
            }));
            
            const totalBytesIn = timeline.reduce((sum, point) => sum + point.bytesIn, 0);
            const totalBytesOut = timeline.reduce((sum, point) => sum + point.bytesOut, 0);
            const totalBytes = totalBytesIn + totalBytesOut;
            
            const peakUsage = timeline.reduce((max, point) => 
                point.totalBytes > max.totalBytes ? point : max, 
                { totalBytes: 0 }
            );
            
            return {
                timeRange,
                interval,
                summary: {
                    totalBytesIn: utils.formatBytes(totalBytesIn),
                    totalBytesOut: utils.formatBytes(totalBytesOut),
                    totalBytes: utils.formatBytes(totalBytes),
                    peakUsage: {
                        timestamp: peakUsage.timestamp,
                        bytes: utils.formatBytes(peakUsage.totalBytes)
                    },
                    avgUsage: utils.formatBytes(Math.round(totalBytes / timeline.length))
                },
                timeline
            };
        } catch (error) {
            return {
                error: error.message,
                timeRange: options.timeRange || '24h',
                timeline: []
            };
        }
    },

    /**
     * Detect network anomalies
     * @param {NetflowIntegration} netflow - NetFlow integration instance
     * @param {Object} options - Detection options
     * @returns {Promise<Object>} Anomaly detection results
     */
    async detectAnomalies(netflow, options = {}) {
        try {
            const timeRange = options.timeRange || '1h';
            const sensitivity = options.sensitivity || 'medium';
            
            const flows = await netflow.getRecentFlows({ 
                time_range: timeRange, 
                limit: options.limit || 1000 
            });
            
            const anomalies = [];
            const securityAlerts = [];
            
            // Analyze each flow for anomalies
            for (const flow of flows.flows || []) {
                const formattedFlow = utils.formatFlowRecord(flow);
                const alerts = utils.detectSecurityIssues(formattedFlow);
                
                if (alerts.length > 0) {
                    securityAlerts.push({
                        flow: formattedFlow,
                        alerts
                    });
                }
                
                // Additional anomaly detection logic can be added here
                // (statistical analysis, machine learning models, etc.)
            }
            
            // Aggregate statistics
            const totalFlows = flows.flows?.length || 0;
            const uniqueIPs = new Set();
            const protocolStats = {};
            let totalBytes = 0;
            
            for (const flow of flows.flows || []) {
                uniqueIPs.add(flow.sourceIP);
                uniqueIPs.add(flow.destinationIP);
                
                const protocol = utils.getProtocolName(flow.protocol);
                protocolStats[protocol] = (protocolStats[protocol] || 0) + 1;
                
                totalBytes += flow.bytes || 0;
            }
            
            return {
                timeRange,
                sensitivity,
                summary: {
                    totalFlows,
                    uniqueIPs: uniqueIPs.size,
                    totalBytes: utils.formatBytes(totalBytes),
                    securityAlerts: securityAlerts.length,
                    anomaliesDetected: anomalies.length
                },
                securityAlerts,
                anomalies,
                protocolDistribution: Object.entries(protocolStats).map(([protocol, count]) => ({
                    protocol,
                    count,
                    percentage: Math.round((count / totalFlows) * 100)
                })).sort((a, b) => b.count - a.count)
            };
        } catch (error) {
            return {
                error: error.message,
                timeRange: options.timeRange || '1h',
                securityAlerts: [],
                anomalies: []
            };
        }
    }
};

module.exports = {
    NetflowIntegration,
    createNetflowIntegration,
    quickSetup,
    defaultConfig,
    utils,
    healthCheck,
    monitor,
    
    // Aliases for convenience
    create: createNetflowIntegration,
    setup: quickSetup,
    Integration: NetflowIntegration
};