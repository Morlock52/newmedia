/**
 * Netflow Integration Module
 * Network flow analysis for media streaming monitoring and optimization
 */

const axios = require('axios');
const EventEmitter = require('events');
const dgram = require('dgram');
const net = require('net');

class NetflowIntegration extends EventEmitter {
    constructor(config = {}) {
        super();
        this.baseURL = config.baseURL || process.env.NETFLOW_COLLECTOR_URL;
        this.apiKey = config.apiKey || process.env.NETFLOW_API_KEY;
        this.collectorPort = config.collectorPort || process.env.NETFLOW_COLLECTOR_PORT || 2055;
        this.analysisEnabled = config.analysisEnabled !== false;
        this.streamingPorts = config.streamingPorts || [32400, 8096, 8080, 9091, 8989, 7878, 9696]; // Common media server ports
        
        // Flow data storage
        this.flows = new Map();
        this.statistics = {
            totalFlows: 0,
            totalBytes: 0,
            totalPackets: 0,
            streamingSessions: new Map(),
            bandwidthUsage: new Map(),
            topTalkers: new Map(),
            protocols: new Map(),
            lastUpdate: new Date()
        };
        
        // Initialize collector if needed
        if (this.analysisEnabled) {
            this.initializeCollector();
        }
        
        // HTTP client for external netflow APIs
        if (this.baseURL) {
            this.client = axios.create({
                baseURL: this.baseURL,
                timeout: 30000,
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (this.apiKey) {
                this.client.defaults.headers['Authorization'] = `Bearer ${this.apiKey}`;
            }
        }

        // Setup periodic analysis
        this.analysisInterval = setInterval(() => {
            this.analyzeFlows();
        }, 30000); // Analyze every 30 seconds
    }

    /**
     * Initialize NetFlow collector
     */
    initializeCollector() {
        this.collector = dgram.createSocket('udp4');
        
        this.collector.on('message', (msg, rinfo) => {
            try {
                this.processNetflowPacket(msg, rinfo);
            } catch (error) {
                this.emit('error', `NetFlow processing error: ${error.message}`);
            }
        });
        
        this.collector.on('error', (err) => {
            this.emit('error', `NetFlow collector error: ${err.message}`);
        });
        
        this.collector.bind(this.collectorPort, () => {
            console.log(`NetFlow collector listening on port ${this.collectorPort}`);
            this.emit('collectorStarted', this.collectorPort);
        });
    }

    /**
     * Process NetFlow packet (simplified NetFlow v5 parser)
     */
    processNetflowPacket(buffer, rinfo) {
        try {
            // NetFlow v5 header parsing
            const header = {
                version: buffer.readUInt16BE(0),
                count: buffer.readUInt16BE(2),
                sysUptime: buffer.readUInt32BE(4),
                unixSecs: buffer.readUInt32BE(8),
                unixNsecs: buffer.readUInt32BE(12),
                flowSequence: buffer.readUInt32BE(16),
                engineType: buffer.readUInt8(20),
                engineId: buffer.readUInt8(21),
                samplingInterval: buffer.readUInt16BE(22)
            };
            
            if (header.version !== 5) {
                // Only support NetFlow v5 for now
                return;
            }
            
            // Parse flow records
            let offset = 24; // Header size
            for (let i = 0; i < header.count; i++) {
                const flow = this.parseFlowRecord(buffer, offset);
                if (flow) {
                    this.processFlow(flow, rinfo);
                }
                offset += 48; // Flow record size for v5
            }
            
        } catch (error) {
            this.emit('error', `NetFlow packet parsing error: ${error.message}`);
        }
    }

    /**
     * Parse individual flow record
     */
    parseFlowRecord(buffer, offset) {
        try {
            return {
                srcAddr: this.readIPAddress(buffer, offset),
                dstAddr: this.readIPAddress(buffer, offset + 4),
                nextHop: this.readIPAddress(buffer, offset + 8),
                input: buffer.readUInt16BE(offset + 12),
                output: buffer.readUInt16BE(offset + 14),
                packets: buffer.readUInt32BE(offset + 16),
                octets: buffer.readUInt32BE(offset + 20),
                first: buffer.readUInt32BE(offset + 24),
                last: buffer.readUInt32BE(offset + 28),
                srcPort: buffer.readUInt16BE(offset + 32),
                dstPort: buffer.readUInt16BE(offset + 34),
                tcpFlags: buffer.readUInt8(offset + 37),
                protocol: buffer.readUInt8(offset + 38),
                tos: buffer.readUInt8(offset + 39),
                srcAs: buffer.readUInt16BE(offset + 40),
                dstAs: buffer.readUInt16BE(offset + 42),
                srcMask: buffer.readUInt8(offset + 44),
                dstMask: buffer.readUInt8(offset + 45),
                timestamp: new Date()
            };
        } catch (error) {
            this.emit('error', `Flow record parsing error: ${error.message}`);
            return null;
        }
    }

    /**
     * Read IP address from buffer
     */
    readIPAddress(buffer, offset) {
        return [
            buffer.readUInt8(offset),
            buffer.readUInt8(offset + 1),
            buffer.readUInt8(offset + 2),
            buffer.readUInt8(offset + 3)
        ].join('.');
    }

    /**
     * Process individual flow
     */
    processFlow(flow, sourceInfo) {
        const flowKey = `${flow.srcAddr}:${flow.srcPort}-${flow.dstAddr}:${flow.dstPort}`;
        
        // Store flow
        this.flows.set(flowKey, {
            ...flow,
            sourceRouter: sourceInfo.address,
            lastSeen: new Date()
        });
        
        // Update statistics
        this.updateStatistics(flow);
        
        // Check if this is media streaming traffic
        if (this.isMediaStreamingFlow(flow)) {
            this.processMediaFlow(flow, flowKey);
        }
        
        this.emit('flowReceived', flow);
    }

    /**
     * Check if flow is media streaming traffic
     */
    isMediaStreamingFlow(flow) {
        return this.streamingPorts.includes(flow.srcPort) || 
               this.streamingPorts.includes(flow.dstPort) ||
               this.isStreamingProtocol(flow);
    }

    /**
     * Check if flow uses streaming protocols
     */
    isStreamingProtocol(flow) {
        // Check for common streaming patterns
        const isHTTP = flow.srcPort === 80 || flow.dstPort === 80 || 
                      flow.srcPort === 443 || flow.dstPort === 443;
        const isLargeTransfer = flow.octets > 1024 * 1024; // > 1MB
        const isVideoPort = (flow.srcPort >= 32400 && flow.srcPort <= 32500) ||
                           (flow.dstPort >= 32400 && flow.dstPort <= 32500);
        
        return (isHTTP && isLargeTransfer) || isVideoPort;
    }

    /**
     * Process media streaming flow
     */
    processMediaFlow(flow, flowKey) {
        const sessionKey = `${flow.srcAddr}-${flow.dstAddr}`;
        
        if (!this.statistics.streamingSessions.has(sessionKey)) {
            this.statistics.streamingSessions.set(sessionKey, {
                srcAddr: flow.srcAddr,
                dstAddr: flow.dstAddr,
                startTime: flow.timestamp,
                totalBytes: 0,
                totalPackets: 0,
                avgBandwidth: 0,
                peakBandwidth: 0,
                quality: 'unknown',
                protocol: this.getProtocolName(flow.protocol),
                ports: new Set()
            });
        }
        
        const session = this.statistics.streamingSessions.get(sessionKey);
        session.totalBytes += flow.octets;
        session.totalPackets += flow.packets;
        session.lastSeen = flow.timestamp;
        session.ports.add(`${flow.srcPort}:${flow.dstPort}`);
        
        // Calculate bandwidth
        const duration = (flow.last - flow.first) / 1000; // Convert to seconds
        if (duration > 0) {
            const bandwidth = (flow.octets * 8) / duration; // bits per second
            session.avgBandwidth = (session.avgBandwidth + bandwidth) / 2;
            if (bandwidth > session.peakBandwidth) {
                session.peakBandwidth = bandwidth;
            }
        }
        
        // Estimate quality based on bandwidth
        session.quality = this.estimateStreamQuality(session.avgBandwidth);
        
        this.emit('mediaFlowDetected', {
            sessionKey,
            flow,
            session: session
        });
    }

    /**
     * Estimate stream quality based on bandwidth
     */
    estimateStreamQuality(bandwidth) {
        const mbps = bandwidth / (1024 * 1024);
        
        if (mbps < 1) return 'audio';
        if (mbps < 3) return 'SD';
        if (mbps < 8) return 'HD';
        if (mbps < 25) return '4K';
        return 'ultra';
    }

    /**
     * Update global statistics
     */
    updateStatistics(flow) {
        this.statistics.totalFlows++;
        this.statistics.totalBytes += flow.octets;
        this.statistics.totalPackets += flow.packets;
        this.statistics.lastUpdate = new Date();
        
        // Update protocol stats
        const protocol = this.getProtocolName(flow.protocol);
        this.statistics.protocols.set(protocol, 
            (this.statistics.protocols.get(protocol) || 0) + 1);
        
        // Update top talkers
        const talker = flow.srcAddr;
        this.statistics.topTalkers.set(talker,
            (this.statistics.topTalkers.get(talker) || 0) + flow.octets);
        
        // Update bandwidth usage by IP
        this.statistics.bandwidthUsage.set(talker,
            (this.statistics.bandwidthUsage.get(talker) || 0) + flow.octets);
    }

    /**
     * Get protocol name from number
     */
    getProtocolName(protocolNumber) {
        const protocols = {
            1: 'ICMP',
            6: 'TCP',
            17: 'UDP',
            47: 'GRE',
            50: 'ESP',
            51: 'AH'
        };
        return protocols[protocolNumber] || `Protocol-${protocolNumber}`;
    }

    /**
     * Analyze flows for insights
     */
    analyzeFlows() {
        try {
            const analysis = {
                timestamp: new Date(),
                totalActiveSessions: this.statistics.streamingSessions.size,
                totalBandwidthUsage: this.getTotalBandwidth(),
                topStreamingSessions: this.getTopStreamingSessions(),
                qualityDistribution: this.getQualityDistribution(),
                protocolDistribution: this.getProtocolDistribution(),
                networkHealth: this.assessNetworkHealth(),
                alerts: this.generateAlerts()
            };
            
            this.emit('analysisComplete', analysis);
            return analysis;
        } catch (error) {
            this.emit('error', `Flow analysis error: ${error.message}`);
        }
    }

    /**
     * Get total bandwidth usage
     */
    getTotalBandwidth() {
        let total = 0;
        for (const session of this.statistics.streamingSessions.values()) {
            total += session.avgBandwidth;
        }
        return total;
    }

    /**
     * Get top streaming sessions by bandwidth
     */
    getTopStreamingSessions(limit = 10) {
        return Array.from(this.statistics.streamingSessions.entries())
            .sort(([,a], [,b]) => b.avgBandwidth - a.avgBandwidth)
            .slice(0, limit)
            .map(([key, session]) => ({ sessionKey: key, ...session }));
    }

    /**
     * Get quality distribution
     */
    getQualityDistribution() {
        const distribution = {};
        for (const session of this.statistics.streamingSessions.values()) {
            distribution[session.quality] = (distribution[session.quality] || 0) + 1;
        }
        return distribution;
    }

    /**
     * Get protocol distribution
     */
    getProtocolDistribution() {
        return Object.fromEntries(this.statistics.protocols);
    }

    /**
     * Assess network health
     */
    assessNetworkHealth() {
        const totalBandwidth = this.getTotalBandwidth();
        const activeSessions = this.statistics.streamingSessions.size;
        const avgBandwidthPerSession = activeSessions > 0 ? totalBandwidth / activeSessions : 0;
        
        let health = 'good';
        const alerts = [];
        
        // Check for high bandwidth usage
        if (totalBandwidth > 100 * 1024 * 1024) { // 100 Mbps
            health = 'warning';
            alerts.push('High total bandwidth usage detected');
        }
        
        // Check for too many concurrent sessions
        if (activeSessions > 20) {
            health = 'warning';
            alerts.push('High number of concurrent streaming sessions');
        }
        
        // Check for low quality streams
        const qualityDist = this.getQualityDistribution();
        const lowQualityCount = (qualityDist.audio || 0) + (qualityDist.SD || 0);
        if (lowQualityCount > activeSessions * 0.5) {
            health = 'poor';
            alerts.push('Many streams are running at low quality');
        }
        
        return {
            status: health,
            totalBandwidth,
            activeSessions,
            avgBandwidthPerSession,
            alerts
        };
    }

    /**
     * Generate alerts based on flow analysis
     */
    generateAlerts() {
        const alerts = [];
        const now = new Date();
        
        // Check for stale sessions
        for (const [key, session] of this.statistics.streamingSessions.entries()) {
            if (now - session.lastSeen > 300000) { // 5 minutes
                alerts.push({
                    type: 'stale_session',
                    message: `Streaming session ${key} appears to be stale`,
                    session: key,
                    lastSeen: session.lastSeen
                });
            }
        }
        
        // Check for bandwidth anomalies
        for (const session of this.statistics.streamingSessions.values()) {
            if (session.peakBandwidth > session.avgBandwidth * 3) {
                alerts.push({
                    type: 'bandwidth_spike',
                    message: `Bandwidth spike detected for ${session.srcAddr} -> ${session.dstAddr}`,
                    avgBandwidth: session.avgBandwidth,
                    peakBandwidth: session.peakBandwidth
                });
            }
        }
        
        return alerts;
    }

    /**
     * Get real-time statistics
     */
    getStatistics() {
        return {
            ...this.statistics,
            streamingSessions: Array.from(this.statistics.streamingSessions.entries())
                .map(([key, session]) => ({ sessionKey: key, ...session })),
            protocols: Object.fromEntries(this.statistics.protocols),
            topTalkers: Array.from(this.statistics.topTalkers.entries())
                .sort(([,a], [,b]) => b - a)
                .slice(0, 10),
            bandwidthUsage: Object.fromEntries(this.statistics.bandwidthUsage)
        };
    }

    /**
     * Get flow history
     */
    getFlowHistory(limit = 100) {
        return Array.from(this.flows.values())
            .sort((a, b) => b.lastSeen - a.lastSeen)
            .slice(0, limit);
    }

    /**
     * Search flows by criteria
     */
    searchFlows(criteria) {
        const results = [];
        
        for (const flow of this.flows.values()) {
            let matches = true;
            
            if (criteria.srcAddr && flow.srcAddr !== criteria.srcAddr) matches = false;
            if (criteria.dstAddr && flow.dstAddr !== criteria.dstAddr) matches = false;
            if (criteria.srcPort && flow.srcPort !== criteria.srcPort) matches = false;
            if (criteria.dstPort && flow.dstPort !== criteria.dstPort) matches = false;
            if (criteria.protocol && flow.protocol !== criteria.protocol) matches = false;
            if (criteria.minBytes && flow.octets < criteria.minBytes) matches = false;
            if (criteria.maxBytes && flow.octets > criteria.maxBytes) matches = false;
            
            if (matches) {
                results.push(flow);
            }
        }
        
        return results;
    }

    /**
     * Export flow data
     */
    exportFlows(format = 'json') {
        const data = {
            timestamp: new Date(),
            statistics: this.getStatistics(),
            flows: this.getFlowHistory(1000)
        };
        
        switch (format.toLowerCase()) {
            case 'csv':
                return this.convertToCSV(data.flows);
            case 'json':
            default:
                return JSON.stringify(data, null, 2);
        }
    }

    /**
     * Convert flows to CSV format
     */
    convertToCSV(flows) {
        const headers = [
            'timestamp', 'srcAddr', 'dstAddr', 'srcPort', 'dstPort',
            'protocol', 'packets', 'octets', 'duration'
        ];
        
        let csv = headers.join(',') + '\n';
        
        for (const flow of flows) {
            const row = [
                flow.timestamp.toISOString(),
                flow.srcAddr,
                flow.dstAddr,
                flow.srcPort,
                flow.dstPort,
                this.getProtocolName(flow.protocol),
                flow.packets,
                flow.octets,
                (flow.last - flow.first) / 1000
            ];
            csv += row.join(',') + '\n';
        }
        
        return csv;
    }

    /**
     * Test connection (if using external API)
     */
    async testConnection() {
        if (!this.client) {
            return {
                success: this.analysisEnabled,
                message: this.analysisEnabled ? 'Local collector active' : 'NetFlow analysis disabled',
                collectorPort: this.collectorPort
            };
        }
        
        try {
            const response = await this.client.get('/health');
            return {
                success: true,
                response: response.data
            };
        } catch (error) {
            return {
                success: false,
                error: error.message
            };
        }
    }

    /**
     * Cleanup resources
     */
    cleanup() {
        if (this.analysisInterval) {
            clearInterval(this.analysisInterval);
        }
        
        if (this.collector) {
            this.collector.close();
        }
        
        this.flows.clear();
        this.statistics.streamingSessions.clear();
        this.statistics.bandwidthUsage.clear();
        this.statistics.topTalkers.clear();
        this.statistics.protocols.clear();
    }

    /**
     * Setup webhook endpoint for external flow data
     */
    setupWebhook(app, path = '/netflow/webhook') {
        app.post(path, (req, res) => {
            try {
                const flowData = req.body;
                
                if (Array.isArray(flowData)) {
                    flowData.forEach(flow => this.processFlow(flow, { address: 'webhook' }));
                } else {
                    this.processFlow(flowData, { address: 'webhook' });
                }
                
                this.emit('webhookFlowReceived', flowData);
                res.status(200).json({ success: true, processed: Array.isArray(flowData) ? flowData.length : 1 });
            } catch (error) {
                console.error('NetFlow webhook error:', error);
                res.status(500).json({ error: 'Flow processing failed' });
            }
        });
    }
}

module.exports = NetflowIntegration;