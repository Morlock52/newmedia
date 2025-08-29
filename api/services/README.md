const logger = require('../../middleware/logger.js');
# Media Server Backend Services

This directory contains 8 comprehensive backend API services for the media server project, each designed to handle specific aspects of the media server infrastructure.

## 🚀 Services Overview

### 1. Web3Service.js
**Web3 wallet connection, NFT media collections, IPFS streaming, smart contracts**
- Wallet connection and management (MetaMask, WalletConnect)
- NFT media collection tracking
- IPFS file upload and streaming
- Smart contract interactions
- Blockchain integration for media ownership

**Key Features:**
- Multiple blockchain support (Ethereum, Polygon, etc.)
- IPFS gateway with Pinata integration
- NFT metadata generation
- Decentralized media streaming

### 2. SmartHomeService.js
**HomeKit, Google Home, Alexa integration, Philips Hue sync**
- Smart home device discovery and control
- Ambient lighting sync with media playback
- Voice assistant integration
- HomeKit accessory management

**Key Features:**
- Philips Hue bridge integration
- Media-reactive lighting
- Scene management
- Cross-platform smart home support

### 3. SecurityService.js
**Zero-trust architecture, OAuth2/OIDC with 2FA, intrusion detection**
- JWT token management and validation
- Two-factor authentication (TOTP)
- Session management and security
- Rate limiting and intrusion detection

**Key Features:**
- Multi-factor authentication
- Role-based access control (RBAC)
- Advanced security monitoring
- Zero-trust architecture principles

### 4. VPNService.js
**Gluetun VPN tunnel management for download protection**
- VPN connection monitoring and control
- Automatic reconnection handling
- Kill switch management
- Speed testing and leak detection

**Key Features:**
- Multiple VPN provider support
- Real-time connection monitoring
- Automatic failover
- DNS leak protection

### 5. MonitoringService.js
**Prometheus metrics, Grafana dashboards, Uptime Kuma**
- System and application metrics collection
- Health monitoring for all services
- Alert management and notifications
- Performance analytics

**Key Features:**
- Prometheus metrics integration
- Grafana dashboard automation
- Comprehensive health checks
- Real-time alerting

### 6. TranscodingService.js
**FileFlows GPU transcoding with hardware acceleration**
- Video/audio transcoding and optimization
- GPU-accelerated processing
- Queue management
- Multiple output formats

**Key Features:**
- Hardware acceleration (NVIDIA, AMD, Intel)
- FileFlows integration
- Batch processing
- Quality presets

### 7. AutheliaService.js
**ForwardAuth with Traefik, TOTP and LDAP support**
- Centralized authentication for all services
- Forward authentication for Traefik
- LDAP integration
- Access policy management

**Key Features:**
- Single sign-on (SSO)
- Policy-based access control
- Multi-factor authentication
- LDAP/Active Directory support

### 8. IndexerService.js
**Prowlarr master indexer with 500+ trackers**
- Comprehensive indexer management
- Multi-source search capabilities
- Sync with *arr applications
- Performance monitoring

**Key Features:**
- 500+ tracker support
- Intelligent result ranking
- Auto-configuration
- Real-time sync

## 🔧 Installation & Setup

### Prerequisites
```bash
# Install Node.js dependencies
npm install

# Additional dependencies for crypto/security features
npm install jsonwebtoken speakeasy crypto form-data multer
```

### Environment Configuration
Create a `.env` file with the following variables:

```env
# Web3 Service
WEB3_PROVIDER_URL=https://mainnet.infura.io/v3/YOUR_KEY
INFURA_API_KEY=your_infura_key
IPFS_GATEWAY=https://gateway.pinata.cloud/ipfs/
PINATA_API_KEY=your_pinata_key
PINATA_SECRET_KEY=your_pinata_secret

# Smart Home Service
HUE_BRIDGE_IP=192.168.1.100
HUE_USERNAME=your_hue_username
HOMEKIT_PIN=123-45-678

# Security Service
JWT_SECRET=your_super_secret_jwt_key
JWT_EXPIRY=24h

# VPN Service
GLUETUN_URL=http://gluetun:8000
VPN_PROVIDER=nordvpn
VPN_REGION=Switzerland

# Monitoring Service
PROMETHEUS_URL=http://prometheus:9090
GRAFANA_URL=http://grafana:3000
GRAFANA_API_KEY=your_grafana_key
UPTIME_KUMA_URL=http://uptime-kuma:3001

# Transcoding Service
FILEFLOWS_URL=http://fileflows:5000
FILEFLOWS_API_KEY=your_fileflows_key

# Authelia Service
AUTHELIA_URL=http://authelia:9091
AUTHELIA_SECRET=your_authelia_secret
SESSION_DOMAIN=.mediaserver.local

# Indexer Service
PROWLARR_URL=http://prowlarr:9696
PROWLARR_API_KEY=your_prowlarr_key
```

## 📖 Usage Examples

### Basic Service Initialization
```javascript
const Web3Service = require('./services/Web3Service');
const SmartHomeService = require('./services/SmartHomeService');
const SecurityService = require('./services/SecurityService');

// Initialize services
const web3Service = new Web3Service();
const smartHomeService = new SmartHomeService();
const securityService = new SecurityService();

// Initialize all services
async function initializeServices() {
  try {
    await Promise.all([
      web3Service.initialize(),
      smartHomeService.initialize(),
      securityService.initialize()
    ]);
    logger.info('✅ All services initialized successfully');
  } catch (error) {
    logger.error('❌ Service initialization failed:', error);
  }
}
```

### Web3 Integration Example
```javascript
// Connect wallet and upload media to IPFS
async function uploadMediaToWeb3(mediaBuffer, metadata) {
  try {
    // Connect user wallet
    const walletResult = await web3Service.connectWallet(
      'metamask',
      '0x742d35Cc6634C0532925a3b8D404fAbCb4614068',
      'signature'
    );

    // Upload to IPFS
    const ipfsResult = await web3Service.uploadToIPFS(mediaBuffer, {
      filename: metadata.title,
      contentType: 'video/mp4',
      name: metadata.title,
      description: metadata.description
    });

    // Create NFT metadata
    const nftMetadata = await web3Service.createNFTMetadata({
      title: metadata.title,
      description: metadata.description,
      imageUrl: ipfsResult.url,
      type: 'Video',
      duration: metadata.duration
    });

    return {
      wallet: walletResult.wallet,
      ipfsUrl: ipfsResult.url,
      metadataUrl: nftMetadata.metadataUri
    };
  } catch (error) {
    logger.error('Web3 upload failed:', error);
    throw error;
  }
}
```

### Smart Home Media Sync
```javascript
// Sync lighting with media playback
async function startMediaWithLighting(mediaInfo) {
  try {
    // Start media sync
    const syncResult = await smartHomeService.startMediaSync({
      title: mediaInfo.title,
      type: 'movie',
      poster: mediaInfo.poster
    });

    // Apply movie scene
    await smartHomeService.applyScene('movie_night');

    logger.info('🎬 Media playback started with ambient lighting');
    return syncResult;
  } catch (error) {
    logger.error('Smart home sync failed:', error);
  }
}
```

### Security & Authentication
```javascript
// Authenticate user with 2FA
async function authenticateUser(username, password, totpCode) {
  try {
    // First factor authentication
    const firstAuth = await securityService.authenticateFirstFactor(
      username,
      password,
      '192.168.1.100'
    );

    if (firstAuth.requiresSecondFactor) {
      // Second factor authentication
      const secondAuth = await securityService.authenticateSecondFactor(
        firstAuth.sessionId,
        totpCode
      );
      
      return {
        success: true,
        user: firstAuth.user,
        sessionId: secondAuth.sessionId,
        authLevel: secondAuth.authenticationLevel
      };
    }

    return firstAuth;
  } catch (error) {
    logger.error('Authentication failed:', error);
    throw error;
  }
}
```

### VPN Management
```javascript
// Manage VPN connection for downloads
async function ensureVPNConnection() {
  try {
    const status = await vpnService.updateVPNStatus();
    
    if (!status.connected) {
      logger.info('🔒 Connecting to VPN...');
      await vpnService.connect({
        provider: 'nordvpn',
        region: 'Switzerland'
      });
    }

    // Test connection speed
    const speedTest = await vpnService.testSpeed();
    logger.info(`📶 VPN Speed: ${speedTest.speedTest.downloadSpeed} Mbps`);

    return status;
  } catch (error) {
    logger.error('VPN management failed:', error);
    throw error;
  }
}
```

### Monitoring & Alerts
```javascript
// Set up comprehensive monitoring
async function setupMonitoring() {
  try {
    // Create custom alert
    await monitoringService.createAlert({
      type: 'SERVICE',
      severity: 'HIGH',
      title: 'High CPU Usage',
      description: 'CPU usage exceeded 80%',
      value: 85,
      threshold: 80
    });

    // Get system metrics
    const metrics = monitoringService.getMetricsSummary();
    logger.info('📊 System Status:', metrics);

    return metrics;
  } catch (error) {
    logger.error('Monitoring setup failed:', error);
  }
}
```

### Transcoding Jobs
```javascript
// Start video transcoding
async function transcodeVideo(inputFile, profile = 'web_optimized') {
  try {
    const job = await transcodingService.startTranscoding(inputFile, {
      profile,
      outputFile: `/media/transcoded/${Date.now()}_output.mp4`
    });

    // Monitor progress
    transcodingService.on('jobProgress', ({ job, progress }) => {
      logger.info(`🎞️ Transcoding progress: ${progress}%`);
    });

    return job;
  } catch (error) {
    logger.error('Transcoding failed:', error);
    throw error;
  }
}
```

### Search & Indexing
```javascript
// Search across all indexers
async function searchMedia(query, type = 'movie') {
  try {
    const searchResult = await indexerService.search(query, {
      type: type === 'movie' ? 'movie' : 'tvsearch',
      categories: type === 'movie' ? [2000] : [5000], // Movies or TV
      limit: 50,
      minSeeders: 5
    });

    logger.info(`🔍 Found ${searchResult.totalResults} results`);
    return searchResult.results;
  } catch (error) {
    logger.error('Search failed:', error);
    throw error;
  }
}
```

## 🔗 Service Integration

### Express.js Integration
```javascript
const express = require('express');
const app = express();

// Initialize all services
const services = {
  web3: new Web3Service(),
  smartHome: new SmartHomeService(),
  security: new SecurityService(),
  vpn: new VPNService(),
  monitoring: new MonitoringService(),
  transcoding: new TranscodingService(),
  authelia: new AutheliaService(),
  indexer: new IndexerService()
};

// API Routes
app.get('/api/services/status', (req, res) => {
  const status = {};
  Object.entries(services).forEach(([name, service]) => {
    status[name] = service.getStatus();
  });
  res.json(status);
});

// Web3 routes
app.post('/api/web3/connect-wallet', async (req, res) => {
  try {
    const result = await services.web3.connectWallet(
      req.body.type,
      req.body.address,
      req.body.signature
    );
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Smart home routes
app.post('/api/smart-home/sync-media', async (req, res) => {
  try {
    const result = await services.smartHome.startMediaSync(req.body.mediaInfo);
    res.json(result);
  } catch (error) {
    res.status(400).json({ error: error.message });
  }
});

// Security middleware
app.use('/api/protected/*', async (req, res, next) => {
  try {
    const authResult = await services.security.verifyToken(
      req.headers.authorization?.replace('Bearer ', '')
    );
    req.user = authResult.user;
    next();
  } catch (error) {
    res.status(401).json({ error: 'Unauthorized' });
  }
});
```

## 🚀 Production Deployment

### Docker Integration
Each service is designed to work seamlessly with Docker containers. Make sure your `docker-compose.yml` includes all the required services:

```yaml
services:
  # Core services
  api-server:
    build: ./api
    environment:
      - NODE_ENV=production
    depends_on:
      - prometheus
      - grafana
      - authelia
      - prowlarr
      - gluetun

  # Supporting services
  prometheus:
    image: prom/prometheus:latest
  
  grafana:
    image: grafana/grafana:latest
    
  authelia:
    image: authelia/authelia:latest
    
  prowlarr:
    image: lscr.io/linuxserver/prowlarr:latest
    
  gluetun:
    image: qmcgaw/gluetun:latest
```

## 📊 Monitoring & Metrics

All services include comprehensive monitoring and emit events for:
- Performance metrics
- Health status
- Error tracking
- Usage statistics
- Security events

## 🔒 Security Considerations

- All services implement proper error handling
- Sensitive data is encrypted at rest
- Rate limiting prevents abuse
- Comprehensive logging for audit trails
- Zero-trust architecture principles

## 🤝 Contributing

When extending these services:
1. Follow the existing event-driven architecture
2. Implement proper error handling and logging
3. Include comprehensive status reporting
4. Maintain Docker container compatibility
5. Add appropriate tests and documentation

## 📝 License

This project is part of the media server infrastructure and follows the same licensing terms as the main project.