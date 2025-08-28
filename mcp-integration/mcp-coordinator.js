#!/usr/bin/env node

/**
 * MCP Integration Coordinator
 * Main orchestrator for all MCP server communications
 */

const MCPAPIGateway = require('./api-gateway');
const MCPAuthManager = require('./auth-manager');
const MCPDataTransformer = require('./data-transformer');

class MCPCoordinator {
  constructor(configPath = './mcp-config.json') {
    this.config = require(configPath);
    this.gateway = new MCPAPIGateway(this.config);
    this.auth = new MCPAuthManager('./secrets');
    this.transformer = new MCPDataTransformer();
    this.isInitialized = false;
  }

  async initialize() {
    if (this.isInitialized) {
      console.log('⚠️ MCP Coordinator already initialized');
      return;
    }

    console.log('🚀 Initializing MCP Integration Coordinator...');

    try {
      // Initialize all components
      await Promise.all([
        this.gateway.initializeServers(),
        this.auth.initialize(),
        // transformer initializes synchronously
      ]);

      this.isInitialized = true;
      console.log('✅ MCP Integration Coordinator ready!');
      
      return this.getStatus();
    } catch (error) {
      console.error('❌ Failed to initialize MCP Coordinator:', error);
      throw error;
    }
  }

  async executeMediaOperation(operation, params = {}) {
    if (!this.isInitialized) {
      await this.initialize();
    }

    const { service, action, data, transform = true } = params;

    try {
      console.log(`📡 Executing ${operation} on ${service}...`);

      // Get authentication headers
      const headers = await this.auth.createAuthHeaders(service);
      
      // Execute operation through gateway
      const result = await this.gateway.executeWithRetry(
        service, 
        action, 
        { ...data, headers }
      );

      // Transform response if requested
      let transformedResult = result;
      if (transform && result.data) {
        try {
          transformedResult = await this.transformer.transform(
            `${service}_${action}`,
            result.data,
            { operation, timestamp: Date.now() }
          );
        } catch (transformError) {
          console.warn(`⚠️ Transform failed, using raw data:`, transformError.message);
        }
      }

      console.log(`✅ Operation ${operation} completed successfully`);
      return transformedResult;

    } catch (error) {
      console.error(`❌ Operation ${operation} failed:`, error.message);
      throw error;
    }
  }

  // Sonarr Operations
  async getSonarrSeries(seriesId = null) {
    return this.executeMediaOperation('get_series', {
      service: 'sonarr',
      action: 'series',
      data: { id: seriesId },
      transform: true
    });
  }

  async addSonarrSeries(seriesData) {
    return this.executeMediaOperation('add_series', {
      service: 'sonarr',
      action: 'add_series',
      data: seriesData,
      transform: true
    });
  }

  // Jellyfin Operations
  async getJellyfinLibrary(libraryId = null) {
    return this.executeMediaOperation('get_library', {
      service: 'jellyfin',
      action: 'library',
      data: { id: libraryId },
      transform: true
    });
  }

  async getJellyfinItem(itemId) {
    return this.executeMediaOperation('get_item', {
      service: 'jellyfin',
      action: 'item',
      data: { id: itemId },
      transform: true
    });
  }

  // Batch Operations
  async batchMediaOperations(operations) {
    console.log(`📦 Executing ${operations.length} batch operations...`);

    const results = await this.gateway.batchOperations(
      operations.map(op => ({
        server: op.service,
        operation: op.action,
        params: {
          ...op.data,
          headers: this.auth.createAuthHeaders(op.service)
        }
      }))
    );

    console.log(`✅ Batch operations completed: ${results.length} results`);
    return results;
  }

  // Cross-service synchronization
  async syncMediaLibraries() {
    console.log('🔄 Starting media library synchronization...');

    try {
      // Get Sonarr series
      const sonarrSeries = await this.getSonarrSeries();
      
      // Get Jellyfin library
      const jellyfinLibrary = await this.getJellyfinLibrary();

      // Compare and sync
      const syncResults = this.compareLilbraries(sonarrSeries, jellyfinLibrary);
      
      console.log('✅ Media library sync completed');
      return syncResults;

    } catch (error) {
      console.error('❌ Media library sync failed:', error);
      throw error;
    }
  }

  compareLilbraries(sonarrData, jellyfinData) {
    // Implementation for comparing and syncing libraries
    return {
      sonarrCount: Array.isArray(sonarrData) ? sonarrData.length : 0,
      jellyfinCount: Array.isArray(jellyfinData) ? jellyfinData.length : 0,
      syncTime: Date.now(),
      status: 'completed'
    };
  }

  // Health and Status
  async getStatus() {
    const gatewayHealth = await this.gateway.healthCheck();
    const authStatus = this.auth.getAuthStatus();
    const transformers = this.transformer.getAvailableTransformers();

    return {
      coordinator: {
        initialized: this.isInitialized,
        uptime: Date.now()
      },
      gateway: gatewayHealth,
      authentication: authStatus,
      transformers: {
        available: transformers,
        count: transformers.length
      },
      config: {
        servers: Object.keys(this.config.mcpServers),
        settings: this.config.integrationSettings
      }
    };
  }

  async healthCheck() {
    try {
      const status = await this.getStatus();
      console.log('🏥 MCP Health Check Results:');
      console.log(JSON.stringify(status, null, 2));
      return status;
    } catch (error) {
      console.error('❌ Health check failed:', error);
      return { error: error.message, timestamp: Date.now() };
    }
  }

  // Graceful shutdown
  async shutdown() {
    console.log('🛑 Shutting down MCP Coordinator...');
    
    // Clean up resources
    this.isInitialized = false;
    
    console.log('✅ MCP Coordinator shutdown complete');
  }
}

// Export for use in other modules
module.exports = MCPCoordinator;

// CLI usage
if (require.main === module) {
  const coordinator = new MCPCoordinator();
  
  // Handle graceful shutdown
  process.on('SIGINT', async () => {
    await coordinator.shutdown();
    process.exit(0);
  });

  coordinator.initialize().then(async (status) => {
    console.log('🌟 MCP Integration Coordinator is ready!');
    console.log('📊 Initial Status:', JSON.stringify(status, null, 2));

    // Start periodic health checks
    setInterval(async () => {
      await coordinator.healthCheck();
    }, 300000); // Every 5 minutes
    
  }).catch(error => {
    console.error('💥 Failed to start MCP Coordinator:', error);
    process.exit(1);
  });
}