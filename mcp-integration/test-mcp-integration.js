#!/usr/bin/env node
/**
 * MCP Integration Test Script
 * Tests MCP connections and service integrations
 */

const MCPClient = require('./MCPClient');

async function testMCPIntegration() {
  console.log('🧪 Testing MCP Integration for Media Server\n');
  console.log('=' .repeat(60));
  
  const client = new MCPClient();
  const results = {
    passed: [],
    failed: [],
    warnings: []
  };

  try {
    // Test 1: Initialize MCP connections
    console.log('\n📋 Test 1: Initialize MCP connections');
    try {
      await client.initialize();
      results.passed.push('MCP initialization');
      console.log('✅ MCP connections initialized');
    } catch (error) {
      results.failed.push(`MCP initialization: ${error.message}`);
      console.log('❌ Failed to initialize MCP connections');
    }

    // Test 2: Check Claude Flow connection
    console.log('\n📋 Test 2: Check Claude Flow MCP');
    const claudeFlow = client.connections.get('claude-flow');
    if (claudeFlow && claudeFlow.status === 'connected') {
      results.passed.push('Claude Flow connection');
      console.log('✅ Claude Flow MCP is connected');
    } else {
      results.warnings.push('Claude Flow MCP not available');
      console.log('⚠️  Claude Flow MCP not available');
    }

    // Test 3: Check RUV Swarm connection
    console.log('\n📋 Test 3: Check RUV Swarm MCP');
    const ruvSwarm = client.connections.get('ruv-swarm');
    if (ruvSwarm && ruvSwarm.status === 'connected') {
      results.passed.push('RUV Swarm connection');
      console.log('✅ RUV Swarm MCP is connected');
    } else {
      results.warnings.push('RUV Swarm MCP not available');
      console.log('⚠️  RUV Swarm MCP not available');
    }

    // Test 4: Check media services
    console.log('\n📋 Test 4: Check media services');
    const services = ['jellyfin', 'plex', 'sonarr', 'radarr', 'qbittorrent'];
    for (const serviceName of services) {
      const service = client.connections.get(`service:${serviceName}`);
      if (service && service.status === 'connected') {
        results.passed.push(`${serviceName} service`);
        console.log(`✅ ${serviceName} is connected`);
      } else {
        results.warnings.push(`${serviceName} service not available`);
        console.log(`⚠️  ${serviceName} not available`);
      }
    }

    // Test 5: Health check
    console.log('\n📋 Test 5: System health check');
    const health = await client.healthCheck();
    console.log('📊 Health Status:');
    console.log(JSON.stringify(health, null, 2));
    results.passed.push('Health check completed');

    // Test 6: Test MCP tools (if services available)
    console.log('\n📋 Test 6: Test MCP tools');
    
    // Test media scan tool
    try {
      const scanResult = await client.executeTool('media.scan', {
        library: 'all',
        deep: false
      });
      results.passed.push('Media scan tool');
      console.log('✅ Media scan tool works');
    } catch (error) {
      results.warnings.push(`Media scan tool: ${error.message}`);
      console.log(`⚠️  Media scan tool: ${error.message}`);
    }

    // Test download status tool
    try {
      const statusResult = await client.executeTool('download.status', {});
      results.passed.push('Download status tool');
      console.log('✅ Download status tool works');
    } catch (error) {
      results.warnings.push(`Download status tool: ${error.message}`);
      console.log(`⚠️  Download status tool: ${error.message}`);
    }

    // Cleanup
    await client.cleanup();

  } catch (error) {
    console.error('\n❌ Test failed with error:', error);
    results.failed.push(`General error: ${error.message}`);
  }

  // Print summary
  console.log('\n' + '='.repeat(60));
  console.log('📊 Test Summary:');
  console.log(`✅ Passed: ${results.passed.length}`);
  console.log(`❌ Failed: ${results.failed.length}`);
  console.log(`⚠️  Warnings: ${results.warnings.length}`);
  
  if (results.passed.length > 0) {
    console.log('\n✅ Passed tests:');
    results.passed.forEach(test => console.log(`  • ${test}`));
  }
  
  if (results.failed.length > 0) {
    console.log('\n❌ Failed tests:');
    results.failed.forEach(test => console.log(`  • ${test}`));
  }
  
  if (results.warnings.length > 0) {
    console.log('\n⚠️  Warnings:');
    results.warnings.forEach(warning => console.log(`  • ${warning}`));
  }

  // Recommendations
  console.log('\n📝 Recommendations:');
  if (!claudeFlow || claudeFlow.status !== 'connected') {
    console.log('  • Install Claude Flow: npx claude-flow@alpha mcp start');
  }
  if (!ruvSwarm || ruvSwarm.status !== 'connected') {
    console.log('  • Install RUV Swarm: npx ruv-swarm@latest mcp start');
  }
  if (results.warnings.length > 0) {
    console.log('  • Start missing services with: docker-compose up -d');
    console.log('  • Check service API keys in .env file');
  }

  process.exit(results.failed.length > 0 ? 1 : 0);
}

// Run tests
if (require.main === module) {
  testMCPIntegration().catch(console.error);
}

module.exports = { testMCPIntegration };