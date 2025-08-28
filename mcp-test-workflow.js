// Test Workflow for MCP Integration

async function testMCPIntegration() {
    console.log("🧪 Testing MCP Integration...\n");
    
    // Test 1: Health Check
    console.log("1. Testing Health Check...");
    try {
        const health = await mcp__unified_media__unified_health_check();
        console.log("✓ Health check successful:", health);
    } catch (error) {
        console.log("✗ Health check failed:", error.message);
    }
    
    // Test 2: Swarm Status
    console.log("\n2. Testing Swarm Status...");
    try {
        const status = await mcp__claude_flow__swarm_status({ verbose: true });
        console.log("✓ Swarm status retrieved:", status);
    } catch (error) {
        console.log("✗ Swarm status failed:", error.message);
    }
    
    // Test 3: Container List
    console.log("\n3. Testing Container List...");
    try {
        const containers = await mcp__unified_media__docker_list_containers();
        console.log("✓ Containers listed:", containers.length, "found");
    } catch (error) {
        console.log("✗ Container list failed:", error.message);
    }
    
    // Test 4: Memory Operations
    console.log("\n4. Testing Memory Operations...");
    try {
        await mcp__claude_flow__memory_usage({
            action: "store",
            key: "test-key",
            value: "test-value"
        });
        console.log("✓ Memory store successful");
        
        const retrieved = await mcp__claude_flow__memory_usage({
            action: "retrieve",
            key: "test-key"
        });
        console.log("✓ Memory retrieve successful:", retrieved);
    } catch (error) {
        console.log("✗ Memory operations failed:", error.message);
    }
    
    console.log("\n✅ MCP Integration tests complete!");
}

// Run the test
testMCPIntegration();
