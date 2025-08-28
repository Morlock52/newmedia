#!/bin/bash

# MCP Integration Quick Start Script
# This script helps you quickly test MCP integration with your media server

echo "🚀 MCP Media Server Integration Quick Start"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if MCP servers are configured
check_mcp_config() {
    echo -e "${YELLOW}Checking MCP configuration...${NC}"
    
    if [ -f ".mcp.json" ]; then
        echo -e "${GREEN}✓ MCP configuration found${NC}"
        cat .mcp.json | grep -E '"(claude-flow|ruv-swarm|unified-media)"' > /dev/null
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ MCP servers configured${NC}"
        else
            echo -e "${RED}✗ No MCP servers found in configuration${NC}"
            exit 1
        fi
    else
        echo -e "${RED}✗ .mcp.json not found${NC}"
        echo "Creating default MCP configuration..."
        cat > .mcp.json << 'EOF'
{
  "mcpServers": {
    "claude-flow": {
      "command": "npx",
      "args": ["claude-flow@alpha", "mcp", "start"],
      "type": "stdio"
    },
    "ruv-swarm": {
      "command": "npx",
      "args": ["ruv-swarm@latest", "mcp", "start"],
      "type": "stdio"
    }
  }
}
EOF
        echo -e "${GREEN}✓ Created .mcp.json${NC}"
    fi
}

# Function to test MCP servers
test_mcp_servers() {
    echo -e "\n${YELLOW}Testing MCP servers...${NC}"
    
    # Test claude-flow
    echo -n "Testing claude-flow... "
    timeout 5 npx claude-flow@alpha --version > /dev/null 2>&1
    if [ $? -eq 0 ] || [ $? -eq 124 ]; then
        echo -e "${GREEN}✓${NC} (available)"
    else
        echo -e "${YELLOW}⚠${NC} (may need installation)"
    fi
    
    # Test ruv-swarm
    echo -n "Testing ruv-swarm... "
    timeout 5 npx ruv-swarm@latest --version > /dev/null 2>&1
    if [ $? -eq 0 ] || [ $? -eq 124 ]; then
        echo -e "${GREEN}✓${NC} (available)"
    else
        echo -e "${YELLOW}⚠${NC} (may need installation)"
    fi
}

# Function to run quick examples
run_quick_examples() {
    echo -e "\n${YELLOW}Running Quick MCP Examples${NC}"
    echo "================================"
    
    echo -e "\n1. ${GREEN}Health Check Example${NC}"
    echo "Check the health of all media services:"
    echo -e "${YELLOW}mcp__unified-media__unified_health_check()${NC}"
    
    echo -e "\n2. ${GREEN}Swarm Initialization Example${NC}"
    echo "Initialize a coordination swarm:"
    echo -e "${YELLOW}mcp__claude-flow__swarm_init({ topology: 'mesh', maxAgents: 5 })${NC}"
    
    echo -e "\n3. ${GREEN}Container Management Example${NC}"
    echo "List all Docker containers:"
    echo -e "${YELLOW}mcp__unified-media__docker_list_containers()${NC}"
    
    echo -e "\n4. ${GREEN}Performance Monitoring Example${NC}"
    echo "Get performance metrics:"
    echo -e "${YELLOW}mcp__claude-flow__benchmark_run({ type: 'all' })${NC}"
    
    echo -e "\n5. ${GREEN}Memory Storage Example${NC}"
    echo "Store data in persistent memory:"
    echo -e "${YELLOW}mcp__claude-flow__memory_usage({ action: 'store', key: 'test', value: 'data' })${NC}"
}

# Function to create test workflow
create_test_workflow() {
    echo -e "\n${YELLOW}Creating Test Workflow${NC}"
    echo "====================="
    
    cat > mcp-test-workflow.js << 'EOF'
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
EOF
    
    echo -e "${GREEN}✓ Created mcp-test-workflow.js${NC}"
    echo "Run this workflow in Claude Code to test MCP integration"
}

# Function to show next steps
show_next_steps() {
    echo -e "\n${GREEN}✅ MCP Integration Setup Complete!${NC}"
    echo "=================================="
    echo
    echo "Next Steps:"
    echo "1. Open Claude Code"
    echo "2. The MCP servers will connect automatically"
    echo "3. Try these commands:"
    echo
    echo -e "   ${YELLOW}mcp__unified-media__unified_health_check()${NC}"
    echo -e "   ${YELLOW}mcp__claude-flow__swarm_init({ topology: 'mesh' })${NC}"
    echo -e "   ${YELLOW}mcp__ruv-swarm__agent_spawn({ type: 'researcher' })${NC}"
    echo
    echo "4. Run the example script:"
    echo -e "   ${YELLOW}node scripts/mcp-examples.js 1${NC}  # Run example 1"
    echo
    echo "5. Check the integration guide:"
    echo -e "   ${YELLOW}MCP_INTEGRATION_GUIDE.md${NC}"
    echo
    echo "Happy orchestrating! 🚀"
}

# Main execution
main() {
    check_mcp_config
    test_mcp_servers
    run_quick_examples
    create_test_workflow
    show_next_steps
}

# Run main function
main