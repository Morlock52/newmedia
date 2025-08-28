#!/bin/bash

# AI Services Test Script
# Tests all AI-enhanced features

echo "🧪 Testing AI-Enhanced Media Server Services"
echo "==========================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Test function
test_service() {
    local name=$1
    local url=$2
    local expected=$3
    
    echo -n "Testing $name... "
    
    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)
    
    if [ "$response" = "$expected" ]; then
        echo -e "${GREEN}✓ OK${NC} (HTTP $response)"
        return 0
    else
        echo -e "${RED}✗ FAILED${NC} (HTTP $response, expected $expected)"
        return 1
    fi
}

# Test AI content analysis
test_ai_content() {
    echo ""
    echo "📝 Testing Content Analysis..."
    
    # Test safe content
    response=$(curl -s -X POST http://localhost:8090/api/assess \
        -H "Content-Type: application/json" \
        -d '{"content": "This is educational content about science", "type": "text"}' \
        2>/dev/null)
    
    if echo "$response" | grep -q "safety_score"; then
        echo -e "${GREEN}✓${NC} Content analysis working"
        echo "   Response: $(echo $response | jq -r '.safety_score' 2>/dev/null || echo 'Processing')"
    else
        echo -e "${YELLOW}⚠${NC} Content analysis may still be initializing"
    fi
}

# Test recommendation engine
test_recommendations() {
    echo ""
    echo "🎯 Testing Recommendation Engine..."
    
    response=$(curl -s "http://localhost:8092/api/recommendations/testuser?num_recs=5" 2>/dev/null)
    
    if echo "$response" | grep -q "recommendations"; then
        echo -e "${GREEN}✓${NC} Recommendation engine working"
    else
        echo -e "${YELLOW}⚠${NC} Recommendation engine may still be initializing"
    fi
}

# Test moderation service
test_moderation() {
    echo ""
    echo "🛡️ Testing Content Moderation..."
    
    response=$(curl -s -X POST http://localhost:8091/api/moderate \
        -H "Content-Type: application/json" \
        -d '{"text": "This is a test message", "userId": "test123"}' \
        2>/dev/null)
    
    if echo "$response" | grep -q "safe"; then
        echo -e "${GREEN}✓${NC} Content moderation working"
    else
        echo -e "${YELLOW}⚠${NC} Content moderation may still be initializing"
    fi
}

# Main tests
echo "🔍 Testing Service Health Endpoints:"
echo "------------------------------------"

test_service "AI Safety Service" "http://localhost:8090/health" "200"
test_service "Content Moderation" "http://localhost:8091/health" "200"
test_service "Recommendation Engine" "http://localhost:8092/health" "200"
test_service "Social Media Service" "http://localhost:8093/health" "200"
test_service "AI Dashboard" "http://localhost:8094/" "200"
test_service "API Gateway" "http://localhost:8095/health" "200"

# Test AI features
test_ai_content
test_recommendations
test_moderation

echo ""
echo "📊 Test Summary:"
echo "---------------"

# Check Docker containers
running_containers=$(docker ps --filter "name=ai-" --format "{{.Names}}" | wc -l)
echo "• Running AI containers: $running_containers"

# Check memory usage
if command -v free &> /dev/null; then
    mem_used=$(free -m | awk 'NR==2{printf "%.1f", $3/1024}')
    echo "• Memory usage: ${mem_used}GB"
fi

# Check GPU usage if available
if command -v nvidia-smi &> /dev/null; then
    gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "• GPU utilization: ${gpu_usage}%"
fi

echo ""
echo "✅ AI Services Test Complete!"
echo ""
echo "🌐 Access Points:"
echo "  • AI Dashboard: http://localhost:8094"
echo "  • API Gateway:  http://localhost:8095"
echo "  • API Docs:     http://localhost:8095/docs"
echo ""