#!/bin/bash

# Test all WebUI API endpoints
BASE_URL="http://localhost:3000"

echo "🧪 Testing Media Server Stack Web UI API Endpoints"
echo "=================================================="

# Test basic endpoints
echo "📊 Testing System Information..."
curl -s "$BASE_URL/api/system-info" | jq '.' || echo "❌ Failed"

echo -e "\n🔍 Testing Environment Status..."
curl -s "$BASE_URL/api/env-status" | jq '.' || echo "❌ Failed"

echo -e "\n🐳 Testing Docker Status..."
curl -s "$BASE_URL/api/docker-status" | jq '.' || echo "❌ Failed"

echo -e "\n📈 Testing System Stats..."
curl -s "$BASE_URL/api/system-stats" | jq '.' || echo "❌ Failed"

echo -e "\n🏥 Testing Health Check..."
curl -s "$BASE_URL/api/health" | jq '.' || echo "❌ Failed"

echo -e "\n📋 Testing Service Status..."
curl -s "$BASE_URL/api/status" | head -3

echo -e "\n🔧 Testing Configuration Validation..."
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"domain":"test.example.com","email":"test@example.com","puid":"1000","vpnProvider":"pia"}' \
  "$BASE_URL/api/validate-config" | jq '.' || echo "❌ Failed"

echo -e "\n📝 Testing Logs Endpoint..."
curl -s "$BASE_URL/api/logs/all" | head -3 || echo "No logs available"

echo -e "\n✅ All endpoints tested!"
echo "🌐 Access the Web UI at: $BASE_URL"
