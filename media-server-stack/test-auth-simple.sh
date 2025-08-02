#!/bin/bash

# Simple Authentication Test - Uses only available ports
# Tests authentication without port conflicts

set -e

echo "🔧 Simple Authentication Test Setup..."

# 1. Clean up any previous attempts
echo "🧹 Cleaning up previous test containers..."
docker stop traefik-auth traefik-auth-alt authelia-auth authelia-auth-alt whoami-test whoami-test-alt 2>/dev/null || true
docker rm traefik-auth traefik-auth-alt authelia-auth authelia-auth-alt whoami-test whoami-test-alt 2>/dev/null || true

# 2. Find available ports
echo "🔍 Finding available ports..."
for port in 8082 8083 8084 8085 8086 8087 8088 8089; do
    if ! lsof -i :$port > /dev/null 2>&1; then
        TRAEFIK_PORT=$port
        break
    fi
done

for port in 8092 8093 8094 8095 8097 8098 8099; do
    if ! lsof -i :$port > /dev/null 2>&1; then
        TRAEFIK_DASH_PORT=$port
        break
    fi
done

echo "Using Traefik on port $TRAEFIK_PORT and dashboard on port $TRAEFIK_DASH_PORT"

# 3. Test Authelia directly first (without Traefik)
echo "🧪 Testing Authelia directly..."
docker run -d \
  --name authelia-direct-test \
  --rm \
  -p 9092:9091 \
  -v "$(pwd)/../config/authelia:/config" \
  -e AUTHELIA_JWT_SECRET=test_jwt_secret_123 \
  -e AUTHELIA_SESSION_SECRET=test_session_secret_123 \
  -e AUTHELIA_STORAGE_ENCRYPTION_KEY=test_encryption_key_123 \
  authelia/authelia:4.38

echo "⏳ Waiting for Authelia to start..."
sleep 10

# 4. Test Authelia connectivity
echo "🔗 Testing Authelia direct connection..."
AUTHELIA_RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9092 || echo "000")

if [ "$AUTHELIA_RESPONSE" = "200" ]; then
    echo "✅ Authelia is working directly on port 9092!"
    echo ""
    echo "📋 Direct Access URLs:"
    echo "   Authelia UI: http://localhost:9092"
    echo "   API Status: http://localhost:9092/api/health"
    echo ""
    echo "🔐 Login Credentials:"
    echo "   Username: admin"
    echo "   Password: admin123"
    echo ""
    echo "✨ Try opening http://localhost:9092 in your browser!"
    echo ""
    
    # Test the API endpoints
    echo "🔍 Testing API endpoints..."
    curl -s http://localhost:9092/api/health | head -c 200
    echo ""
    
else
    echo "❌ Authelia failed to respond (HTTP $AUTHELIA_RESPONSE)"
    echo "📋 Checking logs..."
    docker logs authelia-direct-test --tail 20
fi

# 5. Create a simple HTML test page
echo "📝 Creating simple test page..."
cat > auth-test.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Auth Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .container { max-width: 600px; margin: 0 auto; }
        .test-link { display: block; margin: 10px 0; padding: 10px; background: #f0f0f0; text-decoration: none; border-radius: 4px; }
        .test-link:hover { background: #e0e0e0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Authentication Test Page</h1>
        <p>Test these links to verify authentication is working:</p>
        
        <a href="http://localhost:9092" class="test-link">
            🔐 Authelia Login (Direct)
        </a>
        
        <a href="http://localhost:9092/api/health" class="test-link">
            📊 Authelia API Health
        </a>
        
        <a href="http://localhost:9092/api/state" class="test-link">
            🔍 Authelia State
        </a>
        
        <h2>Login Credentials:</h2>
        <ul>
            <li><strong>Username:</strong> admin</li>
            <li><strong>Password:</strong> admin123</li>
        </ul>
        
        <h2>Expected Behavior:</h2>
        <ol>
            <li>Click "Authelia Login" link</li>
            <li>You should see Authelia login page</li>
            <li>Enter admin/admin123</li>
            <li>You should be authenticated</li>
        </ol>
    </div>
</body>
</html>
EOF

echo "📄 Test page created: auth-test.html"
echo "   Open this file in your browser for easy testing"

echo ""
echo "🎉 Simple authentication test ready!"
echo ""
echo "🔗 Quick Test:"
echo "   1. Open your browser"
echo "   2. Go to: http://localhost:9092"
echo "   3. Login with: admin / admin123"
echo ""

# Keep container running for testing
echo "⏸️  Container will run for 5 minutes for testing..."
echo "   Press Ctrl+C to stop early"
echo ""

# Wait or until user stops
sleep 300 2>/dev/null || echo "Stopping..."

# Cleanup
echo "🧹 Cleaning up test container..."
docker stop authelia-direct-test 2>/dev/null || true