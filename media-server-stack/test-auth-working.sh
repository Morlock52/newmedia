#!/bin/bash

# Working Authentication Test - Fixed Configuration
# Uses proper domain format and configuration

set -e

echo "🔧 Testing Authentication with Fixed Configuration..."

# 1. Clean up any existing containers
echo "🧹 Cleaning up..."
docker stop authelia-working-test 2>/dev/null || true
docker rm authelia-working-test 2>/dev/null || true

# 2. Update hosts file with proper domains
echo "📝 Adding .test domains to /etc/hosts..."
sudo bash -c 'grep -v "localhost.test" /etc/hosts > /tmp/hosts.tmp && mv /tmp/hosts.tmp /etc/hosts'
sudo bash -c 'cat >> /etc/hosts << EOF
# Media Server Auth Routes (Fixed)
127.0.0.1    auth.localhost.test
127.0.0.1    traefik.localhost.test
127.0.0.1    sonarr.localhost.test
127.0.0.1    radarr.localhost.test
127.0.0.1    prowlarr.localhost.test
127.0.0.1    overseerr.localhost.test
127.0.0.1    jellyfin.localhost.test
127.0.0.1    homarr.localhost.test
EOF'

echo "✅ .test domains added to /etc/hosts"

# 3. Start Authelia with fixed configuration
echo "🚀 Starting Authelia with fixed configuration..."
docker run -d \
  --name authelia-working-test \
  --rm \
  -p 9092:9091 \
  -v "$(pwd)/../config/authelia:/config" \
  -e AUTHELIA_JWT_SECRET=test_jwt_secret_123456789012345678901234567890 \
  -e AUTHELIA_SESSION_SECRET=test_session_secret_123456789012345678901234567890 \
  -e AUTHELIA_STORAGE_ENCRYPTION_KEY=test_encryption_key_123456789012345678901234567890 \
  authelia/authelia:4.38 \
  --config /config/configuration-fixed.yml

echo "⏳ Waiting for Authelia to start..."
sleep 15

# 4. Test Authelia connectivity
echo "🔗 Testing Authelia connection..."
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:9092 || echo "000")

if [ "$RESPONSE" = "200" ]; then
    echo "✅ Authelia is working! (HTTP $RESPONSE)"
else
    echo "⚠️  Authelia responded with HTTP $RESPONSE"
    echo "📋 Checking logs..."
    docker logs authelia-working-test --tail 10
fi

# 5. Test API endpoints
echo "🧪 Testing API endpoints..."
echo "Health check:"
curl -s http://localhost:9092/api/health | head -c 200 || echo "Health check failed"
echo ""

echo "Configuration check:"
curl -s http://localhost:9092/api/configuration | head -c 200 || echo "Config check failed"
echo ""

# 6. Create simple test page
echo "📄 Creating browser test page..."
cat > auth-test-working.html << 'EOF'
<!DOCTYPE html>
<html>
<head>
    <title>Working Auth Test</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .test-link { display: block; margin: 15px 0; padding: 15px; background: #e3f2fd; text-decoration: none; border-radius: 6px; color: #1976d2; font-weight: bold; }
        .test-link:hover { background: #bbdefb; }
        .success { background: #e8f5e8; color: #2e7d32; }
        .warning { background: #fff3e0; color: #f57f17; }
        .credentials { background: #f3e5f5; padding: 20px; border-radius: 6px; margin: 20px 0; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🔐 Working Authentication Test</h1>
        <p>The configuration has been fixed! Test these links:</p>
        
        <a href="http://localhost:9092" class="test-link success">
            ✅ Authelia Login (Direct - Port 9092)
        </a>
        
        <a href="http://localhost:9092/api/health" class="test-link">
            📊 Authelia API Health
        </a>
        
        <a href="http://localhost:9092/api/configuration" class="test-link">
            ⚙️ Authelia Configuration
        </a>
        
        <div class="credentials">
            <h2>🔑 Login Credentials:</h2>
            <ul>
                <li><strong>Username:</strong> admin</li>
                <li><strong>Password:</strong> admin123</li>
            </ul>
        </div>
        
        <div class="warning">
            <h2>⚠️ Important Notes:</h2>
            <ul>
                <li>This is a test environment using port 9092</li>
                <li>Configuration has been fixed for Authelia 4.38</li>
                <li>Using .test domains for proper cookie handling</li>
                <li>Container will auto-stop when test script ends</li>
            </ul>
        </div>
        
        <h2>✅ Expected Results:</h2>
        <ol>
            <li>Click "Authelia Login" → Should show login page</li>
            <li>Enter admin/admin123 → Should authenticate successfully</li>
            <li>API Health → Should return {"status":"UP"}</li>
            <li>Configuration → Should return config JSON</li>
        </ol>
    </div>
</body>
</html>
EOF

echo ""
echo "🎉 Authentication test ready!"
echo ""
echo "📋 Quick Test URLs:"
echo "   🌐 Authelia UI: http://localhost:9092"
echo "   📊 API Health: http://localhost:9092/api/health"
echo "   📄 Test Page: open auth-test-working.html"
echo ""
echo "🔐 Login: admin / admin123"
echo ""
echo "⏸️  Test environment will run for 5 minutes..."
echo "   Press Ctrl+C to stop early and cleanup"
echo ""

# Keep running for testing
trap 'echo "🧹 Cleaning up..." && docker stop authelia-working-test 2>/dev/null' EXIT

sleep 300 || echo "Test stopped early"