#!/bin/bash

echo "🐳 Alternative Simple Build (MCP Suite Only)"
echo "============================================"

# Build just the MCP suite as a standalone container
echo "📦 Building MCP Suite container..."

# Create a minimal Dockerfile for just the MCP suite
cat > Dockerfile.mcp-only << 'EOF'
FROM node:20-alpine

# Install runtime dependencies
RUN apk add --no-cache curl tini

# Set working directory
WORKDIR /app

# Copy MCP suite
COPY mcp-architecture/ ./

# Install dependencies
RUN npm ci --only=production

# Create non-root user
RUN addgroup -g 1001 -S nodejs && adduser -S nodejs -u 1001
RUN chown -R nodejs:nodejs /app
USER nodejs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8090/health || exit 1

# Expose ports
EXPOSE 8090 3001 3002 3003 3004 3005

# Use tini as init
ENTRYPOINT ["/sbin/tini", "--"]

# Start the application
CMD ["node", "src/index.js"]
EOF

# Build the MCP-only container
docker build -t mediaserver-mcp -f Dockerfile.mcp-only .

if [ $? -eq 0 ]; then
    echo "✅ MCP Suite build successful!"
    echo ""
    echo "🚀 To run the MCP suite:"
    echo "docker run -d \\"
    echo "  --name mediaserver-mcp \\"
    echo "  --restart unless-stopped \\"
    echo "  -p 8090:8090 \\"
    echo "  -p 3001:3001 \\"
    echo "  -p 3002:3002 \\"
    echo "  -p 3003:3003 \\"
    echo "  -p 3004:3004 \\"
    echo "  -p 3005:3005 \\"
    echo "  -e OPENAI_API_KEY=\"your-openai-key-here\" \\"
    echo "  -e JELLYFIN_URL=\"http://host.docker.internal:8096\" \\"
    echo "  -e SONARR_URL=\"http://host.docker.internal:8989\" \\"
    echo "  -e RADARR_URL=\"http://host.docker.internal:7878\" \\"
    echo "  -e PROWLARR_URL=\"http://host.docker.internal:9696\" \\"
    echo "  -e QBITTORRENT_URL=\"http://host.docker.internal:8080\" \\"
    echo "  mediaserver-mcp"
    echo ""
    echo "📡 MCP Access points:"
    echo "  • AI Dashboard: http://localhost:8090"
    echo "  • Jellyfin MCP: http://localhost:3001"
    echo "  • Sonarr MCP: http://localhost:3002"
    echo "  • Radarr MCP: http://localhost:3003"
    echo "  • Prowlarr MCP: http://localhost:3004"
    echo "  • qBittorrent MCP: http://localhost:3005"
    echo ""
    echo "🧪 Test the MCP suite:"
    echo "  cd mcp-architecture && node test-mcp-servers.js"
    echo ""
    echo "🔗 Connect to Claude Desktop using MCP_CONNECTION_GUIDE.md"
    
    # Clean up temporary Dockerfile
    rm -f Dockerfile.mcp-only
else
    echo "❌ Build failed!"
    # Clean up temporary Dockerfile
    rm -f Dockerfile.mcp-only
fi