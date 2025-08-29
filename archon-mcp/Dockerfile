FROM node:20-slim
WORKDIR /app
RUN npm init -y && npm install express cors body-parser
COPY mcp.js ./mcp.js 2>/dev/null || echo "console.log('Archon MCP Mock');" > mcp.js
COPY ../scripts/console-shim.js ./scripts/console-shim.js
CMD ["node", "-r", "./scripts/console-shim.js", "mcp.js"]
