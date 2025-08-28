FROM node:20-slim
WORKDIR /app
RUN npm init -y && npm install express cors
COPY agents.js ./agents.js 2>/dev/null || echo "console.log('Archon Agents Mock');" > agents.js
CMD ["node", "agents.js"]
