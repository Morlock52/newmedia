FROM node:20-slim
WORKDIR /app
RUN npm init -y && npm install express
COPY ui.js ./ui.js 2>/dev/null || echo "console.log('Archon UI Mock');" > ui.js
CMD ["node", "ui.js"]
