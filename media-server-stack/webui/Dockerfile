FROM node:22-alpine

# Install Docker CLI
RUN apk add --no-cache docker-cli docker-compose

WORKDIR /app
COPY package.json .
RUN npm install --production
COPY . .
ENV PORT=3000
CMD ["node", "server.js"]
