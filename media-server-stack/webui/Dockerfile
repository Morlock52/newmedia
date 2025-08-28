FROM node:22-alpine

# Install Docker CLI and Compose plugin (plus classic for fallback)
RUN apk add --no-cache docker-cli docker-cli-compose docker-compose

WORKDIR /app
COPY package.json .
RUN npm install --production
COPY . .
ENV PORT=3000
CMD ["node", "server.js"]
