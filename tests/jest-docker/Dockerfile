# Jest test runner container that can talk to the host Docker daemon
FROM node:20-alpine

# Install docker CLI to allow `docker ps` inside tests
RUN apk add --no-cache docker-cli bash curl git

WORKDIR /workspace

# Default command will be overridden by docker-compose / run script
CMD ["bash", "-lc", "npm ci && npm test -- --runInBand"]
