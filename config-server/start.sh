#!/bin/bash

# Config Server Startup Script

echo "Config Server Startup"
echo "===================="

# Check if .env file exists
if [ ! -f .env ]; then
    echo "Creating .env file from .env.example..."
    cp .env.example .env
    echo "✓ .env file created"
    echo ""
    echo "⚠️  IMPORTANT: Please update the following in .env:"
    echo "   - JWT_SECRET (use a secure random string)"
    echo "   - ADMIN_PASSWORD_HASH (generate with: node -e \"console.log(require('bcryptjs').hashSync('your-password', 10))\")"
    echo ""
fi

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
    echo "✓ Dependencies installed"
fi

# Check if docker-compose.yml exists
if [ ! -f docker-compose.yml ] && [ -f docker-compose.example.yml ]; then
    echo "No docker-compose.yml found. Would you like to use the example? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        cp docker-compose.example.yml docker-compose.yml
        echo "✓ docker-compose.yml created from example"
    fi
fi

echo ""
echo "Starting server..."
echo "=================="
echo "API URL: http://localhost:3000"
echo "WebSocket URL: ws://localhost:3000"
echo ""

# Start the server
npm start