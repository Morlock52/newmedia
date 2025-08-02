#!/bin/bash

# Environment API Start Script

echo "Starting Environment API Server..."

# Check if node is installed
if ! command -v node &> /dev/null; then
    echo "Error: Node.js is not installed"
    exit 1
fi

# Navigate to script directory
cd "$(dirname "$0")"

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

# Create backups directory if it doesn't exist
mkdir -p backups

# Check if .env file exists in parent directory
if [ ! -f "../.env" ]; then
    echo "Warning: .env file not found in parent directory"
    echo "Creating empty .env file..."
    touch "../.env"
fi

# Start the server
echo "Starting server on http://localhost:3456"
npm start