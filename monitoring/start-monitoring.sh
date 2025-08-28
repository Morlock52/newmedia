#!/bin/bash

# Start Monitoring Stack
echo "🚀 Starting MediaFlow Pro Monitoring Stack..."

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install winston winston-daily-rotate-file winston-elasticsearch @sentry/node node-statsd systeminformation dockerode socket.io express tail chokidar
fi

# Create logs directory
mkdir -p logs
mkdir -p monitoring/public

# Copy dashboard HTML to public directory
cp monitoring/monitoring-dashboard.html monitoring/public/index.html

# Start monitoring dashboard
echo "🎯 Starting Monitoring Dashboard on http://localhost:3005"
node monitoring/monitoring-dashboard.js &
DASHBOARD_PID=$!

# Wait a moment for server to start
sleep 2

# Start log aggregator
echo "📊 Starting Log Aggregator..."
node monitoring/log-aggregator.js &
AGGREGATOR_PID=$!

echo "✅ Monitoring stack started!"
echo "📊 Dashboard: http://localhost:3005"
echo "🔍 Process IDs: Dashboard=$DASHBOARD_PID, Aggregator=$AGGREGATOR_PID"
echo ""
echo "Press Ctrl+C to stop all services"

# Function to cleanup on exit
cleanup() {
    echo "🛑 Stopping monitoring services..."
    kill $DASHBOARD_PID 2>/dev/null
    kill $AGGREGATOR_PID 2>/dev/null
    echo "✅ Monitoring stack stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Keep script running
wait