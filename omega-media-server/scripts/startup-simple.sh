#!/bin/bash
# Simplified startup script for Omega Media Server

echo "Starting Omega Media Server (Simple Version)..."

# Start PostgreSQL
service postgresql start
sleep 5

# Start Redis
service redis-server start
sleep 2

# Initialize database if needed
if [ ! -f /config/.initialized ]; then
    echo "First run detected, initializing..."
    
    # Create PostgreSQL user and database
    sudo -u postgres createuser omega
    sudo -u postgres createdb omega
    
    # Create directories
    mkdir -p /config/{jellyfin,nginx,ssl}
    mkdir -p /media/{movies,tv,music,photos,books}
    
    # Generate self-signed certificate
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout /config/ssl/key.pem \
        -out /config/ssl/cert.pem \
        -subj "/C=US/ST=State/L=City/O=Omega/CN=localhost"
    
    touch /config/.initialized
fi

# Start Nginx
nginx

# Start Jellyfin
/usr/bin/jellyfin \
    --datadir /config/jellyfin \
    --cachedir /config/jellyfin/cache \
    --webdir /usr/share/jellyfin/web &

# Start the main application
cd /opt/omega
npm start &

# Start supervisord to manage all processes
exec /usr/bin/supervisord -n -c /etc/supervisor/supervisord.conf