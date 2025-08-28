#!/bin/bash
# Install Sonarr TV series management

set -e

echo "Installing Sonarr..."

# Add Sonarr repository
curl -o /tmp/sonarr.tar.gz -L "https://download.sonarr.tv/v4/main/4.0.9.2244/Sonarr.main.4.0.9.2244.linux-x64.tar.gz"
tar -xzf /tmp/sonarr.tar.gz -C /opt/
mv /opt/Sonarr /opt/sonarr
chown -R mediaserver:mediaserver /opt/sonarr

# Create service configuration
cat > /etc/supervisor/conf.d/sonarr.conf << 'EOF'
[program:sonarr]
command=/opt/sonarr/Sonarr -nobrowser -data=/config/sonarr
user=mediaserver
autostart=true
autorestart=true
stdout_logfile=/var/log/sonarr.log
stderr_logfile=/var/log/sonarr.error.log
environment=HOME="/app",USER="mediaserver"
EOF

echo "Sonarr installation completed"