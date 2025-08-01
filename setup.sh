#!/bin/bash
# Media Server Automation Setup Script

set -e

echo "============================================"
echo "Media Server Automation Setup"
echo "============================================"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
   echo "This script should not be run as root for security reasons."
   echo "It will request sudo when needed."
   exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to create directory with proper permissions
create_directory() {
    local dir=$1
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        sudo mkdir -p "$dir"
        sudo chown -R $USER:$USER "$dir"
    fi
}

# Check prerequisites
echo ""
echo "Checking prerequisites..."

if ! command_exists docker; then
    echo "Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command_exists docker-compose; then
    echo "Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

if ! command_exists python3; then
    echo "Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ All prerequisites met"

# Create directory structure
echo ""
echo "Creating directory structure..."

BASE_DIR="$HOME/media-server"
create_directory "$BASE_DIR"
create_directory "$BASE_DIR/media/movies"
create_directory "$BASE_DIR/media/tv"
create_directory "$BASE_DIR/media/music"
create_directory "$BASE_DIR/media/books"
create_directory "$BASE_DIR/downloads"
create_directory "$BASE_DIR/incomplete"
create_directory "$BASE_DIR/config"
create_directory "$BASE_DIR/backups"
create_directory "$BASE_DIR/transcode"
create_directory "$BASE_DIR/logs"

# Copy files to appropriate locations
echo ""
echo "Copying configuration files..."

cp -r docker/* "$BASE_DIR/"
cp -r scripts "$BASE_DIR/"
cp -r config "$BASE_DIR/"
cp -r monitoring "$BASE_DIR/"

# Create .env file from example
if [ ! -f "$BASE_DIR/.env" ]; then
    cp "$BASE_DIR/.env.example" "$BASE_DIR/.env"
    echo ""
    echo "Created .env file. Please edit it with your settings:"
    echo "  $BASE_DIR/.env"
fi

# Install Python dependencies
echo ""
echo "Installing Python dependencies..."

# Create virtual environment
python3 -m venv "$BASE_DIR/venv"
source "$BASE_DIR/venv/bin/activate"

# Create requirements.txt
cat > "$BASE_DIR/requirements.txt" << EOF
fastapi==0.104.1
uvicorn==0.24.0
aiohttp==3.9.0
aiofiles==23.2.1
schedule==1.2.0
docker==6.1.3
psutil==5.9.6
prometheus-client==0.19.0
pydantic==2.5.0
python-multipart==0.0.6
requests==2.31.0
websockets==12.0
pyjwt==2.8.0
matplotlib==3.8.2
seaborn==0.13.0
ffmpeg-python==0.2.0
subliminal==2.1.0
PyYAML==6.0.1
EOF

pip install -r "$BASE_DIR/requirements.txt"

# Set up configuration files
echo ""
echo "Setting up configuration files..."

# Create default configuration files if they don't exist
configs=(
    "media_processing.json"
    "content_discovery.json"
    "organization.json"
    "user_experience.json"
    "maintenance.json"
    "orchestrator.json"
)

for config in "${configs[@]}"; do
    if [ ! -f "$BASE_DIR/config/$config" ]; then
        echo "Creating default $config"
        touch "$BASE_DIR/config/$config"
    fi
done

# Create systemd service file
echo ""
echo "Creating systemd service file..."

sudo cp scripts/media-orchestrator.service /etc/systemd/system/
sudo sed -i "s|/opt/media-orchestrator|$BASE_DIR|g" /etc/systemd/system/media-orchestrator.service
sudo systemctl daemon-reload

# Create start/stop scripts
echo ""
echo "Creating management scripts..."

# Start script
cat > "$BASE_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Media Server..."
docker-compose up -d
echo "Waiting for services to start..."
sleep 30
echo "Starting Orchestrator..."
sudo systemctl start media-orchestrator
echo "Media Server started!"
echo "Access the dashboard at: http://localhost:8003"
EOF

# Stop script
cat > "$BASE_DIR/stop.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "Stopping Media Server..."
sudo systemctl stop media-orchestrator
docker-compose down
echo "Media Server stopped!"
EOF

# Status script
cat > "$BASE_DIR/status.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "=== Docker Services ==="
docker-compose ps
echo ""
echo "=== Orchestrator Service ==="
sudo systemctl status media-orchestrator --no-pager
echo ""
echo "=== System Resources ==="
docker stats --no-stream
EOF

# Make scripts executable
chmod +x "$BASE_DIR/start.sh" "$BASE_DIR/stop.sh" "$BASE_DIR/status.sh"

# Create cron job for automated tasks
echo ""
echo "Setting up cron jobs..."

(crontab -l 2>/dev/null; echo "0 2 * * * cd $BASE_DIR && python3 scripts/maintenance_automation.py --backup") | crontab -

# Final instructions
echo ""
echo "============================================"
echo "Setup Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "1. Edit the configuration file: $BASE_DIR/.env"
echo "2. Add your API keys for various services"
echo "3. Configure your media paths"
echo "4. Start the services: cd $BASE_DIR && ./start.sh"
echo ""
echo "Useful commands:"
echo "  Start services: ./start.sh"
echo "  Stop services: ./stop.sh"
echo "  Check status: ./status.sh"
echo ""
echo "Web interfaces will be available at:"
echo "  Orchestrator Dashboard: http://localhost:8003"
echo "  Jellyfin: http://localhost:8096"
echo "  Radarr: http://localhost:7878"
echo "  Sonarr: http://localhost:8989"
echo "  Overseerr: http://localhost:5055"
echo "  Grafana: http://localhost:3000"
echo ""
echo "For more information, see README.md"