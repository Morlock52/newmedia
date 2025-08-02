# Auto-Configuration Container Conflict Fix

## Problem
The auto-configuration scripts were failing with the error:
```
Error response from daemon: Conflict. The container name "/homarr" is already in use
```

This happened because the scripts were trying to create containers that already existed.

## Solution
I've created new scripts that intelligently handle existing containers:

### 1. **configure-media-server.sh** (Main Entry Point)
```bash
./configure-media-server.sh
```
This is the new main script that:
- Checks Docker status
- Fixes container conflicts
- Offers multiple configuration options
- Handles existing containers properly

### 2. **scripts/smart-auto-configure.sh**
The intelligent auto-configuration that:
- Checks for existing containers
- Starts only stopped containers
- Creates only missing containers
- Runs configuration on existing services

### 3. **scripts/fix-container-conflicts.sh**
A utility script that:
- Checks all container statuses
- Starts stopped containers
- Reports on container health
- Provides troubleshooting tips

## Updated Scripts
The following scripts have been updated to handle existing containers:
- `run-auto-config.sh` - Now checks for existing containers
- `scripts/quick-setup.sh` - Won't try to recreate existing containers
- `scripts/auto-configure-all-services.sh` - Handles existing services gracefully

## How to Use

### Option 1: Use the New Launcher (Recommended)
```bash
./configure-media-server.sh
```
Then choose option 1 for Smart Auto-Configure.

### Option 2: Fix Conflicts First
```bash
# First, check and fix any conflicts
./scripts/fix-container-conflicts.sh

# Then run auto-configuration
./scripts/smart-auto-configure.sh
```

### Option 3: Manual Container Management
If you still have issues:
```bash
# Check what's running
docker ps -a

# If homarr is stopped, start it
docker start homarr

# If homarr is broken, remove and recreate
docker rm homarr
docker-compose -f docker-compose-demo.yml up -d homarr
```

## What These Scripts Do

1. **Check Existing Containers**: Before creating any container, check if it already exists
2. **Start Stopped Containers**: If a container exists but is stopped, start it instead of recreating
3. **Only Create Missing**: Only create containers that don't exist
4. **Configure Running Services**: Configure services that are already running
5. **Handle Errors Gracefully**: If a container can't start, offer to recreate it

## Common Issues and Solutions

### Container Already Exists
**Problem**: "Container name already in use"
**Solution**: The scripts now check for this and use existing containers

### Container Won't Start
**Problem**: Container exists but won't start
**Solution**: Check logs with `docker logs <container_name>`, then remove and recreate if needed

### Services Not Connecting
**Problem**: Services can't connect to each other
**Solution**: Ensure all services are on the same Docker network

## Service URLs
After successful configuration:
- **Homarr Dashboard**: http://localhost:7575
- **Homepage Dashboard**: http://localhost:3001
- **Jellyfin**: http://localhost:8096
- **Prowlarr**: http://localhost:9696
- **Sonarr**: http://localhost:8989
- **Radarr**: http://localhost:7878
- **qBittorrent**: http://localhost:8090 (admin/adminadmin)

## Next Steps
1. Run `./configure-media-server.sh`
2. Choose option 1 (Smart Auto-Configure)
3. Wait for services to configure
4. Access your dashboards
5. Complete any manual setup (like Jellyfin initial config)

## Troubleshooting
If you still have issues:
1. Check Docker logs: `docker logs <container_name>`
2. Check container status: `docker ps -a`
3. Restart Docker Desktop
4. Remove all containers and start fresh: `docker-compose down && docker-compose up -d`

The scripts are now much more robust and handle existing containers properly!