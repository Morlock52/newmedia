# Deployment Guide

This guide provides step-by-step instructions on how to deploy the Ultimate Media Server stack. There are two ways to deploy the stack: using the interactive deployment script (recommended) or deploying manually.

## Prerequisites

Before you begin, make sure you have the following installed on your system:

*   **Git:** For cloning the repository.
*   **Docker and Docker Compose:** For running the services. You can download Docker Desktop from the [official website](https://www.docker.com/products/docker-desktop).

## Automated Deployment (Recommended)

The easiest way to deploy the stack is to use the interactive deployment script. This script will guide you through the process of setting up the environment, creating the required directories and configuration files, and starting the services.

### Step 1: Clone the Repository

Open a terminal and clone the repository to your local machine:

```bash
git clone <repository-url>
cd newmedia
```

### Step 2: Run the Deployment Script

Run the `deploy.sh` script from the root of the project:

```bash
./deploy.sh
```

The script will perform the following actions:

1.  **Check for Docker:** It will check if Docker is installed and running on your system.
2.  **Create Directories:** It will create the required directory structure for the services and their configurations.
3.  **Set Up Environment:** It will create a `.env` file from the `.env.example` template and prompt you to edit it with your own values.
4.  **Create Configurations:** It will create the necessary configuration files for Prometheus and Homepage.
5.  **Deploy Services:** It will pull the latest Docker images and start all the services in the stack.
6.  **Health Check:** It will perform a health check to ensure that the core services are running correctly.

### Step 3: Post-Deployment Configuration

Once the script has finished, you will need to perform the following post-deployment configuration steps:

1.  **Configure Prowlarr:** Open Prowlarr at `http://localhost:9696` and add your indexers.
2.  **Connect *ARR Apps:** Connect Sonarr, Radarr, Lidarr, and Readarr to Prowlarr.
3.  **Configure Download Clients:** Configure qBittorrent or SABnzbd with your download settings.
4.  **Set Up Media Servers:** Set up your media libraries in Jellyfin, Plex, or Emby.

## Manual Deployment

If you prefer to deploy the stack manually, you can use `docker-compose`.

### Step 1: Clone the Repository

Open a terminal and clone the repository to your local machine:

```bash
git clone <repository-url>
cd newmedia
```

### Step 2: Create the `.env` File

Copy the `.env.example` file to `.env`:

```bash
cp .env.example .env
```

Then, edit the `.env` file with your own values. You will need to configure the following:

*   Timezone (`TZ`)
*   Media paths (`MEDIA_PATH`, `DOWNLOADS_PATH`, `USENET_PATH`)
*   VPN credentials (`VPN_PROVIDER`, `VPN_PRIVATE_KEY`, `VPN_ADDRESSES`)
*   Cloudflare credentials (`CLOUDFLARE_EMAIL`, `CLOUDFLARE_API_KEY`)
*   Service passwords (`GRAFANA_PASSWORD`)
*   Domain name (`DOMAIN`)

### Step 3: Start the Services

Start the services using `docker-compose`:

```bash
docker-compose up -d
```

This will start all the services in the stack in detached mode.

### Step 4: Post-Deployment Configuration

Once the services are running, you will need to perform the same post-deployment configuration steps as described in the automated deployment section.
