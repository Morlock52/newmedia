-- PostgreSQL Initialization Script - Production Media Stack
-- =========================================================
-- Creates databases and users for all media stack services

-- Create databases for media services
CREATE DATABASE authelia;
CREATE DATABASE sonarr_main;
CREATE DATABASE sonarr_log;
CREATE DATABASE radarr_main;
CREATE DATABASE radarr_log;
CREATE DATABASE prowlarr_main;
CREATE DATABASE prowlarr_log;
CREATE DATABASE lidarr_main;
CREATE DATABASE lidarr_log;
CREATE DATABASE readarr_main;
CREATE DATABASE readarr_log;
CREATE DATABASE grafana;

-- Create extension for better performance
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Grant privileges to media user
GRANT ALL PRIVILEGES ON DATABASE authelia TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE sonarr_main TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE sonarr_log TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE radarr_main TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE radarr_log TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE prowlarr_main TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE prowlarr_log TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE lidarr_main TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE lidarr_log TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE readarr_main TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE readarr_log TO mediauser;
GRANT ALL PRIVILEGES ON DATABASE grafana TO mediauser;

-- Set connection limits per database
ALTER DATABASE authelia CONNECTION LIMIT 50;
ALTER DATABASE sonarr_main CONNECTION LIMIT 25;
ALTER DATABASE radarr_main CONNECTION LIMIT 25;
ALTER DATABASE prowlarr_main CONNECTION LIMIT 15;
ALTER DATABASE lidarr_main CONNECTION LIMIT 15;
ALTER DATABASE readarr_main CONNECTION LIMIT 15;
ALTER DATABASE grafana CONNECTION LIMIT 20;