'use client';

import { useState } from 'react';

export default function DashboardPage() {
  const [services] = useState([
    { name: 'Jellyfin', url: 'http://localhost:8096', port: 8096, status: 'online', category: 'Media', description: 'Media streaming server' },
    { name: 'Sonarr', url: 'http://localhost:8989', port: 8989, status: 'online', category: 'Management', description: 'TV show management' },
    { name: 'Radarr', url: 'http://localhost:7878', port: 7878, status: 'online', category: 'Management', description: 'Movie management' },
    { name: 'Prowlarr', url: 'http://localhost:9696', port: 9696, status: 'online', category: 'Indexer', description: 'Indexer manager' },
    { name: 'Lidarr', url: 'http://localhost:8686', port: 8686, status: 'online', category: 'Management', description: 'Music management' },
    { name: 'Bazarr', url: 'http://localhost:6767', port: 6767, status: 'online', category: 'Management', description: 'Subtitle management' },
    { name: 'qBittorrent', url: 'http://localhost:8080', port: 8080, status: 'online', category: 'Download', description: 'Torrent client' },
    { name: 'Transmission', url: 'http://localhost:9091', port: 9091, status: 'online', category: 'Download', description: 'Torrent client' },
    { name: 'SABnzbd', url: 'http://localhost:8082', port: 8082, status: 'online', category: 'Download', description: 'Usenet client' },
    { name: 'Overseerr', url: 'http://localhost:5056', port: 5056, status: 'online', category: 'Request', description: 'Media requests' },
    { name: 'Jellyseerr', url: 'http://localhost:5055', port: 5055, status: 'online', category: 'Request', description: 'Media requests' },
    { name: 'Portainer', url: 'http://localhost:9000', port: 9000, status: 'online', category: 'System', description: 'Container management' },
    { name: 'Uptime Kuma', url: 'http://localhost:3001', port: 3001, status: 'online', category: 'System', description: 'Service monitoring' },
  ]);

  return (
    <main className="container">
      <header className="header header-section">
        <h1 className="glitch" data-text="Cyberpunk Media Dashboard">🎬 Cyberpunk Media Dashboard</h1>
        <p className="metric-label text-center">Manage all your media services in one place</p>
      </header>

      <div className="grid">
        {services.map((service) => (
          <div
            key={service.name}
            className={`service ${service.status === 'online' ? 'success' : 'error'}`}
            onClick={() => window.open(service.url, '_blank')}
          >
            <div className="row">
              <div className="row-left">
                <span className={`status-indicator ${service.status === 'online' ? 'status-online' : 'status-offline'}`} />
                <h3 className="m0">{service.name}</h3>
              </div>
              <span className="stat-label">Port: {service.port}</span>
            </div>
            <p className="metric-label mt-8">{service.description}</p>
          </div>
        ))}
      </div>
    </main>
  );
}
