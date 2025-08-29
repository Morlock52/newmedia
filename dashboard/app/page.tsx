export default function Page() {
  return (
    <main style={{ padding: 24, fontFamily: 'ui-sans-serif, system-ui' }}>
      <h1 style={{ marginBottom: 16 }}>Media Dashboard</h1>
      <p>Quick links to services:</p>
      <ul>
        <li><a href="http://localhost:7575" target="_blank">Homarr</a></li>
        <li><a href="http://localhost:8096" target="_blank">Jellyfin</a></li>
        <li><a href="http://localhost:8989" target="_blank">Sonarr</a></li>
        <li><a href="http://localhost:7878" target="_blank">Radarr</a></li>
        <li><a href="http://localhost:9696" target="_blank">Prowlarr</a></li>
        <li><a href="http://localhost:8080" target="_blank">qBittorrent</a></li>
        <li><a href="http://localhost:3000" target="_blank">Grafana</a></li>
      </ul>
    </main>
  );
}
