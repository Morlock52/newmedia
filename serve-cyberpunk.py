#!/usr/bin/env python3
"""
Cyberpunk Media Hub Server
Serves the dashboard with proper CORS headers
"""

import http.server
import socketserver
import os

PORT = 8000

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == "__main__":
    os.chdir('/Users/morlock/fun/newmedia')
    
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"""
╔═══════════════════════════════════════════════════════════════╗
║                      NEXUS MEDIA HUB                          ║
║                   CYBERPUNK EDITION 2025                      ║
╠═══════════════════════════════════════════════════════════════╣
║  Dashboard:  http://localhost:{PORT}/cyberpunk-media-hub.html    ║
║  Status:     ONLINE                                           ║
║  Services:   30 CONNECTED                                     ║
║  AI:         ACTIVE                                           ║
╚═══════════════════════════════════════════════════════════════╝
        """)
        print(f"Serving at http://localhost:{PORT}")
        httpd.serve_forever()