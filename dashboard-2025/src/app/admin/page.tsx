'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Server, RefreshCw, Trash2, Power, AlertTriangle } from 'lucide-react';

export default function AdminPage() {
  const [services, setServices] = useState([
    { name: 'Jellyfin', status: 'running', port: 8096, uptime: '3d 14h' },
    { name: 'Sonarr', status: 'running', port: 8989, uptime: '3d 14h' },
    { name: 'Radarr', status: 'running', port: 7878, uptime: '3d 14h' },
    { name: 'qBittorrent', status: 'running', port: 8080, uptime: '3d 14h' },
    { name: 'Plex', status: 'stopped', port: 32400, uptime: '-' }
  ]);

  const handleServiceAction = (service: string, action: string) => {
    console.log(`${action} ${service}`);
    // Implement service control logic here
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center gap-4 mb-8">
          <Link href="/" className="flex items-center gap-2 text-purple-400 hover:text-purple-300">
            <ArrowLeft className="w-5 h-5" />
            Back to Dashboard
          </Link>
          <h1 className="text-4xl font-bold">Admin Panel</h1>
        </div>

        <div className="bg-white/10 backdrop-blur rounded-xl p-6 mb-6">
          <h2 className="text-2xl font-bold mb-4 flex items-center gap-2">
            <Server className="w-6 h-6" />
            Service Management
          </h2>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-white/20">
                  <th className="text-left py-3">Service</th>
                  <th className="text-left py-3">Status</th>
                  <th className="text-left py-3">Port</th>
                  <th className="text-left py-3">Uptime</th>
                  <th className="text-left py-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {services.map((service) => (
                  <tr key={service.name} className="border-b border-white/10">
                    <td className="py-3 font-medium">{service.name}</td>
                    <td className="py-3">
                      <span className={`inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs ${
                        service.status === 'running' 
                          ? 'bg-green-500/20 text-green-400' 
                          : 'bg-red-500/20 text-red-400'
                      }`}>
                        <span className={`w-2 h-2 rounded-full ${
                          service.status === 'running' ? 'bg-green-400' : 'bg-red-400'
                        }`} />
                        {service.status}
                      </span>
                    </td>
                    <td className="py-3">{service.port}</td>
                    <td className="py-3">{service.uptime}</td>
                    <td className="py-3">
                      <div className="flex gap-2">
                        <button
                          onClick={() => handleServiceAction(service.name, 'restart')}
                          className="p-2 bg-blue-600 hover:bg-blue-700 rounded-lg"
                          title="Restart"
                        >
                          <RefreshCw className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleServiceAction(service.name, service.status === 'running' ? 'stop' : 'start')}
                          className={`p-2 rounded-lg ${
                            service.status === 'running'
                              ? 'bg-orange-600 hover:bg-orange-700'
                              : 'bg-green-600 hover:bg-green-700'
                          }`}
                          title={service.status === 'running' ? 'Stop' : 'Start'}
                        >
                          <Power className="w-4 h-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4">System Actions</h3>
            <div className="space-y-3">
              <button className="w-full bg-purple-600 hover:bg-purple-700 px-4 py-3 rounded-lg flex items-center justify-center gap-2">
                <RefreshCw className="w-5 h-5" />
                Restart All Services
              </button>
              <button className="w-full bg-orange-600 hover:bg-orange-700 px-4 py-3 rounded-lg flex items-center justify-center gap-2">
                <AlertTriangle className="w-5 h-5" />
                Clear Cache
              </button>
              <button className="w-full bg-red-600 hover:bg-red-700 px-4 py-3 rounded-lg flex items-center justify-center gap-2">
                <Trash2 className="w-5 h-5" />
                Clean Logs
              </button>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <h3 className="text-xl font-semibold mb-4">Quick Stats</h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span>Total Services</span>
                <span className="font-bold">30</span>
              </div>
              <div className="flex justify-between">
                <span>Running</span>
                <span className="font-bold text-green-400">28</span>
              </div>
              <div className="flex justify-between">
                <span>Stopped</span>
                <span className="font-bold text-red-400">2</span>
              </div>
              <div className="flex justify-between">
                <span>System Uptime</span>
                <span className="font-bold">3d 14h 22m</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}