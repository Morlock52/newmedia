'use client';

import { useEffect, useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, TrendingUp, Users, Database, Activity } from 'lucide-react';

export default function AnalyticsPage() {
  const [metrics, setMetrics] = useState({
    totalViews: 15234,
    activeUsers: 42,
    storageUsed: '2.4 TB',
    bandwidthUsed: '854 GB'
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center gap-4 mb-8">
          <Link href="/" className="flex items-center gap-2 text-purple-400 hover:text-purple-300">
            <ArrowLeft className="w-5 h-5" />
            Back to Dashboard
          </Link>
          <h1 className="text-4xl font-bold">Analytics</h1>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <TrendingUp className="w-8 h-8 text-green-400" />
              <span className="text-2xl font-bold">{metrics.totalViews.toLocaleString()}</span>
            </div>
            <h3 className="text-gray-300">Total Views</h3>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <Users className="w-8 h-8 text-blue-400" />
              <span className="text-2xl font-bold">{metrics.activeUsers}</span>
            </div>
            <h3 className="text-gray-300">Active Users</h3>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <Database className="w-8 h-8 text-purple-400" />
              <span className="text-2xl font-bold">{metrics.storageUsed}</span>
            </div>
            <h3 className="text-gray-300">Storage Used</h3>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <Activity className="w-8 h-8 text-orange-400" />
              <span className="text-2xl font-bold">{metrics.bandwidthUsed}</span>
            </div>
            <h3 className="text-gray-300">Bandwidth Used</h3>
          </div>
        </div>

        <div className="bg-white/10 backdrop-blur rounded-xl p-8">
          <h2 className="text-2xl font-bold mb-6">Usage Trends</h2>
          <div className="h-64 flex items-center justify-center text-gray-400">
            <p>Interactive charts will be displayed here</p>
          </div>
        </div>
      </div>
    </div>
  );
}