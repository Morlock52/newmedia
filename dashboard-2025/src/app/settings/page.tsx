'use client';

import { useState } from 'react';
import Link from 'next/link';
import { ArrowLeft, Save, Bell, Shield, Monitor, Database } from 'lucide-react';

export default function SettingsPage() {
  const [settings, setSettings] = useState({
    notifications: true,
    autoUpdate: true,
    darkMode: true,
    apiPolling: 30
  });

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 text-white">
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center gap-4 mb-8">
          <Link href="/" className="flex items-center gap-2 text-purple-400 hover:text-purple-300">
            <ArrowLeft className="w-5 h-5" />
            Back to Dashboard
          </Link>
          <h1 className="text-4xl font-bold">Settings</h1>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Bell className="w-6 h-6 text-purple-400" />
              <h2 className="text-xl font-semibold">Notifications</h2>
            </div>
            <div className="space-y-4">
              <label className="flex items-center justify-between">
                <span>Enable notifications</span>
                <input
                  type="checkbox"
                  checked={settings.notifications}
                  onChange={(e) => setSettings({...settings, notifications: e.target.checked})}
                  className="w-5 h-5"
                />
              </label>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Shield className="w-6 h-6 text-purple-400" />
              <h2 className="text-xl font-semibold">Security</h2>
            </div>
            <div className="space-y-4">
              <button className="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg">
                Change Password
              </button>
              <button className="w-full bg-purple-600 hover:bg-purple-700 px-4 py-2 rounded-lg">
                Manage API Keys
              </button>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Monitor className="w-6 h-6 text-purple-400" />
              <h2 className="text-xl font-semibold">Display</h2>
            </div>
            <div className="space-y-4">
              <label className="flex items-center justify-between">
                <span>Dark mode</span>
                <input
                  type="checkbox"
                  checked={settings.darkMode}
                  onChange={(e) => setSettings({...settings, darkMode: e.target.checked})}
                  className="w-5 h-5"
                />
              </label>
            </div>
          </div>

          <div className="bg-white/10 backdrop-blur rounded-xl p-6">
            <div className="flex items-center gap-3 mb-4">
              <Database className="w-6 h-6 text-purple-400" />
              <h2 className="text-xl font-semibold">System</h2>
            </div>
            <div className="space-y-4">
              <label className="flex flex-col gap-2">
                <span>API Polling Interval (seconds)</span>
                <input
                  type="number"
                  value={settings.apiPolling}
                  onChange={(e) => setSettings({...settings, apiPolling: parseInt(e.target.value)})}
                  className="bg-white/10 border border-white/20 rounded px-3 py-2"
                />
              </label>
            </div>
          </div>
        </div>

        <div className="mt-8 flex justify-end">
          <button className="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg flex items-center gap-2">
            <Save className="w-5 h-5" />
            Save Settings
          </button>
        </div>
      </div>
    </div>
  );
}