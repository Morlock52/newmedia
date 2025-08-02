import React from 'react';
import { useQuery } from '@tanstack/react-query';
import {
  PlayCircleIcon,
  FilmIcon,
  TvIcon,
  MusicalNoteIcon,
  PhotoIcon,
  BookOpenIcon,
  ServerIcon,
  CpuChipIcon,
  ArrowTrendingUpIcon,
  UsersIcon,
} from '@heroicons/react/24/outline';

import StatCard from '../components/StatCard';
import ActivityFeed from '../components/ActivityFeed';
import QuickActions from '../components/QuickActions';
import SystemHealth from '../components/SystemHealth';
import MediaChart from '../components/MediaChart';
import RecommendationCarousel from '../components/RecommendationCarousel';
import ActiveStreams from '../components/ActiveStreams';

const Dashboard = () => {
  const { data: stats, isLoading: statsLoading } = useQuery({
    queryKey: ['dashboard-stats'],
    queryFn: () => fetch('/api/stats/overview').then(res => res.json()),
    refetchInterval: 30000, // Refresh every 30 seconds
  });

  const { data: health, isLoading: healthLoading } = useQuery({
    queryKey: ['system-health'],
    queryFn: () => fetch('/api/system/health').then(res => res.json()),
    refetchInterval: 10000, // Refresh every 10 seconds
  });

  if (statsLoading || healthLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 p-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Welcome to Omega Media Server
          </h1>
          <p className="text-gray-600 dark:text-gray-400 mt-1">
            Your all-in-one media solution
          </p>
        </div>
        <QuickActions />
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Active Streams"
          value={stats?.activeStreams || 0}
          icon={PlayCircleIcon}
          trend={stats?.streamsTrend}
          color="blue"
        />
        <StatCard
          title="Total Media"
          value={stats?.totalMedia || 0}
          icon={ServerIcon}
          trend={stats?.mediaTrend}
          color="green"
        />
        <StatCard
          title="AI Processed"
          value={stats?.aiProcessed || 0}
          icon={CpuChipIcon}
          trend={stats?.aiTrend}
          color="purple"
        />
        <StatCard
          title="Active Users"
          value={stats?.activeUsers || 0}
          icon={UsersIcon}
          trend={stats?.usersTrend}
          color="orange"
        />
      </div>

      {/* Media Breakdown */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
          <FilmIcon className="h-8 w-8 mx-auto text-blue-500 mb-2" />
          <p className="text-2xl font-semibold">{stats?.movies || 0}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Movies</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
          <TvIcon className="h-8 w-8 mx-auto text-green-500 mb-2" />
          <p className="text-2xl font-semibold">{stats?.tvShows || 0}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">TV Shows</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
          <MusicalNoteIcon className="h-8 w-8 mx-auto text-purple-500 mb-2" />
          <p className="text-2xl font-semibold">{stats?.music || 0}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Songs</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
          <PhotoIcon className="h-8 w-8 mx-auto text-yellow-500 mb-2" />
          <p className="text-2xl font-semibold">{stats?.photos || 0}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Photos</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
          <BookOpenIcon className="h-8 w-8 mx-auto text-red-500 mb-2" />
          <p className="text-2xl font-semibold">{stats?.books || 0}</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Books</p>
        </div>
        <div className="bg-white dark:bg-gray-800 rounded-lg p-4 text-center">
          <ArrowTrendingUpIcon className="h-8 w-8 mx-auto text-indigo-500 mb-2" />
          <p className="text-2xl font-semibold">{stats?.bandwidth || '0'} Mbps</p>
          <p className="text-sm text-gray-600 dark:text-gray-400">Bandwidth</p>
        </div>
      </div>

      {/* AI Recommendations */}
      <RecommendationCarousel />

      {/* Main Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Feed */}
        <div className="lg:col-span-2">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
            <div className="p-6">
              <h2 className="text-xl font-semibold mb-4">Recent Activity</h2>
              <ActivityFeed />
            </div>
          </div>
        </div>

        {/* System Health */}
        <div>
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow">
            <div className="p-6">
              <h2 className="text-xl font-semibold mb-4">System Health</h2>
              <SystemHealth data={health} />
            </div>
          </div>
        </div>
      </div>

      {/* Active Streams */}
      <ActiveStreams />

      {/* Usage Chart */}
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <h2 className="text-xl font-semibold mb-4">Media Usage (Last 7 Days)</h2>
        <MediaChart />
      </div>
    </div>
  );
};

export default Dashboard;