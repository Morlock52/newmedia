import React, { useEffect, useState } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  TouchableOpacity,
  RefreshControl,
  Dimensions,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useDispatch, useSelector } from 'react-redux';
import { Ionicons } from '@expo/vector-icons';
import { RootState, AppDispatch } from '../store';
import {
  fetchServices,
  fetchMediaStats,
  fetchRecentMedia,
  fetchDownloadQueue,
} from '../store/slices/mediaSlice';

const { width } = Dimensions.get('window');

const DashboardScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { services, mediaStats, recentMedia, downloadQueue, loading } = useSelector(
    (state: RootState) => state.media
  );
  const { unreadCount } = useSelector((state: RootState) => state.notifications);
  
  const [refreshing, setRefreshing] = useState(false);
  const [fadeAnim] = useState(new Animated.Value(0));

  useEffect(() => {
    loadDashboardData();
    
    // Animate in
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();

    // Set up periodic refresh
    const interval = setInterval(() => {
      if (!refreshing && !loading.services) {
        loadDashboardData(true);
      }
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, []);

  const loadDashboardData = async (silent = false) => {
    if (!silent) setRefreshing(true);
    
    try {
      await Promise.all([
        dispatch(fetchServices(true)),
        dispatch(fetchMediaStats()),
        dispatch(fetchRecentMedia()),
        dispatch(fetchDownloadQueue()),
      ]);
    } catch (error) {
      console.error('Error loading dashboard data:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const onRefresh = () => {
    loadDashboardData();
  };

  const getServiceStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return '#00ff9f';
      case 'stopped':
        return '#ff6b6b';
      case 'error':
        return '#ff0080';
      default:
        return '#666699';
    }
  };

  const getServiceStatusIcon = (status: string) => {
    switch (status) {
      case 'running':
        return 'checkmark-circle';
      case 'stopped':
        return 'stop-circle';
      case 'error':
        return 'warning';
      default:
        return 'help-circle';
    }
  };

  return (
    <LinearGradient
      colors={['#0a0a0f', '#1a1a2e']}
      style={styles.container}
    >
      <ScrollView
        style={styles.scrollView}
        refreshControl={
          <RefreshControl
            refreshing={refreshing}
            onRefresh={onRefresh}
            tintColor="#00ff9f"
            colors={['#00ff9f']}
          />
        }
        showsVerticalScrollIndicator={false}
      >
        <Animated.View style={[styles.content, { opacity: fadeAnim }]}>
          {/* Header Stats */}
          <View style={styles.statsContainer}>
            <View style={styles.statCard}>
              <LinearGradient
                colors={['rgba(0, 255, 159, 0.1)', 'rgba(0, 255, 159, 0.05)']}
                style={styles.statCardGradient}
              >
                <Ionicons name="film" size={24} color="#00ff9f" />
                <Text style={styles.statNumber}>{mediaStats.movies}</Text>
                <Text style={styles.statLabel}>Movies</Text>
              </LinearGradient>
            </View>

            <View style={styles.statCard}>
              <LinearGradient
                colors={['rgba(255, 0, 128, 0.1)', 'rgba(255, 0, 128, 0.05)']}
                style={styles.statCardGradient}
              >
                <Ionicons name="tv" size={24} color="#ff0080" />
                <Text style={styles.statNumber}>{mediaStats.series}</Text>
                <Text style={styles.statLabel}>Series</Text>
              </LinearGradient>
            </View>

            <View style={styles.statCard}>
              <LinearGradient
                colors={['rgba(255, 170, 0, 0.1)', 'rgba(255, 170, 0, 0.05)']}
                style={styles.statCardGradient}
              >
                <Ionicons name="musical-notes" size={24} color="#ffaa00" />
                <Text style={styles.statNumber}>{mediaStats.tracks}</Text>
                <Text style={styles.statLabel}>Songs</Text>
              </LinearGradient>
            </View>
          </View>

          {/* Services Status */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>System Status</Text>
              <TouchableOpacity>
                <Ionicons name="settings" size={20} color="#666699" />
              </TouchableOpacity>
            </View>
            
            <View style={styles.servicesGrid}>
              {services.slice(0, 6).map((service, index) => (
                <TouchableOpacity
                  key={service.name}
                  style={styles.serviceCard}
                  activeOpacity={0.8}
                >
                  <LinearGradient
                    colors={['rgba(26, 26, 46, 0.8)', 'rgba(22, 33, 62, 0.6)']}
                    style={styles.serviceCardGradient}
                  >
                    <View style={styles.serviceHeader}>
                      <Ionicons
                        name={getServiceStatusIcon(service.status)}
                        size={16}
                        color={getServiceStatusColor(service.status)}
                      />
                      <View style={[
                        styles.statusDot,
                        { backgroundColor: getServiceStatusColor(service.status) }
                      ]} />
                    </View>
                    <Text style={styles.serviceName}>
                      {service.name.charAt(0).toUpperCase() + service.name.slice(1)}
                    </Text>
                    <Text style={styles.serviceStatus}>{service.status}</Text>
                  </LinearGradient>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* Download Queue */}
          {downloadQueue.length > 0 && (
            <View style={styles.section}>
              <View style={styles.sectionHeader}>
                <Text style={styles.sectionTitle}>Active Downloads</Text>
                <TouchableOpacity>
                  <Ionicons name="download" size={20} color="#666699" />
                </TouchableOpacity>
              </View>
              
              <ScrollView horizontal showsHorizontalScrollIndicator={false}>
                {downloadQueue.map((item, index) => (
                  <TouchableOpacity
                    key={item.id}
                    style={styles.downloadCard}
                    activeOpacity={0.8}
                  >
                    <LinearGradient
                      colors={['rgba(0, 255, 159, 0.1)', 'rgba(0, 255, 159, 0.05)']}
                      style={styles.downloadCardGradient}
                    >
                      <Text style={styles.downloadTitle}>{item.title}</Text>
                      <View style={styles.progressContainer}>
                        <View style={styles.progressBar}>
                          <View 
                            style={[
                              styles.progressFill,
                              { width: `${item.progress}%` }
                            ]} 
                          />
                        </View>
                        <Text style={styles.progressText}>{Math.round(item.progress)}%</Text>
                      </View>
                      <Text style={styles.downloadStatus}>{item.status}</Text>
                    </LinearGradient>
                  </TouchableOpacity>
                ))}
              </ScrollView>
            </View>
          )}

          {/* Recent Media */}
          <View style={styles.section}>
            <View style={styles.sectionHeader}>
              <Text style={styles.sectionTitle}>Continue Watching</Text>
              <TouchableOpacity>
                <Ionicons name="chevron-forward" size={20} color="#666699" />
              </TouchableOpacity>
            </View>
            
            <ScrollView horizontal showsHorizontalScrollIndicator={false}>
              {recentMedia.map((item, index) => (
                <TouchableOpacity
                  key={item.id}
                  style={styles.mediaCard}
                  activeOpacity={0.8}
                >
                  <LinearGradient
                    colors={['rgba(26, 26, 46, 0.8)', 'rgba(22, 33, 62, 0.6)']}
                    style={styles.mediaCardGradient}
                  >
                    <View style={styles.mediaHeader}>
                      <Ionicons
                        name={item.type === 'movie' ? 'film' : item.type === 'series' ? 'tv' : 'musical-notes'}
                        size={20}
                        color="#00ff9f"
                      />
                      <Text style={styles.mediaRating}>★ {item.rating}</Text>
                    </View>
                    <Text style={styles.mediaTitle} numberOfLines={2}>
                      {item.title}
                    </Text>
                    <Text style={styles.mediaYear}>{item.year}</Text>
                    <Text style={styles.mediaGenres}>
                      {item.genres?.slice(0, 2).join(', ')}
                    </Text>
                  </LinearGradient>
                </TouchableOpacity>
              ))}
            </ScrollView>
          </View>

          {/* Quick Actions */}
          <View style={styles.section}>
            <Text style={styles.sectionTitle}>Quick Actions</Text>
            
            <View style={styles.actionsGrid}>
              <TouchableOpacity style={styles.actionCard}>
                <LinearGradient
                  colors={['rgba(0, 255, 159, 0.15)', 'rgba(0, 255, 159, 0.05)']}
                  style={styles.actionCardGradient}
                >
                  <Ionicons name="search" size={28} color="#00ff9f" />
                  <Text style={styles.actionText}>Search</Text>
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity style={styles.actionCard}>
                <LinearGradient
                  colors={['rgba(255, 0, 128, 0.15)', 'rgba(255, 0, 128, 0.05)']}
                  style={styles.actionCardGradient}
                >
                  <Ionicons name="camera" size={28} color="#ff0080" />
                  <Text style={styles.actionText}>AR View</Text>
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity style={styles.actionCard}>
                <LinearGradient
                  colors={['rgba(255, 170, 0, 0.15)', 'rgba(255, 170, 0, 0.05)']}
                  style={styles.actionCardGradient}
                >
                  <Ionicons name="cast" size={28} color="#ffaa00" />
                  <Text style={styles.actionText}>Cast</Text>
                </LinearGradient>
              </TouchableOpacity>

              <TouchableOpacity style={styles.actionCard}>
                <LinearGradient
                  colors={['rgba(102, 102, 153, 0.15)', 'rgba(102, 102, 153, 0.05)']}
                  style={styles.actionCardGradient}
                >
                  <Ionicons name="download" size={28} color="#666699" />
                  <Text style={styles.actionText}>Offline</Text>
                </LinearGradient>
              </TouchableOpacity>
            </View>
          </View>
        </Animated.View>
      </ScrollView>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollView: {
    flex: 1,
  },
  content: {
    padding: 16,
  },
  statsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  statCard: {
    flex: 1,
    marginHorizontal: 4,
    borderRadius: 12,
    overflow: 'hidden',
    elevation: 4,
    shadowColor: '#00ff9f',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
  },
  statCardGradient: {
    padding: 16,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginTop: 8,
  },
  statLabel: {
    fontSize: 12,
    color: '#666699',
    marginTop: 4,
  },
  section: {
    marginBottom: 24,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#ffffff',
    textShadowColor: '#00ff9f',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 8,
  },
  servicesGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  serviceCard: {
    width: '48%',
    marginBottom: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  serviceCardGradient: {
    padding: 12,
    borderWidth: 1,
    borderColor: '#16213e',
  },
  serviceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.8,
    shadowRadius: 6,
  },
  serviceName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 4,
  },
  serviceStatus: {
    fontSize: 12,
    color: '#666699',
    textTransform: 'capitalize',
  },
  downloadCard: {
    width: 200,
    marginRight: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  downloadCardGradient: {
    padding: 12,
    borderWidth: 1,
    borderColor: '#00ff9f40',
  },
  downloadTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 8,
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  progressBar: {
    flex: 1,
    height: 4,
    backgroundColor: '#16213e',
    borderRadius: 2,
    marginRight: 8,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#00ff9f',
    borderRadius: 2,
  },
  progressText: {
    fontSize: 12,
    color: '#00ff9f',
    fontWeight: '600',
  },
  downloadStatus: {
    fontSize: 12,
    color: '#666699',
    textTransform: 'capitalize',
  },
  mediaCard: {
    width: 160,
    marginRight: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  mediaCardGradient: {
    padding: 12,
    borderWidth: 1,
    borderColor: '#16213e',
    height: 140,
  },
  mediaHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  mediaRating: {
    fontSize: 12,
    color: '#ffaa00',
    fontWeight: '600',
  },
  mediaTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 4,
    height: 36,
  },
  mediaYear: {
    fontSize: 12,
    color: '#666699',
    marginBottom: 4,
  },
  mediaGenres: {
    fontSize: 11,
    color: '#666699',
  },
  actionsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  actionCard: {
    width: '48%',
    marginBottom: 12,
    borderRadius: 12,
    overflow: 'hidden',
  },
  actionCardGradient: {
    padding: 20,
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#16213e',
  },
  actionText: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
    marginTop: 8,
  },
});

export default DashboardScreen;