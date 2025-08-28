import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Alert,
  Animated,
  Switch,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useDispatch, useSelector } from 'react-redux';
import { Ionicons } from '@expo/vector-icons';
import { RootState, AppDispatch } from '../store';
import {
  removeDownloadedItem,
  pauseDownload,
  resumeDownload,
  cancelDownload,
  calculateStorageUsage,
  setDownloadQuality,
  setDownloadOnlyOnWiFi,
  setAutoCleanupEnabled,
} from '../store/slices/offlineSlice';

const DownloadsScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const {
    downloadedItems,
    downloadQueue,
    totalStorageUsed,
    maxStorageLimit,
    downloadQuality,
    downloadOnlyOnWiFi,
    autoCleanupEnabled,
  } = useSelector((state: RootState) => state.offline);
  
  const [activeTab, setActiveTab] = useState<'downloaded' | 'queue' | 'settings'>('downloaded');
  const [fadeAnim] = useState(new Animated.Value(0));

  useEffect(() => {
    dispatch(calculateStorageUsage());
    
    // Animate in
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();
  }, []);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const handleRemoveDownload = (itemId: string) => {
    Alert.alert(
      'Remove Download',
      'Are you sure you want to remove this downloaded item?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Remove',
          style: 'destructive',
          onPress: () => dispatch(removeDownloadedItem(itemId)),
        },
      ]
    );
  };

  const handlePauseResume = (downloadId: string, status: string) => {
    if (status === 'downloading') {
      dispatch(pauseDownload(downloadId));
    } else if (status === 'paused') {
      dispatch(resumeDownload(downloadId));
    }
  };

  const handleCancelDownload = (downloadId: string) => {
    Alert.alert(
      'Cancel Download',
      'Are you sure you want to cancel this download?',
      [
        { text: 'No', style: 'cancel' },
        {
          text: 'Yes',
          style: 'destructive',
          onPress: () => dispatch(cancelDownload(downloadId)),
        },
      ]
    );
  };

  const getMediaIcon = (type: string) => {
    switch (type) {
      case 'movie':
        return 'film';
      case 'series':
        return 'tv';
      case 'music':
        return 'musical-notes';
      default:
        return 'document';
    }
  };

  const getDownloadStatusColor = (status: string) => {
    switch (status) {
      case 'downloading':
        return '#00ff9f';
      case 'paused':
        return '#ffaa00';
      case 'completed':
        return '#00ff9f';
      case 'failed':
        return '#ff6b6b';
      case 'cancelled':
        return '#666699';
      default:
        return '#666699';
    }
  };

  const renderDownloadedItem = ({ item }: { item: any }) => (
    <TouchableOpacity style={styles.downloadItem} activeOpacity={0.8}>
      <LinearGradient
        colors={['rgba(26, 26, 46, 0.8)', 'rgba(22, 33, 62, 0.6)']}
        style={styles.downloadItemGradient}
      >
        <View style={styles.downloadHeader}>
          <Ionicons
            name={getMediaIcon(item.type)}
            size={20}
            color="#00ff9f"
          />
          <View style={styles.downloadInfo}>
            <Text style={styles.downloadTitle} numberOfLines={1}>
              {item.title}
            </Text>
            <Text style={styles.downloadSize}>
              {formatFileSize(item.fileSize)} • {item.quality}
            </Text>
          </View>
          <TouchableOpacity
            style={styles.removeButton}
            onPress={() => handleRemoveDownload(item.id)}
          >
            <Ionicons name="trash" size={16} color="#ff6b6b" />
          </TouchableOpacity>
        </View>
        
        <View style={styles.downloadMeta}>
          <Text style={styles.downloadDate}>
            Downloaded {new Date(item.downloadDate).toLocaleDateString()}
          </Text>
          {item.watchedOffline && (
            <View style={styles.watchedBadge}>
              <Ionicons name="checkmark" size={12} color="#00ff9f" />
              <Text style={styles.watchedText}>Watched</Text>
            </View>
          )}
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );

  const renderQueueItem = ({ item }: { item: any }) => (
    <View style={styles.queueItem}>
      <LinearGradient
        colors={['rgba(26, 26, 46, 0.8)', 'rgba(22, 33, 62, 0.6)']}
        style={styles.queueItemGradient}
      >
        <View style={styles.queueHeader}>
          <Text style={styles.queueTitle} numberOfLines={1}>
            Download {item.id.split('_')[1]}
          </Text>
          <View style={styles.queueActions}>
            {(item.status === 'downloading' || item.status === 'paused') && (
              <TouchableOpacity
                style={styles.queueActionButton}
                onPress={() => handlePauseResume(item.id, item.status)}
              >
                <Ionicons
                  name={item.status === 'downloading' ? 'pause' : 'play'}
                  size={16}
                  color="#00ff9f"
                />
              </TouchableOpacity>
            )}
            <TouchableOpacity
              style={styles.queueActionButton}
              onPress={() => handleCancelDownload(item.id)}
            >
              <Ionicons name="close" size={16} color="#ff6b6b" />
            </TouchableOpacity>
          </View>
        </View>
        
        <View style={styles.progressContainer}>
          <View style={styles.progressBar}>
            <View
              style={[
                styles.progressFill,
                {
                  width: `${item.progress}%`,
                  backgroundColor: getDownloadStatusColor(item.status),
                },
              ]}
            />
          </View>
          <Text style={styles.progressText}>
            {Math.round(item.progress)}%
          </Text>
        </View>
        
        <View style={styles.queueMeta}>
          <Text style={styles.queueStatus}>
            {item.status.charAt(0).toUpperCase() + item.status.slice(1)}
          </Text>
          <Text style={styles.queueSize}>
            {formatFileSize(item.downloadedSize)} / {formatFileSize(item.totalSize)}
          </Text>
        </View>
      </LinearGradient>
    </View>
  );

  const renderSettings = () => (
    <View style={styles.settingsContainer}>
      
      {/* Storage Usage */}
      <View style={styles.settingsSection}>
        <Text style={styles.settingsTitle}>Storage Usage</Text>
        <View style={styles.storageCard}>
          <LinearGradient
            colors={['rgba(0, 255, 159, 0.1)', 'rgba(0, 255, 159, 0.05)']}
            style={styles.storageCardGradient}
          >
            <View style={styles.storageInfo}>
              <Text style={styles.storageUsed}>
                {formatFileSize(totalStorageUsed)}
              </Text>
              <Text style={styles.storageLimit}>
                of {formatFileSize(maxStorageLimit)} used
              </Text>
            </View>
            <View style={styles.storageBar}>
              <View
                style={[
                  styles.storageBarFill,
                  {
                    width: `${Math.min((totalStorageUsed / maxStorageLimit) * 100, 100)}%`,
                  },
                ]}
              />
            </View>
          </LinearGradient>
        </View>
      </View>

      {/* Download Quality */}
      <View style={styles.settingsSection}>
        <Text style={styles.settingsTitle}>Download Quality</Text>
        <View style={styles.qualityButtons}>
          {['low', 'medium', 'high', 'original'].map((quality) => (
            <TouchableOpacity
              key={quality}
              style={[
                styles.qualityButton,
                downloadQuality === quality && styles.qualityButtonActive,
              ]}
              onPress={() => dispatch(setDownloadQuality(quality as any))}
            >
              <Text
                style={[
                  styles.qualityButtonText,
                  downloadQuality === quality && styles.qualityButtonTextActive,
                ]}
              >
                {quality.charAt(0).toUpperCase() + quality.slice(1)}
              </Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      {/* Download Settings */}
      <View style={styles.settingsSection}>
        <Text style={styles.settingsTitle}>Download Settings</Text>
        
        <View style={styles.settingItem}>
          <View style={styles.settingInfo}>
            <Text style={styles.settingLabel}>Download only on Wi-Fi</Text>
            <Text style={styles.settingDescription}>
              Prevent downloads on cellular data
            </Text>
          </View>
          <Switch
            value={downloadOnlyOnWiFi}
            onValueChange={(value) => dispatch(setDownloadOnlyOnWiFi(value))}
            trackColor={{ false: '#16213e', true: '#00ff9f40' }}
            thumbColor={downloadOnlyOnWiFi ? '#00ff9f' : '#666699'}
          />
        </View>

        <View style={styles.settingItem}>
          <View style={styles.settingInfo}>
            <Text style={styles.settingLabel}>Auto cleanup</Text>
            <Text style={styles.settingDescription}>
              Remove old downloads automatically
            </Text>
          </View>
          <Switch
            value={autoCleanupEnabled}
            onValueChange={(value) => dispatch(setAutoCleanupEnabled(value))}
            trackColor={{ false: '#16213e', true: '#00ff9f40' }}
            thumbColor={autoCleanupEnabled ? '#00ff9f' : '#666699'}
          />
        </View>
      </View>
    </View>
  );

  return (
    <LinearGradient
      colors={['#0a0a0f', '#1a1a2e']}
      style={styles.container}
    >
      <Animated.View style={[styles.content, { opacity: fadeAnim }]}>
        
        {/* Tab Navigation */}
        <View style={styles.tabsContainer}>
          {[
            { key: 'downloaded', label: 'Downloaded', icon: 'download' },
            { key: 'queue', label: 'Queue', icon: 'list' },
            { key: 'settings', label: 'Settings', icon: 'settings' },
          ].map(tab => (
            <TouchableOpacity
              key={tab.key}
              style={[
                styles.tab,
                activeTab === tab.key && styles.activeTab,
              ]}
              onPress={() => setActiveTab(tab.key as any)}
            >
              <Ionicons
                name={tab.icon as any}
                size={16}
                color={activeTab === tab.key ? '#000000' : '#666699'}
              />
              <Text
                style={[
                  styles.tabText,
                  activeTab === tab.key && styles.activeTabText,
                ]}
              >
                {tab.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Content */}
        {activeTab === 'downloaded' && (
          <FlatList
            data={downloadedItems}
            renderItem={renderDownloadedItem}
            keyExtractor={item => item.id}
            contentContainerStyle={styles.listContainer}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <Ionicons name="download" size={64} color="#666699" />
                <Text style={styles.emptyText}>No downloads yet</Text>
                <Text style={styles.emptySubtext}>
                  Download content for offline viewing
                </Text>
              </View>
            }
          />
        )}

        {activeTab === 'queue' && (
          <FlatList
            data={downloadQueue}
            renderItem={renderQueueItem}
            keyExtractor={item => item.id}
            contentContainerStyle={styles.listContainer}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <Ionicons name="list" size={64} color="#666699" />
                <Text style={styles.emptyText}>No active downloads</Text>
                <Text style={styles.emptySubtext}>
                  Start downloading content from the library
                </Text>
              </View>
            }
          />
        )}

        {activeTab === 'settings' && renderSettings()}
      </Animated.View>
    </LinearGradient>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  content: {
    flex: 1,
    padding: 16,
  },
  tabsContainer: {
    flexDirection: 'row',
    marginBottom: 16,
    backgroundColor: 'rgba(26, 26, 46, 0.5)',
    borderRadius: 8,
    padding: 4,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
  },
  activeTab: {
    backgroundColor: '#00ff9f',
    shadowColor: '#00ff9f',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.5,
    shadowRadius: 8,
  },
  tabText: {
    fontSize: 12,
    color: '#666699',
    marginLeft: 4,
    fontWeight: '600',
  },
  activeTabText: {
    color: '#000000',
  },
  listContainer: {
    flexGrow: 1,
  },
  downloadItem: {
    marginBottom: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  downloadItemGradient: {
    padding: 16,
    borderWidth: 1,
    borderColor: '#16213e',
  },
  downloadHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  downloadInfo: {
    flex: 1,
    marginLeft: 12,
  },
  downloadTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 4,
  },
  downloadSize: {
    fontSize: 12,
    color: '#666699',
  },
  removeButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255, 107, 107, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  downloadMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  downloadDate: {
    fontSize: 12,
    color: '#666699',
  },
  watchedBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 255, 159, 0.2)',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  watchedText: {
    fontSize: 10,
    color: '#00ff9f',
    marginLeft: 4,
    fontWeight: '600',
  },
  queueItem: {
    marginBottom: 12,
    borderRadius: 8,
    overflow: 'hidden',
  },
  queueItemGradient: {
    padding: 16,
    borderWidth: 1,
    borderColor: '#16213e',
  },
  queueHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  queueTitle: {
    fontSize: 16,
    fontWeight: '600',
    color: '#ffffff',
    flex: 1,
  },
  queueActions: {
    flexDirection: 'row',
  },
  queueActionButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(102, 102, 153, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
    marginLeft: 8,
  },
  progressContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  progressBar: {
    flex: 1,
    height: 6,
    backgroundColor: '#16213e',
    borderRadius: 3,
    marginRight: 12,
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
  },
  progressText: {
    fontSize: 12,
    color: '#00ff9f',
    fontWeight: '600',
    width: 40,
    textAlign: 'right',
  },
  queueMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  queueStatus: {
    fontSize: 12,
    color: '#666699',
    textTransform: 'capitalize',
  },
  queueSize: {
    fontSize: 12,
    color: '#666699',
  },
  settingsContainer: {
    flex: 1,
  },
  settingsSection: {
    marginBottom: 24,
  },
  settingsTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
  },
  storageCard: {
    borderRadius: 8,
    overflow: 'hidden',
  },
  storageCardGradient: {
    padding: 16,
    borderWidth: 1,
    borderColor: '#00ff9f40',
  },
  storageInfo: {
    alignItems: 'center',
    marginBottom: 12,
  },
  storageUsed: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#00ff9f',
  },
  storageLimit: {
    fontSize: 14,
    color: '#666699',
    marginTop: 4,
  },
  storageBar: {
    height: 8,
    backgroundColor: '#16213e',
    borderRadius: 4,
  },
  storageBarFill: {
    height: '100%',
    backgroundColor: '#00ff9f',
    borderRadius: 4,
  },
  qualityButtons: {
    flexDirection: 'row',
    backgroundColor: 'rgba(26, 26, 46, 0.5)',
    borderRadius: 8,
    padding: 4,
  },
  qualityButton: {
    flex: 1,
    paddingVertical: 8,
    paddingHorizontal: 12,
    borderRadius: 6,
    alignItems: 'center',
  },
  qualityButtonActive: {
    backgroundColor: '#00ff9f',
  },
  qualityButtonText: {
    fontSize: 12,
    color: '#666699',
    fontWeight: '600',
  },
  qualityButtonTextActive: {
    color: '#000000',
  },
  settingItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(26, 26, 46, 0.5)',
    borderRadius: 8,
    padding: 16,
    marginBottom: 12,
  },
  settingInfo: {
    flex: 1,
  },
  settingLabel: {
    fontSize: 16,
    color: '#ffffff',
    fontWeight: '600',
    marginBottom: 4,
  },
  settingDescription: {
    fontSize: 12,
    color: '#666699',
  },
  emptyContainer: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 60,
  },
  emptyText: {
    fontSize: 18,
    color: '#666699',
    marginTop: 16,
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 14,
    color: '#666699',
    textAlign: 'center',
    opacity: 0.7,
  },
});

export default DownloadsScreen;