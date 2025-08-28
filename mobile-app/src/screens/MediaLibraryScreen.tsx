import React, { useState, useEffect } from 'react';
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  TextInput,
  RefreshControl,
  Animated,
} from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { useDispatch, useSelector } from 'react-redux';
import { Ionicons } from '@expo/vector-icons';
import { RootState, AppDispatch } from '../store';
import {
  searchMedia,
  setSearchQuery,
  clearSearchResults,
  fetchRecentMedia,
} from '../store/slices/mediaSlice';

const MediaLibraryScreen: React.FC = () => {
  const dispatch = useDispatch<AppDispatch>();
  const { searchResults, recentMedia, searchQuery, loading } = useSelector(
    (state: RootState) => state.media
  );
  
  const [refreshing, setRefreshing] = useState(false);
  const [activeTab, setActiveTab] = useState<'recent' | 'movies' | 'series' | 'music'>('recent');
  const [fadeAnim] = useState(new Animated.Value(0));

  useEffect(() => {
    loadLibraryData();
    
    // Animate in
    Animated.timing(fadeAnim, {
      toValue: 1,
      duration: 800,
      useNativeDriver: true,
    }).start();
  }, []);

  const loadLibraryData = async () => {
    setRefreshing(true);
    try {
      await dispatch(fetchRecentMedia());
    } catch (error) {
      console.error('Error loading library data:', error);
    } finally {
      setRefreshing(false);
    }
  };

  const handleSearch = (query: string) => {
    dispatch(setSearchQuery(query));
    if (query.trim()) {
      dispatch(searchMedia(query));
    } else {
      dispatch(clearSearchResults());
    }
  };

  const getDisplayData = () => {
    if (searchQuery.trim()) {
      return searchResults;
    }
    
    switch (activeTab) {
      case 'recent':
        return recentMedia;
      case 'movies':
        return recentMedia.filter(item => item.type === 'movie');
      case 'series':
        return recentMedia.filter(item => item.type === 'series');
      case 'music':
        return recentMedia.filter(item => item.type === 'music');
      default:
        return recentMedia;
    }
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
        return 'help';
    }
  };

  const getMediaColor = (type: string) => {
    switch (type) {
      case 'movie':
        return '#00ff9f';
      case 'series':
        return '#ff0080';
      case 'music':
        return '#ffaa00';
      default:
        return '#666699';
    }
  };

  const renderMediaItem = ({ item }: { item: any }) => (
    <TouchableOpacity style={styles.mediaItem} activeOpacity={0.8}>
      <LinearGradient
        colors={['rgba(26, 26, 46, 0.8)', 'rgba(22, 33, 62, 0.6)']}
        style={styles.mediaItemGradient}
      >
        <View style={styles.mediaHeader}>
          <Ionicons
            name={getMediaIcon(item.type)}
            size={20}
            color={getMediaColor(item.type)}
          />
          <View style={styles.mediaRating}>
            <Ionicons name="star" size={12} color="#ffaa00" />
            <Text style={styles.ratingText}>{item.rating}</Text>
          </View>
        </View>
        
        <Text style={styles.mediaTitle} numberOfLines={2}>
          {item.title}
        </Text>
        
        <Text style={styles.mediaYear}>{item.year}</Text>
        
        <Text style={styles.mediaGenres}>
          {item.genres?.slice(0, 2).join(', ')}
        </Text>
        
        <View style={styles.mediaActions}>
          <TouchableOpacity style={styles.actionButton}>
            <Ionicons name="play" size={16} color="#00ff9f" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton}>
            <Ionicons name="download" size={16} color="#666699" />
          </TouchableOpacity>
          <TouchableOpacity style={styles.actionButton}>
            <Ionicons name="cast" size={16} color="#666699" />
          </TouchableOpacity>
        </View>
      </LinearGradient>
    </TouchableOpacity>
  );

  return (
    <LinearGradient
      colors={['#0a0a0f', '#1a1a2e']}
      style={styles.container}
    >
      <Animated.View style={[styles.content, { opacity: fadeAnim }]}>
        
        {/* Search Bar */}
        <View style={styles.searchContainer}>
          <View style={styles.searchInputContainer}>
            <Ionicons name="search" size={20} color="#666699" style={styles.searchIcon} />
            <TextInput
              style={styles.searchInput}
              placeholder="Search movies, series, music..."
              placeholderTextColor="#666699"
              value={searchQuery}
              onChangeText={handleSearch}
            />
            {searchQuery.length > 0 && (
              <TouchableOpacity
                onPress={() => handleSearch('')}
                style={styles.clearButton}
              >
                <Ionicons name="close" size={20} color="#666699" />
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* Filter Tabs */}
        <View style={styles.tabsContainer}>
          {[
            { key: 'recent', label: 'Recent', icon: 'time' },
            { key: 'movies', label: 'Movies', icon: 'film' },
            { key: 'series', label: 'Series', icon: 'tv' },
            { key: 'music', label: 'Music', icon: 'musical-notes' },
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

        {/* Media Grid */}
        <FlatList
          data={getDisplayData()}
          renderItem={renderMediaItem}
          keyExtractor={item => item.id}
          numColumns={2}
          contentContainerStyle={styles.mediaGrid}
          refreshControl={
            <RefreshControl
              refreshing={refreshing}
              onRefresh={loadLibraryData}
              tintColor="#00ff9f"
              colors={['#00ff9f']}
            />
          }
          showsVerticalScrollIndicator={false}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Ionicons
                name={searchQuery ? "search" : "library"}
                size={64}
                color="#666699"
              />
              <Text style={styles.emptyText}>
                {searchQuery ? 'No results found' : 'No media available'}
              </Text>
              <Text style={styles.emptySubtext}>
                {searchQuery
                  ? 'Try a different search term'
                  : 'Add some media to your library to get started'
                }
              </Text>
            </View>
          }
        />
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
  searchContainer: {
    marginBottom: 16,
  },
  searchInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: 'rgba(26, 26, 46, 0.8)',
    borderRadius: 12,
    paddingHorizontal: 16,
    borderWidth: 1,
    borderColor: '#16213e',
    shadowColor: '#00ff9f',
    shadowOffset: { width: 0, height: 0 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
  },
  searchIcon: {
    marginRight: 12,
  },
  searchInput: {
    flex: 1,
    height: 50,
    color: '#ffffff',
    fontSize: 16,
  },
  clearButton: {
    padding: 8,
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
  mediaGrid: {
    flexGrow: 1,
  },
  mediaItem: {
    flex: 1,
    margin: 6,
    borderRadius: 8,
    overflow: 'hidden',
    maxWidth: '48%',
  },
  mediaItemGradient: {
    padding: 12,
    borderWidth: 1,
    borderColor: '#16213e',
    height: 180,
  },
  mediaHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  mediaRating: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  ratingText: {
    fontSize: 12,
    color: '#ffaa00',
    marginLeft: 4,
    fontWeight: '600',
  },
  mediaTitle: {
    fontSize: 14,
    fontWeight: '600',
    color: '#ffffff',
    marginBottom: 8,
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
    marginBottom: 12,
    flex: 1,
  },
  mediaActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 'auto',
  },
  actionButton: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(102, 102, 153, 0.2)',
    justifyContent: 'center',
    alignItems: 'center',
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

export default MediaLibraryScreen;