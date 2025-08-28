import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface MediaSource {
  id: string;
  name: string;
  type: 'movie' | 'tv' | 'music' | 'book' | 'podcast' | 'game';
  provider: string;
  status: 'connected' | 'disconnected' | 'syncing' | 'error';
  icon: string;
  apiEndpoint: string;
  lastSync: Date;
  itemCount: number;
}

interface UnifiedMediaItem {
  id: string;
  title: string;
  type: 'movie' | 'tv' | 'music' | 'book' | 'podcast' | 'game';
  provider: string;
  thumbnail: string;
  description: string;
  rating: number;
  year: number;
  genres: string[];
  metadata: Record<string, any>;
  availability: {
    streaming: boolean;
    download: boolean;
    local: boolean;
  };
}

interface SearchFilters {
  type?: string;
  provider?: string;
  genre?: string;
  year?: { min: number; max: number };
  rating?: { min: number; max: number };
  availability?: string[];
}

const UnifiedMediaAPI: React.FC = () => {
  const [mediaSources, setMediaSources] = useState<MediaSource[]>([]);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<UnifiedMediaItem[]>([]);
  const [filters, setFilters] = useState<SearchFilters>({});
  const [selectedItem, setSelectedItem] = useState<UnifiedMediaItem | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [apiStats, setApiStats] = useState({
    totalItems: 0,
    activeSources: 0,
    lastIndexed: new Date(),
    responseTime: 0
  });
  const [recentSearches, setRecentSearches] = useState<string[]>([]);
  const [favoriteItems, setFavoriteItems] = useState<string[]>([]);

  useEffect(() => {
    initializeMediaSources();
    loadRecentSearches();
    loadFavorites();
  }, []);

  const initializeMediaSources = () => {
    const sources: MediaSource[] = [
      {
        id: 'jellyfin',
        name: 'Jellyfin',
        type: 'movie',
        provider: 'Self-hosted',
        status: 'connected',
        icon: '🎦',
        apiEndpoint: '/api/jellyfin',
        lastSync: new Date(),
        itemCount: 1245
      },
      {
        id: 'plex',
        name: 'Plex',
        type: 'movie',
        provider: 'Self-hosted',
        status: 'connected',
        icon: '📺',
        apiEndpoint: '/api/plex',
        lastSync: new Date(),
        itemCount: 2890
      },
      {
        id: 'emby',
        name: 'Emby',
        type: 'movie',
        provider: 'Self-hosted',
        status: 'syncing',
        icon: '🎥',
        apiEndpoint: '/api/emby',
        lastSync: new Date(Date.now() - 300000),
        itemCount: 876
      },
      {
        id: 'lidarr',
        name: 'Lidarr Music',
        type: 'music',
        provider: 'Automation',
        status: 'connected',
        icon: '🎵',
        apiEndpoint: '/api/lidarr',
        lastSync: new Date(),
        itemCount: 15420
      },
      {
        id: 'readarr',
        name: 'Readarr Books',
        type: 'book',
        provider: 'Automation',
        status: 'error',
        icon: '📚',
        apiEndpoint: '/api/readarr',
        lastSync: new Date(Date.now() - 3600000),
        itemCount: 342
      },
      {
        id: 'tmdb',
        name: 'TMDB',
        type: 'movie',
        provider: 'External API',
        status: 'connected',
        icon: '🎨',
        apiEndpoint: '/api/tmdb',
        lastSync: new Date(),
        itemCount: 850000
      }
    ];
    
    setMediaSources(sources);
    setApiStats({
      totalItems: sources.reduce((sum, source) => sum + source.itemCount, 0),
      activeSources: sources.filter(s => s.status === 'connected').length,
      lastIndexed: new Date(),
      responseTime: Math.round(Math.random() * 200 + 50)
    });
  };

  const searchUnifiedMedia = useCallback(async (query: string, filters: SearchFilters = {}) => {
    if (!query.trim()) return;
    
    setIsLoading(true);
    const startTime = Date.now();
    
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 1000));
      
      // Mock search results
      const mockResults: UnifiedMediaItem[] = [
        {
          id: '1',
          title: `${query} - The Movie`,
          type: 'movie',
          provider: 'Jellyfin',
          thumbnail: 'https://via.placeholder.com/200x300/0a0a0a/00ffff',
          description: 'A cyberpunk thriller set in the near future...',
          rating: 8.5,
          year: 2025,
          genres: ['Sci-Fi', 'Thriller', 'Action'],
          metadata: { director: 'Neural Network', runtime: 145 },
          availability: { streaming: true, download: false, local: true }
        },
        {
          id: '2',
          title: `${query} - Extended Series`,
          type: 'tv',
          provider: 'Plex',
          thumbnail: 'https://via.placeholder.com/200x300/1a1a1a/ff00ff',
          description: 'An epic series spanning multiple seasons...',
          rating: 9.2,
          year: 2024,
          genres: ['Drama', 'Sci-Fi', 'Mystery'],
          metadata: { seasons: 3, episodes: 45 },
          availability: { streaming: true, download: true, local: true }
        },
        {
          id: '3',
          title: `${query} - Original Soundtrack`,
          type: 'music',
          provider: 'Lidarr',
          thumbnail: 'https://via.placeholder.com/200x200/2a2a2a/ffff00',
          description: 'Electronic synthwave album',
          rating: 7.8,
          year: 2025,
          genres: ['Synthwave', 'Electronic', 'Ambient'],
          metadata: { artist: 'Cyber Composer', tracks: 12 },
          availability: { streaming: true, download: true, local: false }
        }
      ];
      
      setSearchResults(mockResults);
      setApiStats(prev => ({ ...prev, responseTime: Date.now() - startTime }));
      
      // Update recent searches
      setRecentSearches(prev => {
        const updated = [query, ...prev.filter(s => s !== query)].slice(0, 5);
        localStorage.setItem('recentSearches', JSON.stringify(updated));
        return updated;
      });
    } catch (error) {
      console.error('Search failed:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadRecentSearches = () => {
    const saved = localStorage.getItem('recentSearches');
    if (saved) {
      setRecentSearches(JSON.parse(saved));
    }
  };

  const loadFavorites = () => {
    const saved = localStorage.getItem('favoriteItems');
    if (saved) {
      setFavoriteItems(JSON.parse(saved));
    }
  };

  const toggleFavorite = (itemId: string) => {
    setFavoriteItems(prev => {
      const updated = prev.includes(itemId) 
        ? prev.filter(id => id !== itemId)
        : [...prev, itemId];
      localStorage.setItem('favoriteItems', JSON.stringify(updated));
      return updated;
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'connected': return '#00FF00';
      case 'disconnected': return '#FF0040';
      case 'syncing': return '#FFFF00';
      case 'error': return '#FF00FF';
      default: return '#666666';
    }
  };

  const getTypeIcon = (type: string) => {
    switch (type) {
      case 'movie': return '🎥';
      case 'tv': return '📺';
      case 'music': return '🎵';
      case 'book': return '📚';
      case 'podcast': return '🎧';
      case 'game': return '🎮';
      default: return '📁';
    }
  };

  return (
    <div 
      style={{
        background: 'linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #16213e 100%)',
        color: '#00FFFF',
        fontFamily: 'Orbitron, monospace',
        minHeight: '100vh',
        padding: '20px'
      }}
    >
      {/* Header */}
      <motion.header
        initial={{ y: -30, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        style={{
          marginBottom: '30px',
          textAlign: 'center',
          position: 'relative'
        }}
      >
        <h1 style={{
          fontSize: '3rem',
          margin: 0,
          background: 'linear-gradient(45deg, #00FFFF, #FF00FF, #FFFF00)',
          backgroundClip: 'text',
          WebkitBackgroundClip: 'text',
          WebkitTextFillColor: 'transparent',
          textShadow: '0 0 20px #00FFFF',
          animation: 'glitch 3s infinite'
        }}>
          UNIFIED MEDIA API
        </h1>
        <p style={{ margin: '10px 0', opacity: 0.8 }}>
          Cross-platform media aggregation and search engine
        </p>
      </motion.header>

      {/* API Statistics */}
      <motion.section
        initial={{ opacity: 0, scale: 0.9 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ delay: 0.2 }}
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
          gap: '15px',
          marginBottom: '30px'
        }}
      >
        {[
          { label: 'Total Items', value: apiStats.totalItems.toLocaleString(), color: '#00FFFF' },
          { label: 'Active Sources', value: `${apiStats.activeSources}/${mediaSources.length}`, color: '#00FF00' },
          { label: 'Response Time', value: `${apiStats.responseTime}ms`, color: '#FFFF00' },
          { label: 'Last Indexed', value: apiStats.lastIndexed.toLocaleTimeString(), color: '#FF00FF' }
        ].map((stat, index) => (
          <div
            key={stat.label}
            style={{
              background: 'rgba(0,0,0,0.7)',
              border: `2px solid ${stat.color}`,
              borderRadius: '12px',
              padding: '20px',
              textAlign: 'center',
              boxShadow: `0 0 20px rgba(${stat.color === '#00FFFF' ? '0,255,255' : stat.color === '#00FF00' ? '0,255,0' : stat.color === '#FFFF00' ? '255,255,0' : '255,0,255'},0.3)`
            }}
          >
            <div style={{ fontSize: '2rem', color: stat.color, fontWeight: 'bold' }}>{stat.value}</div>
            <div style={{ fontSize: '0.9rem', opacity: 0.8 }}>{stat.label}</div>
          </div>
        ))}
      </motion.section>

      {/* Media Sources */}
      <motion.section
        initial={{ opacity: 0, x: -20 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ delay: 0.3 }}
        style={{ marginBottom: '30px' }}
      >
        <h2 style={{ color: '#FFFF00', marginBottom: '20px' }}>CONNECTED SOURCES</h2>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))',
          gap: '15px'
        }}>
          {mediaSources.map((source, index) => (
            <motion.div
              key={source.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 + index * 0.1 }}
              style={{
                background: 'rgba(0,0,0,0.7)',
                border: `2px solid ${getStatusColor(source.status)}`,
                borderRadius: '12px',
                padding: '15px',
                position: 'relative',
                overflow: 'hidden'
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
                  <span style={{ fontSize: '2rem', marginRight: '12px' }}>{source.icon}</span>
                  <div>
                    <h3 style={{ margin: 0, color: '#00FFFF' }}>{source.name}</h3>
                    <p style={{ margin: '4px 0', fontSize: '0.8rem', opacity: 0.7 }}>{source.provider}</p>
                    <p style={{ margin: 0, fontSize: '0.9rem', color: getStatusColor(source.status), textTransform: 'uppercase', fontWeight: 'bold' }}>
                      {source.status}
                    </p>
                  </div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '1.2rem', fontWeight: 'bold', color: '#FFFF00' }}>
                    {source.itemCount.toLocaleString()}
                  </div>
                  <div style={{ fontSize: '0.8rem', opacity: 0.7 }}>items</div>
                </div>
              </div>
              
              {source.status === 'syncing' && (
                <div style={{
                  position: 'absolute',
                  bottom: 0,
                  left: 0,
                  right: 0,
                  height: '4px',
                  background: 'linear-gradient(90deg, #FFFF00, #FF00FF)',
                  animation: 'loading 2s infinite'
                }}
                />
              )}
            </motion.div>
          ))}
        </div>
      </motion.section>

      {/* Search Interface */}
      <motion.section
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        style={{
          background: 'rgba(0,0,0,0.7)',
          border: '2px solid #00FFFF',
          borderRadius: '12px',
          padding: '30px',
          marginBottom: '30px'
        }}
      >
        <h2 style={{ color: '#00FFFF', marginBottom: '20px' }}>UNIVERSAL SEARCH</h2>
        
        <div style={{ display: 'flex', gap: '10px', marginBottom: '20px' }}>
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search across all media sources..."
            onKeyPress={(e) => e.key === 'Enter' && searchUnifiedMedia(searchQuery, filters)}
            style={{
              flex: 1,
              padding: '15px',
              background: 'rgba(0,0,0,0.8)',
              border: '2px solid #FF00FF',
              borderRadius: '8px',
              color: '#00FFFF',
              fontSize: '1.1rem',
              fontFamily: 'inherit'
            }}
          />
          <button
            onClick={() => searchUnifiedMedia(searchQuery, filters)}
            disabled={isLoading || !searchQuery.trim()}
            style={{
              padding: '15px 30px',
              background: 'linear-gradient(45deg, #FF00FF, #FFFF00)',
              border: 'none',
              borderRadius: '8px',
              color: '#000',
              fontWeight: 'bold',
              cursor: 'pointer',
              fontSize: '1.1rem'
            }}
          >
            {isLoading ? 'SEARCHING...' : 'SEARCH'}
          </button>
        </div>
        
        {/* Recent Searches */}
        {recentSearches.length > 0 && (
          <div style={{ marginBottom: '20px' }}>
            <h4 style={{ color: '#FFFF00', marginBottom: '10px' }}>Recent Searches:</h4>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
              {recentSearches.map((search, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setSearchQuery(search);
                    searchUnifiedMedia(search, filters);
                  }}
                  style={{
                    padding: '6px 12px',
                    background: 'rgba(255,255,0,0.2)',
                    border: '1px solid #FFFF00',
                    borderRadius: '16px',
                    color: '#FFFF00',
                    fontSize: '0.9rem',
                    cursor: 'pointer'
                  }}
                >
                  {search}
                </button>
              ))}
            </div>
          </div>
        )}
      </motion.section>

      {/* Search Results */}
      <AnimatePresence>
        {searchResults.length > 0 && (
          <motion.section
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -30 }}
          >
            <h2 style={{ color: '#FF00FF', marginBottom: '20px' }}>SEARCH RESULTS ({searchResults.length})</h2>
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
              gap: '20px'
            }}>
              {searchResults.map((item, index) => (
                <motion.div
                  key={item.id}
                  initial={{ opacity: 0, scale: 0.8 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: index * 0.1 }}
                  whileHover={{ scale: 1.05 }}
                  style={{
                    background: 'rgba(0,0,0,0.8)',
                    border: '2px solid #FF00FF',
                    borderRadius: '12px',
                    padding: '20px',
                    cursor: 'pointer',
                    position: 'relative'
                  }}
                  onClick={() => setSelectedItem(item)}
                >
                  <div style={{ display: 'flex', gap: '15px' }}>
                    <img
                      src={item.thumbnail}
                      alt={item.title}
                      style={{
                        width: '80px',
                        height: '120px',
                        objectFit: 'cover',
                        borderRadius: '8px',
                        border: '1px solid #666'
                      }}
                    />
                    <div style={{ flex: 1 }}>
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                        <h3 style={{ margin: 0, color: '#00FFFF', fontSize: '1.1rem' }}>{item.title}</h3>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleFavorite(item.id);
                          }}
                          style={{
                            background: 'none',
                            border: 'none',
                            color: favoriteItems.includes(item.id) ? '#FF00FF' : '#666',
                            fontSize: '1.5rem',
                            cursor: 'pointer'
                          }}
                        >
                          ♥
                        </button>
                      </div>
                      
                      <div style={{ display: 'flex', alignItems: 'center', gap: '10px', margin: '5px 0' }}>
                        <span style={{ fontSize: '1.2rem' }}>{getTypeIcon(item.type)}</span>
                        <span style={{ color: '#FFFF00', fontSize: '0.9rem' }}>{item.provider}</span>
                        <span style={{ color: '#00FF00', fontSize: '0.9rem' }}>★ {item.rating}/10</span>
                        <span style={{ color: '#FF00FF', fontSize: '0.9rem' }}>{item.year}</span>
                      </div>
                      
                      <p style={{ margin: '10px 0', fontSize: '0.9rem', opacity: 0.8, lineHeight: 1.4 }}>
                        {item.description.substring(0, 100)}...
                      </p>
                      
                      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '5px', marginBottom: '10px' }}>
                        {item.genres.slice(0, 3).map(genre => (
                          <span key={genre} style={{
                            padding: '2px 8px',
                            background: 'rgba(0,255,255,0.2)',
                            border: '1px solid #00FFFF',
                            borderRadius: '10px',
                            fontSize: '0.8rem'
                          }}>
                            {genre}
                          </span>
                        ))}
                      </div>
                      
                      <div style={{ display: 'flex', gap: '8px' }}>
                        {item.availability.streaming && <span style={{ color: '#00FF00', fontSize: '0.8rem' }}>● Stream</span>}
                        {item.availability.download && <span style={{ color: '#FFFF00', fontSize: '0.8rem' }}>● Download</span>}
                        {item.availability.local && <span style={{ color: '#FF00FF', fontSize: '0.8rem' }}>● Local</span>}
                      </div>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </motion.section>
        )}
      </AnimatePresence>

      <style jsx>{`
        @keyframes glitch {
          0%, 100% { filter: hue-rotate(0deg); }
          25% { filter: hue-rotate(90deg); }
          50% { filter: hue-rotate(180deg); }
          75% { filter: hue-rotate(270deg); }
        }
        
        @keyframes loading {
          0% { transform: translateX(-100%); }
          100% { transform: translateX(100%); }
        }
      `}</style>
    </div>
  );
};

export default UnifiedMediaAPI;