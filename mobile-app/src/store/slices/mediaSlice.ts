import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import { apiService } from '../../services/apiService';

export interface MediaItem {
  id: string;
  title: string;
  type: 'movie' | 'series' | 'episode' | 'music';
  poster?: string;
  thumbnail?: string;
  description?: string;
  year?: number;
  duration?: number;
  rating?: number;
  genres?: string[];
  cast?: string[];
  director?: string;
  season?: number;
  episode?: number;
  artist?: string;
  album?: string;
  playUrl?: string;
  downloadUrl?: string;
  subtitles?: Array<{
    language: string;
    url: string;
  }>;
}

export interface Service {
  name: string;
  status: 'running' | 'stopped' | 'error' | 'unknown';
  version?: string;
  message?: string;
  error?: string;
}

export interface MediaStats {
  movies: number;
  series: number;
  episodes: number;
  artists: number;
  albums: number;
  tracks: number;
}

export interface DownloadItem {
  id: string;
  title: string;
  progress: number;
  status: 'downloading' | 'paused' | 'completed' | 'error';
  speed?: string;
  eta?: string;
}

interface MediaState {
  services: Service[];
  recentMedia: MediaItem[];
  searchResults: MediaItem[];
  currentlyPlaying: MediaItem | null;
  downloadQueue: DownloadItem[];
  mediaStats: MediaStats;
  loading: {
    services: boolean;
    media: boolean;
    search: boolean;
  };
  error: string | null;
  searchQuery: string;
  filters: {
    type: string[];
    genres: string[];
    year: [number, number] | null;
  };
}

const initialState: MediaState = {
  services: [],
  recentMedia: [],
  searchResults: [],
  currentlyPlaying: null,
  downloadQueue: [],
  mediaStats: {
    movies: 0,
    series: 0,
    episodes: 0,
    artists: 0,
    albums: 0,
    tracks: 0,
  },
  loading: {
    services: false,
    media: false,
    search: false,
  },
  error: null,
  searchQuery: '',
  filters: {
    type: [],
    genres: [],
    year: null,
  },
};

// Async thunks
export const fetchServices = createAsyncThunk(
  'media/fetchServices',
  async (force = false) => {
    const response = await apiService.get(`/services/status?force=${force}`);
    return response.data;
  }
);

export const fetchMediaStats = createAsyncThunk(
  'media/fetchMediaStats',
  async () => {
    const response = await apiService.get('/media/stats');
    return response.data;
  }
);

export const fetchRecentMedia = createAsyncThunk(
  'media/fetchRecentMedia',
  async () => {
    // Mock data for now - replace with actual API call
    return [
      {
        id: '1',
        title: 'The Matrix Reloaded',
        type: 'movie' as const,
        poster: 'https://image.tmdb.org/t/p/w500/example.jpg',
        year: 2003,
        rating: 8.5,
        genres: ['Action', 'Sci-Fi'],
      },
      {
        id: '2', 
        title: 'Breaking Bad',
        type: 'series' as const,
        poster: 'https://image.tmdb.org/t/p/w500/example2.jpg',
        year: 2008,
        rating: 9.5,
        genres: ['Drama', 'Crime'],
      },
    ];
  }
);

export const searchMedia = createAsyncThunk(
  'media/searchMedia',
  async (query: string) => {
    // Mock search - replace with actual API call
    return [
      {
        id: '3',
        title: `Search result for: ${query}`,
        type: 'movie' as const,
        poster: 'https://image.tmdb.org/t/p/w500/example3.jpg',
        year: 2024,
        rating: 7.8,
        genres: ['Action'],
      },
    ];
  }
);

export const fetchDownloadQueue = createAsyncThunk(
  'media/fetchDownloadQueue',
  async () => {
    const response = await apiService.get('/downloads/queue');
    return response.data;
  }
);

export const startService = createAsyncThunk(
  'media/startService',
  async (serviceName: string) => {
    const response = await apiService.post('/services/start', {
      services: [serviceName],
    });
    return response.data;
  }
);

export const stopService = createAsyncThunk(
  'media/stopService',
  async (serviceName: string) => {
    const response = await apiService.post('/services/stop', {
      services: [serviceName],
    });
    return response.data;
  }
);

export const restartService = createAsyncThunk(
  'media/restartService',
  async (serviceName: string) => {
    const response = await apiService.post('/services/restart', {
      services: [serviceName],
    });
    return response.data;
  }
);

const mediaSlice = createSlice({
  name: 'media',
  initialState,
  reducers: {
    setCurrentlyPlaying: (state, action: PayloadAction<MediaItem | null>) => {
      state.currentlyPlaying = action.payload;
    },
    setSearchQuery: (state, action: PayloadAction<string>) => {
      state.searchQuery = action.payload;
    },
    setFilters: (state, action: PayloadAction<Partial<typeof initialState.filters>>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    clearSearchResults: (state) => {
      state.searchResults = [];
      state.searchQuery = '';
    },
    clearError: (state) => {
      state.error = null;
    },
    updateServiceStatus: (state, action: PayloadAction<{ name: string; status: Service['status'] }>) => {
      const service = state.services.find(s => s.name === action.payload.name);
      if (service) {
        service.status = action.payload.status;
      }
    },
  },
  extraReducers: (builder) => {
    builder
      // Fetch services
      .addCase(fetchServices.pending, (state) => {
        state.loading.services = true;
        state.error = null;
      })
      .addCase(fetchServices.fulfilled, (state, action) => {
        state.loading.services = false;
        state.services = Array.isArray(action.payload) ? action.payload : [];
      })
      .addCase(fetchServices.rejected, (state, action) => {
        state.loading.services = false;
        state.error = action.error.message || 'Failed to fetch services';
      })
      
      // Fetch media stats
      .addCase(fetchMediaStats.fulfilled, (state, action) => {
        state.mediaStats = action.payload;
      })
      
      // Fetch recent media
      .addCase(fetchRecentMedia.pending, (state) => {
        state.loading.media = true;
      })
      .addCase(fetchRecentMedia.fulfilled, (state, action) => {
        state.loading.media = false;
        state.recentMedia = action.payload;
      })
      .addCase(fetchRecentMedia.rejected, (state, action) => {
        state.loading.media = false;
        state.error = action.error.message || 'Failed to fetch recent media';
      })
      
      // Search media
      .addCase(searchMedia.pending, (state) => {
        state.loading.search = true;
      })
      .addCase(searchMedia.fulfilled, (state, action) => {
        state.loading.search = false;
        state.searchResults = action.payload;
      })
      .addCase(searchMedia.rejected, (state, action) => {
        state.loading.search = false;
        state.error = action.error.message || 'Search failed';
      })
      
      // Download queue
      .addCase(fetchDownloadQueue.fulfilled, (state, action) => {
        state.downloadQueue = action.payload;
      });
  },
});

export const {
  setCurrentlyPlaying,
  setSearchQuery,
  setFilters,
  clearSearchResults,
  clearError,
  updateServiceStatus,
} = mediaSlice.actions;

export default mediaSlice.reducer;