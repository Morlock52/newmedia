import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import * as FileSystem from 'expo-file-system';
import { MediaItem } from './mediaSlice';

export interface OfflineMediaItem extends MediaItem {
  localPath: string;
  downloadDate: number;
  fileSize: number;
  quality: 'low' | 'medium' | 'high' | 'original';
  watchedOffline: boolean;
  lastAccessed: number;
}

export interface DownloadProgress {
  id: string;
  progress: number;
  totalSize: number;
  downloadedSize: number;
  speed: number;
  status: 'pending' | 'downloading' | 'paused' | 'completed' | 'failed' | 'cancelled';
  error?: string;
}

interface OfflineState {
  downloadedItems: OfflineMediaItem[];
  downloadQueue: DownloadProgress[];
  isDownloading: boolean;
  totalStorageUsed: number; // in bytes
  maxStorageLimit: number; // in bytes
  autoDownloadEnabled: boolean;
  downloadOnlyOnWiFi: boolean;
  downloadQuality: 'low' | 'medium' | 'high' | 'original';
  autoCleanupEnabled: boolean;
  cleanupDays: number; // Remove items not accessed for X days
  error: string | null;
}

const initialState: OfflineState = {
  downloadedItems: [],
  downloadQueue: [],
  isDownloading: false,
  totalStorageUsed: 0,
  maxStorageLimit: 5 * 1024 * 1024 * 1024, // 5GB default
  autoDownloadEnabled: false,
  downloadOnlyOnWiFi: true,
  downloadQuality: 'medium',
  autoCleanupEnabled: true,
  cleanupDays: 30,
  error: null,
};

// Async thunks
export const calculateStorageUsage = createAsyncThunk(
  'offline/calculateStorageUsage',
  async () => {
    try {
      const downloadDir = `${FileSystem.documentDirectory}downloads/`;
      const dirInfo = await FileSystem.getInfoAsync(downloadDir);
      
      if (dirInfo.exists && dirInfo.isDirectory) {
        const files = await FileSystem.readDirectoryAsync(downloadDir);
        let totalSize = 0;
        
        for (const file of files) {
          const fileInfo = await FileSystem.getInfoAsync(`${downloadDir}${file}`);
          if (fileInfo.exists && !fileInfo.isDirectory) {
            totalSize += fileInfo.size || 0;
          }
        }
        
        return totalSize;
      }
      
      return 0;
    } catch (error) {
      console.error('Error calculating storage usage:', error);
      return 0;
    }
  }
);

export const downloadMediaItem = createAsyncThunk(
  'offline/downloadMediaItem',
  async ({ 
    mediaItem, 
    quality = 'medium' 
  }: { 
    mediaItem: MediaItem; 
    quality?: 'low' | 'medium' | 'high' | 'original';
  }, { getState, dispatch, rejectWithValue }) => {
    try {
      const state = getState() as { offline: OfflineState };
      
      // Check if already downloaded
      const existing = state.offline.downloadedItems.find(item => item.id === mediaItem.id);
      if (existing) {
        throw new Error('Item already downloaded');
      }
      
      // Check storage limit
      const estimatedSize = getEstimatedSize(quality);
      if (state.offline.totalStorageUsed + estimatedSize > state.offline.maxStorageLimit) {
        throw new Error('Storage limit exceeded');
      }
      
      // Create download directory
      const downloadDir = `${FileSystem.documentDirectory}downloads/`;
      await FileSystem.makeDirectoryAsync(downloadDir, { intermediates: true });
      
      const fileName = `${mediaItem.id}_${quality}.${getFileExtension(mediaItem.type)}`;
      const localPath = `${downloadDir}${fileName}`;
      
      // Initialize download progress
      const downloadId = `download_${mediaItem.id}_${Date.now()}`;
      dispatch(addToDownloadQueue({
        id: downloadId,
        progress: 0,
        totalSize: estimatedSize,
        downloadedSize: 0,
        speed: 0,
        status: 'pending',
      }));
      
      // Start download (mock implementation)
      const downloadResult = await simulateDownload(
        mediaItem.downloadUrl || mediaItem.playUrl || '',
        localPath,
        (progress, downloadedSize, speed) => {
          dispatch(updateDownloadProgress({
            id: downloadId,
            progress,
            downloadedSize,
            speed,
            status: 'downloading',
          }));
        }
      );
      
      // Mark as completed
      dispatch(updateDownloadProgress({
        id: downloadId,
        progress: 100,
        downloadedSize: downloadResult.fileSize,
        speed: 0,
        status: 'completed',
      }));
      
      // Create offline media item
      const offlineItem: OfflineMediaItem = {
        ...mediaItem,
        localPath,
        downloadDate: Date.now(),
        fileSize: downloadResult.fileSize,
        quality,
        watchedOffline: false,
        lastAccessed: Date.now(),
      };
      
      return offlineItem;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Download failed');
    }
  }
);

export const removeDownloadedItem = createAsyncThunk(
  'offline/removeDownloadedItem',
  async (itemId: string, { getState }) => {
    try {
      const state = getState() as { offline: OfflineState };
      const item = state.offline.downloadedItems.find(item => item.id === itemId);
      
      if (item) {
        // Delete the file
        const fileInfo = await FileSystem.getInfoAsync(item.localPath);
        if (fileInfo.exists) {
          await FileSystem.deleteAsync(item.localPath);
        }
      }
      
      return itemId;
    } catch (error) {
      console.error('Error removing downloaded item:', error);
      return itemId; // Still remove from state even if file deletion fails
    }
  }
);

export const cleanupOldItems = createAsyncThunk(
  'offline/cleanupOldItems',
  async (_, { getState, dispatch }) => {
    const state = getState() as { offline: OfflineState };
    const { downloadedItems, cleanupDays } = state.offline;
    
    const cutoffDate = Date.now() - (cleanupDays * 24 * 60 * 60 * 1000);
    const itemsToRemove = downloadedItems.filter(item => 
      item.lastAccessed < cutoffDate && !item.watchedOffline
    );
    
    const removedItems = [];
    for (const item of itemsToRemove) {
      try {
        await dispatch(removeDownloadedItem(item.id));
        removedItems.push(item.id);
      } catch (error) {
        console.error('Error removing item during cleanup:', error);
      }
    }
    
    return removedItems;
  }
);

// Helper functions
function getEstimatedSize(quality: string): number {
  const sizes = {
    low: 500 * 1024 * 1024,    // 500MB
    medium: 1024 * 1024 * 1024, // 1GB
    high: 2 * 1024 * 1024 * 1024, // 2GB
    original: 4 * 1024 * 1024 * 1024, // 4GB
  };
  return sizes[quality as keyof typeof sizes] || sizes.medium;
}

function getFileExtension(type: string): string {
  switch (type) {
    case 'movie':
    case 'series':
    case 'episode':
      return 'mp4';
    case 'music':
      return 'mp3';
    default:
      return 'mp4';
  }
}

async function simulateDownload(
  url: string,
  localPath: string,
  onProgress: (progress: number, downloadedSize: number, speed: number) => void
): Promise<{ fileSize: number }> {
  // This is a mock implementation
  // In a real app, you would use FileSystem.downloadAsync with progress callback
  
  const totalSize = getEstimatedSize('medium');
  let downloadedSize = 0;
  const chunkSize = totalSize / 100;
  
  return new Promise((resolve) => {
    const interval = setInterval(() => {
      downloadedSize += chunkSize;
      const progress = Math.min((downloadedSize / totalSize) * 100, 100);
      const speed = chunkSize * 10; // Mock speed
      
      onProgress(progress, downloadedSize, speed);
      
      if (progress >= 100) {
        clearInterval(interval);
        resolve({ fileSize: totalSize });
      }
    }, 100);
  });
}

const offlineSlice = createSlice({
  name: 'offline',
  initialState,
  reducers: {
    addToDownloadQueue: (state, action: PayloadAction<DownloadProgress>) => {
      state.downloadQueue.push(action.payload);
      state.isDownloading = true;
    },
    
    updateDownloadProgress: (state, action: PayloadAction<Partial<DownloadProgress> & { id: string }>) => {
      const download = state.downloadQueue.find(d => d.id === action.payload.id);
      if (download) {
        Object.assign(download, action.payload);
        
        if (action.payload.status === 'completed' || action.payload.status === 'failed' || action.payload.status === 'cancelled') {
          // Check if any downloads are still active
          state.isDownloading = state.downloadQueue.some(d => 
            d.status === 'downloading' || d.status === 'pending'
          );
        }
      }
    },
    
    removeFromDownloadQueue: (state, action: PayloadAction<string>) => {
      state.downloadQueue = state.downloadQueue.filter(d => d.id !== action.payload);
      state.isDownloading = state.downloadQueue.some(d => 
        d.status === 'downloading' || d.status === 'pending'
      );
    },
    
    pauseDownload: (state, action: PayloadAction<string>) => {
      const download = state.downloadQueue.find(d => d.id === action.payload);
      if (download && download.status === 'downloading') {
        download.status = 'paused';
      }
    },
    
    resumeDownload: (state, action: PayloadAction<string>) => {
      const download = state.downloadQueue.find(d => d.id === action.payload);
      if (download && download.status === 'paused') {
        download.status = 'downloading';
        state.isDownloading = true;
      }
    },
    
    cancelDownload: (state, action: PayloadAction<string>) => {
      const download = state.downloadQueue.find(d => d.id === action.payload);
      if (download) {
        download.status = 'cancelled';
      }
    },
    
    markAsWatchedOffline: (state, action: PayloadAction<string>) => {
      const item = state.downloadedItems.find(item => item.id === action.payload);
      if (item) {
        item.watchedOffline = true;
        item.lastAccessed = Date.now();
      }
    },
    
    updateLastAccessed: (state, action: PayloadAction<string>) => {
      const item = state.downloadedItems.find(item => item.id === action.payload);
      if (item) {
        item.lastAccessed = Date.now();
      }
    },
    
    setMaxStorageLimit: (state, action: PayloadAction<number>) => {
      state.maxStorageLimit = action.payload;
    },
    
    setDownloadQuality: (state, action: PayloadAction<OfflineState['downloadQuality']>) => {
      state.downloadQuality = action.payload;
    },
    
    setAutoDownloadEnabled: (state, action: PayloadAction<boolean>) => {
      state.autoDownloadEnabled = action.payload;
    },
    
    setDownloadOnlyOnWiFi: (state, action: PayloadAction<boolean>) => {
      state.downloadOnlyOnWiFi = action.payload;
    },
    
    setAutoCleanupEnabled: (state, action: PayloadAction<boolean>) => {
      state.autoCleanupEnabled = action.payload;
    },
    
    setCleanupDays: (state, action: PayloadAction<number>) => {
      state.cleanupDays = action.payload;
    },
    
    clearError: (state) => {
      state.error = null;
    },
  },
  extraReducers: (builder) => {
    builder
      // Calculate storage usage
      .addCase(calculateStorageUsage.fulfilled, (state, action) => {
        state.totalStorageUsed = action.payload;
      })
      
      // Download media item
      .addCase(downloadMediaItem.fulfilled, (state, action) => {
        state.downloadedItems.push(action.payload);
        state.totalStorageUsed += action.payload.fileSize;
      })
      .addCase(downloadMediaItem.rejected, (state, action) => {
        state.error = action.payload as string;
        state.isDownloading = false;
      })
      
      // Remove downloaded item
      .addCase(removeDownloadedItem.fulfilled, (state, action) => {
        const index = state.downloadedItems.findIndex(item => item.id === action.payload);
        if (index !== -1) {
          const item = state.downloadedItems[index];
          state.totalStorageUsed -= item.fileSize;
          state.downloadedItems.splice(index, 1);
        }
      })
      
      // Cleanup old items
      .addCase(cleanupOldItems.fulfilled, (state, action) => {
        action.payload.forEach(itemId => {
          const index = state.downloadedItems.findIndex(item => item.id === itemId);
          if (index !== -1) {
            const item = state.downloadedItems[index];
            state.totalStorageUsed -= item.fileSize;
            state.downloadedItems.splice(index, 1);
          }
        });
      });
  },
});

export const {
  addToDownloadQueue,
  updateDownloadProgress,
  removeFromDownloadQueue,
  pauseDownload,
  resumeDownload,
  cancelDownload,
  markAsWatchedOffline,
  updateLastAccessed,
  setMaxStorageLimit,
  setDownloadQuality,
  setAutoDownloadEnabled,
  setDownloadOnlyOnWiFi,
  setAutoCleanupEnabled,
  setCleanupDays,
  clearError,
} = offlineSlice.actions;

export default offlineSlice.reducer;