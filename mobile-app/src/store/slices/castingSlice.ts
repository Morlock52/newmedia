import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';

export interface CastDevice {
  id: string;
  name: string;
  type: 'chromecast' | 'airplay' | 'dlna';
  status: 'available' | 'connecting' | 'connected' | 'unavailable';
  capabilities: string[];
  volume?: number;
  muted?: boolean;
}

export interface CastSession {
  id: string;
  deviceId: string;
  deviceName: string;
  type: 'chromecast' | 'airplay' | 'dlna';
  status: 'idle' | 'playing' | 'paused' | 'buffering' | 'ended' | 'error';
  mediaInfo?: {
    title: string;
    subtitle?: string;
    imageUrl?: string;
    contentUrl: string;
    contentType: string;
    duration?: number;
    currentTime?: number;
  };
  volume: number;
  muted: boolean;
}

interface CastingState {
  devices: CastDevice[];
  availableDevices: CastDevice[];
  currentSession: CastSession | null;
  isScanning: boolean;
  isConnecting: boolean;
  isPlaying: boolean;
  error: string | null;
  castingEnabled: boolean;
  autoDiscovery: boolean;
}

const initialState: CastingState = {
  devices: [],
  availableDevices: [],
  currentSession: null,
  isScanning: false,
  isConnecting: false,
  isPlaying: false,
  error: null,
  castingEnabled: true,
  autoDiscovery: true,
};

// Async thunks
export const initializeCasting = createAsyncThunk(
  'casting/initialize',
  async (_, { rejectWithValue }) => {
    try {
      // Initialize Google Cast SDK
      // Note: This would use react-native-google-cast in a real implementation
      console.log('Initializing casting capabilities...');
      
      // For AirPlay on iOS, it's handled automatically by the system
      // For Chromecast, we need to initialize the SDK
      
      return true;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Casting initialization failed');
    }
  }
);

export const scanForDevices = createAsyncThunk(
  'casting/scanForDevices',
  async (_, { rejectWithValue }) => {
    try {
      // Mock device discovery - replace with actual implementation
      const mockDevices: CastDevice[] = [
        {
          id: 'chromecast_1',
          name: 'Living Room TV',
          type: 'chromecast',
          status: 'available',
          capabilities: ['video', 'audio'],
        },
        {
          id: 'airplay_1',
          name: 'Apple TV',
          type: 'airplay',
          status: 'available',
          capabilities: ['video', 'audio', 'mirroring'],
        },
        {
          id: 'dlna_1',
          name: 'Smart TV',
          type: 'dlna',
          status: 'available',
          capabilities: ['video', 'audio'],
        },
      ];
      
      // Simulate scan delay
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      return mockDevices;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Device scan failed');
    }
  }
);

export const connectToDevice = createAsyncThunk(
  'casting/connectToDevice',
  async (deviceId: string, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { casting: CastingState };
      const device = state.casting.availableDevices.find(d => d.id === deviceId);
      
      if (!device) {
        throw new Error('Device not found');
      }
      
      // Mock connection - replace with actual implementation
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const session: CastSession = {
        id: `session_${Date.now()}`,
        deviceId: device.id,
        deviceName: device.name,
        type: device.type,
        status: 'idle',
        volume: 50,
        muted: false,
      };
      
      return session;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Connection failed');
    }
  }
);

export const disconnectFromDevice = createAsyncThunk(
  'casting/disconnectFromDevice',
  async () => {
    // Mock disconnection
    await new Promise(resolve => setTimeout(resolve, 500));
    return true;
  }
);

export const castMedia = createAsyncThunk(
  'casting/castMedia',
  async ({ 
    mediaUrl, 
    title, 
    subtitle, 
    imageUrl, 
    contentType 
  }: { 
    mediaUrl: string; 
    title: string; 
    subtitle?: string; 
    imageUrl?: string; 
    contentType: string; 
  }, { getState, rejectWithValue }) => {
    try {
      const state = getState() as { casting: CastingState };
      
      if (!state.casting.currentSession) {
        throw new Error('No active cast session');
      }
      
      // Mock media casting
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const mediaInfo = {
        title,
        subtitle,
        imageUrl,
        contentUrl: mediaUrl,
        contentType,
        duration: 7200, // Mock duration
        currentTime: 0,
      };
      
      return mediaInfo;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Media casting failed');
    }
  }
);

export const controlPlayback = createAsyncThunk(
  'casting/controlPlayback',
  async (action: 'play' | 'pause' | 'stop' | 'seek', { rejectWithValue }) => {
    try {
      // Mock playback control
      await new Promise(resolve => setTimeout(resolve, 300));
      return action;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Playback control failed');
    }
  }
);

export const setVolume = createAsyncThunk(
  'casting/setVolume',
  async ({ volume, muted }: { volume?: number; muted?: boolean }, { rejectWithValue }) => {
    try {
      // Mock volume control
      await new Promise(resolve => setTimeout(resolve, 200));
      return { volume, muted };
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Volume control failed');
    }
  }
);

const castingSlice = createSlice({
  name: 'casting',
  initialState,
  reducers: {
    clearError: (state) => {
      state.error = null;
    },
    
    setCastingEnabled: (state, action: PayloadAction<boolean>) => {
      state.castingEnabled = action.payload;
    },
    
    setAutoDiscovery: (state, action: PayloadAction<boolean>) => {
      state.autoDiscovery = action.payload;
    },
    
    updateDeviceStatus: (state, action: PayloadAction<{ deviceId: string; status: CastDevice['status'] }>) => {
      const device = state.availableDevices.find(d => d.id === action.payload.deviceId);
      if (device) {
        device.status = action.payload.status;
      }
    },
    
    updateSessionStatus: (state, action: PayloadAction<CastSession['status']>) => {
      if (state.currentSession) {
        state.currentSession.status = action.payload;
        state.isPlaying = action.payload === 'playing';
      }
    },
    
    updateMediaTime: (state, action: PayloadAction<number>) => {
      if (state.currentSession?.mediaInfo) {
        state.currentSession.mediaInfo.currentTime = action.payload;
      }
    },
    
    // Handle session events from native modules
    handleSessionEvent: (state, action: PayloadAction<{ 
      type: 'connected' | 'disconnected' | 'suspended' | 'resumed' | 'ended' | 'error';
      data?: any;
    }>) => {
      const { type, data } = action.payload;
      
      switch (type) {
        case 'connected':
          state.isConnecting = false;
          if (state.currentSession) {
            state.currentSession.status = 'idle';
          }
          break;
          
        case 'disconnected':
        case 'ended':
          state.currentSession = null;
          state.isPlaying = false;
          state.isConnecting = false;
          break;
          
        case 'error':
          state.error = data?.message || 'Cast session error';
          state.isConnecting = false;
          state.isPlaying = false;
          break;
      }
    },
  },
  extraReducers: (builder) => {
    builder
      // Initialize casting
      .addCase(initializeCasting.fulfilled, (state) => {
        state.castingEnabled = true;
      })
      .addCase(initializeCasting.rejected, (state, action) => {
        state.error = action.payload as string;
        state.castingEnabled = false;
      })
      
      // Scan for devices
      .addCase(scanForDevices.pending, (state) => {
        state.isScanning = true;
        state.error = null;
      })
      .addCase(scanForDevices.fulfilled, (state, action) => {
        state.isScanning = false;
        state.availableDevices = action.payload;
      })
      .addCase(scanForDevices.rejected, (state, action) => {
        state.isScanning = false;
        state.error = action.payload as string;
      })
      
      // Connect to device
      .addCase(connectToDevice.pending, (state) => {
        state.isConnecting = true;
        state.error = null;
      })
      .addCase(connectToDevice.fulfilled, (state, action) => {
        state.isConnecting = false;
        state.currentSession = action.payload;
      })
      .addCase(connectToDevice.rejected, (state, action) => {
        state.isConnecting = false;
        state.error = action.payload as string;
      })
      
      // Disconnect from device
      .addCase(disconnectFromDevice.fulfilled, (state) => {
        state.currentSession = null;
        state.isPlaying = false;
      })
      
      // Cast media
      .addCase(castMedia.fulfilled, (state, action) => {
        if (state.currentSession) {
          state.currentSession.mediaInfo = action.payload;
          state.currentSession.status = 'playing';
          state.isPlaying = true;
        }
      })
      .addCase(castMedia.rejected, (state, action) => {
        state.error = action.payload as string;
      })
      
      // Control playback
      .addCase(controlPlayback.fulfilled, (state, action) => {
        if (state.currentSession) {
          switch (action.payload) {
            case 'play':
              state.currentSession.status = 'playing';
              state.isPlaying = true;
              break;
            case 'pause':
              state.currentSession.status = 'paused';
              state.isPlaying = false;
              break;
            case 'stop':
              state.currentSession.status = 'idle';
              state.currentSession.mediaInfo = undefined;
              state.isPlaying = false;
              break;
          }
        }
      })
      
      // Set volume
      .addCase(setVolume.fulfilled, (state, action) => {
        if (state.currentSession) {
          if (action.payload.volume !== undefined) {
            state.currentSession.volume = action.payload.volume;
          }
          if (action.payload.muted !== undefined) {
            state.currentSession.muted = action.payload.muted;
          }
        }
      });
  },
});

export const {
  clearError,
  setCastingEnabled,
  setAutoDiscovery,
  updateDeviceStatus,
  updateSessionStatus,
  updateMediaTime,
  handleSessionEvent,
} = castingSlice.actions;

export default castingSlice.reducer;