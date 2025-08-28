import { createSlice, PayloadAction } from '@reduxjs/toolkit';

export interface ServerSettings {
  baseUrl: string;
  port: number;
  useHttps: boolean;
  apiKey?: string;
}

export interface AppSettings {
  theme: 'dark' | 'light' | 'auto';
  language: string;
  autoPlay: boolean;
  downloadQuality: 'low' | 'medium' | 'high' | 'original';
  offlineMode: boolean;
  cacheLimit: number; // in GB
  backgroundSync: boolean;
}

export interface NotificationSettings {
  enabled: boolean;
  downloadComplete: boolean;
  newContent: boolean;
  serviceAlerts: boolean;
  quiet: boolean;
  quietHours: {
    enabled: boolean;
    start: string; // HH:MM format
    end: string;   // HH:MM format
  };
}

export interface PlaybackSettings {
  defaultQuality: 'auto' | '480p' | '720p' | '1080p' | '4k';
  autoplay: boolean;
  subtitlesEnabled: boolean;
  defaultSubtitleLanguage: string;
  playbackSpeed: number;
  skipIntroEnabled: boolean;
  skipCreditsEnabled: boolean;
}

export interface SecuritySettings {
  biometricAuth: boolean;
  autoLock: boolean;
  autoLockTimeout: number; // in minutes
  requireAuthForPlayback: boolean;
  requireAuthForDownloads: boolean;
  hideContentInRecents: boolean;
}

interface SettingsState {
  server: ServerSettings;
  app: AppSettings;
  notifications: NotificationSettings;
  playback: PlaybackSettings;
  security: SecuritySettings;
  initialized: boolean;
}

const initialState: SettingsState = {
  server: {
    baseUrl: 'localhost',
    port: 3333,
    useHttps: false,
  },
  app: {
    theme: 'dark',
    language: 'en',
    autoPlay: false,
    downloadQuality: 'high',
    offlineMode: false,
    cacheLimit: 5, // 5GB default
    backgroundSync: true,
  },
  notifications: {
    enabled: true,
    downloadComplete: true,
    newContent: true,
    serviceAlerts: true,
    quiet: false,
    quietHours: {
      enabled: false,
      start: '22:00',
      end: '08:00',
    },
  },
  playback: {
    defaultQuality: 'auto',
    autoplay: false,
    subtitlesEnabled: true,
    defaultSubtitleLanguage: 'en',
    playbackSpeed: 1.0,
    skipIntroEnabled: true,
    skipCreditsEnabled: false,
  },
  security: {
    biometricAuth: false,
    autoLock: true,
    autoLockTimeout: 5, // 5 minutes
    requireAuthForPlayback: false,
    requireAuthForDownloads: true,
    hideContentInRecents: false,
  },
  initialized: false,
};

const settingsSlice = createSlice({
  name: 'settings',
  initialState,
  reducers: {
    updateServerSettings: (state, action: PayloadAction<Partial<ServerSettings>>) => {
      state.server = { ...state.server, ...action.payload };
    },
    
    updateAppSettings: (state, action: PayloadAction<Partial<AppSettings>>) => {
      state.app = { ...state.app, ...action.payload };
    },
    
    updateNotificationSettings: (state, action: PayloadAction<Partial<NotificationSettings>>) => {
      state.notifications = { ...state.notifications, ...action.payload };
    },
    
    updatePlaybackSettings: (state, action: PayloadAction<Partial<PlaybackSettings>>) => {
      state.playback = { ...state.playback, ...action.payload };
    },
    
    updateSecuritySettings: (state, action: PayloadAction<Partial<SecuritySettings>>) => {
      state.security = { ...state.security, ...action.payload };
    },
    
    resetToDefaults: (state) => {
      return { ...initialState, initialized: true };
    },
    
    setInitialized: (state, action: PayloadAction<boolean>) => {
      state.initialized = action.payload;
    },
    
    // Quick toggles for common settings
    toggleBiometricAuth: (state) => {
      state.security.biometricAuth = !state.security.biometricAuth;
    },
    
    toggleOfflineMode: (state) => {
      state.app.offlineMode = !state.app.offlineMode;
    },
    
    toggleNotifications: (state) => {
      state.notifications.enabled = !state.notifications.enabled;
    },
    
    toggleAutoPlay: (state) => {
      state.app.autoPlay = !state.app.autoPlay;
      state.playback.autoplay = !state.playback.autoplay;
    },
    
    setServerUrl: (state, action: PayloadAction<{ baseUrl: string; port: number; useHttps: boolean }>) => {
      state.server = { ...state.server, ...action.payload };
    },
    
    setTheme: (state, action: PayloadAction<'dark' | 'light' | 'auto'>) => {
      state.app.theme = action.payload;
    },
    
    setLanguage: (state, action: PayloadAction<string>) => {
      state.app.language = action.payload;
    },
    
    setCacheLimit: (state, action: PayloadAction<number>) => {
      state.app.cacheLimit = Math.max(1, Math.min(50, action.payload)); // Limit between 1-50 GB
    },
    
    setPlaybackQuality: (state, action: PayloadAction<PlaybackSettings['defaultQuality']>) => {
      state.playback.defaultQuality = action.payload;
    },
    
    setDownloadQuality: (state, action: PayloadAction<AppSettings['downloadQuality']>) => {
      state.app.downloadQuality = action.payload;
    },
    
    setAutoLockTimeout: (state, action: PayloadAction<number>) => {
      state.security.autoLockTimeout = Math.max(1, Math.min(60, action.payload)); // 1-60 minutes
    },
    
    setPlaybackSpeed: (state, action: PayloadAction<number>) => {
      state.playback.playbackSpeed = Math.max(0.25, Math.min(3.0, action.payload)); // 0.25x to 3.0x
    },
    
    setQuietHours: (state, action: PayloadAction<{ start: string; end: string }>) => {
      state.notifications.quietHours = {
        ...state.notifications.quietHours,
        ...action.payload,
      };
    },
    
    toggleQuietHours: (state) => {
      state.notifications.quietHours.enabled = !state.notifications.quietHours.enabled;
    },
  },
});

export const {
  updateServerSettings,
  updateAppSettings,
  updateNotificationSettings,
  updatePlaybackSettings,
  updateSecuritySettings,
  resetToDefaults,
  setInitialized,
  toggleBiometricAuth,
  toggleOfflineMode,
  toggleNotifications,
  toggleAutoPlay,
  setServerUrl,
  setTheme,
  setLanguage,
  setCacheLimit,
  setPlaybackQuality,
  setDownloadQuality,
  setAutoLockTimeout,
  setPlaybackSpeed,
  setQuietHours,
  toggleQuietHours,
} = settingsSlice.actions;

export default settingsSlice.reducer;