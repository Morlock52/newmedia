# Media Server Remote Control - React Native Mobile App

A cutting-edge React Native mobile application for controlling your media server with cyberpunk aesthetics and advanced features.

## 🚀 Features

### 🔐 **Biometric Authentication**
- Face ID & Touch ID support
- Secure credential storage
- Fallback to device passcode

### 📱 **Push Notifications**
- Download completion alerts
- New content notifications
- Service status updates
- Cast session notifications

### 🎮 **Remote Control**
- Media playback controls
- Volume and seek controls
- Subtitle management
- Quality settings

### 📱 **Offline Mode**
- Download content for offline viewing
- Smart storage management
- Auto-cleanup of old downloads
- Multiple quality options

### 🥽 **AR Content Discovery**
- Camera-based content finder
- 3D media positioning
- Interactive AR overlays
- Gesture-based controls

### 📺 **Casting Support**
- Chromecast integration
- AirPlay support
- DLNA compatibility
- Real-time cast controls

### 🌈 **Cyberpunk Theme**
- Neon color scheme (#00ff9f, #ff0080, #ffaa00)
- Glowing effects and shadows
- Animated UI elements
- Grid-based backgrounds

## 🛠 Technology Stack

- **React Native 0.73.4** - Cross-platform mobile framework
- **Expo 50** - Development platform and services
- **Redux Toolkit** - State management
- **React Navigation 6** - Navigation library
- **TypeScript** - Type safety
- **Expo AV** - Audio/Video playback
- **Expo Camera** - AR features
- **Expo Notifications** - Push notifications
- **Expo Local Authentication** - Biometric auth
- **React Native Google Cast** - Chromecast support

## 📁 Project Structure

```
mobile-app/
├── src/
│   ├── components/          # Reusable UI components
│   │   └── LoadingScreen.tsx
│   ├── contexts/            # React contexts
│   │   └── AuthContext.tsx
│   ├── hooks/              # Custom React hooks
│   ├── navigation/         # Navigation configuration
│   │   └── AppNavigator.tsx
│   ├── screens/            # Screen components
│   │   ├── LoginScreen.tsx
│   │   ├── DashboardScreen.tsx
│   │   ├── MediaLibraryScreen.tsx
│   │   ├── DownloadsScreen.tsx
│   │   ├── CastingScreen.tsx
│   │   ├── ARViewScreen.tsx
│   │   ├── SettingsScreen.tsx
│   │   ├── MediaPlayerScreen.tsx
│   │   ├── NotificationsScreen.tsx
│   │   └── ServiceControlScreen.tsx
│   ├── services/           # API and external services
│   │   ├── apiService.ts
│   │   └── notificationService.ts
│   ├── store/              # Redux store and slices
│   │   ├── index.ts
│   │   └── slices/
│   │       ├── authSlice.ts
│   │       ├── mediaSlice.ts
│   │       ├── notificationsSlice.ts
│   │       ├── settingsSlice.ts
│   │       ├── castingSlice.ts
│   │       └── offlineSlice.ts
│   ├── types/              # TypeScript type definitions
│   └── utils/              # Utility functions
├── assets/                 # Static assets (icons, images)
├── App.tsx                 # Main app component
├── app.json               # Expo configuration
├── package.json           # Dependencies and scripts
├── tsconfig.json          # TypeScript configuration
└── babel.config.js        # Babel configuration
```

## 🚦 Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Expo CLI (`npm install -g @expo/cli`)
- iOS Simulator (for iOS development)
- Android Studio/Emulator (for Android development)

### Installation

1. **Clone the repository**
   ```bash
   cd /Users/morlock/fun/newmedia/mobile-app
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start the development server**
   ```bash
   npm start
   ```

4. **Run on iOS Simulator**
   ```bash
   npm run ios
   ```

5. **Run on Android Emulator**
   ```bash
   npm run android
   ```

### Environment Setup

1. **Configure API endpoint**
   - The app connects to `localhost:3333` by default
   - Update in `src/services/apiService.ts` if needed

2. **Setup push notifications**
   - Update `projectId` in `app.json`
   - Configure Firebase/Expo push services

3. **Configure biometric authentication**
   - Ensure Face ID/Touch ID is set up on device
   - iOS: Add `NSFaceIDUsageDescription` in `app.json`

## 📱 App Features

### Authentication
- Secure login with username/password
- Biometric authentication support
- Token-based session management
- Auto-login with stored credentials

### Dashboard
- System status overview
- Media statistics
- Recent content
- Quick actions
- Download progress

### Media Library
- Search functionality
- Filter by type (movies, series, music)
- Grid view with media cards
- Play, download, and cast actions

### Downloads/Offline
- Download content for offline viewing
- Storage usage tracking
- Download queue management
- Quality selection
- Auto-cleanup settings

### Casting
- Device discovery and connection
- Media casting controls
- Volume and playback control
- Support for multiple protocols

### AR View
- Camera-based content discovery
- 3D positioning of media items
- Interactive AR overlays
- Tap to select and interact

### Settings
- Account management
- Server configuration
- Security settings
- App preferences
- Theme selection

## 🔧 Configuration

### API Configuration
```typescript
// src/services/apiService.ts
const config = {
  baseURL: 'http://localhost:3333/api',
  timeout: 30000,
  retries: 3,
};
```

### Notification Configuration
```typescript
// Push notification setup
const projectId = 'media-server-remote-uuid';
```

### Redux State Structure
```typescript
interface RootState {
  auth: AuthState;           // Authentication state
  media: MediaState;         // Media content and services
  notifications: NotificationState; // Push notifications
  settings: SettingsState;   // App and user settings
  casting: CastingState;     // Cast device management
  offline: OfflineState;     // Download and offline content
}
```

## 🎨 Theming

The app uses a cyberpunk-inspired color scheme:

```typescript
const colors = {
  primary: '#00ff9f',        // Neon green
  secondary: '#ff0080',      // Neon pink
  accent: '#ffaa00',         // Neon orange
  background: '#0a0a0f',     // Dark blue-black
  surface: '#1a1a2e',       // Dark blue
  border: '#16213e',        // Blue-gray
  text: '#ffffff',          // White
  textSecondary: '#666699', // Light gray
};
```

## 🔒 Security Features

- **Biometric Authentication**: Face ID, Touch ID, fingerprint
- **Secure Storage**: Encrypted credential storage
- **Token Management**: JWT token refresh and validation
- **API Security**: Request signing and rate limiting
- **Data Encryption**: Sensitive data encryption at rest

## 📊 Performance

- **Lazy Loading**: Components loaded on demand
- **Image Optimization**: Cached and optimized images
- **Network Optimization**: Request queuing and retry logic
- **Memory Management**: Proper cleanup and disposal
- **Offline Support**: Cached data and offline functionality

## 🐛 Debugging

1. **Enable debug mode**
   ```bash
   __DEV__ && console.log('Debug info');
   ```

2. **Network debugging**
   - Check API connectivity in `apiService.ts`
   - Monitor network requests in debugger

3. **Redux debugging**
   - Use Redux DevTools
   - Enable state logging in development

## 📱 Platform-Specific Features

### iOS
- Face ID authentication
- AirPlay casting
- iOS-style navigation
- Push notification badges

### Android
- Fingerprint authentication
- Chromecast support
- Material design elements
- Android notification channels

## 🚀 Building for Production

### iOS
```bash
eas build --platform ios
```

### Android
```bash
eas build --platform android
```

### Configuration
- Update app signing certificates
- Configure push notification keys
- Set production API endpoints
- Enable crash reporting

## 📈 Monitoring

- **Crash Reporting**: Expo crash reporting
- **Analytics**: User interaction tracking
- **Performance**: App performance monitoring
- **Error Logging**: Centralized error collection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is part of the Media Server Remote Control suite.

## 🔗 Related Projects

- **Media Server API**: Backend API service
- **Web Dashboard**: Browser-based control panel
- **Docker Setup**: Container orchestration

---

**Built with ❤️ using React Native, Expo, and modern mobile development practices.**