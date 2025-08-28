import React from 'react';
import { createStackNavigator } from '@react-navigation/stack';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { useSelector } from 'react-redux';
import { RootState } from '../store';
import { useAuth } from '../contexts/AuthContext';

// Import screens
import LoginScreen from '../screens/LoginScreen';
import DashboardScreen from '../screens/DashboardScreen';
import MediaLibraryScreen from '../screens/MediaLibraryScreen';
import DownloadsScreen from '../screens/DownloadsScreen';
import CastingScreen from '../screens/CastingScreen';
import ARViewScreen from '../screens/ARViewScreen';
import SettingsScreen from '../screens/SettingsScreen';
import MediaPlayerScreen from '../screens/MediaPlayerScreen';
import ServiceControlScreen from '../screens/ServiceControlScreen';
import NotificationsScreen from '../screens/NotificationsScreen';
import { LoadingScreen } from '../components/LoadingScreen';

// Import icons
import { Ionicons } from '@expo/vector-icons';
import { Platform } from 'react-native';

// Navigation types
export type RootStackParamList = {
  Auth: undefined;
  Main: undefined;
  MediaPlayer: {
    mediaItem: any;
    playbackUrl?: string;
  };
  ARView: {
    searchQuery?: string;
  };
  Notifications: undefined;
  ServiceControl: undefined;
};

export type MainTabParamList = {
  Dashboard: undefined;
  Library: undefined;
  Downloads: undefined;
  Casting: undefined;
  Settings: undefined;
};

const Stack = createStackNavigator<RootStackParamList>();
const Tab = createBottomTabNavigator<MainTabParamList>();

// Custom tab bar icon component
const TabBarIcon: React.FC<{
  name: keyof typeof Ionicons.glyphMap;
  color: string;
  focused: boolean;
}> = ({ name, color, focused }) => {
  return (
    <Ionicons 
      name={name} 
      size={focused ? 28 : 24} 
      color={color}
      style={{
        textShadowColor: focused ? color : 'transparent',
        textShadowOffset: { width: 0, height: 0 },
        textShadowRadius: focused ? 8 : 0,
      }}
    />
  );
};

// Main tab navigator
const MainTabNavigator: React.FC = () => {
  return (
    <Tab.Navigator
      screenOptions={({ route }) => ({
        tabBarIcon: ({ focused, color }) => {
          let iconName: keyof typeof Ionicons.glyphMap;

          switch (route.name) {
            case 'Dashboard':
              iconName = focused ? 'grid' : 'grid-outline';
              break;
            case 'Library':
              iconName = focused ? 'library' : 'library-outline';
              break;
            case 'Downloads':
              iconName = focused ? 'download' : 'download-outline';
              break;
            case 'Casting':
              iconName = focused ? 'cast' : 'cast-outline';
              break;
            case 'Settings':
              iconName = focused ? 'settings' : 'settings-outline';
              break;
            default:
              iconName = 'help-outline';
          }

          return <TabBarIcon name={iconName} color={color} focused={focused} />;
        },
        tabBarActiveTintColor: '#00ff9f',
        tabBarInactiveTintColor: '#666699',
        tabBarStyle: {
          backgroundColor: '#1a1a2e',
          borderTopColor: '#16213e',
          borderTopWidth: 1,
          paddingBottom: Platform.OS === 'ios' ? 20 : 5,
          paddingTop: 5,
          height: Platform.OS === 'ios' ? 85 : 60,
          elevation: 8,
          shadowColor: '#00ff9f',
          shadowOffset: { width: 0, height: -2 },
          shadowOpacity: 0.1,
          shadowRadius: 8,
        },
        tabBarLabelStyle: {
          fontSize: 12,
          fontWeight: '600',
          marginTop: -2,
        },
        headerStyle: {
          backgroundColor: '#1a1a2e',
          borderBottomColor: '#16213e',
          elevation: 0,
          shadowOpacity: 0,
        },
        headerTintColor: '#ffffff',
        headerTitleStyle: {
          fontWeight: 'bold',
          fontSize: 18,
        },
      })}
    >
      <Tab.Screen 
        name="Dashboard" 
        component={DashboardScreen}
        options={{
          title: 'Media Hub',
          headerRight: () => (
            <NotificationButton />
          ),
        }}
      />
      <Tab.Screen 
        name="Library" 
        component={MediaLibraryScreen}
        options={{
          title: 'Library',
          headerRight: () => (
            <ARViewButton />
          ),
        }}
      />
      <Tab.Screen 
        name="Downloads" 
        component={DownloadsScreen}
        options={{
          title: 'Offline',
        }}
      />
      <Tab.Screen 
        name="Casting" 
        component={CastingScreen}
        options={{
          title: 'Cast & Control',
        }}
      />
      <Tab.Screen 
        name="Settings" 
        component={SettingsScreen}
        options={{
          title: 'Settings',
        }}
      />
    </Tab.Navigator>
  );
};

// Header button components
const NotificationButton: React.FC = () => {
  const { unreadCount } = useSelector((state: RootState) => state.notifications);
  
  return (
    <Ionicons 
      name={unreadCount > 0 ? 'notifications' : 'notifications-outline'} 
      size={24} 
      color="#00ff9f"
      style={{ 
        marginRight: 15,
        textShadowColor: unreadCount > 0 ? '#00ff9f' : 'transparent',
        textShadowOffset: { width: 0, height: 0 },
        textShadowRadius: unreadCount > 0 ? 8 : 0,
      }}
    />
  );
};

const ARViewButton: React.FC = () => {
  return (
    <Ionicons 
      name="camera" 
      size={24} 
      color="#ff0080"
      style={{ 
        marginRight: 15,
        textShadowColor: '#ff0080',
        textShadowOffset: { width: 0, height: 0 },
        textShadowRadius: 8,
      }}
    />
  );
};

// Main app navigator
const AppNavigator: React.FC = () => {
  const { isAuthenticated } = useSelector((state: RootState) => state.auth);
  const { isLoading } = useAuth();

  if (isLoading) {
    return <LoadingScreen />;
  }

  return (
    <Stack.Navigator
      screenOptions={{
        headerStyle: {
          backgroundColor: '#1a1a2e',
          borderBottomColor: '#16213e',
        },
        headerTintColor: '#ffffff',
        headerTitleStyle: {
          fontWeight: 'bold',
        },
        cardStyle: {
          backgroundColor: '#0a0a0f',
        },
        animationEnabled: true,
        gestureEnabled: true,
      }}
    >
      {!isAuthenticated ? (
        <Stack.Screen 
          name="Auth" 
          component={LoginScreen}
          options={{ 
            headerShown: false,
            animationTypeForReplace: 'push',
          }}
        />
      ) : (
        <>
          <Stack.Screen 
            name="Main" 
            component={MainTabNavigator}
            options={{ 
              headerShown: false,
              animationTypeForReplace: 'push',
            }}
          />
          <Stack.Screen 
            name="MediaPlayer" 
            component={MediaPlayerScreen}
            options={{ 
              title: 'Now Playing',
              headerShown: false,
              presentation: 'modal',
              animationTypeForReplace: 'push',
            }}
          />
          <Stack.Screen 
            name="ARView" 
            component={ARViewScreen}
            options={{ 
              title: 'AR Content Finder',
              headerShown: false,
              presentation: 'modal',
              animationTypeForReplace: 'push',
            }}
          />
          <Stack.Screen 
            name="Notifications" 
            component={NotificationsScreen}
            options={{ 
              title: 'Notifications',
              presentation: 'modal',
            }}
          />
          <Stack.Screen 
            name="ServiceControl" 
            component={ServiceControlScreen}
            options={{ 
              title: 'Service Control',
              presentation: 'modal',
            }}
          />
        </>
      )}
    </Stack.Navigator>
  );
};

export default AppNavigator;