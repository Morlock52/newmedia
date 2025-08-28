import { createSlice, createAsyncThunk, PayloadAction } from '@reduxjs/toolkit';
import * as Notifications from 'expo-notifications';

export interface NotificationData {
  id: string;
  title: string;
  body: string;
  data?: any;
  timestamp: number;
  read: boolean;
  type: 'download_complete' | 'new_content' | 'service_alert' | 'system' | 'cast';
  action?: {
    type: 'navigate' | 'play' | 'download';
    payload: any;
  };
}

interface NotificationsState {
  notifications: NotificationData[];
  unreadCount: number;
  pushToken: string | null;
  permissionGranted: boolean;
  loading: boolean;
  error: string | null;
}

const initialState: NotificationsState = {
  notifications: [],
  unreadCount: 0,
  pushToken: null,
  permissionGranted: false,
  loading: false,
  error: null,
};

// Async thunks
export const requestNotificationPermissions = createAsyncThunk(
  'notifications/requestPermissions',
  async (_, { rejectWithValue }) => {
    try {
      const { status: existingStatus } = await Notifications.getPermissionsAsync();
      let finalStatus = existingStatus;
      
      if (existingStatus !== 'granted') {
        const { status } = await Notifications.requestPermissionsAsync();
        finalStatus = status;
      }
      
      if (finalStatus !== 'granted') {
        throw new Error('Notification permissions not granted');
      }
      
      return true;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Permission request failed');
    }
  }
);

export const registerForPushNotifications = createAsyncThunk(
  'notifications/registerForPush',
  async (_, { rejectWithValue }) => {
    try {
      const token = await Notifications.getExpoPushTokenAsync({
        projectId: 'media-server-remote-uuid', // Replace with your actual project ID
      });
      
      // Send token to your server
      // await apiService.post('/notifications/register', { token: token.data });
      
      return token.data;
    } catch (error) {
      return rejectWithValue(error instanceof Error ? error.message : 'Token registration failed');
    }
  }
);

export const scheduleLocalNotification = createAsyncThunk(
  'notifications/scheduleLocal',
  async ({ 
    title, 
    body, 
    data, 
    trigger 
  }: { 
    title: string; 
    body: string; 
    data?: any; 
    trigger?: Notifications.NotificationTriggerInput 
  }) => {
    const notificationId = await Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data,
        sound: 'default',
        priority: Notifications.AndroidNotificationPriority.HIGH,
        vibrate: [0, 250, 250, 250],
      },
      trigger: trigger || null, // null means immediate
    });
    
    return {
      id: notificationId,
      title,
      body,
      data,
      timestamp: Date.now(),
      read: false,
      type: data?.type || 'system' as const,
    };
  }
);

const notificationsSlice = createSlice({
  name: 'notifications',
  initialState,
  reducers: {
    addNotification: (state, action: PayloadAction<Omit<NotificationData, 'id' | 'timestamp' | 'read'>>) => {
      const notification: NotificationData = {
        ...action.payload,
        id: `notif_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: Date.now(),
        read: false,
      };
      
      state.notifications.unshift(notification);
      state.unreadCount += 1;
    },
    
    markAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification && !notification.read) {
        notification.read = true;
        state.unreadCount = Math.max(0, state.unreadCount - 1);
      }
    },
    
    markAllAsRead: (state) => {
      state.notifications.forEach(notification => {
        notification.read = true;
      });
      state.unreadCount = 0;
    },
    
    removeNotification: (state, action: PayloadAction<string>) => {
      const index = state.notifications.findIndex(n => n.id === action.payload);
      if (index !== -1) {
        const notification = state.notifications[index];
        if (!notification.read) {
          state.unreadCount = Math.max(0, state.unreadCount - 1);
        }
        state.notifications.splice(index, 1);
      }
    },
    
    clearAllNotifications: (state) => {
      state.notifications = [];
      state.unreadCount = 0;
    },
    
    setPushToken: (state, action: PayloadAction<string>) => {
      state.pushToken = action.payload;
    },
    
    // Handle incoming push notifications
    handlePushNotification: (state, action: PayloadAction<any>) => {
      const { title, body, data } = action.payload;
      
      const notification: NotificationData = {
        id: `push_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        title,
        body,
        data,
        timestamp: Date.now(),
        read: false,
        type: data?.type || 'system',
        action: data?.action,
      };
      
      state.notifications.unshift(notification);
      state.unreadCount += 1;
    },
  },
  extraReducers: (builder) => {
    builder
      // Request permissions
      .addCase(requestNotificationPermissions.pending, (state) => {
        state.loading = true;
        state.error = null;
      })
      .addCase(requestNotificationPermissions.fulfilled, (state) => {
        state.loading = false;
        state.permissionGranted = true;
      })
      .addCase(requestNotificationPermissions.rejected, (state, action) => {
        state.loading = false;
        state.error = action.payload as string;
        state.permissionGranted = false;
      })
      
      // Register for push notifications
      .addCase(registerForPushNotifications.fulfilled, (state, action) => {
        state.pushToken = action.payload;
      })
      .addCase(registerForPushNotifications.rejected, (state, action) => {
        state.error = action.payload as string;
      })
      
      // Schedule local notification
      .addCase(scheduleLocalNotification.fulfilled, (state, action) => {
        state.notifications.unshift(action.payload);
        state.unreadCount += 1;
      });
  },
});

export const {
  addNotification,
  markAsRead,
  markAllAsRead,
  removeNotification,
  clearAllNotifications,
  setPushToken,
  handlePushNotification,
} = notificationsSlice.actions;

export default notificationsSlice.reducer;