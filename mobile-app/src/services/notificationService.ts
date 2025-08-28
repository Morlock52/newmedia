import * as Notifications from 'expo-notifications';
import * as Device from 'expo-device';
import Constants from 'expo-constants';
import { Platform } from 'react-native';
import { store } from '../store';
import { 
  requestNotificationPermissions, 
  registerForPushNotifications,
  addNotification,
  handlePushNotification
} from '../store/slices/notificationsSlice';

// Configure notification behavior
Notifications.setNotificationHandler({
  handleNotification: async () => ({
    shouldShowAlert: true,
    shouldPlaySound: true,
    shouldSetBadge: true,
    priority: Notifications.AndroidNotificationPriority.HIGH,
  }),
});

export class NotificationService {
  private static instance: NotificationService;
  private notificationListener: any;
  private responseListener: any;

  public static getInstance(): NotificationService {
    if (!NotificationService.instance) {
      NotificationService.instance = new NotificationService();
    }
    return NotificationService.instance;
  }

  public async initialize() {
    // Request permissions
    await store.dispatch(requestNotificationPermissions());

    // Register for push notifications if on physical device
    if (Device.isDevice) {
      await store.dispatch(registerForPushNotifications());
    }

    // Set up notification listeners
    this.setupNotificationListeners();

    // Configure notification channels for Android
    if (Platform.OS === 'android') {
      await this.setupAndroidChannels();
    }
  }

  private setupNotificationListeners() {
    // Handle notifications received while app is foregrounded
    this.notificationListener = Notifications.addNotificationReceivedListener(notification => {
      const { title, body, data } = notification.request.content;
      
      store.dispatch(handlePushNotification({
        title: title || '',
        body: body || '',
        data,
      }));
    });

    // Handle notification responses (when user taps notification)
    this.responseListener = Notifications.addNotificationResponseReceivedListener(response => {
      const { notification } = response;
      const { data } = notification.request.content;
      
      if (data?.action) {
        this.handleNotificationAction(data.action);
      }
    });
  }

  private async setupAndroidChannels() {
    await Notifications.setNotificationChannelAsync('downloads', {
      name: 'Downloads',
      description: 'Download progress and completion notifications',
      importance: Notifications.AndroidImportance.DEFAULT,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#00ff9f',
      sound: 'default',
    });

    await Notifications.setNotificationChannelAsync('content', {
      name: 'New Content',
      description: 'Notifications about new movies, shows, and music',
      importance: Notifications.AndroidImportance.DEFAULT,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#ff0080',
      sound: 'default',
    });

    await Notifications.setNotificationChannelAsync('services', {
      name: 'Service Alerts',
      description: 'Alerts about service status and system issues',
      importance: Notifications.AndroidImportance.HIGH,
      vibrationPattern: [0, 250, 250, 250],
      lightColor: '#ffaa00',
      sound: 'default',
    });

    await Notifications.setNotificationChannelAsync('casting', {
      name: 'Casting',
      description: 'Cast session and playback notifications',
      importance: Notifications.AndroidImportance.LOW,
      vibrationPattern: [0, 250],
      lightColor: '#0099ff',
      sound: 'default',
    });
  }

  private handleNotificationAction(action: any) {
    // Handle different notification actions
    switch (action.type) {
      case 'navigate':
        // Navigate to specific screen
        // This would typically use navigation service
        console.log('Navigate to:', action.payload);
        break;
        
      case 'play':
        // Start playing media
        console.log('Play media:', action.payload);
        break;
        
      case 'download':
        // Start download
        console.log('Start download:', action.payload);
        break;
    }
  }

  // Public methods for scheduling notifications
  public async scheduleDownloadCompleteNotification(title: string, mediaType: string) {
    return Notifications.scheduleNotificationAsync({
      content: {
        title: 'Download Complete',
        body: `${title} is now available for offline viewing`,
        data: {
          type: 'download_complete',
          mediaTitle: title,
          action: {
            type: 'navigate',
            payload: { screen: 'Offline' }
          }
        },
        sound: 'default',
        priority: Notifications.AndroidNotificationPriority.DEFAULT,
        vibrate: [0, 250, 250, 250],
      },
      trigger: null, // Immediate
      identifier: `download_complete_${Date.now()}`,
    });
  }

  public async scheduleNewContentNotification(
    title: string, 
    type: 'movie' | 'series' | 'episode' | 'music',
    imageUrl?: string
  ) {
    return Notifications.scheduleNotificationAsync({
      content: {
        title: 'New Content Available',
        body: `${title} has been added to your media library`,
        data: {
          type: 'new_content',
          mediaTitle: title,
          mediaType: type,
          action: {
            type: 'navigate',
            payload: { screen: 'MediaDetails', params: { title, type } }
          }
        },
        sound: 'default',
        priority: Notifications.AndroidNotificationPriority.DEFAULT,
        vibrate: [0, 250, 250, 250],
        ...(imageUrl && Platform.OS === 'android' && {
          largeIcon: imageUrl,
          bigPicture: imageUrl,
        }),
      },
      trigger: null,
      identifier: `new_content_${Date.now()}`,
    });
  }

  public async scheduleServiceAlertNotification(serviceName: string, status: string, message?: string) {
    const isError = status === 'error' || status === 'stopped';
    
    return Notifications.scheduleNotificationAsync({
      content: {
        title: isError ? 'Service Alert' : 'Service Update',
        body: message || `${serviceName} is now ${status}`,
        data: {
          type: 'service_alert',
          serviceName,
          status,
          action: {
            type: 'navigate',
            payload: { screen: 'Services' }
          }
        },
        sound: 'default',
        priority: isError 
          ? Notifications.AndroidNotificationPriority.HIGH 
          : Notifications.AndroidNotificationPriority.DEFAULT,
        vibrate: isError ? [0, 250, 250, 250, 250, 250] : [0, 250, 250, 250],
      },
      trigger: null,
      identifier: `service_alert_${serviceName}_${Date.now()}`,
    });
  }

  public async scheduleCastingNotification(action: 'started' | 'stopped' | 'failed', deviceName: string, mediaTitle?: string) {
    let title: string;
    let body: string;

    switch (action) {
      case 'started':
        title = 'Casting Started';
        body = mediaTitle ? `Now casting "${mediaTitle}" to ${deviceName}` : `Connected to ${deviceName}`;
        break;
      case 'stopped':
        title = 'Casting Stopped';
        body = `Disconnected from ${deviceName}`;
        break;
      case 'failed':
        title = 'Casting Failed';
        body = `Could not connect to ${deviceName}`;
        break;
    }

    return Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data: {
          type: 'cast',
          action,
          deviceName,
          mediaTitle,
        },
        sound: action === 'failed' ? 'default' : undefined,
        priority: Notifications.AndroidNotificationPriority.LOW,
        vibrate: action === 'failed' ? [0, 250, 250, 250] : [0, 250],
      },
      trigger: null,
      identifier: `casting_${action}_${Date.now()}`,
    });
  }

  public async scheduleReminder(title: string, body: string, triggerDate: Date) {
    return Notifications.scheduleNotificationAsync({
      content: {
        title,
        body,
        data: {
          type: 'reminder',
        },
        sound: 'default',
        priority: Notifications.AndroidNotificationPriority.DEFAULT,
      },
      trigger: {
        date: triggerDate,
      },
      identifier: `reminder_${Date.now()}`,
    });
  }

  // Progress notification for downloads
  public async updateDownloadProgressNotification(
    notificationId: string,
    title: string,
    progress: number,
    isCompleted = false
  ) {
    if (Platform.OS === 'android') {
      const content = {
        title: isCompleted ? 'Download Complete' : 'Downloading',
        body: isCompleted ? `${title} is ready to watch` : `${title} - ${Math.round(progress)}%`,
        data: {
          type: 'download_progress',
          mediaTitle: title,
          progress,
          isCompleted,
        },
        sound: isCompleted ? 'default' : undefined,
        priority: Notifications.AndroidNotificationPriority.LOW,
        sticky: !isCompleted,
        ongoing: !isCompleted,
        ...(Platform.OS === 'android' && !isCompleted && {
          progress: {
            max: 100,
            current: Math.round(progress),
            indeterminate: false,
          },
        }),
      };

      return Notifications.scheduleNotificationAsync({
        content,
        trigger: null,
        identifier: notificationId,
      });
    }
  }

  // Cancel specific notification
  public async cancelNotification(identifier: string) {
    return Notifications.cancelScheduledNotificationAsync(identifier);
  }

  // Cancel all notifications
  public async cancelAllNotifications() {
    return Notifications.cancelAllScheduledNotificationsAsync();
  }

  // Set badge count (iOS)
  public async setBadgeCount(count: number) {
    if (Platform.OS === 'ios') {
      return Notifications.setBadgeCountAsync(count);
    }
  }

  // Clean up listeners
  public cleanup() {
    if (this.notificationListener) {
      Notifications.removeNotificationSubscription(this.notificationListener);
    }
    if (this.responseListener) {
      Notifications.removeNotificationSubscription(this.responseListener);
    }
  }
}

// Setup function to be called from App.tsx
export async function setupNotifications() {
  const notificationService = NotificationService.getInstance();
  await notificationService.initialize();
  return notificationService;
}

// Export singleton instance
export const notificationService = NotificationService.getInstance();