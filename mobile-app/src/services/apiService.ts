import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import * as SecureStore from 'expo-secure-store';
import NetInfo from '@react-native-community/netinfo';

export interface ApiConfig {
  baseURL: string;
  timeout: number;
  retries: number;
  retryDelay: number;
}

class ApiService {
  private api: AxiosInstance;
  private config: ApiConfig;
  private isOnline: boolean = true;
  private requestQueue: Array<() => Promise<any>> = [];

  constructor() {
    this.config = {
      baseURL: 'http://localhost:3333/api',
      timeout: 30000,
      retries: 3,
      retryDelay: 1000,
    };

    this.api = axios.create({
      baseURL: this.config.baseURL,
      timeout: this.config.timeout,
      headers: {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
      },
    });

    this.setupInterceptors();
    this.setupNetworkListener();
  }

  private setupInterceptors() {
    // Request interceptor
    this.api.interceptors.request.use(
      async (config) => {
        // Add auth token if available
        try {
          const token = await SecureStore.getItemAsync('auth_token');
          if (token) {
            config.headers.Authorization = `Bearer ${token}`;
          }
        } catch (error) {
          console.warn('Failed to get auth token:', error);
        }

        // Add API key if configured
        const apiKey = await SecureStore.getItemAsync('api_key');
        if (apiKey) {
          config.headers['X-API-Key'] = apiKey;
        }

        return config;
      },
      (error) => {
        return Promise.reject(error);
      }
    );

    // Response interceptor
    this.api.interceptors.response.use(
      (response: AxiosResponse) => {
        return response;
      },
      async (error) => {
        const originalRequest = error.config;

        // Handle 401 (Unauthorized) - token might be expired
        if (error.response?.status === 401 && !originalRequest._retry) {
          originalRequest._retry = true;

          try {
            // Try to refresh token
            const refreshToken = await SecureStore.getItemAsync('refresh_token');
            if (refreshToken) {
              const response = await this.api.post('/auth/refresh', {
                refreshToken,
              });

              if (response.data.success) {
                const { token } = response.data.data;
                await SecureStore.setItemAsync('auth_token', token);
                
                // Retry original request with new token
                originalRequest.headers.Authorization = `Bearer ${token}`;
                return this.api(originalRequest);
              }
            }
          } catch (refreshError) {
            // Refresh failed, clear tokens and redirect to login
            await SecureStore.deleteItemAsync('auth_token');
            await SecureStore.deleteItemAsync('refresh_token');
            // You could emit an event here to trigger logout in the app
          }
        }

        // Handle network errors
        if (!error.response && error.code === 'NETWORK_ERROR') {
          if (this.isOnline) {
            // Queue request for retry when back online
            this.requestQueue.push(() => this.api(originalRequest));
          }
          return Promise.reject(new Error('Network error - request queued for retry'));
        }

        // Handle timeout
        if (error.code === 'ECONNABORTED') {
          return this.retryRequest(originalRequest);
        }

        return Promise.reject(error);
      }
    );
  }

  private setupNetworkListener() {
    NetInfo.addEventListener(state => {
      const wasOffline = !this.isOnline;
      this.isOnline = state.isConnected ?? false;

      // Process queued requests when back online
      if (wasOffline && this.isOnline && this.requestQueue.length > 0) {
        console.log(`Back online! Processing ${this.requestQueue.length} queued requests`);
        
        const queue = [...this.requestQueue];
        this.requestQueue = [];
        
        queue.forEach(request => {
          request().catch(error => {
            console.warn('Queued request failed:', error);
          });
        });
      }
    });
  }

  private async retryRequest(config: AxiosRequestConfig, retryCount = 0): Promise<any> {
    if (retryCount >= this.config.retries) {
      throw new Error(`Request failed after ${this.config.retries} retries`);
    }

    await new Promise(resolve => 
      setTimeout(resolve, this.config.retryDelay * Math.pow(2, retryCount))
    );

    try {
      return await this.api(config);
    } catch (error) {
      return this.retryRequest(config, retryCount + 1);
    }
  }

  // Configuration methods
  public updateConfig(newConfig: Partial<ApiConfig>) {
    this.config = { ...this.config, ...newConfig };
    this.api.defaults.baseURL = this.config.baseURL;
    this.api.defaults.timeout = this.config.timeout;
  }

  public setBaseUrl(baseURL: string) {
    this.config.baseURL = baseURL;
    this.api.defaults.baseURL = baseURL;
  }

  // HTTP Methods
  public async get<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.api.get(url, config);
  }

  public async post<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.api.post(url, data, config);
  }

  public async put<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.api.put(url, data, config);
  }

  public async patch<T = any>(url: string, data?: any, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.api.patch(url, data, config);
  }

  public async delete<T = any>(url: string, config?: AxiosRequestConfig): Promise<AxiosResponse<T>> {
    return this.api.delete(url, config);
  }

  // Specific API methods
  public async getServices(force = false) {
    return this.get(`/services/status?force=${force}`);
  }

  public async getMediaStats() {
    return this.get('/media/stats');
  }

  public async getDownloadQueue() {
    return this.get('/downloads/queue');
  }

  public async getHealthStatus() {
    return this.get('/health');
  }

  public async startServices(services: string[]) {
    return this.post('/services/start', { services });
  }

  public async stopServices(services: string[]) {
    return this.post('/services/stop', { services });
  }

  public async restartServices(services: string[]) {
    return this.post('/services/restart', { services });
  }

  public async getServiceLogs(serviceName: string, lines = 100) {
    return this.get(`/services/${serviceName}/logs?lines=${lines}`);
  }

  public async getSystemMetrics() {
    return this.get('/health/metrics');
  }

  public async searchMedia(query: string, filters?: any) {
    return this.post('/search', { query, filters });
  }

  public async getMediaDetails(id: string, type: string) {
    return this.get(`/media/${type}/${id}`);
  }

  public async requestDownload(mediaId: string, quality = 'medium') {
    return this.post('/downloads/request', { mediaId, quality });
  }

  public async cancelDownload(downloadId: string) {
    return this.delete(`/downloads/${downloadId}`);
  }

  // Utility methods
  public isConnected(): boolean {
    return this.isOnline;
  }

  public getQueuedRequestsCount(): number {
    return this.requestQueue.length;
  }

  public clearRequestQueue() {
    this.requestQueue = [];
  }

  // Test connection
  public async testConnection(): Promise<boolean> {
    try {
      const response = await this.get('/system', { timeout: 5000 });
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }

  // Upload file (for logs, screenshots, etc.)
  public async uploadFile(file: File | Blob, endpoint: string, onProgress?: (progress: number) => void) {
    const formData = new FormData();
    formData.append('file', file);

    return this.post(endpoint, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (onProgress && progressEvent.total) {
          const progress = (progressEvent.loaded / progressEvent.total) * 100;
          onProgress(progress);
        }
      },
    });
  }

  // WebSocket URL helper
  public getWebSocketUrl(): string {
    const wsProtocol = this.config.baseURL.startsWith('https') ? 'wss' : 'ws';
    const baseUrl = this.config.baseURL.replace(/^https?/, wsProtocol).replace('/api', '');
    return baseUrl;
  }
}

// Create singleton instance
export const apiService = new ApiService();

// Export types
export type { AxiosResponse, AxiosRequestConfig };