/**
 * Graceful Degradation System for Media Server Infrastructure
 * Provides fallback UI and functionality when services are unavailable
 */

import React, { useState, useEffect, useContext, createContext } from 'react';

// Service Status Context
const ServiceStatusContext = createContext({
    services: {},
    isServiceAvailable: () => false,
    retryService: () => {},
    addFallback: () => {}
});

export const useServiceStatus = () => {
    return useContext(ServiceStatusContext);
};

// Service Status Provider
export const ServiceStatusProvider = ({ children }) => {
    const [services, setServices] = useState({});
    const [fallbacks, setFallbacks] = useState({});
    const [retryAttempts, setRetryAttempts] = useState({});

    const checkServiceHealth = async (serviceName, url) => {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 5000);

            const response = await fetch(url, {
                signal: controller.signal,
                headers: { 'Cache-Control': 'no-cache' }
            });

            clearTimeout(timeoutId);

            const isHealthy = response.ok;
            
            setServices(prev => ({
                ...prev,
                [serviceName]: {
                    status: isHealthy ? 'healthy' : 'unhealthy',
                    lastCheck: new Date().toISOString(),
                    responseTime: Date.now() - performance.now(),
                    error: isHealthy ? null : `HTTP ${response.status}`
                }
            }));

            return isHealthy;
        } catch (error) {
            setServices(prev => ({
                ...prev,
                [serviceName]: {
                    status: 'unreachable',
                    lastCheck: new Date().toISOString(),
                    responseTime: null,
                    error: error.message
                }
            }));

            return false;
        }
    };

    const isServiceAvailable = (serviceName) => {
        return services[serviceName]?.status === 'healthy';
    };

    const retryService = async (serviceName) => {
        const attempts = retryAttempts[serviceName] || 0;
        if (attempts >= 3) {
            console.warn(`Max retry attempts reached for ${serviceName}`);
            return false;
        }

        setRetryAttempts(prev => ({
            ...prev,
            [serviceName]: attempts + 1
        }));

        // Exponential backoff
        const delay = Math.pow(2, attempts) * 1000;
        await new Promise(resolve => setTimeout(resolve, delay));

        const serviceConfig = getServiceConfig(serviceName);
        if (serviceConfig) {
            return await checkServiceHealth(serviceName, serviceConfig.healthUrl);
        }

        return false;
    };

    const addFallback = (serviceName, fallbackComponent) => {
        setFallbacks(prev => ({
            ...prev,
            [serviceName]: fallbackComponent
        }));
    };

    const getServiceConfig = (serviceName) => {
        const configs = {
            jellyfin: { healthUrl: '/api/health', port: 8096 },
            sonarr: { healthUrl: '/ping', port: 8989 },
            radarr: { healthUrl: '/ping', port: 7878 },
            prowlarr: { healthUrl: '/ping', port: 9696 },
            qbittorrent: { healthUrl: '/api/v2/app/version', port: 8080 },
            plex: { healthUrl: '/identity', port: 32400 }
        };
        return configs[serviceName];
    };

    // Monitor services periodically
    useEffect(() => {
        const monitorServices = async () => {
            const serviceNames = ['jellyfin', 'sonarr', 'radarr', 'prowlarr', 'qbittorrent', 'plex'];
            
            for (const serviceName of serviceNames) {
                const config = getServiceConfig(serviceName);
                if (config) {
                    const url = `http://localhost:${config.port}${config.healthUrl}`;
                    await checkServiceHealth(serviceName, url);
                }
            }
        };

        // Initial check
        monitorServices();

        // Set up periodic monitoring
        const interval = setInterval(monitorServices, 30000); // Every 30 seconds

        return () => clearInterval(interval);
    }, []);

    const value = {
        services,
        isServiceAvailable,
        retryService,
        addFallback
    };

    return (
        <ServiceStatusContext.Provider value={value}>
            {children}
        </ServiceStatusContext.Provider>
    );
};

// Graceful Service Component
export const GracefulService = ({ 
    serviceName, 
    children, 
    fallback, 
    loadingComponent,
    errorComponent,
    retryable = true 
}) => {
    const { services, isServiceAvailable, retryService } = useServiceStatus();
    const [isRetrying, setIsRetrying] = useState(false);
    
    const service = services[serviceName];
    const isAvailable = isServiceAvailable(serviceName);

    const handleRetry = async () => {
        setIsRetrying(true);
        await retryService(serviceName);
        setIsRetrying(false);
    };

    // Loading state
    if (!service) {
        return loadingComponent || <ServiceLoadingFallback serviceName={serviceName} />;
    }

    // Service is healthy
    if (isAvailable) {
        return children;
    }

    // Service is unhealthy - show fallback
    if (fallback) {
        return fallback;
    }

    // Default error fallback
    return errorComponent || (
        <ServiceErrorFallback 
            serviceName={serviceName}
            service={service}
            onRetry={retryable ? handleRetry : undefined}
            isRetrying={isRetrying}
        />
    );
};

// Default Fallback Components
const ServiceLoadingFallback = ({ serviceName }) => (
    <div className="service-loading animate-pulse bg-gray-100 rounded-lg p-6">
        <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-3 text-gray-600">Checking {serviceName} status...</span>
        </div>
    </div>
);

const ServiceErrorFallback = ({ serviceName, service, onRetry, isRetrying }) => (
    <div className="service-error bg-red-50 border border-red-200 rounded-lg p-6">
        <div className="flex items-start">
            <div className="flex-shrink-0">
                <svg className="h-6 w-6 text-red-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L3.732 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
            </div>
            <div className="ml-3 flex-1">
                <h3 className="text-sm font-medium text-red-800">
                    {serviceName} is currently unavailable
                </h3>
                <div className="mt-2 text-sm text-red-700">
                    <p>Status: {service.status}</p>
                    {service.error && <p>Error: {service.error}</p>}
                    <p>Last checked: {new Date(service.lastCheck).toLocaleTimeString()}</p>
                </div>
                {onRetry && (
                    <div className="mt-4">
                        <button
                            onClick={onRetry}
                            disabled={isRetrying}
                            className="inline-flex items-center px-3 py-2 border border-transparent text-sm leading-4 font-medium rounded-md text-red-700 bg-red-100 hover:bg-red-200 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50"
                        >
                            {isRetrying ? (
                                <>
                                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Retrying...
                                </>
                            ) : (
                                'Retry Connection'
                            )}
                        </button>
                    </div>
                )}
            </div>
        </div>
    </div>
);

// Specific service fallbacks
export const JellyfinFallback = () => (
    <div className="jellyfin-fallback bg-purple-50 border border-purple-200 rounded-lg p-6">
        <div className="text-center">
            <svg className="mx-auto h-12 w-12 text-purple-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 3a2 2 0 00-2 2v10a2 2 0 002 2h12a2 2 0 002-2V5a2 2 0 00-2-2H4zm12 12H4l4-8 3 6 2-4 3 6z" clipRule="evenodd" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">Jellyfin Unavailable</h3>
            <p className="mt-1 text-sm text-gray-500">
                Media streaming is temporarily unavailable. Your content is safe and will be accessible once the service is restored.
            </p>
            <div className="mt-4">
                <div className="bg-purple-100 rounded-md p-3">
                    <h4 className="text-xs font-medium text-purple-800">Available Alternatives:</h4>
                    <ul className="mt-1 text-xs text-purple-700">
                        <li>• Direct file access via File Browser</li>
                        <li>• Download content via qBittorrent</li>
                        <li>• Use Plex if available</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
);

export const SonarrFallback = () => (
    <div className="sonarr-fallback bg-blue-50 border border-blue-200 rounded-lg p-6">
        <div className="text-center">
            <svg className="mx-auto h-12 w-12 text-blue-400" fill="currentColor" viewBox="0 0 20 20">
                <path d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">TV Show Management Unavailable</h3>
            <p className="mt-1 text-sm text-gray-500">
                Automatic TV show downloads are paused. Manual downloads are still available.
            </p>
            <div className="mt-4 space-y-2">
                <button className="w-full bg-blue-100 hover:bg-blue-200 text-blue-800 py-2 px-4 rounded text-sm">
                    Manual Search via Prowlarr
                </button>
                <button className="w-full bg-blue-100 hover:bg-blue-200 text-blue-800 py-2 px-4 rounded text-sm">
                    Check Download Queue
                </button>
            </div>
        </div>
    </div>
);

export const QBittorrentFallback = () => (
    <div className="qbittorrent-fallback bg-green-50 border border-green-200 rounded-lg p-6">
        <div className="text-center">
            <svg className="mx-auto h-12 w-12 text-green-400" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">Download Client Unavailable</h3>
            <p className="mt-1 text-sm text-gray-500">
                BitTorrent downloads are temporarily paused. Existing downloads will resume automatically.
            </p>
            <div className="mt-4">
                <div className="bg-green-100 rounded-md p-3 text-left">
                    <h4 className="text-xs font-medium text-green-800">Current Status:</h4>
                    <ul className="mt-1 text-xs text-green-700">
                        <li>• Downloads: Paused</li>
                        <li>• Seeding: May continue</li>
                        <li>• Queue: Preserved</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
);

// Network Status Component
export const NetworkStatus = () => {
    const { services } = useServiceStatus();
    const [isOnline, setIsOnline] = useState(navigator.onLine);

    useEffect(() => {
        const handleOnline = () => setIsOnline(true);
        const handleOffline = () => setIsOnline(false);

        window.addEventListener('online', handleOnline);
        window.addEventListener('offline', handleOffline);

        return () => {
            window.removeEventListener('online', handleOnline);
            window.removeEventListener('offline', handleOffline);
        };
    }, []);

    if (!isOnline) {
        return (
            <div className="network-status bg-red-600 text-white p-2 text-center text-sm">
                <span className="inline-flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    You are currently offline. Some features may not be available.
                </span>
            </div>
        );
    }

    const unhealthyServices = Object.entries(services)
        .filter(([_, service]) => service.status !== 'healthy')
        .length;

    if (unhealthyServices > 0) {
        return (
            <div className="network-status bg-yellow-600 text-white p-2 text-center text-sm">
                <span className="inline-flex items-center">
                    <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    {unhealthyServices} service{unhealthyServices > 1 ? 's' : ''} currently unavailable
                </span>
            </div>
        );
    }

    return null;
};

// Offline Storage Hook
export const useOfflineStorage = (key, initialValue) => {
    const [value, setValue] = useState(() => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : initialValue;
        } catch (error) {
            console.warn(`Error reading from localStorage key "${key}":`, error);
            return initialValue;
        }
    });

    const setStoredValue = (newValue) => {
        try {
            setValue(newValue);
            localStorage.setItem(key, JSON.stringify(newValue));
        } catch (error) {
            console.warn(`Error writing to localStorage key "${key}":`, error);
        }
    };

    return [value, setStoredValue];
};

// Cache Hook for API responses
export const useApiCache = (key, fetcher, options = {}) => {
    const { cacheTime = 5 * 60 * 1000, staleTime = 30 * 1000 } = options; // 5min cache, 30s stale
    const [cache, setCache] = useOfflineStorage(`api_cache_${key}`, null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);

    const isStale = cache && (Date.now() - cache.timestamp > staleTime);
    const isExpired = cache && (Date.now() - cache.timestamp > cacheTime);

    const fetchData = async (force = false) => {
        if (!force && cache && !isExpired) {
            return cache.data;
        }

        setIsLoading(true);
        setError(null);

        try {
            const data = await fetcher();
            const cacheEntry = {
                data,
                timestamp: Date.now()
            };
            setCache(cacheEntry);
            return data;
        } catch (err) {
            setError(err);
            // Return stale data if available
            if (cache) {
                return cache.data;
            }
            throw err;
        } finally {
            setIsLoading(false);
        }
    };

    useEffect(() => {
        if (!cache || isStale) {
            fetchData();
        }
    }, [key]);

    return {
        data: cache?.data,
        isLoading,
        error,
        refetch: () => fetchData(true),
        isStale
    };
};

export default {
    ServiceStatusProvider,
    GracefulService,
    NetworkStatus,
    JellyfinFallback,
    SonarrFallback,
    QBittorrentFallback,
    useServiceStatus,
    useOfflineStorage,
    useApiCache
};