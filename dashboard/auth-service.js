/**
 * Authentication Service for Media Dashboard
 * Handles user authentication, token management, and session persistence
 */

class AuthService {
    constructor(apiBaseUrl) {
        this.apiBaseUrl = apiBaseUrl;
        this.token = localStorage.getItem('auth_token');
        this.refreshToken = localStorage.getItem('refresh_token');
        this.user = null;
        this.listeners = new Map();
        
        // Initialize user if token exists
        if (this.token) {
            this.validateToken();
        }
    }

    // Event system
    on(event, callback) {
        if (!this.listeners.has(event)) {
            this.listeners.set(event, new Set());
        }
        this.listeners.get(event).add(callback);
    }

    off(event, callback) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).delete(callback);
        }
    }

    emit(event, data) {
        if (this.listeners.has(event)) {
            this.listeners.get(event).forEach(callback => {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in auth event listener for ${event}:`, error);
                }
            });
        }
    }

    // Authentication methods
    async login(credentials) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/auth/login`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(credentials)
            });

            const data = await response.json();

            if (data.success) {
                this.setTokens(data.data.token, data.data.refreshToken);
                this.user = data.data.user;
                this.emit('login', this.user);
                return { success: true, user: this.user };
            } else {
                this.emit('loginError', data.error);
                return { success: false, error: data.error };
            }
        } catch (error) {
            console.error('Login error:', error);
            this.emit('loginError', 'Network error');
            return { success: false, error: 'Network error' };
        }
    }

    async logout() {
        try {
            if (this.token) {
                await fetch(`${this.apiBaseUrl}/api/auth/logout`, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${this.token}`
                    }
                });
            }
        } catch (error) {
            console.error('Logout error:', error);
        } finally {
            this.clearTokens();
            this.user = null;
            this.emit('logout');
        }
    }

    async validateToken() {
        if (!this.token) {
            return false;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/auth/me`, {
                headers: {
                    'Authorization': `Bearer ${this.token}`
                }
            });

            const data = await response.json();

            if (data.success) {
                this.user = data.data;
                this.emit('authenticated', this.user);
                return true;
            } else {
                // Token is invalid, try to refresh
                return await this.attemptTokenRefresh();
            }
        } catch (error) {
            console.error('Token validation error:', error);
            return await this.attemptTokenRefresh();
        }
    }

    async attemptTokenRefresh() {
        if (!this.refreshToken) {
            this.clearTokens();
            this.emit('unauthenticated');
            return false;
        }

        try {
            const response = await fetch(`${this.apiBaseUrl}/api/auth/refresh`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    refreshToken: this.refreshToken
                })
            });

            const data = await response.json();

            if (data.success) {
                this.setTokens(data.data.token, data.data.refreshToken);
                this.user = data.data.user;
                this.emit('tokenRefreshed', this.user);
                return true;
            } else {
                this.clearTokens();
                this.emit('unauthenticated');
                return false;
            }
        } catch (error) {
            console.error('Token refresh error:', error);
            this.clearTokens();
            this.emit('unauthenticated');
            return false;
        }
    }

    // Token management
    setTokens(token, refreshToken) {
        this.token = token;
        this.refreshToken = refreshToken;
        localStorage.setItem('auth_token', token);
        if (refreshToken) {
            localStorage.setItem('refresh_token', refreshToken);
        }
    }

    clearTokens() {
        this.token = null;
        this.refreshToken = null;
        localStorage.removeItem('auth_token');
        localStorage.removeItem('refresh_token');
    }

    // Utility methods
    isAuthenticated() {
        return !!this.token && !!this.user;
    }

    getToken() {
        return this.token;
    }

    getUser() {
        return this.user;
    }

    // HTTP request helper with automatic token handling
    async authenticatedFetch(url, options = {}) {
        if (!this.token) {
            throw new Error('Not authenticated');
        }

        const headers = {
            'Authorization': `Bearer ${this.token}`,
            'Content-Type': 'application/json',
            ...options.headers
        };

        let response = await fetch(url, {
            ...options,
            headers
        });

        // If unauthorized, try to refresh token and retry
        if (response.status === 401) {
            const refreshSuccess = await this.attemptTokenRefresh();
            if (refreshSuccess) {
                headers['Authorization'] = `Bearer ${this.token}`;
                response = await fetch(url, {
                    ...options,
                    headers
                });
            } else {
                throw new Error('Authentication failed');
            }
        }

        return response;
    }

    // Role and permission checking
    hasRole(role) {
        return this.user && this.user.role === role;
    }

    hasPermission(permission) {
        return this.user && this.user.permissions && this.user.permissions.includes(permission);
    }

    canAccessServices() {
        return this.hasPermission('services:read') || this.hasRole('admin');
    }

    canControlServices() {
        return this.hasPermission('services:write') || this.hasRole('admin');
    }

    canAccessLogs() {
        return this.hasPermission('logs:read') || this.hasRole('admin');
    }

    canManageSettings() {
        return this.hasPermission('settings:write') || this.hasRole('admin');
    }

    // Session management
    extendSession() {
        if (this.token) {
            // Update the token timestamp to prevent auto-logout
            localStorage.setItem('auth_timestamp', Date.now().toString());
        }
    }

    checkSessionExpiry() {
        const timestamp = localStorage.getItem('auth_timestamp');
        if (timestamp) {
            const elapsed = Date.now() - parseInt(timestamp);
            const sessionTimeout = 24 * 60 * 60 * 1000; // 24 hours
            
            if (elapsed > sessionTimeout) {
                this.logout();
                this.emit('sessionExpired');
                return false;
            }
        }
        return true;
    }

    // Password management
    async changePassword(currentPassword, newPassword) {
        try {
            const response = await this.authenticatedFetch(`${this.apiBaseUrl}/api/auth/change-password`, {
                method: 'POST',
                body: JSON.stringify({
                    currentPassword,
                    newPassword
                })
            });

            const data = await response.json();
            
            if (data.success) {
                this.emit('passwordChanged');
                return { success: true };
            } else {
                return { success: false, error: data.error };
            }
        } catch (error) {
            console.error('Password change error:', error);
            return { success: false, error: 'Network error' };
        }
    }

    async resetPassword(email) {
        try {
            const response = await fetch(`${this.apiBaseUrl}/api/auth/reset-password`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ email })
            });

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Password reset error:', error);
            return { success: false, error: 'Network error' };
        }
    }

    // Two-factor authentication
    async enableTwoFactor() {
        try {
            const response = await this.authenticatedFetch(`${this.apiBaseUrl}/api/auth/2fa/enable`, {
                method: 'POST'
            });

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('2FA enable error:', error);
            return { success: false, error: 'Network error' };
        }
    }

    async verifyTwoFactor(code) {
        try {
            const response = await this.authenticatedFetch(`${this.apiBaseUrl}/api/auth/2fa/verify`, {
                method: 'POST',
                body: JSON.stringify({ code })
            });

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('2FA verify error:', error);
            return { success: false, error: 'Network error' };
        }
    }
}

// Auto-refresh token every 30 minutes
setInterval(() => {
    if (window.authService && window.authService.isAuthenticated()) {
        window.authService.extendSession();
        window.authService.checkSessionExpiry();
    }
}, 30 * 60 * 1000);

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = AuthService;
} else if (typeof window !== 'undefined') {
    window.AuthService = AuthService;
}