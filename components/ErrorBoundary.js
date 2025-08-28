/**
 * React Error Boundary Component with Advanced Error Handling
 * Provides fallback UI, error reporting, and recovery mechanisms
 */

import React from 'react';

class ErrorBoundary extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            hasError: false,
            error: null,
            errorInfo: null,
            errorId: null,
            retryCount: 0,
            isRetrying: false
        };
        
        this.maxRetries = props.maxRetries || 3;
        this.retryDelay = props.retryDelay || 1000;
        this.onError = props.onError || this.defaultErrorHandler;
    }

    static getDerivedStateFromError(error) {
        // Update state so the next render will show the fallback UI
        return {
            hasError: true,
            errorId: `error_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
        };
    }

    componentDidCatch(error, errorInfo) {
        // Log error details
        console.error('ErrorBoundary caught an error:', error, errorInfo);
        
        this.setState({
            error,
            errorInfo
        });

        // Report error to monitoring service
        this.reportError(error, errorInfo);
        
        // Call custom error handler
        this.onError(error, errorInfo);
    }

    reportError = async (error, errorInfo) => {
        const errorReport = {
            errorId: this.state.errorId,
            timestamp: new Date().toISOString(),
            message: error.message,
            stack: error.stack,
            componentStack: errorInfo.componentStack,
            url: window.location.href,
            userAgent: navigator.userAgent,
            userId: this.props.userId || 'anonymous',
            buildVersion: process.env.REACT_APP_VERSION || 'unknown'
        };

        try {
            // Send to error reporting service
            await fetch('/api/errors', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(errorReport)
            });
        } catch (reportingError) {
            console.error('Failed to report error:', reportingError);
            
            // Fallback: store in localStorage for later reporting
            try {
                const storedErrors = JSON.parse(localStorage.getItem('pendingErrors') || '[]');
                storedErrors.push(errorReport);
                localStorage.setItem('pendingErrors', JSON.stringify(storedErrors.slice(-10))); // Keep last 10
            } catch (storageError) {
                console.error('Failed to store error locally:', storageError);
            }
        }
    };

    defaultErrorHandler = (error, errorInfo) => {
        // Default error handling logic
        console.group('🚨 Error Boundary - Default Handler');
        console.error('Error:', error);
        console.error('Error Info:', errorInfo);
        console.groupEnd();
    };

    handleRetry = async () => {
        if (this.state.retryCount >= this.maxRetries) {
            console.warn('Max retries exceeded');
            return;
        }

        this.setState({ isRetrying: true });
        
        // Wait before retry with exponential backoff
        const delay = this.retryDelay * Math.pow(2, this.state.retryCount);
        await new Promise(resolve => setTimeout(resolve, delay));

        this.setState(prevState => ({
            hasError: false,
            error: null,
            errorInfo: null,
            errorId: null,
            retryCount: prevState.retryCount + 1,
            isRetrying: false
        }));
    };

    handleReload = () => {
        window.location.reload();
    };

    handleGoHome = () => {
        window.location.href = '/';
    };

    renderErrorDetails = () => {
        if (!this.props.showDetails && process.env.NODE_ENV === 'production') {
            return null;
        }

        return (
            <details className="error-details mt-4">
                <summary className="cursor-pointer text-sm text-gray-600 hover:text-gray-800">
                    Technical Details
                </summary>
                <div className="mt-2 p-4 bg-gray-100 rounded-lg text-xs font-mono">
                    <div className="mb-2">
                        <strong>Error ID:</strong> {this.state.errorId}
                    </div>
                    <div className="mb-2">
                        <strong>Message:</strong> {this.state.error?.message}
                    </div>
                    <div className="mb-2">
                        <strong>Stack Trace:</strong>
                        <pre className="mt-1 text-red-600 whitespace-pre-wrap">
                            {this.state.error?.stack}
                        </pre>
                    </div>
                    {this.state.errorInfo && (
                        <div>
                            <strong>Component Stack:</strong>
                            <pre className="mt-1 text-blue-600 whitespace-pre-wrap">
                                {this.state.errorInfo.componentStack}
                            </pre>
                        </div>
                    )}
                </div>
            </details>
        );
    };

    render() {
        if (this.state.hasError) {
            // Custom fallback UI
            if (this.props.fallback) {
                return this.props.fallback(this.state.error, this.handleRetry);
            }

            // Default fallback UI
            return (
                <div className="error-boundary min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8">
                    <div className="sm:mx-auto sm:w-full sm:max-w-md">
                        <div className="bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10">
                            <div className="text-center">
                                <div className="mx-auto h-12 w-12 text-red-600">
                                    <svg fill="currentColor" viewBox="0 0 20 20">
                                        <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                    </svg>
                                </div>
                                
                                <h2 className="mt-4 text-lg font-medium text-gray-900">
                                    Something went wrong
                                </h2>
                                
                                <p className="mt-2 text-sm text-gray-600">
                                    {this.props.title || 'An unexpected error occurred while loading this component.'}
                                </p>
                                
                                {this.state.retryCount < this.maxRetries && (
                                    <p className="mt-1 text-xs text-gray-500">
                                        Retry attempt {this.state.retryCount} of {this.maxRetries}
                                    </p>
                                )}
                            </div>

                            <div className="mt-6 flex flex-col space-y-3">
                                {this.state.retryCount < this.maxRetries && (
                                    <button
                                        onClick={this.handleRetry}
                                        disabled={this.state.isRetrying}
                                        className="w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed"
                                    >
                                        {this.state.isRetrying ? (
                                            <>
                                                <svg className="animate-spin -ml-1 mr-3 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                                                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                                    <path className="opacity-75" fill="currentColor" d="m4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                                </svg>
                                                Retrying...
                                            </>
                                        ) : (
                                            'Try Again'
                                        )}
                                    </button>
                                )}
                                
                                <button
                                    onClick={this.handleReload}
                                    className="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                                >
                                    Reload Page
                                </button>
                                
                                <button
                                    onClick={this.handleGoHome}
                                    className="w-full flex justify-center py-2 px-4 border border-gray-300 rounded-md shadow-sm text-sm font-medium text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
                                >
                                    Go to Home
                                </button>
                            </div>

                            {this.renderErrorDetails()}
                        </div>
                    </div>
                </div>
            );
        }

        return this.props.children;
    }
}

// Hook version for functional components
export const useErrorHandler = (onError) => {
    return React.useCallback((error, errorInfo) => {
        console.error('useErrorHandler caught:', error, errorInfo);
        if (onError) {
            onError(error, errorInfo);
        }
    }, [onError]);
};

// Higher-order component for wrapping components with error boundary
export const withErrorBoundary = (Component, errorBoundaryProps = {}) => {
    return function WrappedComponent(props) {
        return (
            <ErrorBoundary {...errorBoundaryProps}>
                <Component {...props} />
            </ErrorBoundary>
        );
    };
};

// Specialized error boundaries for different component types
export class ServiceErrorBoundary extends ErrorBoundary {
    constructor(props) {
        super(props);
        this.serviceName = props.serviceName || 'Unknown Service';
    }

    renderServiceError = () => {
        return (
            <div className="service-error bg-red-50 border border-red-200 rounded-lg p-4">
                <div className="flex">
                    <div className="flex-shrink-0">
                        <svg className="h-5 w-5 text-red-400" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                        </svg>
                    </div>
                    <div className="ml-3">
                        <h3 className="text-sm font-medium text-red-800">
                            {this.serviceName} Service Error
                        </h3>
                        <div className="mt-2 text-sm text-red-700">
                            <p>
                                The {this.serviceName} service encountered an error and needs to be restarted.
                            </p>
                        </div>
                        <div className="mt-4">
                            <div className="-mx-2 -my-1.5 flex">
                                <button
                                    onClick={this.handleRetry}
                                    className="bg-red-50 px-2 py-1.5 rounded-md text-sm font-medium text-red-800 hover:bg-red-100 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-red-50 focus:ring-red-600"
                                >
                                    Restart Service
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        );
    };

    render() {
        if (this.state.hasError) {
            return this.renderServiceError();
        }
        return this.props.children;
    }
}

export class APIErrorBoundary extends ErrorBoundary {
    render() {
        if (this.state.hasError) {
            return (
                <div className="api-error bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <div className="flex">
                        <div className="flex-shrink-0">
                            <svg className="h-5 w-5 text-yellow-400" fill="currentColor" viewBox="0 0 20 20">
                                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                            </svg>
                        </div>
                        <div className="ml-3">
                            <h3 className="text-sm font-medium text-yellow-800">
                                API Connection Error
                            </h3>
                            <div className="mt-2 text-sm text-yellow-700">
                                <p>Unable to connect to the API. Please check your connection and try again.</p>
                            </div>
                            <div className="mt-4">
                                <button
                                    onClick={this.handleRetry}
                                    className="bg-yellow-50 px-2 py-1.5 rounded-md text-sm font-medium text-yellow-800 hover:bg-yellow-100"
                                >
                                    Retry Connection
                                </button>
                            </div>
                        </div>
                    </div>
                </div>
            );
        }
        return this.props.children;
    }
}

export default ErrorBoundary;