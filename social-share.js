// Social Share Widget for MediaFlow 2025
// Modern social media integration with share tracking

class SocialShareWidget {
    constructor() {
        this.platforms = {
            twitter: {
                name: 'Twitter',
                icon: `<svg viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M23.953 4.57a10 10 0 01-2.825.775 4.958 4.958 0 002.163-2.723c-.951.555-2.005.959-3.127 1.184a4.92 4.92 0 00-8.384 4.482C7.69 8.095 4.067 6.13 1.64 3.162a4.822 4.822 0 00-.666 2.475c0 1.71.87 3.213 2.188 4.096a4.904 4.904 0 01-2.228-.616v.06a4.923 4.923 0 003.946 4.827 4.996 4.996 0 01-2.212.085 4.936 4.936 0 004.604 3.417 9.867 9.867 0 01-6.102 2.105c-.39 0-.779-.023-1.17-.067a13.995 13.995 0 007.557 2.209c9.053 0 13.998-7.496 13.998-13.985 0-.21 0-.42-.015-.63A9.935 9.935 0 0024 4.59z"/>
                </svg>`,
                color: '#1DA1F2',
                shareUrl: (url, text) => `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`
            },
            facebook: {
                name: 'Facebook',
                icon: `<svg viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M24 12.073c0-6.627-5.373-12-12-12s-12 5.373-12 12c0 5.99 4.388 10.954 10.125 11.854v-8.385H7.078v-3.47h3.047V9.43c0-3.007 1.792-4.669 4.533-4.669 1.312 0 2.686.235 2.686.235v2.953H15.83c-1.491 0-1.956.925-1.956 1.874v2.25h3.328l-.532 3.47h-2.796v8.385C19.612 23.027 24 18.062 24 12.073z"/>
                </svg>`,
                color: '#1877F2',
                shareUrl: (url) => `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(url)}`
            },
            instagram: {
                name: 'Instagram',
                icon: `<svg viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M12 2.163c3.204 0 3.584.012 4.85.07 3.252.148 4.771 1.691 4.919 4.919.058 1.265.069 1.645.069 4.849 0 3.205-.012 3.584-.069 4.849-.149 3.225-1.664 4.771-4.919 4.919-1.266.058-1.644.07-4.85.07-3.204 0-3.584-.012-4.849-.07-3.26-.149-4.771-1.699-4.919-4.92-.058-1.265-.07-1.644-.07-4.849 0-3.204.013-3.583.07-4.849.149-3.227 1.664-4.771 4.919-4.919 1.266-.057 1.645-.069 4.849-.069zm0-2.163c-3.259 0-3.667.014-4.947.072-4.358.2-6.78 2.618-6.98 6.98-.059 1.281-.073 1.689-.073 4.948 0 3.259.014 3.668.072 4.948.2 4.358 2.618 6.78 6.98 6.98 1.281.058 1.689.072 4.948.072 3.259 0 3.668-.014 4.948-.072 4.354-.2 6.782-2.618 6.979-6.98.059-1.28.073-1.689.073-4.948 0-3.259-.014-3.667-.072-4.947-.196-4.354-2.617-6.78-6.979-6.98-1.281-.059-1.69-.073-4.949-.073zM5.838 12a6.162 6.162 0 1112.324 0 6.162 6.162 0 01-12.324 0zM12 16a4 4 0 110-8 4 4 0 010 8zm4.965-10.405a1.44 1.44 0 112.881.001 1.44 1.44 0 01-2.881-.001z"/>
                </svg>`,
                color: '#E4405F',
                shareUrl: () => null // Instagram doesn't support direct URL sharing
            },
            tiktok: {
                name: 'TikTok',
                icon: `<svg viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M12.525.02c1.31-.02 2.61-.01 3.91-.02.08 1.53.63 3.09 1.75 4.17 1.12 1.11 2.7 1.62 4.24 1.79v4.03c-1.44-.05-2.89-.35-4.2-.97-.57-.26-1.1-.59-1.62-.93-.01 2.92.01 5.84-.02 8.75-.08 1.4-.54 2.79-1.35 3.94-1.31 1.92-3.58 3.17-5.91 3.21-1.43.08-2.86-.31-4.08-1.03-2.02-1.19-3.44-3.37-3.65-5.71-.02-.5-.03-1-.01-1.49.18-1.9 1.12-3.72 2.58-4.96 1.66-1.44 3.98-2.13 6.15-1.72.02 1.48-.04 2.96-.04 4.44-.99-.32-2.15-.23-3.02.37-.63.41-1.11 1.04-1.36 1.75-.21.51-.15 1.07-.14 1.61.24 1.64 1.82 3.02 3.5 2.87 1.12-.01 2.19-.66 2.77-1.61.19-.33.4-.67.41-1.06.1-1.79.06-3.57.07-5.36.01-4.03-.01-8.05.02-12.07z"/>
                </svg>`,
                color: '#000000',
                shareUrl: () => null // TikTok sharing requires app integration
            },
            linkedin: {
                name: 'LinkedIn',
                icon: `<svg viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433c-1.144 0-2.063-.926-2.063-2.065 0-1.138.92-2.063 2.063-2.063 1.14 0 2.064.925 2.064 2.063 0 1.139-.925 2.065-2.064 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
                </svg>`,
                color: '#0A66C2',
                shareUrl: (url, title) => `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(url)}`
            },
            reddit: {
                name: 'Reddit',
                icon: `<svg viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5">
                    <path d="M12 0A12 12 0 0 0 0 12a12 12 0 0 0 12 12 12 12 0 0 0 12-12A12 12 0 0 0 12 0zm5.01 4.744c.688 0 1.25.561 1.25 1.249a1.25 1.25 0 0 1-2.498.056l-2.597-.547-.8 3.747c1.824.07 3.48.632 4.674 1.488.308-.309.73-.491 1.207-.491.968 0 1.754.786 1.754 1.754 0 .716-.435 1.333-1.01 1.614a3.111 3.111 0 0 1 .042.52c0 2.694-3.13 4.87-7.004 4.87-3.874 0-7.004-2.176-7.004-4.87 0-.183.015-.366.043-.534A1.748 1.748 0 0 1 4.028 12c0-.968.786-1.754 1.754-1.754.463 0 .898.196 1.207.49 1.207-.883 2.878-1.43 4.744-1.487l.885-4.182a.342.342 0 0 1 .14-.197.35.35 0 0 1 .238-.042l2.906.617a1.214 1.214 0 0 1 1.108-.701zM9.25 12C8.561 12 8 12.562 8 13.25c0 .687.561 1.248 1.25 1.248.687 0 1.248-.561 1.248-1.249 0-.688-.561-1.249-1.249-1.249zm5.5 0c-.687 0-1.248.561-1.248 1.25 0 .687.561 1.248 1.249 1.248.688 0 1.249-.561 1.249-1.249 0-.687-.562-1.249-1.25-1.249zm-5.466 3.99a.327.327 0 0 0-.231.094.33.33 0 0 0 0 .463c.842.842 2.484.913 2.961.913.477 0 2.105-.056 2.961-.913a.361.361 0 0 0 .029-.463.33.33 0 0 0-.464 0c-.547.533-1.684.73-2.512.73-.828 0-1.979-.196-2.512-.73a.326.326 0 0 0-.232-.095z"/>
                </svg>`,
                color: '#FF4500',
                shareUrl: (url, title) => `https://reddit.com/submit?url=${encodeURIComponent(url)}&title=${encodeURIComponent(title)}`
            }
        };
        
        this.analytics = {
            shares: {},
            clicks: {}
        };
    }

    init(containerId, options = {}) {
        const container = document.getElementById(containerId);
        if (!container) return;

        const {
            url = window.location.href,
            title = document.title,
            description = '',
            platforms = ['twitter', 'facebook', 'instagram', 'tiktok'],
            floating = false,
            theme = 'glass'
        } = options;

        container.innerHTML = '';
        container.className = floating ? 'social-share-floating' : 'social-share-widget';

        // Add floating styles if needed
        if (floating) {
            container.style.cssText = `
                position: fixed;
                bottom: 24px;
                right: 24px;
                display: flex;
                flex-direction: column;
                gap: 12px;
                z-index: 1000;
            `;
        }

        // Create share buttons
        platforms.forEach(platform => {
            if (!this.platforms[platform]) return;

            const button = this.createShareButton(platform, url, title, description, theme);
            container.appendChild(button);
        });

        // Add copy link button
        const copyButton = this.createCopyButton(url, theme);
        container.appendChild(copyButton);
    }

    createShareButton(platform, url, title, description, theme) {
        const config = this.platforms[platform];
        const button = document.createElement('button');
        
        button.className = `social-share-button ${theme}`;
        button.innerHTML = config.icon;
        button.style.cssText = `
            width: 48px;
            height: 48px;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
            position: relative;
            overflow: hidden;
        `;

        // Add platform-specific hover color
        button.addEventListener('mouseenter', () => {
            button.style.borderColor = config.color;
            button.style.boxShadow = `0 0 20px ${config.color}40`;
            button.style.transform = 'translateY(-2px) scale(1.05)';
        });

        button.addEventListener('mouseleave', () => {
            button.style.borderColor = 'rgba(255, 255, 255, 0.2)';
            button.style.boxShadow = 'none';
            button.style.transform = 'translateY(0) scale(1)';
        });

        // Handle clicks
        button.addEventListener('click', () => {
            this.handleShare(platform, url, title, description);
            this.showShareFeedback(button);
        });

        // Add tooltip
        const tooltip = document.createElement('span');
        tooltip.textContent = `Share on ${config.name}`;
        tooltip.style.cssText = `
            position: absolute;
            bottom: 100%;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            font-size: 12px;
            white-space: nowrap;
            opacity: 0;
            pointer-events: none;
            transition: opacity 0.3s ease;
            margin-bottom: 8px;
        `;

        button.appendChild(tooltip);
        button.addEventListener('mouseenter', () => { tooltip.style.opacity = '1'; });
        button.addEventListener('mouseleave', () => { tooltip.style.opacity = '0'; });

        return button;
    }

    createCopyButton(url, theme) {
        const button = document.createElement('button');
        button.className = `social-share-button ${theme}`;
        button.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="w-5 h-5">
                <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"></path>
                <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"></path>
            </svg>
        `;
        
        button.style.cssText = `
            width: 48px;
            height: 48px;
            border-radius: 24px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: all 0.3s ease;
            color: white;
        `;

        button.addEventListener('click', () => {
            this.copyToClipboard(url);
            this.showCopyFeedback(button);
        });

        return button;
    }

    handleShare(platform, url, title, description) {
        const config = this.platforms[platform];
        
        // Track share
        this.trackShare(platform);

        // Special handling for platforms without direct URL sharing
        if (platform === 'instagram' || platform === 'tiktok') {
            this.showMobileSharePrompt(platform, url, title);
            return;
        }

        // Generate share URL
        const shareUrl = config.shareUrl(url, `${title} - ${description}`);
        
        // Open in new window
        window.open(shareUrl, '_blank', 'width=600,height=400');
    }

    showMobileSharePrompt(platform, url, title) {
        const modal = document.createElement('div');
        modal.style.cssText = `
            position: fixed;
            inset: 0;
            background: rgba(0, 0, 0, 0.8);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            display: flex;
            align-items: center;
            justify-content: center;
            z-index: 10000;
            animation: fadeIn 0.3s ease;
        `;

        const content = document.createElement('div');
        content.style.cssText = `
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            -webkit-backdrop-filter: blur(20px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 24px;
            padding: 32px;
            max-width: 400px;
            text-align: center;
            color: white;
        `;

        content.innerHTML = `
            <h3 style="font-size: 24px; margin-bottom: 16px; background: linear-gradient(45deg, #FF006E, #8338EC); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Share on ${this.platforms[platform].name}
            </h3>
            <p style="margin-bottom: 24px; color: rgba(255, 255, 255, 0.8);">
                ${platform === 'instagram' ? 'Take a screenshot and share it to your story!' : 'Create a video about this and share it!'}
            </p>
            <div style="background: rgba(0, 0, 0, 0.3); padding: 16px; border-radius: 12px; margin-bottom: 24px;">
                <p style="font-size: 14px; margin-bottom: 8px; color: rgba(255, 255, 255, 0.6);">Link copied to clipboard:</p>
                <p style="font-size: 12px; word-break: break-all; color: rgba(255, 255, 255, 0.8);">${url}</p>
            </div>
            <button onclick="this.parentElement.parentElement.remove()" style="
                background: linear-gradient(45deg, #FF006E, #8338EC);
                border: none;
                color: white;
                padding: 12px 32px;
                border-radius: 24px;
                font-size: 16px;
                cursor: pointer;
                transition: transform 0.2s ease;
            " onmouseover="this.style.transform='scale(1.05)'" onmouseout="this.style.transform='scale(1)'">
                Got it!
            </button>
        `;

        modal.appendChild(content);
        document.body.appendChild(modal);

        // Copy URL to clipboard
        this.copyToClipboard(url);

        // Auto-remove after 10 seconds
        setTimeout(() => {
            if (modal.parentElement) {
                modal.remove();
            }
        }, 10000);
    }

    copyToClipboard(text) {
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text);
        } else {
            // Fallback for older browsers
            const textarea = document.createElement('textarea');
            textarea.value = text;
            textarea.style.position = 'fixed';
            textarea.style.opacity = '0';
            document.body.appendChild(textarea);
            textarea.select();
            document.execCommand('copy');
            document.body.removeChild(textarea);
        }
    }

    showShareFeedback(button) {
        const feedback = document.createElement('div');
        feedback.style.cssText = `
            position: absolute;
            inset: 0;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            animation: ripple 0.6s ease-out;
        `;

        button.appendChild(feedback);
        setTimeout(() => feedback.remove(), 600);
    }

    showCopyFeedback(button) {
        const originalHTML = button.innerHTML;
        button.innerHTML = `
            <svg viewBox="0 0 24 24" fill="currentColor" class="w-5 h-5" style="color: #06FFA5;">
                <path d="M9 16.17L4.83 12l-1.42 1.41L9 19 21 7l-1.41-1.41z"/>
            </svg>
        `;
        
        setTimeout(() => {
            button.innerHTML = originalHTML;
        }, 2000);
    }

    trackShare(platform) {
        if (!this.analytics.shares[platform]) {
            this.analytics.shares[platform] = 0;
        }
        this.analytics.shares[platform]++;
        
        // Send to analytics service if configured
        if (window.gtag) {
            window.gtag('event', 'share', {
                method: platform,
                content_type: 'media',
                item_id: window.location.pathname
            });
        }
    }

    getAnalytics() {
        return this.analytics;
    }
}

// Create global instance
window.socialShare = new SocialShareWidget();

// Export initialization function
window.initSocialShare = (containerId, options) => {
    window.socialShare.init(containerId, options);
};

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    
    @keyframes ripple {
        0% {
            transform: scale(0);
            opacity: 1;
        }
        100% {
            transform: scale(4);
            opacity: 0;
        }
    }
    
    .social-share-button:active {
        transform: scale(0.95);
    }
`;
document.head.appendChild(style);