/**
 * Navigation Header Component
 * Provides consistent navigation across all pages with back button, breadcrumbs, and responsive design
 */

class NavigationHeader extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        this.loadResponsiveCSS();
        this.render();
        this.setupEventListeners();
    }

    loadResponsiveCSS() {
        // Load responsive navigation CSS if not already loaded
        if (!document.getElementById('responsive-navigation-css')) {
            const link = document.createElement('link');
            link.id = 'responsive-navigation-css';
            link.rel = 'stylesheet';
            link.href = 'css/responsive-navigation.css';
            document.head.appendChild(link);
        }
    }

    render() {
        const pageTitle = this.getAttribute('title') || 'HoloMedia Hub';
        const showBackButton = this.getAttribute('show-back') !== 'false';
        const breadcrumbs = this.getAttribute('breadcrumbs') || '';
        const currentPage = this.getAttribute('current-page') || '';

        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    position: sticky;
                    top: 0;
                    z-index: 1000;
                    width: 100%;
                }

                .nav-header {
                    background: linear-gradient(135deg, 
                        rgba(10, 10, 20, 0.95) 0%, 
                        rgba(15, 15, 25, 0.98) 100%
                    );
                    backdrop-filter: blur(20px) saturate(150%);
                    border-bottom: 1px solid rgba(0, 255, 255, 0.2);
                    padding: 1rem 2rem;
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 1rem;
                    position: relative;
                    overflow: hidden;
                }

                .nav-header::before {
                    content: '';
                    position: absolute;
                    top: 0;
                    left: -100%;
                    width: 300%;
                    height: 1px;
                    background: linear-gradient(90deg,
                        transparent,
                        rgba(0, 255, 255, 0.5),
                        rgba(255, 0, 255, 0.5),
                        rgba(255, 255, 0, 0.5),
                        transparent
                    );
                    animation: scan 4s linear infinite;
                }

                @keyframes scan {
                    0% { transform: translateX(-50%); }
                    100% { transform: translateX(50%); }
                }

                .nav-left {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    flex: 1;
                }

                .back-button {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    padding: 0.75rem 1rem;
                    background: rgba(0, 255, 255, 0.1);
                    border: 1px solid rgba(0, 255, 255, 0.3);
                    border-radius: 12px;
                    color: #00ffff;
                    text-decoration: none;
                    font-weight: 500;
                    transition: all 0.3s ease;
                    font-size: 0.9rem;
                    cursor: pointer;
                }

                .back-button:hover {
                    background: rgba(0, 255, 255, 0.2);
                    border-color: rgba(0, 255, 255, 0.5);
                    transform: translateX(-2px);
                    box-shadow: 0 0 20px rgba(0, 255, 255, 0.3);
                }

                .back-button:active {
                    transform: translateX(-1px);
                }

                .back-icon {
                    font-size: 1.2em;
                    transition: transform 0.3s ease;
                }

                .back-button:hover .back-icon {
                    transform: translateX(-2px);
                }

                .nav-title-section {
                    display: flex;
                    flex-direction: column;
                    gap: 0.25rem;
                    flex: 1;
                    min-width: 0;
                }

                .page-title {
                    font-family: 'Orbitron', monospace;
                    font-size: 1.5rem;
                    font-weight: 700;
                    background: linear-gradient(90deg,
                        #00ffff 0%,
                        #ff00ff 50%,
                        #ffff00 100%
                    );
                    background-size: 200% auto;
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                    animation: gradient-shift 3s linear infinite;
                    margin: 0;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }

                @keyframes gradient-shift {
                    0% { background-position: 0% center; }
                    100% { background-position: 200% center; }
                }

                .breadcrumbs {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.85rem;
                    color: rgba(255, 255, 255, 0.7);
                    flex-wrap: wrap;
                }

                .breadcrumb-item {
                    color: rgba(0, 255, 255, 0.8);
                    text-decoration: none;
                    transition: color 0.3s ease;
                    white-space: nowrap;
                }

                .breadcrumb-item:hover {
                    color: #00ffff;
                    text-shadow: 0 0 5px rgba(0, 255, 255, 0.5);
                }

                .breadcrumb-separator {
                    color: rgba(255, 255, 255, 0.4);
                    font-size: 0.8em;
                }

                .breadcrumb-current {
                    color: rgba(255, 255, 255, 0.9);
                    font-weight: 500;
                }

                .nav-actions {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }

                .nav-action {
                    padding: 0.5rem;
                    background: rgba(255, 255, 255, 0.05);
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 8px;
                    color: rgba(255, 255, 255, 0.8);
                    cursor: pointer;
                    transition: all 0.3s ease;
                    font-size: 1.1rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    width: 40px;
                    height: 40px;
                }

                .nav-action:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border-color: rgba(0, 255, 255, 0.5);
                    color: #00ffff;
                    transform: translateY(-1px);
                }

                .mobile-menu {
                    display: none;
                    flex-direction: column;
                    gap: 0.5rem;
                    width: 100%;
                    padding: 1rem 0;
                    border-top: 1px solid rgba(255, 255, 255, 0.1);
                    margin-top: 1rem;
                }

                .mobile-menu.active {
                    display: flex;
                }

                .mobile-nav-item {
                    padding: 0.75rem 1rem;
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    color: rgba(255, 255, 255, 0.9);
                    text-decoration: none;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                }

                .mobile-nav-item:hover {
                    background: rgba(0, 255, 255, 0.1);
                    color: #00ffff;
                }

                /* Responsive Design */
                @media (max-width: 768px) {
                    .nav-header {
                        padding: 1rem;
                        flex-wrap: wrap;
                    }

                    .nav-left {
                        width: 100%;
                        justify-content: space-between;
                    }

                    .page-title {
                        font-size: 1.2rem;
                    }

                    .breadcrumbs {
                        display: none;
                    }

                    .nav-actions {
                        gap: 0.5rem;
                    }

                    .nav-action {
                        width: 36px;
                        height: 36px;
                        font-size: 1rem;
                    }

                    .back-button {
                        padding: 0.5rem 0.75rem;
                        font-size: 0.85rem;
                    }
                }

                @media (max-width: 480px) {
                    .nav-header {
                        padding: 0.75rem;
                    }

                    .page-title {
                        font-size: 1rem;
                    }

                    .back-button .back-text {
                        display: none;
                    }

                    .nav-title-section {
                        flex: unset;
                        width: 100%;
                        text-align: center;
                    }

                    .nav-left {
                        flex-direction: column;
                        gap: 0.5rem;
                    }
                }

                /* Dark mode compatibility */
                @media (prefers-color-scheme: dark) {
                    .nav-header {
                        background: linear-gradient(135deg, 
                            rgba(5, 5, 10, 0.98) 0%, 
                            rgba(10, 10, 15, 0.99) 100%
                        );
                    }
                }

                /* High contrast mode */
                @media (prefers-contrast: high) {
                    .nav-header {
                        border-bottom: 2px solid #00ffff;
                    }
                    
                    .back-button {
                        border: 2px solid #00ffff;
                    }
                }

                /* Reduced motion */
                @media (prefers-reduced-motion: reduce) {
                    .nav-header::before,
                    .page-title {
                        animation: none;
                    }
                    
                    .back-button,
                    .nav-action {
                        transition: none;
                    }
                }
            </style>

            <header class="nav-header">
                <div class="nav-left">
                    ${showBackButton ? `
                        <button class="back-button" id="backButton">
                            <span class="back-icon">←</span>
                            <span class="back-text">Back</span>
                        </button>
                    ` : ''}
                    
                    <div class="nav-title-section">
                        <h1 class="page-title">${pageTitle}</h1>
                        ${breadcrumbs ? `
                            <nav class="breadcrumbs" aria-label="Breadcrumb">
                                ${this.renderBreadcrumbs(breadcrumbs, currentPage)}
                            </nav>
                        ` : ''}
                    </div>
                </div>

                <div class="nav-actions">
                    <button class="nav-action" id="homeButton" title="Home" aria-label="Go to home page">
                        🏠
                    </button>
                    <button class="nav-action" id="refreshButton" title="Refresh" aria-label="Refresh page">
                        🔄
                    </button>
                    <button class="nav-action" id="settingsButton" title="Settings" aria-label="Open settings">
                        ⚙️
                    </button>
                    <button class="nav-action" id="menuButton" title="Menu" aria-label="Toggle menu" style="display: none;">
                        ☰
                    </button>
                </div>

                <div class="mobile-menu" id="mobileMenu">
                    <a href="main-app.html" class="mobile-nav-item">
                        <span>🏠</span>
                        <span>Dashboard</span>
                    </a>
                    <a href="config-manager-fixed.html" class="mobile-nav-item">
                        <span>⚙️</span>
                        <span>Configuration</span>
                    </a>
                    <a href="smart-env-editor.html" class="mobile-nav-item">
                        <span>🔧</span>
                        <span>Env Editor</span>
                    </a>
                    <a href="media-assistant.html" class="mobile-nav-item">
                        <span>🤖</span>
                        <span>AI Assistant</span>
                    </a>
                </div>
            </header>
        `;
    }

    renderBreadcrumbs(breadcrumbsStr, currentPage) {
        if (!breadcrumbsStr) return '';
        
        const items = breadcrumbsStr.split(',').map(item => item.trim());
        let breadcrumbHTML = '';
        
        items.forEach((item, index) => {
            if (index > 0) {
                breadcrumbHTML += '<span class="breadcrumb-separator">></span>';
            }
            
            if (index === items.length - 1 || item === currentPage) {
                breadcrumbHTML += `<span class="breadcrumb-current">${item}</span>`;
            } else {
                const href = this.getBreadcrumbHref(item);
                breadcrumbHTML += `<a href="${href}" class="breadcrumb-item">${item}</a>`;
            }
        });
        
        return breadcrumbHTML;
    }

    getBreadcrumbHref(item) {
        const itemMap = {
            'Home': 'main-app.html',
            'Dashboard': 'main-app.html',
            'Configuration': 'config-manager-fixed.html',
            'Env Editor': 'smart-env-editor.html',
            'AI Assistant': 'media-assistant.html',
            'Environment': 'env-viewer.html',
            'AI Config': 'ai-config-manager.html'
        };
        
        return itemMap[item] || 'main-app.html';
    }

    setupEventListeners() {
        const backButton = this.shadowRoot.getElementById('backButton');
        const homeButton = this.shadowRoot.getElementById('homeButton');
        const refreshButton = this.shadowRoot.getElementById('refreshButton');
        const settingsButton = this.shadowRoot.getElementById('settingsButton');
        const menuButton = this.shadowRoot.getElementById('menuButton');
        const mobileMenu = this.shadowRoot.getElementById('mobileMenu');

        if (backButton) {
            backButton.addEventListener('click', (e) => {
                e.preventDefault();
                this.handleBackNavigation();
            });
        }

        if (homeButton) {
            homeButton.addEventListener('click', () => {
                window.location.href = 'main-app.html';
            });
        }

        if (refreshButton) {
            refreshButton.addEventListener('click', () => {
                window.location.reload();
            });
        }

        if (settingsButton) {
            settingsButton.addEventListener('click', () => {
                this.dispatchEvent(new CustomEvent('settings-requested', {
                    bubbles: true,
                    detail: { source: 'navigation-header' }
                }));
            });
        }

        if (menuButton && mobileMenu) {
            menuButton.addEventListener('click', () => {
                mobileMenu.classList.toggle('active');
                const isActive = mobileMenu.classList.contains('active');
                menuButton.setAttribute('aria-expanded', isActive.toString());
            });
        }

        // Handle keyboard navigation
        this.shadowRoot.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && mobileMenu) {
                mobileMenu.classList.remove('active');
                menuButton.setAttribute('aria-expanded', 'false');
            }
        });

        // Handle responsive menu visibility
        this.handleResponsiveMenu();
        window.addEventListener('resize', () => this.handleResponsiveMenu());
    }

    handleBackNavigation() {
        const backUrl = this.getAttribute('back-url');
        
        if (backUrl) {
            window.location.href = backUrl;
        } else if (window.history.length > 1) {
            window.history.back();
        } else {
            window.location.href = 'main-app.html';
        }

        // Dispatch custom event for analytics or other listeners
        this.dispatchEvent(new CustomEvent('navigation-back', {
            bubbles: true,
            detail: { 
                from: window.location.pathname,
                to: backUrl || 'history-back'
            }
        }));
    }

    handleResponsiveMenu() {
        const menuButton = this.shadowRoot.getElementById('menuButton');
        const navActions = this.shadowRoot.querySelector('.nav-actions');
        
        if (window.innerWidth <= 768) {
            if (menuButton) menuButton.style.display = 'flex';
        } else {
            if (menuButton) menuButton.style.display = 'none';
            const mobileMenu = this.shadowRoot.getElementById('mobileMenu');
            if (mobileMenu) mobileMenu.classList.remove('active');
        }
    }

    // Public methods for external control
    updateTitle(newTitle) {
        const titleElement = this.shadowRoot.querySelector('.page-title');
        if (titleElement) {
            titleElement.textContent = newTitle;
        }
        this.setAttribute('title', newTitle);
    }

    updateBreadcrumbs(newBreadcrumbs, currentPage) {
        const breadcrumbsElement = this.shadowRoot.querySelector('.breadcrumbs');
        if (breadcrumbsElement) {
            breadcrumbsElement.innerHTML = this.renderBreadcrumbs(newBreadcrumbs, currentPage);
        }
        this.setAttribute('breadcrumbs', newBreadcrumbs);
        if (currentPage) this.setAttribute('current-page', currentPage);
    }

    showLoading() {
        const refreshButton = this.shadowRoot.getElementById('refreshButton');
        if (refreshButton) {
            refreshButton.style.animation = 'spin 1s linear infinite';
        }
    }

    hideLoading() {
        const refreshButton = this.shadowRoot.getElementById('refreshButton');
        if (refreshButton) {
            refreshButton.style.animation = '';
        }
    }
}

// Define the custom element
customElements.define('navigation-header', NavigationHeader);

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = NavigationHeader;
}