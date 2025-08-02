// Navigation Component for HoloMedia Hub
// Modern web component for consistent navigation across all pages

class NavigationHeader extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.currentPath = window.location.pathname;
        this.isMenuOpen = false;
    }

    connectedCallback() {
        this.render();
        this.setupEventListeners();
        this.updateActiveState();
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    position: sticky;
                    top: 0;
                    z-index: 1000;
                    --header-height: 70px;
                    --neon-cyan: #00ffff;
                    --neon-pink: #ff006e;
                    --neon-purple: #8338ec;
                    --dark-bg: #0a0a0f;
                    --glass-bg: rgba(10, 10, 15, 0.8);
                    --glass-border: rgba(255, 255, 255, 0.1);
                    --text-primary: #ffffff;
                    --text-secondary: #a0a0b8;
                }

                .nav-header {
                    height: var(--header-height);
                    background: var(--glass-bg);
                    backdrop-filter: blur(20px);
                    -webkit-backdrop-filter: blur(20px);
                    border-bottom: 1px solid var(--glass-border);
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    padding: 0 2rem;
                    transition: all 0.3s ease;
                }

                .nav-header.scrolled {
                    background: rgba(10, 10, 15, 0.95);
                    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
                }

                .nav-brand {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    text-decoration: none;
                    color: var(--text-primary);
                    transition: transform 0.3s ease;
                }

                .nav-brand:hover {
                    transform: translateY(-2px);
                }

                .brand-icon {
                    width: 40px;
                    height: 40px;
                    background: linear-gradient(45deg, var(--neon-cyan), var(--neon-pink));
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 900;
                    font-size: 1.5rem;
                    color: var(--dark-bg);
                    animation: glow-pulse 2s ease-in-out infinite;
                }

                @keyframes glow-pulse {
                    0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.5); }
                    50% { box-shadow: 0 0 40px rgba(0, 255, 255, 0.8); }
                }

                .brand-text {
                    font-size: 1.5rem;
                    font-weight: 700;
                    background: linear-gradient(45deg, var(--neon-cyan), var(--neon-pink));
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }

                .nav-menu {
                    display: flex;
                    align-items: center;
                    gap: 2rem;
                    list-style: none;
                    margin: 0;
                    padding: 0;
                }

                .nav-item {
                    position: relative;
                }

                .nav-link {
                    color: var(--text-secondary);
                    text-decoration: none;
                    padding: 0.5rem 1rem;
                    border-radius: 8px;
                    transition: all 0.3s ease;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    position: relative;
                    overflow: hidden;
                }

                .nav-link::before {
                    content: '';
                    position: absolute;
                    bottom: 0;
                    left: 50%;
                    width: 0;
                    height: 2px;
                    background: var(--neon-cyan);
                    transform: translateX(-50%);
                    transition: width 0.3s ease;
                }

                .nav-link:hover {
                    color: var(--text-primary);
                    background: rgba(255, 255, 255, 0.05);
                }

                .nav-link:hover::before {
                    width: 80%;
                }

                .nav-link.active {
                    color: var(--neon-cyan);
                    background: rgba(0, 255, 255, 0.1);
                }

                .nav-link.active::before {
                    width: 100%;
                    height: 3px;
                }

                .nav-icon {
                    font-size: 1.2rem;
                }

                .nav-actions {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                }

                .search-container {
                    position: relative;
                    width: 250px;
                }

                .search-input {
                    width: 100%;
                    padding: 0.6rem 1rem 0.6rem 2.5rem;
                    background: var(--glass-bg);
                    border: 1px solid var(--glass-border);
                    border-radius: 8px;
                    color: var(--text-primary);
                    font-size: 0.9rem;
                    transition: all 0.3s ease;
                    outline: none;
                }

                .search-input:focus {
                    border-color: var(--neon-cyan);
                    box-shadow: 0 0 20px rgba(0, 255, 255, 0.2);
                    width: 300px;
                }

                .search-icon {
                    position: absolute;
                    left: 0.8rem;
                    top: 50%;
                    transform: translateY(-50%);
                    color: var(--text-secondary);
                    pointer-events: none;
                }

                .notification-btn {
                    position: relative;
                    background: transparent;
                    border: none;
                    color: var(--text-secondary);
                    cursor: pointer;
                    padding: 0.5rem;
                    border-radius: 8px;
                    transition: all 0.3s ease;
                }

                .notification-btn:hover {
                    color: var(--text-primary);
                    background: rgba(255, 255, 255, 0.05);
                }

                .notification-badge {
                    position: absolute;
                    top: 0;
                    right: 0;
                    background: var(--neon-pink);
                    color: white;
                    font-size: 0.7rem;
                    font-weight: 600;
                    padding: 2px 6px;
                    border-radius: 10px;
                    min-width: 18px;
                    text-align: center;
                }

                .user-menu {
                    display: flex;
                    align-items: center;
                    gap: 0.75rem;
                    padding: 0.5rem 1rem;
                    background: var(--glass-bg);
                    border: 1px solid var(--glass-border);
                    border-radius: 8px;
                    cursor: pointer;
                    transition: all 0.3s ease;
                }

                .user-menu:hover {
                    border-color: var(--neon-cyan);
                    background: rgba(0, 255, 255, 0.1);
                }

                .user-avatar {
                    width: 32px;
                    height: 32px;
                    border-radius: 8px;
                    background: linear-gradient(45deg, var(--neon-purple), var(--neon-pink));
                }

                .mobile-menu-btn {
                    display: none;
                    background: transparent;
                    border: none;
                    color: var(--text-primary);
                    cursor: pointer;
                    padding: 0.5rem;
                }

                .hamburger {
                    width: 24px;
                    height: 20px;
                    position: relative;
                }

                .hamburger span {
                    position: absolute;
                    width: 100%;
                    height: 2px;
                    background: currentColor;
                    transition: all 0.3s ease;
                    left: 0;
                }

                .hamburger span:nth-child(1) { top: 0; }
                .hamburger span:nth-child(2) { top: 9px; }
                .hamburger span:nth-child(3) { bottom: 0; }

                .hamburger.open span:nth-child(1) {
                    transform: rotate(45deg);
                    top: 9px;
                }

                .hamburger.open span:nth-child(2) {
                    opacity: 0;
                }

                .hamburger.open span:nth-child(3) {
                    transform: rotate(-45deg);
                    bottom: 9px;
                }

                /* Mobile Styles */
                @media (max-width: 768px) {
                    .nav-header {
                        padding: 0 1rem;
                    }

                    .mobile-menu-btn {
                        display: block;
                    }

                    .nav-menu {
                        position: fixed;
                        top: var(--header-height);
                        left: -100%;
                        width: 100%;
                        height: calc(100vh - var(--header-height));
                        background: var(--glass-bg);
                        backdrop-filter: blur(20px);
                        flex-direction: column;
                        padding: 2rem;
                        gap: 1rem;
                        transition: left 0.3s ease;
                    }

                    .nav-menu.open {
                        left: 0;
                    }

                    .nav-item {
                        width: 100%;
                    }

                    .nav-link {
                        width: 100%;
                        padding: 1rem;
                        font-size: 1.1rem;
                    }

                    .search-container {
                        display: none;
                    }

                    .nav-actions {
                        margin-left: auto;
                    }
                }

                /* Loading indicator */
                .nav-loading {
                    position: absolute;
                    bottom: 0;
                    left: 0;
                    right: 0;
                    height: 2px;
                    background: linear-gradient(90deg, 
                        transparent, 
                        var(--neon-cyan), 
                        transparent
                    );
                    transform: translateX(-100%);
                    animation: loading-slide 2s linear infinite;
                    opacity: 0;
                    transition: opacity 0.3s ease;
                }

                .nav-loading.active {
                    opacity: 1;
                }

                @keyframes loading-slide {
                    to { transform: translateX(100%); }
                }

                /* Breadcrumbs */
                .breadcrumbs {
                    position: absolute;
                    bottom: -30px;
                    left: 2rem;
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                    font-size: 0.875rem;
                    color: var(--text-secondary);
                    opacity: 0;
                    animation: fade-in 0.5s ease forwards;
                    animation-delay: 0.3s;
                }

                @keyframes fade-in {
                    to { opacity: 1; }
                }

                .breadcrumb-item {
                    display: flex;
                    align-items: center;
                    gap: 0.5rem;
                }

                .breadcrumb-link {
                    color: var(--text-secondary);
                    text-decoration: none;
                    transition: color 0.3s ease;
                }

                .breadcrumb-link:hover {
                    color: var(--neon-cyan);
                }

                .breadcrumb-separator {
                    color: var(--text-secondary);
                    opacity: 0.5;
                }
            </style>

            <nav class="nav-header" id="navHeader">
                <a href="/" class="nav-brand">
                    <div class="brand-icon">H</div>
                    <span class="brand-text">HoloMedia Hub</span>
                </a>

                <ul class="nav-menu" id="navMenu">
                    <li class="nav-item">
                        <a href="/holographic-dashboard/main-app.html" class="nav-link" data-route="/dashboard">
                            <span class="nav-icon">📊</span>
                            <span>Dashboard</span>
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="/holographic-dashboard/main-app.html#media-library" class="nav-link" data-route="/media">
                            <span class="nav-icon">🎬</span>
                            <span>Media</span>
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="/holographic-dashboard/main-app.html#config-manager" class="nav-link" data-route="/config">
                            <span class="nav-icon">⚙️</span>
                            <span>Config</span>
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="/holographic-dashboard/main-app.html#ai-assistant" class="nav-link" data-route="/ai">
                            <span class="nav-icon">🤖</span>
                            <span>AI</span>
                        </a>
                    </li>
                    <li class="nav-item">
                        <a href="/holographic-dashboard/main-app.html#workflow-builder" class="nav-link" data-route="/workflows">
                            <span class="nav-icon">🔄</span>
                            <span>Workflows</span>
                        </a>
                    </li>
                </ul>

                <div class="nav-actions">
                    <div class="search-container">
                        <span class="search-icon">🔍</span>
                        <input type="text" class="search-input" placeholder="Search...">
                    </div>
                    
                    <button class="notification-btn">
                        <span class="nav-icon">🔔</span>
                        <span class="notification-badge">3</span>
                    </button>

                    <div class="user-menu">
                        <div class="user-avatar"></div>
                        <span>Admin</span>
                    </div>

                    <button class="mobile-menu-btn" id="mobileMenuBtn">
                        <div class="hamburger" id="hamburger">
                            <span></span>
                            <span></span>
                            <span></span>
                        </div>
                    </button>
                </div>

                <div class="nav-loading" id="navLoading"></div>
            </nav>

            <div class="breadcrumbs" id="breadcrumbs"></div>
        `;
    }

    setupEventListeners() {
        const shadowRoot = this.shadowRoot;
        
        // Mobile menu toggle
        const mobileMenuBtn = shadowRoot.getElementById('mobileMenuBtn');
        const navMenu = shadowRoot.getElementById('navMenu');
        const hamburger = shadowRoot.getElementById('hamburger');

        mobileMenuBtn.addEventListener('click', () => {
            this.isMenuOpen = !this.isMenuOpen;
            navMenu.classList.toggle('open');
            hamburger.classList.toggle('open');
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.contains(e.target) && this.isMenuOpen) {
                this.isMenuOpen = false;
                navMenu.classList.remove('open');
                hamburger.classList.remove('open');
            }
        });

        // Scroll effect
        let lastScroll = 0;
        window.addEventListener('scroll', () => {
            const currentScroll = window.pageYOffset;
            const navHeader = shadowRoot.getElementById('navHeader');
            
            if (currentScroll > 50) {
                navHeader.classList.add('scrolled');
            } else {
                navHeader.classList.remove('scrolled');
            }

            lastScroll = currentScroll;
        });

        // Navigation loading indicator
        window.addEventListener('beforeunload', () => {
            shadowRoot.getElementById('navLoading').classList.add('active');
        });

        // Search functionality
        const searchInput = shadowRoot.querySelector('.search-input');
        searchInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                this.performSearch(e.target.value);
            }
        });

        // Update active state on navigation
        window.addEventListener('popstate', () => {
            this.updateActiveState();
            this.updateBreadcrumbs();
        });
    }

    updateActiveState() {
        const currentPath = window.location.pathname + window.location.hash;
        const navLinks = this.shadowRoot.querySelectorAll('.nav-link');
        
        navLinks.forEach(link => {
            const route = link.getAttribute('data-route');
            const href = link.getAttribute('href');
            const isActive = currentPath.includes(route) || currentPath === href;
            link.classList.toggle('active', isActive);
        });
    }

    updateBreadcrumbs() {
        const breadcrumbs = this.shadowRoot.getElementById('breadcrumbs');
        const path = window.location.pathname + window.location.hash;
        const parts = path.split('/').filter(p => p);
        
        let html = '<div class="breadcrumb-item">';
        html += '<a href="/" class="breadcrumb-link">Home</a>';
        
        if (parts.length > 0) {
            html += '<span class="breadcrumb-separator">›</span>';
            
            parts.forEach((part, index) => {
                const isLast = index === parts.length - 1;
                const url = '/' + parts.slice(0, index + 1).join('/');
                const name = part.replace(/-/g, ' ').replace(/#/g, '');
                
                if (isLast) {
                    html += `<span>${name}</span>`;
                } else {
                    html += `<a href="${url}" class="breadcrumb-link">${name}</a>`;
                    html += '<span class="breadcrumb-separator">›</span>';
                }
            });
        }
        
        html += '</div>';
        breadcrumbs.innerHTML = html;
    }

    performSearch(query) {
        // Dispatch custom event for search
        this.dispatchEvent(new CustomEvent('navsearch', {
            detail: { query },
            bubbles: true,
            composed: true
        }));
    }

    showLoading() {
        this.shadowRoot.getElementById('navLoading').classList.add('active');
    }

    hideLoading() {
        this.shadowRoot.getElementById('navLoading').classList.remove('active');
    }

    updateNotificationCount(count) {
        const badge = this.shadowRoot.querySelector('.notification-badge');
        badge.textContent = count;
        badge.style.display = count > 0 ? 'block' : 'none';
    }
}

// Register the custom element
customElements.define('navigation-header', NavigationHeader);

// Export for use in other modules
window.NavigationHeader = NavigationHeader;