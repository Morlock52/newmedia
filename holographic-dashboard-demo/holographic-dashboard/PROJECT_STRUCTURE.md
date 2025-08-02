# Project Structure

This document provides an overview of the Holographic Media Dashboard project structure and organization.

## 📁 Directory Structure

```
holographic-dashboard/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 CONTRIBUTING.md               # Contribution guidelines
├── 📄 SECURITY.md                   # Security policy
├── 📄 PROJECT_STRUCTURE.md          # This file
├── 📄 package.json                  # Node.js dependencies and scripts
├── 📄 package-lock.json             # Exact dependency versions
├── 📄 .gitignore                    # Git ignore rules
├── 📄 .eslintrc.js                  # ESLint configuration
├── 📄 lighthouse.config.js          # Lighthouse CI configuration
│
├── 🎨 css/                          # Stylesheets
│   ├── main.css                     # Core application styles
│   ├── holographic.css              # 3D holographic effects
│   ├── navigation.css               # Navigation components
│   ├── responsive-navigation.css    # Mobile navigation
│   └── ui-components.css            # UI component styles
│
├── 🔧 js/                           # JavaScript modules
│   ├── main.js                      # Application entry point
│   ├── config.js                    # Configuration settings
│   ├── holographic-scene.js         # Main 3D scene manager
│   ├── particles.js                 # Particle system effects
│   ├── shaders.js                   # WebGL shader definitions
│   ├── audio-visualizer.js          # Audio visualization
│   ├── media-cards.js               # 3D media card system
│   ├── websocket-client.js          # WebSocket communication
│   ├── ui-controller.js             # UI interaction handling
│   ├── navigation-manager.js        # Navigation system
│   ├── page-manager.js              # Page/view management
│   ├── router.js                    # Client-side routing
│   ├── utils.js                     # Utility functions
│   ├── debug.js                     # Debug and performance tools
│   └── lib/                         # Third-party libraries
│       ├── three.min.js             # Three.js WebGL library
│       └── OrbitControls.js         # Camera orbit controls
│
├── 🖼️ assets/                       # Static assets
│   ├── icons/                       # Application icons
│   ├── images/                      # Images and textures
│   └── fonts/                       # Custom fonts (if any)
│
├── 📚 docs/                         # Documentation
│   ├── README.md                    # Documentation overview
│   ├── API.md                       # API reference
│   ├── ARCHITECTURE.md              # Technical architecture
│   ├── DEPLOYMENT.md                # Deployment guide
│   ├── FEATURES.md                  # Feature descriptions
│   ├── TROUBLESHOOTING.md           # Common issues and solutions
│   ├── CHANGELOG.md                 # Version history
│   └── screenshots/                 # Visual documentation
│
├── 🧪 examples/                     # Usage examples
│   ├── basic-integration.html       # Simple integration example
│   ├── custom-theme.html            # Theming example
│   └── websocket-integration.js     # WebSocket server example
│
├── ⚙️ .github/                      # GitHub configuration
│   ├── workflows/                   # GitHub Actions
│   │   └── ci.yml                   # Continuous integration
│   ├── ISSUE_TEMPLATE/              # Issue templates
│   │   ├── bug_report.md            # Bug report template
│   │   └── feature_request.md       # Feature request template
│   └── pull_request_template.md     # PR template
│
├── 🌐 HTML Files                    # Application pages
│   ├── index.html                   # Main dashboard page
│   ├── complete-dashboard.html      # Full-featured version
│   ├── demo-showcase.html           # Feature showcase
│   └── [other-html-files]           # Various development versions
│
├── 🏗️ Build Files                   # Build and development
│   ├── demo-server.js               # Development server
│   └── dist/                        # Built/minified files (generated)
│       ├── app.min.js               # Minified JavaScript
│       └── styles.min.css           # Minified CSS
│
└── 📊 Development Reports           # Analysis and reports
    ├── INITIALIZATION_ANALYSIS_REPORT.md
    ├── MEDIA_SERVER_CONFIG_GUIDE.md
    └── NAVIGATION_IMPLEMENTATION_SUMMARY.md
```

## 🔍 File Categories

### Core Application Files
- **HTML Pages**: Entry points for different versions of the dashboard
- **CSS Stylesheets**: Visual styling and 3D effects
- **JavaScript Modules**: Core functionality and WebGL rendering
- **Configuration**: Settings and environment configuration

### Documentation
- **User Documentation**: README, features, troubleshooting
- **Developer Documentation**: API reference, architecture, contributing
- **Project Documentation**: Changelog, security policy, structure

### Development & Build
- **Package Configuration**: package.json, dependencies
- **Build Tools**: ESLint, Lighthouse, GitHub Actions
- **Development Server**: Node.js server for local development
- **Examples**: Integration examples and demos

### Assets & Resources
- **Static Assets**: Images, icons, fonts
- **Third-party Libraries**: Three.js, WebGL utilities
- **Generated Content**: Build artifacts, minified files

## 🎯 Key File Relationships

### Entry Points
```
User Access Points:
├── index.html                 → main.js → holographic-scene.js
├── complete-dashboard.html    → enhanced-main.js → full feature set
└── demo-showcase.html         → demo-specific features
```

### Core Dependencies
```
3D Rendering Pipeline:
main.js
├── holographic-scene.js       # Main 3D scene
│   ├── particles.js           # Particle effects
│   ├── shaders.js             # Custom materials
│   ├── media-cards.js         # Interactive cards
│   └── audio-visualizer.js    # Audio visualization
├── websocket-client.js        # Real-time data
├── ui-controller.js           # User interaction
└── config.js                  # Configuration
```

### Styling System
```
CSS Architecture:
main.css                       # Base styles and layout
├── holographic.css            # 3D effects and materials
├── navigation.css             # Navigation components
├── responsive-navigation.css  # Mobile adaptations
└── ui-components.css          # UI element styles
```

## 📦 Module Organization

### JavaScript Modules
- **Scene Management**: holographic-scene.js, particles.js, shaders.js
- **User Interface**: ui-controller.js, navigation-manager.js, page-manager.js
- **Communication**: websocket-client.js, api-client.js
- **Utilities**: utils.js, debug.js, config.js
- **Libraries**: lib/ directory with third-party code

### CSS Organization
- **Base Styles**: Typography, colors, layout fundamentals
- **Components**: Reusable UI component styles
- **Effects**: 3D transforms, animations, WebGL integration
- **Responsive**: Mobile and tablet adaptations

## 🔧 Development Workflow

### Local Development
1. **Setup**: npm install dependencies
2. **Development**: npm run dev (starts demo server)
3. **Testing**: Open browser to localhost:9999
4. **Linting**: npm run lint for code quality
5. **Building**: npm run build for production files

### File Modification Patterns
- **New Features**: Add to js/ directory, update main.js imports
- **Styling**: Modify CSS files, consider responsive impact
- **Configuration**: Update config.js for new settings
- **Documentation**: Update relevant docs/ files
- **Examples**: Add to examples/ directory with clear naming

### Build Process
```
Source Files → Processing → Distribution
├── JavaScript: Minification, bundling
├── CSS: Minification, prefixing
├── HTML: Validation, optimization
└── Assets: Compression, optimization
```

## 🚀 Deployment Structure

### Static Hosting
```
Deployment Package:
├── HTML files (entry points)
├── css/ (stylesheets)
├── js/ (JavaScript modules)
├── assets/ (static assets)
├── examples/ (usage examples)
└── docs/ (documentation)
```

### CDN Organization
- **Static Assets**: Images, fonts, large libraries
- **Application Code**: Main JavaScript and CSS
- **Documentation**: Hosted docs and guides

## 🔍 Finding Files

### Common Tasks
- **Add new feature**: Create file in js/, import in main.js
- **Modify styles**: Edit relevant CSS file in css/
- **Update documentation**: Edit appropriate file in docs/
- **Fix bug**: Locate in js/ directory, consider css/ if visual
- **Add example**: Create new file in examples/
- **Configure build**: Modify package.json or config files

### File Naming Conventions
- **JavaScript**: kebab-case.js (e.g., holographic-scene.js)
- **CSS**: kebab-case.css (e.g., responsive-navigation.css)
- **HTML**: kebab-case.html (e.g., demo-showcase.html)
- **Documentation**: UPPER_CASE.md (e.g., CONTRIBUTING.md)
- **Examples**: descriptive-name.html (e.g., basic-integration.html)

## 📊 File Size Guidelines

### Performance Targets
- **Individual JS files**: < 50KB (unminified)
- **Individual CSS files**: < 20KB
- **Total bundle size**: < 500KB (minified + gzipped)
- **Images/Assets**: Optimized for web delivery
- **Documentation**: No size limits, prioritize clarity

### Optimization Strategies
- **Code Splitting**: Separate features into modules
- **Lazy Loading**: Load non-critical resources later
- **Minification**: Use build tools for production
- **Compression**: Enable gzip/brotli on server
- **CDN Usage**: Serve static assets from CDN

---

This structure is designed for maintainability, scalability, and clear separation of concerns. Each directory and file has a specific purpose and follows established conventions for web development projects.