# Media Server Management Dashboard Design

## 🎨 Design Overview

A modern, gamified media server management dashboard with dual-mode interface (Simple & Advanced) featuring service toggles, .env management, and achievement system.

## 🎯 Design Principles

### Core Design Philosophy
- **Zero Interface Approach**: Anticipate user needs with AI-powered suggestions
- **Hyper-Minimalism**: Clean, purposeful design with maximum functional impact
- **Real-Time Interactivity**: Live updates and dynamic visualizations
- **Mobile-First Adaptive**: Intelligent UI that adapts to device and context
- **Gamification Integration**: Motivational elements without overwhelming functionality

## 🎮 Simple Mode (Newbie View)

### Layout Structure
```
┌─────────────────────────────────────────────────────────────┐
│  🏠 Media Server Hub            [Simple] [Advanced] 👤 Alex  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Welcome back, Alex! 🎉                                    │
│  Your server is 92% healthy                                │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │  Daily Streak: 🔥 7 days    Level 5 Media Master    │  │
│  │  ████████████████████░░░░  450/500 XP              │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Quick Actions                                              │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐         │
│  │   📺    │ │   🎬    │ │   📚    │ │   🎮    │         │
│  │  Watch  │ │ Download│ │ Organize│ │  Play   │         │
│  └─────────┘ └─────────┘ └─────────┘ └─────────┘         │
│                                                             │
│  Essential Services                                         │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Plex          ● Running     [━━━━━━━━━] ON/OFF     │  │
│  │ 📺 Stream anywhere          CPU: 12%  MEM: 2.1GB   │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Sonarr        ● Running     [━━━━━━━━━] ON/OFF     │  │
│  │ 📺 TV Shows automated       CPU: 8%   MEM: 512MB   │  │
│  └─────────────────────────────────────────────────────┘  │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Radarr        ● Running     [━━━━━━━━━] ON/OFF     │  │
│  │ 🎬 Movies automated         CPU: 6%   MEM: 489MB   │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Recent Achievements 🏆                                     │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ ┌────┐                      │
│  │ 🎬 │ │ 📺 │ │ 🔧 │ │ 🚀 │ │ ❓ │ → Next: 10 movies  │
│  └────┘ └────┘ └────┘ └────┘ └────┘                      │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Service Toggle Design
```
┌─────────────────────────────────────────────────────┐
│ Service Name      ● Status     [━━━━━━━━━] Toggle  │
│ 🎯 Simple description         Resources: Low/Med/Hi │
│                                                     │
│ [Expand ▼] for quick settings                      │
└─────────────────────────────────────────────────────┘
```

### Gamification Elements - Simple Mode
- **XP Bar**: Visual progress to next level
- **Daily Streak**: Fire emoji with day count
- **Quick Achievements**: Icon-based recent unlocks
- **Health Meter**: Server status as percentage
- **Resource Indicators**: Simple Low/Med/High labels

## 🚀 Advanced Mode (Techie View)

### Layout Structure
```
┌─────────────────────────────────────────────────────────────┐
│  🖥️ Media Server Control Center  [Simple] [Advanced] ⚙️     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  System Overview                          Achievements      │
│  ┌─────────────────────┐  ┌──────────────────────────┐    │
│  │ CPU █████░░░░░ 52%  │  │ Server Uptime Master     │    │
│  │ RAM ████████░░ 84%  │  │ ████████████░ 95/100 hrs│    │
│  │ SSD ██████░░░░ 61%  │  │                          │    │
│  │ NET ↓2.4GB/s ↑458MB│  │ Automation Wizard        │    │
│  └─────────────────────┘  │ ████████░░░░ 42/50 flows│    │
│                           └──────────────────────────┘    │
│                                                             │
│  Service Matrix                                             │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ Service    Status  CPU   RAM    Disk   Net   Health │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │ Plex       ✓ UP   12%   2.1GB  142GB  ↓24MB  100%  │  │
│  │ Sonarr     ✓ UP   8%    512MB  89GB   ↓2MB   100%  │  │
│  │ Radarr     ✓ UP   6%    489MB  124GB  ↓4MB   100%  │  │
│  │ Prowlarr   ✓ UP   2%    128MB  2GB    ↓1MB   100%  │  │
│  │ Bazarr     ⚠ WARN 15%   768MB  45GB   ↓8MB   85%   │  │
│  │ Overseerr  ✓ UP   3%    256MB  512MB  ↓1MB   100%  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
│  Container Orchestra                    Performance Graph   │
│  ┌─────────────────────┐  ┌──────────────────────────┐    │
│  │ ┌─┐ ┌─┐ ┌─┐ ┌─┐   │  │     CPU Usage (24h)      │    │
│  │ │P│→│S│→│R│→│O│   │  │ 100%┤                     │    │
│  │ └┬┘ └┬┘ └┬┘ └┬┘   │  │  80%┤    ╱╲               │    │
│  │  ↓   ↓   ↓   ↓    │  │  60%┤   ╱  ╲    ╱╲       │    │
│  │ ┌─┐ ┌─┐ ┌─┐ ┌─┐   │  │  40%┤__╱    ╲__╱  ╲____  │    │
│  │ │B│ │T│ │N│ │J│   │  │  20%┤                     │    │
│  │ └─┘ └─┘ └─┘ └─┘   │  │   0%└────────────────────┘│    │
│  └─────────────────────┘  └──────────────────────────┘    │
│                                                             │
│  Environment Configuration (.env)                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │ [Search...] [Filter: All|Modified|Security]         │  │
│  ├─────────────────────────────────────────────────────┤  │
│  │ PLEX_CLAIM         = claim-xxxxxxxxxxxx    [Edit]  │  │
│  │ SONARR_API_KEY     = ••••••••••••••••••    [Show]  │  │
│  │ RADARR_API_KEY     = ••••••••••••••••••    [Show]  │  │
│  │ DOWNLOAD_PATH      = /media/downloads      [Edit]  │  │
│  │ MEDIA_PATH         = /media/library        [Edit]  │  │
│  └─────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Advanced Service Control
```
┌─────────────────────────────────────────────────────────────┐
│ Plex Media Server          Status: Running ● Healthy        │
├─────────────────────────────────────────────────────────────┤
│ Container: plex_1          Uptime: 14d 7h 23m              │
│ Image: linuxserver/plex:latest                              │
│                                                             │
│ Resources                           Network                 │
│ CPU:  ████░░░░░░ 42% (4 cores)     IN:  24.3 MB/s         │
│ RAM:  ██████░░░░ 2.1/4.0 GB        OUT: 458 KB/s          │
│ Disk: ████████░░ 142/200 GB        Connections: 12        │
│                                                             │
│ Quick Actions                                               │
│ [Restart] [Logs] [Shell] [Update] [Backup] [━━━━━] OFF    │
│                                                             │
│ Dependencies: ✓ Database  ✓ Transcoder  ⚠ Hardware Accel  │
└─────────────────────────────────────────────────────────────┘
```

## 🎯 Gamification System

### Achievement Categories
1. **Uptime Warrior** - Server uptime milestones
2. **Storage Master** - Efficient storage management
3. **Speed Demon** - Performance optimization
4. **Automation Wizard** - Workflow automation
5. **Security Guardian** - Security best practices
6. **Update Champion** - Keeping services updated

### Progress Tracking
```
Level 12: Media Server Elite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━░░░░ 8,450/10,000 XP

Daily Quests (3/5 complete)
✓ Check server health
✓ Review recent downloads
✓ Update one service
○ Optimize storage (2.3GB reclaimable)
○ Review security logs

Weekly Challenge: Zero Downtime Week
Progress: ████████████████████░░░░ 6/7 days
```

## 🎨 Visual Design System

### Color Palette
```scss
// Primary Colors
$primary-dark: #0A0E27;      // Dark background
$primary-light: #F8F9FA;     // Light background
$accent-blue: #3B82F6;       // Primary actions
$accent-green: #10B981;      // Success/Running
$accent-yellow: #F59E0B;     // Warning
$accent-red: #EF4444;        // Error/Stopped

// Gradient Overlays
$gradient-hero: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
$gradient-achievement: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);

// Dark Mode
$dark-surface: #1A1F3A;
$dark-card: #242B47;
$dark-border: #2D3561;
```

### Typography Scale
```scss
// Font: Inter or system font stack
$font-display: 36px/1.2;     // Dashboard title
$font-heading: 24px/1.3;     // Section headers
$font-subheading: 18px/1.4;  // Card titles
$font-body: 16px/1.5;        // Default text
$font-small: 14px/1.5;       // Secondary text
$font-tiny: 12px/1.4;        // Labels
```

### Component Styling

#### Service Toggle Switch
```scss
.service-toggle {
  width: 60px;
  height: 32px;
  background: $dark-border;
  border-radius: 16px;
  position: relative;
  transition: all 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
  
  &.active {
    background: $accent-green;
    
    .toggle-thumb {
      transform: translateX(28px);
    }
  }
  
  .toggle-thumb {
    width: 28px;
    height: 28px;
    background: white;
    border-radius: 14px;
    position: absolute;
    top: 2px;
    left: 2px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
    transition: transform 0.3s cubic-bezier(0.68, -0.55, 0.265, 1.55);
  }
}
```

#### Achievement Badge
```scss
.achievement-badge {
  width: 48px;
  height: 48px;
  background: $gradient-achievement;
  border-radius: 12px;
  display: flex;
  align-items: center;
  justify-content: center;
  box-shadow: 0 4px 12px rgba(245, 87, 108, 0.3);
  animation: pulse 2s infinite;
  
  &.locked {
    background: $dark-border;
    opacity: 0.5;
    animation: none;
  }
}

@keyframes pulse {
  0% { transform: scale(1); }
  50% { transform: scale(1.05); }
  100% { transform: scale(1); }
}
```

## 📱 Mobile Responsive Design

### Mobile Layout (Simple Mode)
```
┌─────────────────────┐
│ 🏠 Media Hub    👤  │
├─────────────────────┤
│ Level 5 🔥7 days    │
│ ████████░░ 450/500  │
├─────────────────────┤
│ Quick Actions       │
│ [📺][🎬][📚][🎮]   │
├─────────────────────┤
│ Services            │
│ ┌─────────────────┐ │
│ │ Plex        ● ▣ │ │
│ │ Stream anywhere │ │
│ └─────────────────┘ │
│ ┌─────────────────┐ │
│ │ Sonarr      ● ▣ │ │
│ │ TV automated    │ │
│ └─────────────────┘ │
├─────────────────────┤
│ [🏆] Recent Wins    │
└─────────────────────┘
```

### Mobile Gestures
- **Swipe Right**: Quick toggle service on/off
- **Swipe Left**: Access service settings
- **Long Press**: View detailed stats
- **Pull Down**: Refresh all statuses
- **Pinch**: Switch between Simple/Advanced

## 🔧 .env Settings Management Interface

### Visual Editor Mode
```
┌─────────────────────────────────────────────────────────────┐
│ Environment Settings                    [Visual] [Raw] [AI]  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 🔒 Security Settings                                        │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ API Keys & Tokens                                   │   │
│ │ Plex Claim Token    [••••••••••] [Show] [Generate] │   │
│ │ Sonarr API Key      [••••••••••] [Show] [Copy]     │   │
│ │ Radarr API Key      [••••••••••] [Show] [Copy]     │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ 📁 Path Configuration                                       │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Downloads      [/media/downloads    ] [Browse]     │   │
│ │ Movies         [/media/movies       ] [Browse]     │   │
│ │ TV Shows       [/media/tv           ] [Browse]     │   │
│ │ Backups        [/backups            ] [Browse]     │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ 🌐 Network Settings                                         │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Domain         [media.home.local    ] [Test]       │   │
│ │ External URL   [https://media.me.com] [Test]       │   │
│ │ VPN Enabled    [━━━━━━━━━━━━━━] ON                 │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ [💾 Save] [🔄 Reset] [📋 Export] [🚀 Apply & Restart]     │
└─────────────────────────────────────────────────────────────┘
```

### AI Assistant Mode
```
┌─────────────────────────────────────────────────────────────┐
│ 🤖 AI Configuration Assistant                               │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ "I'll help you configure your media server. What would     │
│  you like to set up?"                                      │
│                                                             │
│ Suggestions:                                                │
│ • "Set up automated TV show downloads"                     │
│ • "Configure remote access"                                │
│ • "Optimize for 4K streaming"                              │
│ • "Enable hardware transcoding"                            │
│                                                             │
│ ┌─────────────────────────────────────────────────────┐   │
│ │ Type your request...                                │   │
│ └─────────────────────────────────────────────────────┘   │
│                                                             │
│ Recent Configurations:                                      │
│ ✓ Enabled SSL for external access                         │
│ ✓ Set up automated backups every 3 days                   │
│ ✓ Configured Plex for hardware acceleration               │
└─────────────────────────────────────────────────────────────┘
```

## 🎮 Interactive Elements

### Microinteractions
1. **Toggle Animation**: Smooth slide with subtle bounce
2. **Hover States**: Service cards lift with shadow
3. **Loading States**: Skeleton screens with shimmer
4. **Success Feedback**: Green pulse + checkmark
5. **Error States**: Red shake + error icon

### Real-time Updates
- **Live CPU/RAM graphs**: Update every second
- **Download progress**: Real-time speed and ETA
- **Active streams**: User avatars appear/disappear
- **Log streaming**: New entries slide in from bottom

## 🛡️ Security & Privacy Features

### Security Dashboard
```
┌─────────────────────────────────────────────────────────────┐
│ Security Status: 🟢 Secure                   [Details ▼]    │
├─────────────────────────────────────────────────────────────┤
│ • SSL Certificates: Valid (expires in 67 days)             │
│ • API Keys: All encrypted                                   │
│ • Failed Logins: 0 in last 24h                            │
│ • Firewall: Active (12 rules)                             │
│ • Updates: All services current                            │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Notes

### Technology Stack
- **Frontend**: React/Vue.js with TypeScript
- **Styling**: Tailwind CSS with custom components
- **State**: Redux/Pinia for global state
- **Real-time**: WebSockets for live updates
- **Charts**: Chart.js or D3.js for visualizations
- **Animations**: Framer Motion or GSAP

### Accessibility Features
- **ARIA labels** on all interactive elements
- **Keyboard navigation** support
- **High contrast mode** option
- **Screen reader** optimized
- **Reduced motion** preference respected

### Performance Optimizations
- **Lazy loading** for service details
- **Virtual scrolling** for large lists
- **Debounced** search and filters
- **Optimistic UI** updates
- **Service worker** for offline capability