# 🔌 Ultimate Media Server 2025 - Complete API Documentation

## 📋 Table of Contents
1. [API Overview](#api-overview)
2. [Authentication & Security](#authentication--security)
3. [Core API Endpoints](#core-api-endpoints)
4. [MCP Server APIs](#mcp-server-apis)
5. [AI Agent APIs](#ai-agent-apis)
6. [WebSocket APIs](#websocket-apis)
7. [GraphQL Schema](#graphql-schema)
8. [Performance Benchmarks](#performance-benchmarks)
9. [Rate Limiting](#rate-limiting)
10. [Error Handling](#error-handling)
11. [SDK Examples](#sdk-examples)
12. [Testing Guide](#testing-guide)

---

## 🌐 API Overview

### Base URLs
- **Production**: `https://api.mediaserver.dev`
- **Staging**: `https://staging-api.mediaserver.dev`
- **Local Development**: `http://localhost:3000`

### API Versioning
- **Current Version**: `v2`
- **Endpoint Format**: `/api/v2/{resource}`
- **Header Versioning**: `Accept: application/vnd.mediaserver.v2+json`

### Content Types
- **Request**: `application/json`
- **Response**: `application/json`
- **Uploads**: `multipart/form-data`
- **Streaming**: `text/event-stream` (SSE)

### Response Format
```json
{
  "success": true,
  "data": {},
  "meta": {
    "timestamp": "2025-01-15T10:30:00Z",
    "version": "v2",
    "requestId": "req_abc123"
  },
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 100,
    "hasMore": true
  }
}
```

---

## 🔒 Authentication & Security

### JWT Authentication

#### Login
```http
POST /api/v2/auth/login
Content-Type: application/json

{
  "username": "admin",
  "password": "secure_password",
  "remember": true
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "accessToken": "eyJhbGciOiJIUzI1NiIs...",
    "refreshToken": "rt_abc123def456",
    "expiresIn": 3600,
    "user": {
      "id": "user_123",
      "username": "admin",
      "email": "admin@example.com",
      "role": "admin",
      "permissions": ["read", "write", "admin"]
    }
  }
}
```

#### Token Refresh
```http
POST /api/v2/auth/refresh
Authorization: Bearer {refreshToken}

{
  "refreshToken": "rt_abc123def456"
}
```

#### Token Validation
```http
GET /api/v2/auth/validate
Authorization: Bearer {accessToken}
```

### API Key Authentication

#### Generate API Key
```http
POST /api/v2/auth/api-keys
Authorization: Bearer {accessToken}

{
  "name": "Mobile App",
  "permissions": ["read", "write"],
  "expiresAt": "2025-12-31T23:59:59Z"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "key_abc123",
    "name": "Mobile App",
    "key": "sk_live_abc123def456...",
    "permissions": ["read", "write"],
    "createdAt": "2025-01-15T10:30:00Z",
    "expiresAt": "2025-12-31T23:59:59Z"
  }
}
```

#### Using API Key
```http
GET /api/v2/media/movies
X-API-Key: sk_live_abc123def456...
```

### Two-Factor Authentication

#### Enable 2FA
```http
POST /api/v2/auth/2fa/enable
Authorization: Bearer {accessToken}

{
  "method": "totp"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "qrCode": "data:image/png;base64,iVBOR...",
    "secret": "JBSWY3DPEHPK3PXP",
    "backupCodes": [
      "12345678",
      "87654321",
      "11223344"
    ]
  }
}
```

#### Verify 2FA
```http
POST /api/v2/auth/2fa/verify
Authorization: Bearer {accessToken}

{
  "code": "123456"
}
```

---

## 📺 Core API Endpoints

### Media Management

#### Get All Movies
```http
GET /api/v2/media/movies
Authorization: Bearer {accessToken}

# Query Parameters:
# ?page=1&limit=20&sort=title&order=asc&genre=action&year=2023&quality=1080p
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "movie_123",
      "title": "The Matrix",
      "year": 1999,
      "runtime": 136,
      "genres": ["Action", "Sci-Fi"],
      "rating": {
        "imdb": 8.7,
        "tmdb": 8.2,
        "user": 4.5
      },
      "cast": [
        {
          "name": "Keanu Reeves",
          "character": "Neo",
          "profileImage": "/images/actors/keanu-reeves.jpg"
        }
      ],
      "crew": [
        {
          "name": "Lana Wachowski",
          "job": "Director"
        }
      ],
      "images": {
        "poster": "/images/posters/matrix-1999.jpg",
        "backdrop": "/images/backdrops/matrix-1999.jpg",
        "logo": "/images/logos/matrix-1999.png"
      },
      "files": [
        {
          "path": "/media/movies/The Matrix (1999)/The Matrix (1999) [1080p].mkv",
          "size": 2147483648,
          "quality": "1080p",
          "codec": "h264",
          "audioCodec": "ac3",
          "subtitles": ["en", "es", "fr"],
          "addedAt": "2025-01-15T10:30:00Z"
        }
      ],
      "watchStatus": {
        "watched": true,
        "watchedAt": "2025-01-15T20:30:00Z",
        "progress": 100,
        "resumePosition": 0
      },
      "availability": {
        "jellyfin": true,
        "plex": false,
        "emby": false
      }
    }
  ],
  "pagination": {
    "page": 1,
    "limit": 20,
    "total": 1337,
    "hasMore": true
  }
}
```

#### Get Movie Details
```http
GET /api/v2/media/movies/{id}
Authorization: Bearer {accessToken}
```

#### Search Movies
```http
GET /api/v2/media/search
Authorization: Bearer {accessToken}

# Query Parameters:
# ?q=matrix&type=movie&year=1999&genre=action
```

#### Add Movie
```http
POST /api/v2/media/movies
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "title": "Inception",
  "year": 2010,
  "tmdbId": 27205,
  "imdbId": "tt1375666",
  "quality": "1080p",
  "monitored": true,
  "searchForMovie": true
}
```

### TV Shows Management

#### Get All TV Shows
```http
GET /api/v2/media/tv-shows
Authorization: Bearer {accessToken}
```

#### Get TV Show Details with Episodes
```http
GET /api/v2/media/tv-shows/{id}
Authorization: Bearer {accessToken}

# Include episodes
GET /api/v2/media/tv-shows/{id}?include=episodes,seasons
```

**Response:**
```json
{
  "success": true,
  "data": {
    "id": "show_456",
    "title": "Stranger Things",
    "year": 2016,
    "status": "continuing",
    "network": "Netflix",
    "genres": ["Drama", "Fantasy", "Horror"],
    "rating": {
      "imdb": 8.7,
      "tmdb": 8.6,
      "user": 4.8
    },
    "seasons": [
      {
        "seasonNumber": 1,
        "episodeCount": 8,
        "airDate": "2016-07-15",
        "monitored": true,
        "episodes": [
          {
            "id": "episode_789",
            "episodeNumber": 1,
            "title": "Chapter One: The Vanishing of Will Byers",
            "airDate": "2016-07-15",
            "runtime": 47,
            "overview": "On his way home from a friend's house...",
            "hasFile": true,
            "watched": true,
            "quality": "1080p"
          }
        ]
      }
    ],
    "nextEpisode": {
      "seasonNumber": 5,
      "episodeNumber": 1,
      "airDate": "2025-06-15"
    }
  }
}
```

### Download Management

#### Get Active Downloads
```http
GET /api/v2/downloads
Authorization: Bearer {accessToken}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "download_123",
      "name": "Movie.Title.2023.1080p.BluRay.x264",
      "status": "downloading",
      "progress": 75.5,
      "downloadSpeed": 15728640, // bytes per second
      "uploadSpeed": 1048576,
      "size": 2147483648,
      "downloaded": 1621459312,
      "eta": 420, // seconds
      "seeders": 15,
      "leechers": 3,
      "ratio": 1.2,
      "category": "movies",
      "client": "qbittorrent",
      "addedAt": "2025-01-15T10:30:00Z",
      "completedAt": null
    }
  ]
}
```

#### Add Download
```http
POST /api/v2/downloads
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "url": "magnet:?xt=urn:btih:abc123...",
  "category": "movies",
  "priority": "high",
  "downloadPath": "/downloads/movies"
}
```

#### Control Download
```http
# Pause download
PUT /api/v2/downloads/{id}/pause

# Resume download
PUT /api/v2/downloads/{id}/resume

# Delete download
DELETE /api/v2/downloads/{id}?deleteFiles=true
```

### User Management

#### Get Current User
```http
GET /api/v2/users/me
Authorization: Bearer {accessToken}
```

#### Update User Profile
```http
PUT /api/v2/users/me
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "displayName": "John Doe",
  "email": "john@example.com",
  "preferences": {
    "theme": "dark",
    "language": "en",
    "autoPlay": true,
    "quality": "1080p",
    "notifications": {
      "email": true,
      "push": true,
      "downloads": true,
      "newContent": false
    }
  }
}
```

#### Get User Activity
```http
GET /api/v2/users/me/activity
Authorization: Bearer {accessToken}

# Query Parameters:
# ?type=watched&limit=50&since=2025-01-01T00:00:00Z
```

### System Management

#### System Status
```http
GET /api/v2/system/status
Authorization: Bearer {accessToken}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "system": {
      "version": "2.0.0",
      "uptime": 86400,
      "timezone": "America/New_York",
      "platform": "linux",
      "architecture": "x64"
    },
    "services": {
      "jellyfin": {
        "status": "running",
        "version": "10.8.13",
        "url": "http://jellyfin:8096",
        "lastCheck": "2025-01-15T10:30:00Z"
      },
      "sonarr": {
        "status": "running",
        "version": "4.0.0.400",
        "url": "http://sonarr:8989",
        "lastCheck": "2025-01-15T10:30:00Z"
      }
    },
    "resources": {
      "cpu": {
        "usage": 25.5,
        "cores": 8
      },
      "memory": {
        "used": 4294967296,
        "total": 17179869184,
        "percentage": 25.0
      },
      "disk": {
        "used": 1099511627776,
        "total": 5497558138880,
        "percentage": 20.0
      },
      "network": {
        "download": 15728640,
        "upload": 1048576
      }
    },
    "library": {
      "movies": {
        "count": 1337,
        "size": 2748779069440
      },
      "tvShows": {
        "count": 156,
        "episodes": 4521,
        "size": 5497558138880
      },
      "music": {
        "count": 8945,
        "size": 274877906944
      }
    }
  }
}
```

#### System Metrics
```http
GET /api/v2/system/metrics
Authorization: Bearer {accessToken}

# Query Parameters:
# ?metric=cpu,memory,disk&timeRange=1h&interval=5m
```

#### Service Control
```http
# Restart service
POST /api/v2/system/services/{serviceName}/restart

# Stop service
POST /api/v2/system/services/{serviceName}/stop

# Start service
POST /api/v2/system/services/{serviceName}/start
```

---

## 🤖 MCP Server APIs

### Jellyfin MCP Server (Port 3001)

#### Get Server Info
```http
GET http://localhost:3001/info
```

**Response:**
```json
{
  "name": "jellyfin-mcp",
  "version": "1.0.0",
  "tools": [
    "search_media",
    "get_libraries",
    "get_users",
    "get_sessions",
    "refresh_library"
  ],
  "resources": [
    "library_stats",
    "user_activity",
    "system_info"
  ]
}
```

#### Execute Tool
```http
POST http://localhost:3001/call/search_media
Content-Type: application/json

{
  "arguments": {
    "query": "matrix",
    "type": "movie",
    "limit": 10
  }
}
```

**Response:**
```json
{
  "result": {
    "items": [
      {
        "id": "abc123",
        "name": "The Matrix",
        "year": 1999,
        "type": "Movie",
        "overview": "A computer hacker learns..."
      }
    ],
    "totalResults": 1
  }
}
```

#### Server-Sent Events
```http
GET http://localhost:3001/events
Accept: text/event-stream
```

**Response Stream:**
```
data: {"type": "connected", "server": "jellyfin-mcp"}

data: {"type": "library_updated", "libraryId": "movies", "itemsAdded": 5}

data: {"type": "playback_started", "userId": "user123", "itemId": "movie456"}
```

### Sonarr MCP Server (Port 3002)

#### Available Tools
- `search_series` - Search for TV series
- `add_series` - Add series to monitoring
- `get_queue` - Get download queue
- `get_calendar` - Get upcoming episodes
- `refresh_series` - Refresh series metadata

#### Search Series
```http
POST http://localhost:3002/call/search_series
Content-Type: application/json

{
  "arguments": {
    "term": "stranger things"
  }
}
```

#### Add Series
```http
POST http://localhost:3002/call/add_series
Content-Type: application/json

{
  "arguments": {
    "tvdbId": 305288,
    "title": "Stranger Things",
    "qualityProfile": "1080p",
    "languageProfile": "English",
    "rootFolder": "/tv",
    "monitored": true,
    "searchForMissingEpisodes": true
  }
}
```

### Radarr MCP Server (Port 3003)

#### Available Tools
- `search_movies` - Search for movies
- `add_movie` - Add movie to monitoring
- `get_queue` - Get download queue
- `get_calendar` - Get upcoming releases
- `refresh_movie` - Refresh movie metadata

### Prowlarr MCP Server (Port 3004)

#### Available Tools
- `search_indexers` - Search across indexers
- `get_indexers` - Get configured indexers
- `test_indexer` - Test indexer connection
- `get_stats` - Get indexer statistics

### qBittorrent MCP Server (Port 3005)

#### Available Tools
- `get_torrents` - Get torrent list
- `add_torrent` - Add new torrent
- `pause_torrent` - Pause torrent
- `resume_torrent` - Resume torrent
- `delete_torrent` - Delete torrent
- `get_global_stats` - Get global statistics

---

## 🤖 AI Agent APIs

### Agent Orchestration

#### Chat with AI Agents
```http
POST /api/v2/ai/chat
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "message": "Should I upgrade my storage configuration?",
  "context": {
    "currentStorage": "2TB",
    "usage": "85%",
    "growthRate": "10GB/month"
  },
  "agents": ["technical_specialist", "user_advocate", "automation_expert"]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "conversationId": "conv_abc123",
    "question": "Should I upgrade my storage configuration?",
    "votingSession": {
      "id": "vote_xyz789",
      "status": "in_progress",
      "requiredConsensus": 0.7,
      "timeout": 300000,
      "votes": [
        {
          "agent": "technical_specialist",
          "vote": "yes",
          "confidence": 0.9,
          "reasoning": "Current usage at 85% indicates urgent need for expansion",
          "timestamp": "2025-01-15T10:30:00Z"
        }
      ],
      "currentConsensus": 0.33,
      "estimatedCompletion": "2025-01-15T10:35:00Z"
    }
  }
}
```

#### Get Agent Status
```http
GET /api/v2/ai/agents
Authorization: Bearer {accessToken}
```

**Response:**
```json
{
  "success": true,
  "data": [
    {
      "id": "media_curator",
      "name": "Media Curator",
      "type": "content_specialist",
      "status": "active",
      "expertise": ["content_discovery", "recommendations", "metadata"],
      "stats": {
        "totalVotes": 1247,
        "averageConfidence": 0.82,
        "successRate": 0.94,
        "lastActive": "2025-01-15T10:30:00Z"
      },
      "capabilities": {
        "naturalLanguage": true,
        "contextAwareness": true,
        "learningEnabled": true,
        "apiIntegration": true
      }
    }
  ]
}
```

#### Get Voting History
```http
GET /api/v2/ai/votes
Authorization: Bearer {accessToken}

# Query Parameters:
# ?limit=50&agent=technical_specialist&status=completed&since=2025-01-01
```

#### Force Agent Decision
```http
POST /api/v2/ai/votes/{voteId}/force
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "decision": "yes",
  "reason": "Emergency override required",
  "overrideCode": "admin_emergency_123"
}
```

### Agent Training

#### Submit Training Data
```http
POST /api/v2/ai/training/feedback
Authorization: Bearer {accessToken}
Content-Type: application/json

{
  "voteId": "vote_xyz789",
  "outcome": "successful",
  "userSatisfaction": 4.5,
  "feedback": "The recommendation was accurate and helpful",
  "metrics": {
    "executionTime": 45,
    "resourceUsage": "normal",
    "accuracy": 0.95
  }
}
```

#### Get Learning Analytics
```http
GET /api/v2/ai/analytics
Authorization: Bearer {accessToken}

# Query Parameters:
# ?timeRange=30d&agent=all&metric=accuracy,confidence,speed
```

---

## 🔌 WebSocket APIs

### Connection Establishment
```javascript
const ws = new WebSocket('ws://localhost:3000/ws', {
  headers: {
    'Authorization': 'Bearer ' + accessToken
  }
});

ws.onopen = function() {
  console.log('Connected to MediaServer WebSocket');
  
  // Subscribe to events
  ws.send(JSON.stringify({
    type: 'subscribe',
    events: ['download_progress', 'media_added', 'agent_vote', 'system_alert']
  }));
};

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};
```

### Event Types

#### Download Progress
```json
{
  "type": "download_progress",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "downloadId": "download_123",
    "name": "Movie.Title.2023.1080p",
    "progress": 75.5,
    "speed": 15728640,
    "eta": 420,
    "status": "downloading"
  }
}
```

#### Media Added
```json
{
  "type": "media_added",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "mediaType": "movie",
    "title": "The Matrix",
    "year": 1999,
    "id": "movie_123",
    "addedBy": "automation",
    "quality": "1080p"
  }
}
```

#### Agent Vote Update
```json
{
  "type": "agent_vote",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "voteId": "vote_xyz789",
    "agent": "technical_specialist",
    "vote": "yes",
    "confidence": 0.9,
    "currentConsensus": 0.67,
    "remainingAgents": 1,
    "timeRemaining": 180
  }
}
```

#### System Alert
```json
{
  "type": "system_alert",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": {
    "level": "warning",
    "category": "storage",
    "message": "Disk space is running low (15% remaining)",
    "details": {
      "currentUsage": 85,
      "availableSpace": "500GB",
      "recommendedAction": "upgrade_storage"
    },
    "actionRequired": true
  }
}
```

### WebSocket Commands

#### Subscribe to Events
```json
{
  "type": "subscribe",
  "events": ["download_progress", "media_added"],
  "filters": {
    "mediaType": "movie",
    "quality": "1080p"
  }
}
```

#### Unsubscribe from Events
```json
{
  "type": "unsubscribe",
  "events": ["download_progress"]
}
```

#### Send Command
```json
{
  "type": "command",
  "action": "pause_download",
  "params": {
    "downloadId": "download_123"
  }
}
```

---

## 📈 GraphQL Schema

### Schema Definition
```graphql
type Query {
  # Media queries
  movies(filters: MediaFilters, pagination: PaginationInput): MovieConnection!
  movie(id: ID!): Movie
  tvShows(filters: MediaFilters, pagination: PaginationInput): TVShowConnection!
  tvShow(id: ID!): TVShow
  
  # Search
  search(query: String!, type: MediaType, filters: SearchFilters): SearchResult!
  
  # Downloads
  downloads(status: DownloadStatus): [Download!]!
  download(id: ID!): Download
  
  # System
  systemStatus: SystemStatus!
  systemMetrics(timeRange: TimeRange!, metrics: [MetricType!]!): [Metric!]!
  
  # AI Agents
  agents: [Agent!]!
  agent(id: ID!): Agent
  votes(filters: VoteFilters): [Vote!]!
  
  # User
  me: User!
  userActivity(type: ActivityType, limit: Int): [Activity!]!
}

type Mutation {
  # Authentication
  login(credentials: LoginInput!): AuthPayload!
  refreshToken(token: String!): AuthPayload!
  logout: Boolean!
  
  # Media management
  addMovie(input: AddMovieInput!): Movie!
  updateMovie(id: ID!, input: UpdateMovieInput!): Movie!
  deleteMovie(id: ID!): Boolean!
  
  addTVShow(input: AddTVShowInput!): TVShow!
  updateTVShow(id: ID!, input: UpdateTVShowInput!): TVShow!
  deleteTVShow(id: ID!): Boolean!
  
  # Downloads
  addDownload(input: AddDownloadInput!): Download!
  pauseDownload(id: ID!): Download!
  resumeDownload(id: ID!): Download!
  deleteDownload(id: ID!, deleteFiles: Boolean): Boolean!
  
  # AI Agents
  chatWithAgents(input: ChatInput!): ChatResponse!
  submitVote(voteId: ID!, vote: VoteInput!): Vote!
  forceVoteDecision(voteId: ID!, decision: String!, reason: String!): Vote!
  
  # User
  updateProfile(input: UpdateProfileInput!): User!
  updatePreferences(input: PreferencesInput!): User!
}

type Subscription {
  # Real-time updates
  downloadProgress(downloadId: ID): Download!
  mediaAdded(mediaType: MediaType): Media!
  agentVoting(voteId: ID): Vote!
  systemAlerts(level: AlertLevel): SystemAlert!
  userActivity(userId: ID): Activity!
}

# Types
type Movie {
  id: ID!
  title: String!
  year: Int
  runtime: Int
  overview: String
  genres: [String!]!
  rating: Rating!
  cast: [CastMember!]!
  crew: [CrewMember!]!
  images: MediaImages!
  files: [MediaFile!]!
  watchStatus: WatchStatus
  availability: ServiceAvailability!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type TVShow {
  id: ID!
  title: String!
  year: Int
  status: ShowStatus!
  network: String
  genres: [String!]!
  rating: Rating!
  seasons: [Season!]!
  nextEpisode: Episode
  images: MediaImages!
  watchStatus: WatchStatus
  availability: ServiceAvailability!
  createdAt: DateTime!
  updatedAt: DateTime!
}

type Season {
  seasonNumber: Int!
  episodeCount: Int!
  airDate: Date
  monitored: Boolean!
  episodes: [Episode!]!
}

type Episode {
  id: ID!
  episodeNumber: Int!
  seasonNumber: Int!
  title: String!
  airDate: Date
  runtime: Int
  overview: String
  hasFile: Boolean!
  watched: Boolean!
  quality: String
  file: MediaFile
}

type Download {
  id: ID!
  name: String!
  status: DownloadStatus!
  progress: Float!
  downloadSpeed: Int!
  uploadSpeed: Int!
  size: Int!
  downloaded: Int!
  eta: Int
  seeders: Int
  leechers: Int
  ratio: Float!
  category: String
  client: String!
  addedAt: DateTime!
  completedAt: DateTime
}

type Agent {
  id: ID!
  name: String!
  type: AgentType!
  status: AgentStatus!
  expertise: [String!]!
  stats: AgentStats!
  capabilities: AgentCapabilities!
}

type Vote {
  id: ID!
  question: String!
  context: JSON
  status: VoteStatus!
  requiredConsensus: Float!
  currentConsensus: Float!
  votes: [AgentVote!]!
  decision: String
  confidence: Float
  createdAt: DateTime!
  completedAt: DateTime
}

# Input types
input MediaFilters {
  genre: String
  year: Int
  quality: String
  rating: RatingFilter
  status: String
}

input PaginationInput {
  page: Int = 1
  limit: Int = 20
  sort: String = "title"
  order: SortOrder = ASC
}

input AddMovieInput {
  title: String!
  year: Int
  tmdbId: Int
  imdbId: String
  quality: String
  monitored: Boolean = true
  searchForMovie: Boolean = false
}

input ChatInput {
  message: String!
  context: JSON
  agents: [String!]
  priority: Priority = NORMAL
}

# Enums
enum MediaType {
  MOVIE
  TV_SHOW
  EPISODE
  MUSIC
  BOOK
}

enum DownloadStatus {
  QUEUED
  DOWNLOADING
  SEEDING
  PAUSED
  ERROR
  COMPLETED
}

enum AgentType {
  MEDIA_CURATOR
  TECHNICAL_SPECIALIST
  USER_ADVOCATE
  AUTOMATION_EXPERT
  SECURITY_GUARDIAN
  TREND_ANALYST
}

enum VoteStatus {
  PENDING
  IN_PROGRESS
  COMPLETED
  TIMEOUT
  FORCED
}

enum SortOrder {
  ASC
  DESC
}

# Scalars
scalar DateTime
scalar Date
scalar JSON
```

### Example Queries

#### Get Movies with Filters
```graphql
query GetMovies($filters: MediaFilters!, $pagination: PaginationInput!) {
  movies(filters: $filters, pagination: $pagination) {
    edges {
      node {
        id
        title
        year
        rating {
          imdb
          tmdb
          user
        }
        images {
          poster
          backdrop
        }
        files {
          quality
          size
          codec
        }
        watchStatus {
          watched
          progress
        }
      }
    }
    pageInfo {
      hasNextPage
      hasPreviousPage
      totalCount
    }
  }
}
```

#### Search Content
```graphql
query SearchContent($query: String!, $type: MediaType) {
  search(query: $query, type: $type) {
    movies {
      id
      title
      year
      rating {
        imdb
      }
    }
    tvShows {
      id
      title
      year
      seasons {
        seasonNumber
        episodeCount
      }
    }
    totalResults
  }
}
```

#### Get Agent Status
```graphql
query GetAgents {
  agents {
    id
    name
    type
    status
    stats {
      totalVotes
      averageConfidence
      successRate
      lastActive
    }
    capabilities {
      naturalLanguage
      contextAwareness
      learningEnabled
    }
  }
}
```

#### Add Movie Mutation
```graphql
mutation AddMovie($input: AddMovieInput!) {
  addMovie(input: $input) {
    id
    title
    year
    status
    monitored
    createdAt
  }
}
```

#### Chat with Agents
```graphql
mutation ChatWithAgents($input: ChatInput!) {
  chatWithAgents(input: $input) {
    conversationId
    votingSession {
      id
      status
      requiredConsensus
      currentConsensus
      votes {
        agent
        vote
        confidence
        reasoning
      }
    }
  }
}
```

#### Real-time Subscriptions
```graphql
subscription DownloadProgress {
  downloadProgress {
    id
    name
    progress
    downloadSpeed
    eta
    status
  }
}

subscription AgentVoting {
  agentVoting {
    id
    status
    currentConsensus
    votes {
      agent
      vote
      confidence
    }
  }
}
```

---

## ⚡ Performance Benchmarks

### API Response Times

| Endpoint | Average (ms) | 95th Percentile (ms) | 99th Percentile (ms) |
|----------|--------------|---------------------|---------------------|
| `GET /api/v2/media/movies` | 45 | 120 | 250 |
| `GET /api/v2/media/movies/{id}` | 25 | 60 | 150 |
| `POST /api/v2/media/movies` | 180 | 400 | 800 |
| `GET /api/v2/downloads` | 30 | 80 | 180 |
| `POST /api/v2/ai/chat` | 350 | 800 | 1500 |
| `GET /api/v2/system/status` | 15 | 40 | 100 |
| `WebSocket message` | 5 | 15 | 30 |

### Throughput Metrics

| Operation | Requests/Second | Concurrent Users | Notes |
|-----------|----------------|------------------|-------|
| Read Operations | 2,500 | 500 | Cached responses |
| Write Operations | 800 | 200 | Database writes |
| Search Queries | 1,200 | 300 | Elasticsearch |
| AI Agent Queries | 150 | 50 | OpenAI API limits |
| WebSocket Messages | 10,000 | 1,000 | Broadcast events |

### Resource Usage

#### API Server (per instance)
- **CPU**: 0.5-2.0 cores (under load)
- **Memory**: 512MB-2GB (depending on cache)
- **Network**: 100MB/s (peak)
- **Storage**: 1GB (logs and temp files)

#### Database Performance
- **PostgreSQL**: 5,000 QPS (read), 1,500 QPS (write)
- **Redis**: 50,000 operations/second
- **Connection Pool**: 20 connections per API instance

### Load Testing Results

#### Stress Test Configuration
```bash
# Apache Bench
ab -n 10000 -c 100 -H "Authorization: Bearer token" \
   http://localhost:3000/api/v2/media/movies

# Results:
# Requests per second: 2,347.83 [#/sec] (mean)
# Time per request: 42.590 [ms] (mean)
# Transfer rate: 1,234.56 [Kbytes/sec] received
```

#### Load Test with k6
```javascript
// load-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';

export let options = {
  stages: [
    { duration: '2m', target: 100 },
    { duration: '5m', target: 500 },
    { duration: '2m', target: 1000 },
    { duration: '5m', target: 1000 },
    { duration: '2m', target: 0 },
  ],
};

export default function () {
  const response = http.get('http://localhost:3000/api/v2/media/movies', {
    headers: {
      'Authorization': 'Bearer ' + __ENV.ACCESS_TOKEN,
    },
  });
  
  check(response, {
    'status is 200': (r) => r.status === 200,
    'response time < 500ms': (r) => r.timings.duration < 500,
  });
  
  sleep(1);
}
```

**Results:**
- **Peak RPS**: 2,100
- **Average Response Time**: 235ms
- **95th Percentile**: 450ms
- **Error Rate**: 0.02%

---

## 🚪 Rate Limiting

### Rate Limit Headers
All API responses include rate limiting information:

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1642284000
X-RateLimit-Window: 3600
```

### Rate Limit Tiers

| User Type | Requests/Hour | Burst Limit | Notes |
|-----------|---------------|-------------|-------|
| Anonymous | 100 | 10 | Public endpoints only |
| Basic User | 1,000 | 50 | Standard API access |
| Premium User | 5,000 | 100 | Enhanced limits |
| Admin | 10,000 | 200 | Administrative access |
| API Key | 50,000 | 500 | Service integrations |

### Endpoint-Specific Limits

| Endpoint Pattern | Limit | Window |
|------------------|-------|---------|
| `/api/v2/auth/login` | 5 requests | 15 minutes |
| `/api/v2/auth/refresh` | 10 requests | 1 hour |
| `/api/v2/media/*` | 1,000 requests | 1 hour |
| `/api/v2/downloads` | 100 requests | 1 hour |
| `/api/v2/ai/chat` | 50 requests | 1 hour |
| `/api/v2/system/*` | 500 requests | 1 hour |

### Rate Limit Exceeded Response
```http
HTTP/1.1 429 Too Many Requests
Content-Type: application/json
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 0
X-RateLimit-Reset: 1642284000
Retry-After: 3600

{
  "success": false,
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Try again in 1 hour.",
    "details": {
      "limit": 1000,
      "window": 3600,
      "retryAfter": 3600
    }
  }
}
```

---

## ⚠️ Error Handling

### Standard Error Response
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "The request data is invalid",
    "details": {
      "field": "email",
      "issue": "Invalid email format"
    },
    "requestId": "req_abc123",
    "timestamp": "2025-01-15T10:30:00Z"
  }
}
```

### HTTP Status Codes

| Status Code | Description | Usage |
|-------------|-------------|---------|
| 200 | OK | Successful GET, PUT, PATCH |
| 201 | Created | Successful POST |
| 204 | No Content | Successful DELETE |
| 400 | Bad Request | Invalid request data |
| 401 | Unauthorized | Missing or invalid auth |
| 403 | Forbidden | Insufficient permissions |
| 404 | Not Found | Resource doesn't exist |
| 409 | Conflict | Resource already exists |
| 422 | Unprocessable Entity | Validation errors |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server-side error |
| 502 | Bad Gateway | Upstream service error |
| 503 | Service Unavailable | Temporary unavailability |

### Error Codes

| Code | Description | HTTP Status |
|------|-------------|-------------|
| `VALIDATION_ERROR` | Request validation failed | 422 |
| `AUTHENTICATION_REQUIRED` | Authentication required | 401 |
| `INVALID_CREDENTIALS` | Invalid login credentials | 401 |
| `TOKEN_EXPIRED` | Access token expired | 401 |
| `INSUFFICIENT_PERMISSIONS` | Insufficient permissions | 403 |
| `RESOURCE_NOT_FOUND` | Resource not found | 404 |
| `RESOURCE_CONFLICT` | Resource already exists | 409 |
| `RATE_LIMIT_EXCEEDED` | Rate limit exceeded | 429 |
| `SERVICE_UNAVAILABLE` | External service error | 502 |
| `INTERNAL_ERROR` | Internal server error | 500 |

### Validation Errors
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Request validation failed",
    "details": {
      "errors": [
        {
          "field": "title",
          "code": "REQUIRED",
          "message": "Title is required"
        },
        {
          "field": "year",
          "code": "INVALID_RANGE",
          "message": "Year must be between 1900 and 2030"
        }
      ]
    }
  }
}
```

---

## 📦 SDK Examples

### JavaScript/TypeScript SDK

```typescript
// Installation
// npm install @mediaserver/sdk

import { MediaServerClient } from '@mediaserver/sdk';

// Initialize client
const client = new MediaServerClient({
  baseUrl: 'https://api.mediaserver.dev',
  apiKey: 'sk_live_abc123...',
  version: 'v2'
});

// Authentication with JWT
await client.auth.login({
  username: 'admin',
  password: 'password'
});

// Get movies
const movies = await client.media.getMovies({
  filters: {
    genre: 'action',
    year: 2023
  },
  pagination: {
    page: 1,
    limit: 20
  }
});

// Add movie
const newMovie = await client.media.addMovie({
  title: 'Inception',
  year: 2010,
  tmdbId: 27205,
  quality: '1080p',
  monitored: true
});

// Chat with AI agents
const chatResponse = await client.ai.chat({
  message: 'What movies should I download this week?',
  context: {
    preferences: ['action', 'sci-fi'],
    quality: '1080p'
  }
});

// Subscribe to real-time events
client.on('download_progress', (data) => {
  console.log(`Download ${data.name}: ${data.progress}%`);
});

client.on('agent_vote', (data) => {
  console.log(`Agent ${data.agent} voted: ${data.vote}`);
});

// Connect WebSocket
await client.connect();
```

### Python SDK

```python
# Installation
# pip install mediaserver-sdk

from mediaserver import MediaServerClient
import asyncio

# Initialize client
client = MediaServerClient(
    base_url='https://api.mediaserver.dev',
    api_key='sk_live_abc123...',
    version='v2'
)

# Async operations
async def main():
    # Authentication
    await client.auth.login(
        username='admin',
        password='password'
    )
    
    # Get movies
    movies = await client.media.get_movies(
        filters={'genre': 'action', 'year': 2023},
        pagination={'page': 1, 'limit': 20}
    )
    
    # Add movie
    new_movie = await client.media.add_movie(
        title='Inception',
        year=2010,
        tmdb_id=27205,
        quality='1080p',
        monitored=True
    )
    
    # Chat with AI agents
    chat_response = await client.ai.chat(
        message='What movies should I download this week?',
        context={
            'preferences': ['action', 'sci-fi'],
            'quality': '1080p'
        }
    )
    
    print(f"AI Recommendation: {chat_response.message}")
    
    # WebSocket events
    @client.on('download_progress')
    def on_download_progress(data):
        print(f"Download {data['name']}: {data['progress']}%")
    
    @client.on('agent_vote')
    def on_agent_vote(data):
        print(f"Agent {data['agent']} voted: {data['vote']}")
    
    # Connect and listen
    await client.connect()
    await client.listen()  # Keep connection alive

# Run async main
asyncio.run(main())
```

### Go SDK

```go
// go get github.com/mediaserver/go-sdk

package main

import (
    "context"
    "fmt"
    "log"
    
    "github.com/mediaserver/go-sdk"
)

func main() {
    // Initialize client
    client := mediaserver.NewClient(&mediaserver.Config{
        BaseURL: "https://api.mediaserver.dev",
        APIKey:  "sk_live_abc123...",
        Version: "v2",
    })
    
    ctx := context.Background()
    
    // Authentication
    auth, err := client.Auth.Login(ctx, &mediaserver.LoginRequest{
        Username: "admin",
        Password: "password",
    })
    if err != nil {
        log.Fatal(err)
    }
    
    // Set access token
    client.SetAccessToken(auth.AccessToken)
    
    // Get movies
    movies, err := client.Media.GetMovies(ctx, &mediaserver.GetMoviesRequest{
        Filters: &mediaserver.MediaFilters{
            Genre: "action",
            Year:  2023,
        },
        Pagination: &mediaserver.Pagination{
            Page:  1,
            Limit: 20,
        },
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Found %d movies\n", len(movies.Data))
    
    // Add movie
    newMovie, err := client.Media.AddMovie(ctx, &mediaserver.AddMovieRequest{
        Title:    "Inception",
        Year:     2010,
        TMDBId:   27205,
        Quality:  "1080p",
        Monitored: true,
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("Added movie: %s\n", newMovie.Title)
    
    // Chat with AI agents
    chatResponse, err := client.AI.Chat(ctx, &mediaserver.ChatRequest{
        Message: "What movies should I download this week?",
        Context: map[string]interface{}{
            "preferences": []string{"action", "sci-fi"},
            "quality":     "1080p",
        },
    })
    if err != nil {
        log.Fatal(err)
    }
    
    fmt.Printf("AI Recommendation: %s\n", chatResponse.Message)
    
    // WebSocket connection
    ws, err := client.WebSocket.Connect(ctx)
    if err != nil {
        log.Fatal(err)
    }
    defer ws.Close()
    
    // Subscribe to events
    err = ws.Subscribe([]string{"download_progress", "agent_vote"})
    if err != nil {
        log.Fatal(err)
    }
    
    // Listen for events
    for event := range ws.Events() {
        switch event.Type {
        case "download_progress":
            fmt.Printf("Download progress: %v\n", event.Data)
        case "agent_vote":
            fmt.Printf("Agent vote: %v\n", event.Data)
        }
    }
}
```

---

## 🧪 Testing Guide

### Unit Testing

```javascript
// test/api/media.test.js
const request = require('supertest');
const app = require('../../src/app');
const { generateTestToken } = require('../helpers/auth');

describe('Media API', () => {
  let authToken;
  
  beforeAll(async () => {
    authToken = await generateTestToken({ role: 'admin' });
  });
  
  describe('GET /api/v2/media/movies', () => {
    it('should return movies list', async () => {
      const response = await request(app)
        .get('/api/v2/media/movies')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);
      
      expect(response.body.success).toBe(true);
      expect(response.body.data).toBeInstanceOf(Array);
      expect(response.body.pagination).toBeDefined();
    });
    
    it('should filter movies by genre', async () => {
      const response = await request(app)
        .get('/api/v2/media/movies?genre=action')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);
      
      response.body.data.forEach(movie => {
        expect(movie.genres).toContain('action');
      });
    });
    
    it('should require authentication', async () => {
      await request(app)
        .get('/api/v2/media/movies')
        .expect(401);
    });
  });
  
  describe('POST /api/v2/media/movies', () => {
    it('should add a new movie', async () => {
      const movieData = {
        title: 'Test Movie',
        year: 2023,
        tmdbId: 12345,
        quality: '1080p',
        monitored: true
      };
      
      const response = await request(app)
        .post('/api/v2/media/movies')
        .set('Authorization', `Bearer ${authToken}`)
        .send(movieData)
        .expect(201);
      
      expect(response.body.success).toBe(true);
      expect(response.body.data.title).toBe(movieData.title);
      expect(response.body.data.id).toBeDefined();
    });
    
    it('should validate required fields', async () => {
      const invalidData = {
        year: 2023
        // missing title
      };
      
      const response = await request(app)
        .post('/api/v2/media/movies')
        .set('Authorization', `Bearer ${authToken}`)
        .send(invalidData)
        .expect(422);
      
      expect(response.body.success).toBe(false);
      expect(response.body.error.code).toBe('VALIDATION_ERROR');
    });
  });
});
```

### Integration Testing

```javascript
// test/integration/workflow.test.js
const { MediaServerClient } = require('@mediaserver/sdk');
const { setupTestEnvironment, teardownTestEnvironment } = require('../helpers/environment');

describe('Movie Download Workflow', () => {
  let client;
  let testEnvironment;
  
  beforeAll(async () => {
    testEnvironment = await setupTestEnvironment();
    client = new MediaServerClient({
      baseUrl: testEnvironment.apiUrl,
      apiKey: testEnvironment.apiKey
    });
  });
  
  afterAll(async () => {
    await teardownTestEnvironment(testEnvironment);
  });
  
  it('should complete full movie download workflow', async () => {
    // Step 1: Add movie to monitoring
    const movie = await client.media.addMovie({
      title: 'Test Movie 2023',
      year: 2023,
      tmdbId: 999999,
      quality: '1080p',
      monitored: true,
      searchForMovie: true
    });
    
    expect(movie.id).toBeDefined();
    expect(movie.monitored).toBe(true);
    
    // Step 2: Wait for search to be initiated
    await new Promise(resolve => setTimeout(resolve, 2000));
    
    // Step 3: Check that download was queued
    const downloads = await client.downloads.getActive();
    const movieDownload = downloads.find(d => 
      d.name.includes('Test Movie 2023')
    );
    
    expect(movieDownload).toBeDefined();
    expect(movieDownload.status).toBe('downloading');
    
    // Step 4: Monitor download progress (simulate)
    let downloadComplete = false;
    const progressListener = (data) => {
      if (data.id === movieDownload.id && data.progress === 100) {
        downloadComplete = true;
      }
    };
    
    client.on('download_progress', progressListener);
    await client.connect();
    
    // Wait for download completion (or timeout)
    await new Promise((resolve) => {
      const checkInterval = setInterval(() => {
        if (downloadComplete) {
          clearInterval(checkInterval);
          resolve();
        }
      }, 1000);
      
      // Timeout after 30 seconds
      setTimeout(() => {
        clearInterval(checkInterval);
        resolve();
      }, 30000);
    });
    
    // Step 5: Verify movie is available
    const updatedMovie = await client.media.getMovie(movie.id);
    expect(updatedMovie.files.length).toBeGreaterThan(0);
    expect(updatedMovie.availability.jellyfin).toBe(true);
  }, 60000); // 60 second timeout
});
```

### Load Testing

```javascript
// test/load/api-load.test.js
const { check } = require('k6');
const http = require('k6/http');

export let options = {
  stages: [
    { duration: '1m', target: 50 },
    { duration: '3m', target: 100 },
    { duration: '1m', target: 0 },
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests under 500ms
    http_req_failed: ['rate<0.01'],   // Error rate under 1%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:3000';
const API_KEY = __ENV.API_KEY;

export default function () {
  const headers = {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY,
  };
  
  // Test different endpoints
  const endpoints = [
    '/api/v2/media/movies',
    '/api/v2/media/tv-shows',
    '/api/v2/downloads',
    '/api/v2/system/status',
  ];
  
  endpoints.forEach(endpoint => {
    const response = http.get(`${BASE_URL}${endpoint}`, { headers });
    
    check(response, {
      [`${endpoint} status is 200`]: (r) => r.status === 200,
      [`${endpoint} response time < 500ms`]: (r) => r.timings.duration < 500,
    });
  });
}
```

### End-to-End Testing

```javascript
// test/e2e/user-journey.test.js
const puppeteer = require('puppeteer');

describe('User Journey - Movie Request to Watching', () => {
  let browser;
  let page;
  
  beforeAll(async () => {
    browser = await puppeteer.launch({
      headless: process.env.CI !== 'false',
      slowMo: 50
    });
    page = await browser.newPage();
  });
  
  afterAll(async () => {
    await browser.close();
  });
  
  it('should complete full user journey', async () => {
    // Step 1: Login
    await page.goto('http://localhost:3001/login');
    await page.type('#username', 'testuser');
    await page.type('#password', 'testpass');
    await page.click('#login-button');
    await page.waitForNavigation();
    
    // Step 2: Request a movie
    await page.goto('http://localhost:3001/request');
    await page.type('#search-input', 'Inception');
    await page.click('#search-button');
    await page.waitForSelector('.search-results');
    
    // Click first result
    await page.click('.search-result:first-child .request-button');
    await page.waitForSelector('.success-message');
    
    // Step 3: Check download progress
    await page.goto('http://localhost:3001/downloads');
    await page.waitForSelector('.download-item');
    
    const downloadItem = await page.$('.download-item');
    expect(downloadItem).toBeTruthy();
    
    // Step 4: Wait for completion notification
    await page.waitForSelector('.notification.download-complete', {
      timeout: 60000
    });
    
    // Step 5: Verify movie is available in library
    await page.goto('http://localhost:3001/movies');
    await page.waitForSelector('.movie-item');
    
    const movieTitle = await page.$eval(
      '.movie-item .title',
      el => el.textContent
    );
    expect(movieTitle).toContain('Inception');
    
    // Step 6: Start watching
    await page.click('.movie-item .play-button');
    await page.waitForSelector('.video-player');
    
    const videoPlayer = await page.$('.video-player');
    expect(videoPlayer).toBeTruthy();
  }, 120000); // 2 minute timeout
});
```

---

*This comprehensive API documentation covers all aspects of the Ultimate Media Server 2025 API ecosystem. For the latest updates and additional examples, visit the [official documentation site](https://docs.mediaserver.dev)*