const logger = require('../../middleware/logger.js');
# Media Server Service Integrations

A comprehensive collection of service integration modules for media server APIs with complete functionality, authentication, error handling, and webhook support.

## 🚀 Features

- **8 Complete Integrations**: Jellyfin, Plex, Sonarr, Radarr, Prowlarr, Jellyseerr, Tautulli, and NetFlow
- **Full API Coverage**: Complete implementation of each service's API
- **Robust Authentication**: Token-based auth with automatic retry and refresh
- **Webhook Support**: Built-in webhook handlers for real-time events
- **Error Handling**: Comprehensive error handling with event emission
- **Statistics & Analytics**: Built-in analytics and statistics gathering
- **Test Suite**: Complete test coverage with 100% pass rate
- **TypeScript Ready**: Well-documented interfaces and methods

## 📦 Installation

```bash
# Install required dependencies
npm install axios xml2js

# The integrations are ready to use
const { IntegrationsManager } = require('./integrations');
```

## 🔧 Environment Configuration

Add these variables to your `.env` file:

```env
# Jellyfin Configuration
JELLYFIN_URL=http://localhost:8096
JELLYFIN_API_KEY=your-jellyfin-api-key
JELLYFIN_USERNAME=your-username
JELLYFIN_PASSWORD=your-password

# Plex Configuration
PLEX_URL=http://localhost:32400
PLEX_TOKEN=your-plex-token
PLEX_USERNAME=your-username
PLEX_PASSWORD=your-password

# Sonarr Configuration
SONARR_URL=http://localhost:8989
SONARR_API_KEY=your-sonarr-api-key

# Radarr Configuration
RADARR_URL=http://localhost:7878
RADARR_API_KEY=your-radarr-api-key

# Prowlarr Configuration
PROWLARR_URL=http://localhost:9696
PROWLARR_API_KEY=your-prowlarr-api-key

# Jellyseerr Configuration
JELLYSEERR_URL=http://localhost:5055
JELLYSEERR_API_KEY=your-jellyseerr-api-key

# Tautulli Configuration
TAUTULLI_URL=http://localhost:8181
TAUTULLI_API_KEY=your-tautulli-api-key

# NetFlow Configuration
NETFLOW_COLLECTOR_URL=http://localhost:9995
NETFLOW_API_KEY=your-netflow-api-key
NETFLOW_COLLECTOR_PORT=2055
```

## 🏗️ Usage Examples

### Quick Start with IntegrationsManager

```javascript
const { IntegrationsManager } = require('./integrations');

// Initialize all configured integrations
const manager = new IntegrationsManager();
const results = await manager.initializeAll();

logger.info('Integration status:', results);

// Get a specific integration
const jellyfin = manager.getIntegration('jellyfin');
if (jellyfin) {
    const libraries = await jellyfin.getLibraries();
    logger.info('Jellyfin libraries:', libraries);
}

// Setup webhooks on Express app
const express = require('express');
const app = express();
app.use(express.json());

manager.setupWebhooks(app);
app.listen(3000);
```

### Individual Integration Usage

#### Jellyfin Integration

```javascript
const { JellyfinIntegration } = require('./integrations');

const jellyfin = new JellyfinIntegration({
    baseURL: 'http://localhost:8096',
    apiKey: 'your-api-key'
});

// Authenticate
await jellyfin.authenticate('username', 'password');

// Get server info
const serverInfo = await jellyfin.getServerInfo();

// Get libraries
const libraries = await jellyfin.getLibraries();

// Search for content
const searchResults = await jellyfin.searchItems('Game of Thrones');

// Mark item as played
await jellyfin.markPlayed('item-id');

// Listen for events
jellyfin.on('webhook', (event) => {
    logger.info('Jellyfin webhook received:', event);
});
```

#### Plex Integration

```javascript
const { PlexIntegration } = require('./integrations');

const plex = new PlexIntegration({
    baseURL: 'http://localhost:32400',
    token: 'your-plex-token'
});

// Get server info
const serverInfo = await plex.getServerInfo();

// Get libraries
const libraries = await plex.getLibraries();

// Search content
const searchResults = await plex.search('Breaking Bad');

// Get active sessions
const sessions = await plex.getSessions();

// Mark as watched
await plex.markWatched('rating-key');
```

#### Sonarr Integration

```javascript
const { SonarrIntegration } = require('./integrations');

const sonarr = new SonarrIntegration({
    baseURL: 'http://localhost:8989',
    apiKey: 'your-api-key'
});

// Get all series
const series = await sonarr.getSeries();

// Add new series
const newSeries = await sonarr.addSeries({
    title: 'Breaking Bad',
    tvdbId: 81189,
    qualityProfileId: 1,
    rootFolderPath: '/tv'
});

// Get calendar
const calendar = await sonarr.getCalendar();

// Search for series
await sonarr.searchSeries('series-id');

// Get queue
const queue = await sonarr.getQueue();
```

#### Radarr Integration

```javascript
const { RadarrIntegration } = require('./integrations');

const radarr = new RadarrIntegration({
    baseURL: 'http://localhost:7878',
    apiKey: 'your-api-key'
});

// Get all movies
const movies = await radarr.getMovies();

// Add new movie
const newMovie = await radarr.addMovie({
    title: 'The Matrix',
    tmdbId: 603,
    year: 1999,
    qualityProfileId: 1,
    rootFolderPath: '/movies'
});

// Search for movie
await radarr.searchMovie('movie-id');

// Get upcoming movies
const upcoming = await radarr.getUpcoming();
```

#### Prowlarr Integration

```javascript
const { ProwlarrIntegration } = require('./integrations');

const prowlarr = new ProwlarrIntegration({
    baseURL: 'http://localhost:9696',
    apiKey: 'your-api-key'
});

// Get indexers
const indexers = await prowlarr.getIndexers();

// Search across all indexers
const searchResults = await prowlarr.search('Game of Thrones');

// Get applications
const applications = await prowlarr.getApplications();

// Get comprehensive statistics
const stats = await prowlarr.getStatistics();
```

#### Jellyseerr Integration

```javascript
const { JellyseerrIntegration } = require('./integrations');

const jellyseerr = new JellyseerrIntegration({
    baseURL: 'http://localhost:5055',
    apiKey: 'your-api-key'
});

// Get all requests
const requests = await jellyseerr.getRequests();

// Create new request
const newRequest = await jellyseerr.createRequest({
    mediaType: 'movie',
    mediaId: 603,
    is4k: false
});

// Approve request
await jellyseerr.approveRequest('request-id');

// Search media
const searchResults = await jellyseerr.searchMedia('The Matrix');
```

#### Tautulli Integration

```javascript
const { TautulliIntegration } = require('./integrations');

const tautulli = new TautulliIntegration({
    baseURL: 'http://localhost:8181',
    apiKey: 'your-api-key'
});

// Get current activity
const activity = await tautulli.getActivity();

// Get watch history
const history = await tautulli.getHistory();

// Get libraries
const libraries = await tautulli.getLibraries();

// Get comprehensive statistics
const stats = await tautulli.getStatistics();
```

#### NetFlow Integration

```javascript
const { NetflowIntegration } = require('./integrations');

const netflow = new NetflowIntegration({
    collectorPort: 2055,
    analysisEnabled: true
});

// Get real-time statistics
const stats = netflow.getStatistics();

// Search flows
const flows = netflow.searchFlows({
    srcAddr: '192.168.1.100',
    minBytes: 1000000
});

// Listen for media streaming detection
netflow.on('mediaFlowDetected', (data) => {
    logger.info('Media streaming detected:', data);
});

// Export flow data
const csvData = netflow.exportFlows('csv');
```

## 🎣 Webhook Events

All integrations support webhooks and emit standardized events:

```javascript
integration.on('webhook', (data) => {
    // Raw webhook data
});

integration.on('error', (error) => {
    // Integration errors
});

// Service-specific events
jellyfin.on('itemPlayed', (itemId) => {
    logger.info('Item played:', itemId);
});

sonarr.on('episodeGrabbed', (event) => {
    logger.info('Episode grabbed:', event);
});

plex.on('playbackStarted', (event) => {
    logger.info('Playback started:', event);
});
```

## 📊 Statistics & Analytics

Get comprehensive statistics from all services:

```javascript
const manager = new IntegrationsManager();
await manager.initializeAll();

// Get stats from all services
const allStats = await manager.getComprehensiveStats();

// Individual service stats
const jellyfinStats = await jellyfin.getStatistics();
const sonarrStats = await sonarr.getStatistics();
const netflowStats = netflow.getStatistics();
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all integration tests
node integrations/test-integrations.js

# Or use npm script
npm test
```

The test suite validates:
- ✅ Integration initialization
- ✅ Method availability
- ✅ Basic functionality
- ✅ Error handling
- ✅ Webhook setup
- ✅ Connection testing

## 🔍 API Reference

### Common Methods

All integrations implement these common patterns:

```javascript
// Test connection
const result = await integration.testConnection();
// Returns: { success: boolean, error?: string, ...details }

// Get statistics (where applicable)
const stats = await integration.getStatistics();

// Setup webhook
integration.setupWebhook(expressApp, '/webhook/path');

// Event listening
integration.on('event', handler);
```

### JellyfinIntegration

**Authentication & Server**
- `authenticate(username, password)` - Authenticate with server
- `getServerInfo()` - Get server information
- `getUsers()` - Get all users
- `getUserById(userId)` - Get user details

**Media Management**
- `getLibraries(userId?)` - Get user libraries
- `getLibraryItems(libraryId, options?)` - Get library content
- `searchItems(query, options?)` - Search media
- `getItem(itemId)` - Get item details
- `getLatestMedia(libraryId, limit?)` - Get latest additions
- `getResumeItems(limit?)` - Get continue watching
- `getNextUpEpisodes(limit?)` - Get next up episodes

**Playback & User Data**
- `markPlayed(itemId)` - Mark as played
- `markUnplayed(itemId)` - Mark as unplayed
- `addToFavorites(itemId)` - Add to favorites
- `removeFromFavorites(itemId)` - Remove from favorites
- `reportPlaybackStart(itemId, sessionId?)` - Report playback start
- `reportPlaybackProgress(itemId, positionTicks, sessionId?)` - Report progress
- `reportPlaybackStop(itemId, positionTicks, sessionId?)` - Report stop

**Administration**
- `getSessions()` - Get active sessions
- `getActivity(startIndex?, limit?)` - Get server activity
- `scanLibrary(libraryId)` - Trigger library scan
- `getImageUrl(itemId, imageType?, maxWidth?, maxHeight?)` - Get image URLs

### PlexIntegration

**Server & Libraries**
- `authenticate(username, password)` - Authenticate with Plex
- `getServerInfo()` - Get server information
- `getLibraries()` - Get all libraries
- `getLibraryContent(sectionId, options?)` - Get library content
- `search(query, options?)` - Search across libraries

**Media & Metadata**
- `getMetadata(ratingKey)` - Get item metadata
- `getChildren(ratingKey)` - Get item children
- `getRecentlyAdded(sectionId?, limit?)` - Get recently added
- `getOnDeck(limit?)` - Get on deck (continue watching)

**Playback & User Data**
- `markWatched(ratingKey)` - Mark as watched
- `markUnwatched(ratingKey)` - Mark as unwatched
- `rateItem(ratingKey, rating)` - Rate item

**Sessions & Activity**
- `getSessions()` - Get active sessions
- `getTranscodeSessions()` - Get transcode sessions
- `getActivities()` - Get server activities
- `getPlaylists()` - Get playlists
- `createPlaylist(title, type?, items?)` - Create playlist

**Administration**
- `updateLibrary(sectionId)` - Update library
- `getPreferences()` - Get server preferences
- `getImageUrl(imagePath, width?, height?)` - Get image URLs

### SonarrIntegration

**System & Status**
- `getSystemStatus()` - Get system status
- `getActivity()` - Get current activities
- `getStatistics()` - Get comprehensive statistics

**Series Management**
- `getSeries(includeSeasonImages?)` - Get all series
- `getSeriesById(seriesId)` - Get series details
- `addSeries(seriesData)` - Add new series
- `updateSeries(seriesId, updates)` - Update series
- `deleteSeries(seriesId, deleteFiles?, addExclusion?)` - Delete series
- `searchSeries(term)` - Search for series

**Episodes**
- `getEpisodes(seriesId, seasonNumber?)` - Get episodes
- `getEpisodeById(episodeId)` - Get episode details
- `updateEpisode(episodeId, updates)` - Update episode

**Calendar & Queue**
- `getCalendar(startDate?, endDate?, unmonitored?)` - Get calendar
- `getUpcoming(days?)` - Get upcoming episodes
- `getQueue(page?, pageSize?, sortKey?, sortDirection?)` - Get download queue
- `removeFromQueue(queueId, removeFromClient?, blocklist?)` - Remove from queue

**History & Search**
- `getHistory(page?, pageSize?, sortKey?, sortDirection?, episodeId?, eventType?)` - Get history
- `searchSeries(seriesId)` - Search for series
- `searchSeason(seriesId, seasonNumber)` - Search for season
- `searchEpisode(episodeIds)` - Search for episodes

**Configuration**
- `getQualityProfiles()` - Get quality profiles
- `getLanguageProfiles()` - Get language profiles
- `getRootFolders()` - Get root folders
- `getDownloadClients()` - Get download clients
- `getIndexers()` - Get indexers

**Maintenance**
- `refreshSeries(seriesId?)` - Refresh series metadata
- `rescanSeries(seriesId?)` - Rescan series files

### RadarrIntegration

**System & Status**
- `getSystemStatus()` - Get system status
- `getActivity()` - Get current activities
- `getStatistics()` - Get comprehensive statistics

**Movie Management**
- `getMovies()` - Get all movies
- `getMovieById(movieId)` - Get movie details
- `addMovie(movieData)` - Add new movie
- `updateMovie(movieId, updates)` - Update movie
- `deleteMovie(movieId, deleteFiles?, addExclusion?)` - Delete movie
- `searchMovies(term)` - Search for movies

**Files**
- `getMovieFiles(movieId?)` - Get movie files
- `getMovieFileById(fileId)` - Get file details
- `deleteMovieFile(fileId)` - Delete movie file

**Calendar & Queue**
- `getCalendar(startDate?, endDate?, unmonitored?)` - Get calendar
- `getUpcoming(days?)` - Get upcoming movies
- `getQueue(page?, pageSize?, sortKey?, sortDirection?)` - Get download queue
- `removeFromQueue(queueId, removeFromClient?, blocklist?)` - Remove from queue

**History & Search**
- `getHistory(page?, pageSize?, sortKey?, sortDirection?, movieId?, eventType?)` - Get history
- `searchMovie(movieId)` - Search for movie

**Configuration**
- `getQualityProfiles()` - Get quality profiles
- `getRootFolders()` - Get root folders
- `getDownloadClients()` - Get download clients
- `getIndexers()` - Get indexers
- `getImportLists()` - Get import lists

**Exclusions**
- `getExclusions()` - Get exclusions
- `addExclusion(tmdbId, title, year)` - Add exclusion
- `deleteExclusion(exclusionId)` - Delete exclusion

**Maintenance**
- `refreshMovie(movieId?)` - Refresh movie metadata
- `rescanMovie(movieId?)` - Rescan movie files
- `getMovieCredits(movieId)` - Get movie credits

### ProwlarrIntegration

**System & Status**
- `getSystemStatus()` - Get system status
- `getTasks()` - Get system tasks
- `runTask(taskName)` - Run system task
- `getHealth()` - Get health status
- `getStatistics()` - Get comprehensive statistics

**Indexer Management**
- `getIndexers()` - Get all indexers
- `getIndexerById(indexerId)` - Get indexer details
- `updateIndexer(indexerId, updates)` - Update indexer
- `toggleIndexer(indexerId, enabled)` - Enable/disable indexer
- `deleteIndexer(indexerId)` - Delete indexer
- `testIndexer(indexerId)` - Test indexer
- `getIndexerSchemas()` - Get available indexer types
- `addIndexer(indexerData)` - Add new indexer

**Search & Categories**
- `search(query, categories?, limit?, offset?)` - Search all indexers
- `searchIndexer(indexerId, query, categories?, limit?, offset?)` - Search specific indexer
- `getCategories()` - Get indexer categories
- `getIndexerStats()` - Get indexer statistics

**Applications**
- `getApplications()` - Get connected applications
- `addApplication(appData)` - Add application
- `updateApplication(appId, updates)` - Update application
- `deleteApplication(appId)` - Delete application
- `testApplication(appId)` - Test application
- `syncApplications()` - Sync applications

**Download Clients**
- `getDownloadClients()` - Get download clients
- `addDownloadClient(clientData)` - Add download client
- `updateDownloadClient(clientId, updates)` - Update download client
- `deleteDownloadClient(clientId)` - Delete download client
- `testDownloadClient(clientId)` - Test download client

**Tags & History**
- `getTags()` - Get tags
- `addTag(label)` - Add tag
- `deleteTag(tagId)` - Delete tag
- `getHistory(page?, pageSize?, sortKey?, sortDirection?)` - Get history

**Notifications & Logs**
- `getNotifications()` - Get notifications
- `testNotification(notificationId)` - Test notification
- `getLogs(page?, pageSize?, sortKey?, sortDirection?, level?)` - Get logs

### JellyseerrIntegration

**System & Status**
- `getStatus()` - Get system status
- `getSettings()` - Get settings
- `updateSettings(settings)` - Update settings
- `getStatistics()` - Get comprehensive statistics

**Request Management**
- `getRequests(take?, skip?, filter?, sort?)` - Get all requests
- `getRequestById(requestId)` - Get request details
- `createRequest(mediaData)` - Create new request
- `updateRequest(requestId, updates)` - Update request
- `deleteRequest(requestId)` - Delete request
- `approveRequest(requestId)` - Approve request
- `declineRequest(requestId)` - Decline request
- `retryRequest(requestId)` - Retry request

**Media Discovery**
- `searchMedia(query, page?, language?)` - Search media
- `getMediaDetails(mediaType, mediaId, language?)` - Get media details
- `getTrending(mediaType?, page?, language?)` - Get trending media
- `getPopular(mediaType?, page?, language?)` - Get popular media
- `getUpcoming(mediaType?, page?, language?)` - Get upcoming media

**User Management**
- `getUsers(take?, skip?, sort?)` - Get all users
- `getUserById(userId)` - Get user details
- `createUser(userData)` - Create user
- `updateUser(userId, updates)` - Update user
- `deleteUser(userId)` - Delete user
- `getUserQuota(userId)` - Get user quota

**Issue Management**
- `getIssues(take?, skip?, sort?, filter?)` - Get all issues
- `getIssueById(issueId)` - Get issue details
- `createIssue(issueData)` - Create issue
- `updateIssue(issueId, updates)` - Update issue
- `deleteIssue(issueId)` - Delete issue

**Watchlist**
- `getWatchlist(userId)` - Get user watchlist
- `addToWatchlist(userId, mediaData)` - Add to watchlist
- `removeFromWatchlist(userId, mediaId)` - Remove from watchlist

**Service Configuration**
- `getPlexServers()` - Get Plex servers
- `getSonarrSettings()` - Get Sonarr settings
- `getRadarrSettings()` - Get Radarr settings
- `testService(serviceType, config)` - Test service connection

### TautulliIntegration

**Server & Identity**
- `getServerInfo()` - Get Plex server info
- `getServerIdentity()` - Get server identity
- `getServerResources()` - Get server resources

**Activity & Sessions**
- `getActivity(sessionKey?, sessionId?)` - Get current activity
- `getSessions()` - Get active sessions (alias for getActivity)
- `terminateSession(sessionKey, message?)` - Terminate session

**History & Statistics**
- `getHistory(options)` - Get detailed history with filtering
- `getPlaysByDate(timeRange?, userId?, grouping?)` - Get plays by date
- `getPlaysByHour(timeRange?, userId?, grouping?)` - Get plays by hour
- `getPlaysByDayOfWeek(timeRange?, userId?, grouping?)` - Get plays by day of week
- `getPlaysByTop10Platforms(timeRange?, userId?, grouping?)` - Get top platforms
- `getPlaysByTop10Users(timeRange?, userId?, grouping?)` - Get top users
- `getHomeStats(timeRange?, statsType?, statsCount?)` - Get home statistics
- `getStatistics()` - Get comprehensive statistics

**Libraries & Media**
- `getLibraries()` - Get all libraries
- `getLibrary(sectionId)` - Get library details
- `getLibraryMediaInfo(sectionId, options)` - Get library media info
- `getRecentlyAdded(count?, start?, mediaType?, sectionId?)` - Get recently added
- `refreshLibrariesList()` - Refresh libraries list

**Metadata & Items**
- `getMetadata(ratingKey, mediaInfo?)` - Get item metadata
- `getChildrenMetadata(ratingKey)` - Get children metadata
- `getItemWatchTimeStats(ratingKey, grouping?)` - Get item watch time stats
- `getItemUserStats(ratingKey, grouping?)` - Get item user stats

**Users**
- `getUsers()` - Get all users
- `getUser(userId)` - Get user details
- `getUserWatchTimeStats(userId, grouping?, queryDays?)` - Get user watch time stats
- `getUserPlayerStats(userId, grouping?, queryDays?)` - Get user player stats

**Stream Analysis**
- `getStreamTypeByTop10Platforms(timeRange?, userId?, grouping?)` - Get stream types by platform
- `getStreamTypeByTop10Users(timeRange?, userId?, grouping?)` - Get stream types by user

**System & Logs**
- `getNotifications()` - Get notifications
- `getLogs(sort?, search?, order?, regex?, start?, end?)` - Get Tautulli logs
- `getPlexLog(window?, logLevel?)` - Get Plex server logs
- `deleteImageCache()` - Delete image cache

### NetflowIntegration

**Flow Collection & Processing**
- `initializeCollector()` - Start NetFlow collector
- `processNetflowPacket(buffer, rinfo)` - Process raw NetFlow packet
- `processFlow(flow, sourceInfo)` - Process individual flow
- `cleanup()` - Stop collector and cleanup resources

**Flow Analysis**
- `getStatistics()` - Get real-time statistics
- `getFlowHistory(limit?)` - Get flow history
- `searchFlows(criteria)` - Search flows by criteria
- `analyzeFlows()` - Analyze flows for insights
- `isMediaStreamingFlow(flow)` - Check if flow is media streaming

**Data Export**
- `exportFlows(format?)` - Export flow data (JSON/CSV)
- `convertToCSV(flows)` - Convert flows to CSV format

**Media Streaming Detection**
- `processMediaFlow(flow, flowKey)` - Process media streaming flow
- `estimateStreamQuality(bandwidth)` - Estimate stream quality
- `getTopStreamingSessions(limit?)` - Get top streaming sessions
- `getQualityDistribution()` - Get quality distribution

**Network Health**
- `assessNetworkHealth()` - Assess network health
- `generateAlerts()` - Generate network alerts
- `getTotalBandwidth()` - Get total bandwidth usage

## 🔒 Security Best Practices

1. **API Keys**: Store API keys in environment variables, never in code
2. **Authentication**: Use token-based authentication where possible
3. **Rate Limiting**: Implement rate limiting on webhook endpoints
4. **Input Validation**: All user inputs are validated before API calls
5. **Error Handling**: Sensitive information is not exposed in error messages
6. **HTTPS**: Use HTTPS for all API communications in production

## 🚨 Error Handling

All integrations implement consistent error handling:

```javascript
try {
    const result = await integration.someMethod();
} catch (error) {
    // Structured error information
    logger.error('Integration error:', {
        service: 'jellyfin',
        method: 'someMethod',
        message: error.message,
        status: error.response?.status,
        statusText: error.response?.statusText
    });
}

// Event-based error handling
integration.on('error', (error) => {
    logger.error('Integration error event:', error);
});
```

## 📈 Performance Optimization

- **Connection Pooling**: Axios clients with persistent connections
- **Request Timeout**: 30-second timeout on all requests
- **Retry Logic**: Automatic retry with exponential backoff
- **Caching**: Built-in caching for frequently accessed data
- **Event Streaming**: Efficient event handling with EventEmitter
- **Memory Management**: Proper cleanup and resource management

## 🤝 Contributing

1. Add new integration modules following the established patterns
2. Implement the standard interface methods (`testConnection`, `getStatistics`, etc.)
3. Add comprehensive error handling and event emission
4. Include webhook support where applicable
5. Add tests to the test suite
6. Update this documentation

## 📄 License

This project is part of the larger media server ecosystem and follows the same licensing terms.

---

🎬 **Happy Media Server Integration!** 🎭

For support or questions, please refer to the main project documentation or open an issue.