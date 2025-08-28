# Service Catalog

This document provides a detailed overview of each service included in the Ultimate Media Server stack.

## Media Servers

Media servers are the core of the stack, responsible for organizing, streaming, and playing your media files on various devices.

### 🎬 Jellyfin

Jellyfin is a free and open-source media server that puts you in control of your media. It's a community-built fork of Emby that has no premium features, licenses, or hidden agendas.

**Key Features:**

*   **Wide Device Support:** Stream to any device with a modern web browser. Clients are also available for Android, iOS, Roku, Apple TV, and more.
*   **Live TV & DVR:** Watch live TV and set recordings with a compatible tuner.
*   **Customization:** A wide range of plugins and customization options to tailor your experience.
*   **User Management:** Create multiple user accounts with their own watch history, progress, and parental controls.

**Screenshot Details:**

> A screenshot of the Jellyfin dashboard would show a vibrant and modern interface. The main view would be dominated by colorful posters of movies and TV shows, organized into horizontal carousels like "Latest Movies," "Continue Watching," and "Next Up." A sidebar on the left would provide navigation to different libraries (Movies, TV Shows, Music) and administrative settings. The overall feel is clean, intuitive, and focused on showcasing your media library in a visually appealing way.

### 🎭 Plex

Plex is a popular and feature-rich media server that offers a polished user experience and a wide range of client applications. It has a freemium model, with some advanced features requiring a Plex Pass subscription.

**Key Features:**

*   **Polished UI:** A highly polished and user-friendly interface that is consistent across all devices.
*   **Remote Access:** Simple and secure remote access to your media library from anywhere in the world.
*   **Plex Pass (Premium):** Optional subscription for features like mobile sync, live TV & DVR, parental controls, and hardware-accelerated streaming.
*   **Discover & Watchlists:** Integrated discovery features to find new content and a universal watchlist for tracking movies and shows.

**Screenshot Details:**

> A screenshot of the Plex interface would look very sleek and professional. Similar to Jellyfin, the main screen would feature a grid of movie and TV show posters, but with a slightly more refined and minimalist design. The left-hand sidebar would allow you to navigate between your libraries and Plex's own streaming services. A prominent search bar at the top would make it easy to find content. The overall impression is one of a premium, user-friendly streaming service like Netflix or Hulu, but populated with your own media.

### 📺 Emby

Emby is a media server that offers a similar feature set to Plex and Jellyfin. It also follows a freemium model, with some features requiring an Emby Premiere subscription. Emby is known for its performance and wide range of customization options.

**Key Features:**

*   **Performance:** Known for its fast and efficient media scanning and streaming.
*   **Live TV & DVR:** Robust Live TV and DVR capabilities with a compatible tuner.
*   **Parental Controls:** Advanced parental control features to manage what your kids can watch.
*   **Emby Premiere (Premium):** Subscription for features like mobile sync, offline media, and cloud sync.

**Screenshot Details:**

> A screenshot of the Emby dashboard would present a clean and organized layout. The main content area would display your media libraries with poster art, similar to Plex and Jellyfin. The color scheme is typically a dark theme with blue accents, giving it a modern look. The navigation is usually a top bar with dropdown menus for accessing different libraries, settings, and live TV. Emby's interface is highly customizable, so the layout and appearance can vary, but it generally feels powerful and feature-rich, with a focus on providing a lot of information and control to the user.

## Content Management

Content management services, often referred to as the "*ARR" suite, automate the process of finding, downloading, and organizing your media files.

### 📺 Sonarr

Sonarr is a TV show management tool that automates the process of finding, downloading, and organizing TV series. It monitors your favorite shows for new episodes and automatically sends them to your download client.

**Key Features:**

*   **Automatic Episode Fetching:** Monitors RSS feeds for new episodes of your favorite shows and grabs them as soon as they're available.
*   **Quality Profiles:** Define the quality of video you want, from standard definition to 4K.
*   **Calendar View:** A calendar view to see all upcoming episodes.
*   **Failed Download Handling:** Automatically retries failed downloads and can block releases that are known to be fake or problematic.

**Screenshot Details:**

> A screenshot of the Sonarr interface would show a list of TV series in your library, each with its poster art, the number of episodes you have, and the total number of episodes in the series. The color-coded progress bar for each series would give you a quick overview of how complete your collection is. The top navigation bar would have links to the series list, a calendar of upcoming episodes, and settings. The overall look is clean, data-rich, and focused on managing your TV show collection efficiently.

### 🎬 Radarr

Radarr is a movie management tool, similar to Sonarr but for movies. It automates the process of finding, downloading, and organizing your movie collection.

**Key Features:**

*   **Automated Movie Downloading:** Monitors your watchlist for movies and automatically sends them to your download client when they become available in your desired quality.
*   **Quality Profiles:** Set your preferred movie quality, from DVD to 4K.
*   **Calendar View:** A calendar to see upcoming movie releases.
*   **Custom Lists:** Integrate with popular movie lists (like IMDb Top 250) to automatically add movies to your library.

**Screenshot Details:**

> A screenshot of the Radarr interface would be very similar to Sonarr's, but for movies. The main view would be a grid of movie posters, each with information about its status (e.g., "Downloaded," "Missing"). You'd see a search bar at the top to add new movies, and the navigation bar would have links to your movie list, a calendar of upcoming releases, and settings. The interface is clean, organized, and designed to make managing a large movie collection as easy as possible.

### 🎵 Lidarr

Lidarr is a music management tool that automates the process of finding, downloading, and organizing your music collection. It's particularly useful for managing large music libraries and ensuring you have all the albums from your favorite artists.

**Key Features:**

*   **Automated Music Downloading:** Monitors artists and automatically downloads new albums when they are released.
*   **Quality Profiles:** Define your preferred audio quality, from MP3 to FLAC.
*   **Artist and Album Management:** A comprehensive interface for managing your music library by artist and album.
*   **Integration with Music Players:** Integrates with various music players and library managers.

**Screenshot Details:**

> A screenshot of the Lidarr interface would show a list of artists in your library, each with their banner image. For each artist, you'd see a list of their albums, with a status indicator to show which albums you have and which are missing. The navigation would allow you to switch between artists, albums, and a calendar of upcoming releases. The design is focused on providing a detailed overview of your music collection and making it easy to fill in the gaps.

### 📚 Readarr

Readarr is a book management tool that automates the process of finding, downloading, and organizing your e-book collection. It's a great tool for avid readers who want to manage a large digital library.

**Key Features:**

*   **Automated Book Downloading:** Monitors authors and automatically downloads new books when they are released.
*   **Quality Profiles:** Define your preferred e-book format (e.g., EPUB, MOBI).
*   **Author and Book Management:** A comprehensive interface for managing your e-book library by author and book.
*   **Integration with E-book Readers:** Integrates with various e-book readers and library managers like Calibre.

**Screenshot Details:**

> A screenshot of the Readarr interface would be similar to the other *ARR applications, but tailored for books. You would see a list of authors, and for each author, a list of their books with their status. The interface would be clean and organized, with a focus on managing your e-book collection. You'd be able to see at a glance which books you have and which you're missing from your favorite authors.

### 💬 Bazarr

Bazarr is a subtitle management tool that works in conjunction with Sonarr and Radarr. It automatically finds, downloads, and organizes subtitles for your movies and TV shows.

**Key Features:**

*   **Automatic Subtitle Downloading:** Monitors your Sonarr and Radarr libraries and automatically downloads matching subtitles.
*   **Subtitle Language Support:** Supports a wide range of subtitle languages.
*   **Subtitle Quality and Syncing:** Can be configured to find subtitles that are perfectly synced with your video files.
*   **Manual Search:** A manual search feature to find subtitles for specific movies or episodes.

**Screenshot Details:**

> A screenshot of the Bazarr interface would show two main sections: one for movies and one for TV shows, mirroring your Radarr and Sonarr libraries. For each movie or episode, you'd see the status of its subtitles (e.g., "Downloaded," "Missing"). You could then drill down to see the available subtitles in different languages and choose which ones to download. The interface is functional and focused on one task: getting you the right subtitles for your media.

### 🔍 Prowlarr

Prowlarr is an indexer manager for the *ARR suite. It allows you to manage all of your torrent and Usenet indexers in one place and automatically syncs them to your other *ARR applications.

**Key Features:**

*   **Centralized Indexer Management:** Add, configure, and manage all of your indexers in one place.
*   **Automatic Syncing:** Automatically syncs your indexers to Sonarr, Radarr, Lidarr, and Readarr.
*   **Indexer Health Checks:** Monitors the status of your indexers and notifies you if they go down.
*   **Statistics:** Provides detailed statistics on your indexer usage.

**Screenshot Details:**

> A screenshot of the Prowlarr interface would show a list of all your configured indexers. For each indexer, you'd see its name, type (e.g., "Torznab," "Newznab"), and status. You could add new indexers, test existing ones, and see detailed statistics on their usage. The interface is clean, technical, and designed for power users who want to fine-tune their automated download setup.

## Security & VPN

These services help to secure your media server and protect your privacy while downloading files.

### 🛡️ Gluetun

Gluetun is a lightweight and easy-to-use VPN client in a Docker container. It allows you to connect your other Docker containers to the internet through a VPN, ensuring your privacy and security.

**Key Features:**

*   **Wide VPN Provider Support:** Supports a large number of VPN providers, including NordVPN, ExpressVPN, and Mullvad.
*   **Kill Switch:** A built-in kill switch to prevent traffic from leaking if the VPN connection drops.
*   **Proxy Support:** Can act as a proxy server for your other devices.
*   **DNS over TLS:** Encrypts your DNS queries to prevent them from being snooped on.

**Screenshot Details:**

> Gluetun runs in the background and doesn't have a traditional web interface for you to interact with. A screenshot would typically show the container logs, which would display the status of the VPN connection. You would see lines of text indicating that the container has successfully connected to a VPN server, the IP address it has been assigned, and the status of the kill switch. The logs are the primary way to monitor Gluetun's activity.

### 🚫 Pi-hole

Pi-hole is a network-wide ad blocker that protects your devices from unwanted content without installing any client-side software. It acts as a DNS sinkhole, blocking ads at the network level.

**Key Features:**

*   **Network-wide Ad Blocking:** Block ads on all your devices, including smart TVs and mobile devices.
*   **Dashboard:** A web interface with detailed statistics on your network's ad-blocking activity.
*   **Whitelist/Blacklist:** Easily whitelist or blacklist domains to customize your blocking.
*   **DHCP Server:** Can act as a DHCP server for your network.

**Screenshot Details:**

> A screenshot of the Pi-hole dashboard would show a clean and informative interface with a variety of charts and statistics. The main dashboard would have graphs showing the total number of DNS queries, the number of blocked queries, and the percentage of blocked queries. You would also see a list of the top blocked domains and the top clients on your network. The interface is designed to give you a comprehensive overview of your network's ad-blocking activity at a glance.

### 🛡️ AdGuard Home

AdGuard Home is another network-wide ad and tracker blocker. It provides a more modern and feature-rich interface than Pi-hole, with a focus on providing more granular control over your network's filtering.

**Key Features:**

*   **Advanced Ad Blocking:** Blocks ads, trackers, and malicious websites.
*   **Parental Controls:** Built-in parental control features to block access to adult content.
*   **DNS-over-HTTPS and DNS-over-TLS:** Supports encrypted DNS protocols for enhanced privacy.
*   **Modern Interface:** A clean and modern web interface with detailed statistics and filtering options.

**Screenshot Details:**

> A screenshot of the AdGuard Home dashboard would look very modern and polished. The main screen would feature a set of colorful charts and graphs showing your DNS query statistics, similar to Pi-hole but with a more contemporary design. You would see a top navigation bar for accessing the query log, filter settings, and parental controls. The interface is designed to be very user-friendly and provides a lot of control over your network's filtering, with easy-to-use toggles for enabling and disabling various blocking features.

## Request Management

Request management services allow your users to request new content in a streamlined and automated way.

### 🎯 Jellyseerr

Jellyseerr is a request management and media discovery tool for Jellyfin and Emby. It's a fork of Overseerr, adapted to work with Jellyfin's ecosystem.

**Key Features:**

*   **Jellyfin and Emby Integration:** Specifically designed to work with Jellyfin and Emby media servers.
*   **User-friendly Interface:** A clean and modern interface for users to browse and request content.
*   **Automated Request Handling:** Automatically sends requests to Sonarr and Radarr for downloading.
*   **Granular Permissions:** Control who can request content and set approval requirements.

**Screenshot Details:**

> A screenshot of the Jellyseerr interface would look very similar to Overseerr, with a modern and visually appealing design. The main dashboard would feature carousels of trending, popular, and recently added media. Users would see a prominent search bar to find and request movies or TV shows. The interface is designed to be very user-friendly, resembling a modern streaming service, making it easy for non-technical users to request content.

### 📋 Overseerr

Overseerr is a request management and media discovery tool for Plex. It provides a user-friendly interface for Plex users to request content.

**Key Features:**

*   **Plex Integration:** Deep integration with Plex for user authentication and library management.
*   **Sleek and Modern UI:** A beautiful and intuitive interface for browsing and requesting content.
*   **Automated Request Handling:** Automatically sends requests to Sonarr and Radarr.
*   **User Management:** Granular control over user permissions and request approvals.

**Screenshot Details:**

> A screenshot of the Overseerr dashboard would showcase a polished and professional-looking interface. The main view would be filled with high-quality posters of movies and TV shows, organized into categories like "Popular on Plex," "Trending," and "New Releases." The design is very clean and modern, with a focus on making it easy for users to discover and request new content. It looks and feels like a premium application.

### 📝 Ombi

Ombi is another popular request management tool that works with Plex, Emby, and Jellyfin. It offers a wide range of features and integrations.

**Key Features:**

*   **Multi-server Support:** Works with Plex, Emby, and Jellyfin.
*   **Mobile App:** A mobile app for users to request content on the go.
*   **Notifications:** A robust notification system to keep users updated on the status of their requests.
*   **Issues and Reporting:** A built-in issue reporting system for users to report problems with their media.

**Screenshot Details:**

> A screenshot of the Ombi interface would show a clean and functional design. The main dashboard would display recently requested and available media. The interface is more focused on functionality than the other two, with a clear and easy-to-navigate layout. Users can easily search for content, make requests, and track the status of their requests. It's a powerful and feature-rich interface that provides a lot of information and control to both users and administrators.
