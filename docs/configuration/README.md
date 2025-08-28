# Configuration Guide

This guide explains how to configure the Ultimate Media Server stack using the `.env` file. Before you start, make sure you have copied `.env.example` to `.env`.

## General Settings

These are the basic settings for your media server.

### `TZ`

*   **Description:** The timezone for all services. This ensures that all services have the correct time for logging and scheduling.
*   **Example:** `America/New_York`
*   **Action:** Set this to your local timezone. You can find a list of valid timezones [here](https://en.wikipedia.org/wiki/List_of_tz_database_time_zones).

## Media Paths

These settings define the locations where your media files and downloads are stored.

### `MEDIA_PATH`

*   **Description:** The root directory for all your media files.
*   **Default:** `./media-data`
*   **Action:** You can change this to any path on your host machine. It's recommended to use an absolute path to a directory with plenty of storage space.

### `DOWNLOADS_PATH`

*   **Description:** The directory where your download clients will store completed downloads.
*   **Default:** `./media-data/downloads`
*   **Action:** This should be a subdirectory of `MEDIA_PATH`.

### `USENET_PATH`

*   **Description:** The directory where your Usenet clients will store completed downloads.
*   **Default:** `./media-data/usenet`
*   **Action:** This should be a subdirectory of `MEDIA_PATH`.

## VPN Configuration

These settings are for configuring the Gluetun VPN container, which is used to secure your download traffic.

### `VPN_PROVIDER`

*   **Description:** The VPN provider you are using.
*   **Default:** `mullvad`
*   **Action:** Set this to your VPN provider. A list of supported providers can be found in the [Gluetun documentation](https://github.com/qdm12/gluetun-wiki/blob/main/setup/providers/providers.md).

### `VPN_PRIVATE_KEY`

*   **Description:** Your WireGuard private key.
*   **Default:** `your_wireguard_private_key_here`
*   **Action:** Set this to your WireGuard private key from your VPN provider.

### `VPN_ADDRESSES`

*   **Description:** The WireGuard IP addresses to use.
*   **Default:** `10.x.x.x/32`
*   **Action:** Set this to the WireGuard IP addresses provided by your VPN provider.

## Cloudflare Configuration

These settings are for configuring Cloudflare for DNS and reverse proxying.

### `CLOUDFLARE_EMAIL`

*   **Description:** The email address for your Cloudflare account.
*   **Default:** `your_email@example.com`
*   **Action:** Set this to your Cloudflare account email.

### `CLOUDFLARE_API_KEY`

*   **Description:** Your Cloudflare API key.
*   **Default:** `your_cloudflare_api_key_here`
*   **Action:** You can find your Cloudflare API key in your Cloudflare account settings.

## Service Passwords

These settings are for configuring the passwords for various services.

### `GRAFANA_USER`

*   **Description:** The username for the Grafana admin user.
*   **Default:** `admin`
*   **Action:** You can change this to a different username if you wish.

### `GRAFANA_PASSWORD`

*   **Description:** The password for the Grafana admin user.
*   **Default:** `secure_password_here`
*   **Action:** **It is critical that you change this to a strong, unique password.**

## Domain Configuration

This setting is for configuring the domain name for your media server.

### `DOMAIN`

*   **Description:** The domain name you will use to access your media server.
*   **Default:** `yourdomain.com`
*   **Action:** Set this to your own domain name. This is used by Nginx Proxy Manager to set up the reverse proxy and obtain SSL certificates.
