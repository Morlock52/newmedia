# Varnish Configuration for Edge Caching
# Optimized for 8K video streaming and ultra-low latency

vcl 4.1;

import std;
import directors;
import vsthrottle;
import xkey;

# Define backend servers
backend jellyfin_primary {
    .host = "jellyfin_gpu";
    .port = "8096";
    .probe = {
        .url = "/health";
        .interval = 5s;
        .timeout = 1s;
        .window = 5;
        .threshold = 3;
    }
    .first_byte_timeout = 300s;
    .connect_timeout = 5s;
    .between_bytes_timeout = 60s;
    .max_connections = 1000;
}

backend jellyfin_secondary {
    .host = "jellyfin_gpu_secondary";
    .port = "8096";
    .probe = {
        .url = "/health";
        .interval = 5s;
        .timeout = 1s;
        .window = 5;
        .threshold = 3;
    }
    .first_byte_timeout = 300s;
    .connect_timeout = 5s;
    .between_bytes_timeout = 60s;
    .max_connections = 1000;
}

# Initialize director for load balancing
sub vcl_init {
    new media_director = directors.round_robin();
    media_director.add_backend(jellyfin_primary);
    media_director.add_backend(jellyfin_secondary);
}

# ACL for purge requests
acl purge {
    "localhost";
    "127.0.0.1";
    "172.20.0.0"/24;
}

# ACL for local networks (bypass certain restrictions)
acl local_net {
    "10.0.0.0"/8;
    "172.16.0.0"/12;
    "192.168.0.0"/16;
}

sub vcl_recv {
    # Set backend
    set req.backend_hint = media_director.backend();
    
    # Normalize the host header
    set req.http.Host = regsub(req.http.Host, ":[0-9]+", "");
    
    # Normalize the URL
    set req.url = std.querysort(req.url);
    
    # Remove tracking parameters
    if (req.url ~ "(\?|&)(utm_source|utm_medium|utm_campaign|utm_content|gclid|cx|ie|cof|siteurl)=") {
        set req.url = regsuball(req.url, "&(utm_source|utm_medium|utm_campaign|utm_content|gclid|cx|ie|cof|siteurl)=[^&]+", "");
        set req.url = regsuball(req.url, "\?(utm_source|utm_medium|utm_campaign|utm_content|gclid|cx|ie|cof|siteurl)=[^&]+", "?");
        set req.url = regsub(req.url, "\?&", "?");
        set req.url = regsub(req.url, "\?$", "");
    }
    
    # Handle PURGE requests
    if (req.method == "PURGE") {
        if (!client.ip ~ purge) {
            return(synth(405, "Not allowed."));
        }
        
        if (req.http.xkey) {
            set req.http.n-gone = xkey.purge(req.http.xkey);
            return (synth(200, "Purged " + req.http.n-gone + " objects"));
        } else {
            return (purge);
        }
    }
    
    # Handle BAN requests for pattern-based cache invalidation
    if (req.method == "BAN") {
        if (!client.ip ~ purge) {
            return(synth(405, "Not allowed."));
        }
        
        ban("obj.http.x-url ~ " + req.url);
        return(synth(200, "Ban added"));
    }
    
    # Pipe video streaming requests
    if (req.url ~ "\.(mp4|webm|mkv|avi|mov|flv|wmv|m4v)$") {
        return (pipe);
    }
    
    # Handle Range requests for video segments
    if (req.http.Range) {
        set req.http.x-range = req.http.Range;
        return (hash);
    }
    
    # Cache HLS/DASH manifests with short TTL
    if (req.url ~ "\.(m3u8|mpd)$") {
        unset req.http.Cookie;
        return (hash);
    }
    
    # Cache video segments aggressively
    if (req.url ~ "\.(ts|m4s|mp4|webm)$") {
        unset req.http.Cookie;
        unset req.http.Cache-Control;
        return (hash);
    }
    
    # Cache images
    if (req.url ~ "\.(jpg|jpeg|png|gif|webp|avif|ico|svg)$") {
        unset req.http.Cookie;
        return (hash);
    }
    
    # Cache static assets
    if (req.url ~ "\.(css|js|woff|woff2|ttf|otf|eot)$") {
        unset req.http.Cookie;
        return (hash);
    }
    
    # WebSocket pass-through
    if (req.http.Upgrade ~ "(?i)websocket") {
        return (pipe);
    }
    
    # API requests - intelligent caching
    if (req.url ~ "^/api/") {
        # Cache GET requests
        if (req.method == "GET") {
            unset req.http.Cookie;
            return (hash);
        }
        # Pass through other methods
        return (pass);
    }
    
    # Health check endpoint
    if (req.url == "/varnish-health") {
        return (synth(200, "OK"));
    }
    
    # Strip cookies for cacheable content
    if (req.method == "GET" || req.method == "HEAD") {
        if (req.url !~ "^/(login|logout|admin|auth)") {
            unset req.http.Cookie;
        }
    }
    
    # Grace mode for stale content delivery
    if (std.healthy(req.backend_hint)) {
        set req.grace = 10s;
    } else {
        set req.grace = 24h;
    }
    
    return (hash);
}

sub vcl_hash {
    hash_data(req.url);
    
    if (req.http.host) {
        hash_data(req.http.host);
    } else {
        hash_data(server.ip);
    }
    
    # Include protocol in hash
    if (req.http.X-Forwarded-Proto) {
        hash_data(req.http.X-Forwarded-Proto);
    }
    
    # Hash based on device type for responsive content
    if (req.http.User-Agent ~ "(Mobile|Android|iPhone|iPad)") {
        hash_data("mobile");
    } else {
        hash_data("desktop");
    }
    
    # Hash range requests separately
    if (req.http.Range) {
        hash_data(req.http.Range);
    }
    
    return (lookup);
}

sub vcl_backend_response {
    # Store URL for ban operations
    set beresp.http.x-url = bereq.url;
    set beresp.http.x-host = bereq.http.host;
    
    # Enable grace mode
    set beresp.grace = 24h;
    
    # Set cache TTLs based on content type
    if (bereq.url ~ "\.(mp4|webm|mkv|avi|mov)$") {
        set beresp.ttl = 30d;
        set beresp.http.Cache-Control = "public, max-age=2592000";
        set beresp.do_stream = true;
    } elsif (bereq.url ~ "\.(ts|m4s)$") {
        # Video segments - cache for 7 days
        set beresp.ttl = 7d;
        set beresp.http.Cache-Control = "public, max-age=604800";
    } elsif (bereq.url ~ "\.(m3u8|mpd)$") {
        # Manifests - short cache
        set beresp.ttl = 10s;
        set beresp.http.Cache-Control = "public, max-age=10";
    } elsif (bereq.url ~ "\.(jpg|jpeg|png|gif|webp|avif)$") {
        # Images - cache for 30 days
        set beresp.ttl = 30d;
        set beresp.http.Cache-Control = "public, max-age=2592000";
    } elsif (bereq.url ~ "\.(css|js)$") {
        # Static assets - cache for 7 days
        set beresp.ttl = 7d;
        set beresp.http.Cache-Control = "public, max-age=604800";
    } elsif (bereq.url ~ "^/api/" && bereq.method == "GET") {
        # API responses - cache for 1 minute
        set beresp.ttl = 1m;
        set beresp.http.Cache-Control = "public, max-age=60";
    }
    
    # Handle errors with stale content
    if (beresp.status >= 500 && beresp.status < 600) {
        if (beresp.http.Content-Length == "0") {
            return (abandon);
        }
        set beresp.saintmode = 10s;
        return (deliver);
    }
    
    # Compress text responses
    if (beresp.http.Content-Type ~ "^(text/|application/(json|javascript|xml))") {
        set beresp.do_gzip = true;
    }
    
    # Set Surrogate-Key for xkey purging
    if (bereq.url ~ "^/media/([^/]+)") {
        set beresp.http.xkey = "media " + regsub(bereq.url, "^/media/([^/]+).*", "\1");
    }
    
    # Add custom headers for debugging
    set beresp.http.X-Cache-TTL = beresp.ttl;
    set beresp.http.X-Backend-Name = beresp.backend.name;
    
    return (deliver);
}

sub vcl_deliver {
    # Add cache hit/miss header
    if (obj.hits > 0) {
        set resp.http.X-Cache = "HIT";
        set resp.http.X-Cache-Hits = obj.hits;
    } else {
        set resp.http.X-Cache = "MISS";
    }
    
    # Security headers
    set resp.http.X-Frame-Options = "SAMEORIGIN";
    set resp.http.X-Content-Type-Options = "nosniff";
    set resp.http.X-XSS-Protection = "1; mode=block";
    set resp.http.Referrer-Policy = "strict-origin-when-cross-origin";
    
    # CORS headers for media content
    if (req.url ~ "\.(mp4|webm|m3u8|mpd|ts|m4s)$") {
        set resp.http.Access-Control-Allow-Origin = "*";
        set resp.http.Access-Control-Allow-Methods = "GET, OPTIONS";
        set resp.http.Access-Control-Allow-Headers = "Range";
        set resp.http.Access-Control-Expose-Headers = "Content-Length, Content-Range";
    }
    
    # Performance headers
    set resp.http.X-Served-By = server.hostname;
    set resp.http.X-Varnish-Cache = regsub(obj.storage, "^s0", "memory");
    
    # Clean up internal headers
    unset resp.http.x-url;
    unset resp.http.x-host;
    unset resp.http.X-Varnish;
    unset resp.http.Via;
    unset resp.http.X-Backend-Name;
    unset resp.http.X-Cache-TTL;
    
    return (deliver);
}

sub vcl_backend_error {
    # Serve stale content on backend error
    if (bereq.retries < 3) {
        return (retry);
    }
    
    # Generate a simple error page
    synthetic({"
<!DOCTYPE html>
<html>
<head>
    <title>Service Temporarily Unavailable</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 50px; }
        h1 { color: #e74c3c; }
        p { color: #7f8c8d; }
    </style>
</head>
<body>
    <h1>Service Temporarily Unavailable</h1>
    <p>The media server is currently experiencing issues. Please try again later.</p>
    <p>Error: "} + beresp.status + " " + beresp.reason + {"</p>
</body>
</html>
"});
    
    return (deliver);
}

sub vcl_synth {
    set resp.http.Content-Type = "text/html; charset=utf-8";
    set resp.http.Retry-After = "5";
    
    if (resp.status == 720) {
        # Redirect
        set resp.http.Location = resp.reason;
        set resp.status = 301;
        return (deliver);
    } elsif (resp.status == 721) {
        # Temporary redirect
        set resp.http.Location = resp.reason;
        set resp.status = 302;
        return (deliver);
    }
    
    return (deliver);
}

sub vcl_fini {
    return (ok);
}