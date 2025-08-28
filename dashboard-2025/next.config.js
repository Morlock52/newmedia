/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    turbo: {
      loaders: {
        '.svg': ['@svgr/webpack'],
      },
    },
  },
  
  // PWA Configuration
  async headers() {
    return [
      {
        source: '/sw.js',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=0, must-revalidate',
          },
          {
            key: 'Service-Worker-Allowed',
            value: '/',
          },
        ],
      },
      {
        source: '/manifest.json',
        headers: [
          {
            key: 'Cache-Control',
            value: 'public, max-age=86400',
          },
        ],
      },
      {
        source: '/(.*)',
        headers: [
          {
            key: 'X-Frame-Options',
            value: 'DENY',
          },
          {
            key: 'X-Content-Type-Options',
            value: 'nosniff',
          },
          {
            key: 'Referrer-Policy',
            value: 'strict-origin-when-cross-origin',
          },
          {
            key: 'Permissions-Policy',
            value: 'camera=(), microphone=(), geolocation=()',
          },
        ],
      },
    ]
  },

  // Webpack configuration for Three.js and 3D libraries
  webpack: (config, { isServer }) => {
    // Handle Three.js modules
    config.module.rules.push({
      test: /\.(glsl|vs|fs|vert|frag)$/,
      exclude: /node_modules/,
      use: ['raw-loader', 'glslify-loader'],
    })

    // Handle canvas for server-side rendering
    if (isServer) {
      config.externals.push('canvas')
    }

    // Optimize bundle splitting
    config.optimization.splitChunks = {
      chunks: 'all',
      cacheGroups: {
        vendor: {
          test: /[\\/]node_modules[\\/]/,
          name: 'vendors',
          priority: 10,
          chunks: 'all',
        },
        three: {
          test: /[\\/]node_modules[\\/](three|@react-three)[\\/]/,
          name: 'three',
          priority: 20,
          chunks: 'all',
        },
        ui: {
          test: /[\\/]node_modules[\\/](@radix-ui|framer-motion|lucide-react)[\\/]/,
          name: 'ui',
          priority: 15,
          chunks: 'all',
        },
      },
    }

    return config
  },

  // Image optimization
  images: {
    domains: ['localhost', '127.0.0.1'],
    formats: ['image/webp', 'image/avif'],
    deviceSizes: [640, 750, 828, 1080, 1200, 1920, 2048, 3840],
    imageSizes: [16, 32, 48, 64, 96, 128, 256, 384],
  },

  // TypeScript configuration
  typescript: {
    ignoreBuildErrors: false,
  },

  // ESLint configuration
  eslint: {
    dirs: ['src'],
    ignoreDuringBuilds: false,
  },

  // Experimental features
  experimental: {
    optimizeCss: true,
    scrollRestoration: true,
  },

  // Performance optimizations
  compress: true,
  poweredByHeader: false,

  // Output configuration
  output: 'standalone',
  trailingSlash: false,

  // Environment variables
  env: {
    CUSTOM_KEY: 'ultimate-media-server-2025',
  },

  // Redirects for legacy URLs
  async redirects() {
    return [
      {
        source: '/home',
        destination: '/',
        permanent: true,
      },
      {
        source: '/dashboard',
        destination: '/',
        permanent: true,
      },
    ]
  },

  // API rewrites for service integration
  async rewrites() {
    return [
      {
        source: '/api/jellyfin/:path*',
        destination: 'http://localhost:8096/api/:path*',
      },
      {
        source: '/api/plex/:path*',
        destination: 'http://localhost:32400/:path*',
      },
      {
        source: '/api/sonarr/:path*',
        destination: 'http://localhost:8989/api/v3/:path*',
      },
      {
        source: '/api/radarr/:path*',
        destination: 'http://localhost:7878/api/v3/:path*',
      },
    ]
  },
}

module.exports = nextConfig