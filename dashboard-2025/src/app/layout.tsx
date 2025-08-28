import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Ultimate Media Server 2025',
  description: 'Advanced 3D media server dashboard with immersive UI and comprehensive service management',
  keywords: ['media server', 'jellyfin', 'plex', 'sonarr', 'radarr', '3D dashboard', 'automation'],
  authors: [{ name: 'Ultimate Media Server Team' }],
  manifest: '/manifest.json',
  metadataBase: new URL('http://localhost:3002'),
  openGraph: {
    title: 'Ultimate Media Server 2025',
    description: 'Advanced 3D media server dashboard with immersive UI',
    type: 'website',
    url: '/',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Ultimate Media Server 2025',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Ultimate Media Server 2025',
    description: 'Advanced 3D media server dashboard with immersive UI',
    images: ['/og-image.png'],
  },
  robots: {
    index: false,
    follow: false,
  },
}

export const viewport = {
  width: 'device-width',
  initialScale: 1,
  themeColor: '#8B5CF6',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
        <meta name="mobile-web-app-capable" content="yes" />
      </head>
      <body className={`${inter.className} antialiased`}>
        <div id="root">
          {children}
        </div>
        
        {/* Service Worker Registration */}
        <script dangerouslySetInnerHTML={{
          __html: `
            if ('serviceWorker' in navigator) {
              window.addEventListener('load', function() {
                navigator.serviceWorker.register('/sw.js')
                  .then(function(registration) {
                    console.log('SW registered: ', registration);
                  })
                  .catch(function(registrationError) {
                    console.log('SW registration failed: ', registrationError);
                  });
              });
            }
          `
        }} />
        
        {/* WebGL Detection */}
        <script dangerouslySetInnerHTML={{
          __html: `
            (function() {
              const canvas = document.createElement('canvas');
              const gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
              if (!gl) {
                console.warn('WebGL not supported, falling back to 2D mode');
                document.body.classList.add('no-webgl');
              }
            })();
          `
        }} />
      </body>
    </html>
  )
}