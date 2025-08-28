// Performance-Optimized Babel Configuration
module.exports = {
  presets: [
    [
      '@babel/preset-env',
      {
        targets: {
          browsers: ['> 1%', 'last 2 versions', 'not ie <= 11']
        },
        modules: false, // Let webpack handle modules
        useBuiltIns: 'usage',
        corejs: 3,
        // Enable optimization features
        loose: true,
        bugfixes: true,
        // Exclude transforms for modern browsers
        exclude: [
          '@babel/plugin-transform-typeof-symbol',
          '@babel/plugin-transform-unicode-regex'
        ]
      }
    ],
    [
      '@babel/preset-react',
      {
        runtime: 'automatic', // Use new JSX transform
        development: process.env.NODE_ENV === 'development'
      }
    ]
  ],
  plugins: [
    // Class properties support
    '@babel/plugin-proposal-class-properties',
    
    // Dynamic imports
    '@babel/plugin-syntax-dynamic-import',
    
    // Optional chaining and nullish coalescing
    '@babel/plugin-proposal-optional-chaining',
    '@babel/plugin-proposal-nullish-coalescing-operator',
    
    // Production optimizations
    ...(process.env.NODE_ENV === 'production' ? [
      ['babel-plugin-transform-react-remove-prop-types', { removeImport: true }],
      'babel-plugin-transform-react-inline-elements',
      'babel-plugin-transform-react-constant-elements'
    ] : []),
    
    // Development enhancements
    ...(process.env.NODE_ENV === 'development' ? [
      'react-refresh/babel'
    ] : [])
  ],
  
  // Environment-specific configurations
  env: {
    test: {
      presets: [
        [
          '@babel/preset-env',
          {
            targets: { node: 'current' },
            modules: 'commonjs'
          }
        ]
      ]
    }
  },
  
  // Caching configuration
  cacheDirectory: true,
  cacheCompression: false,
  compact: process.env.NODE_ENV === 'production'
};