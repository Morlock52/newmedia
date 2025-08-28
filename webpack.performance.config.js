// Webpack Performance-Optimized Configuration
// Implements advanced bundling strategies for maximum performance

const path = require('path');
const webpack = require('webpack');
const TerserPlugin = require('terser-webpack-plugin');
const CompressionPlugin = require('compression-webpack-plugin');
const BundleAnalyzerPlugin = require('webpack-bundle-analyzer').BundleAnalyzerPlugin;
const WorkboxPlugin = require('workbox-webpack-plugin');
const CssMinimizerPlugin = require('css-minimizer-webpack-plugin');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');
const PreloadWebpackPlugin = require('@vue/preload-webpack-plugin');
const ImageMinimizerPlugin = require('image-minimizer-webpack-plugin');

const isProduction = process.env.NODE_ENV === 'production';
const isDevelopment = !isProduction;

module.exports = {
    mode: isProduction ? 'production' : 'development',
    
    // Entry points with code splitting
    entry: {
        main: './src/index.js',
        performance: './src/performance.js',
        vendor: ['react', 'react-dom'], // Separate vendor bundle
        polyfills: './src/polyfills.js'
    },
    
    output: {
        path: path.resolve(__dirname, 'dist'),
        filename: isProduction 
            ? 'js/[name].[contenthash:8].js'
            : 'js/[name].js',
        chunkFilename: isProduction
            ? 'js/[name].[contenthash:8].chunk.js'
            : 'js/[name].chunk.js',
        assetModuleFilename: 'assets/[name].[contenthash:8][ext]',
        publicPath: '/',
        clean: true,
        pathinfo: false // Reduce bundle info in production
    },
    
    // Optimization configuration
    optimization: {
        minimize: isProduction,
        minimizer: [
            // JavaScript minification
            new TerserPlugin({
                terserOptions: {
                    parse: {
                        ecma: 8
                    },
                    compress: {
                        ecma: 5,
                        warnings: false,
                        comparisons: false,
                        inline: 2,
                        drop_console: isProduction,
                        drop_debugger: isProduction,
                        pure_getters: true,
                        unsafe: true,
                        unsafe_comps: true,
                        unsafe_math: true,
                        unsafe_methods: true,
                        passes: 2
                    },
                    mangle: {
                        safari10: true
                    },
                    output: {
                        ecma: 5,
                        comments: false,
                        ascii_only: true
                    }
                },
                parallel: true,
                extractComments: false
            }),
            
            // CSS minification
            new CssMinimizerPlugin({
                minimizerOptions: {
                    preset: [
                        'default',
                        {
                            discardComments: { removeAll: true },
                            normalizeUnicode: false,
                            minifySelectors: true,
                            reduceIdents: true
                        }
                    ]
                }
            }),

            // Image optimization
            new ImageMinimizerPlugin({
                minimizer: {
                    implementation: ImageMinimizerPlugin.sharpMinify,
                    options: {
                        encodeOptions: {
                            jpeg: { quality: 85, progressive: true },
                            png: { quality: 85, progressive: true },
                            webp: { quality: 85 }
                        }
                    }
                },
                generator: [
                    {
                        type: 'asset',
                        preset: 'webp-custom-name',
                        implementation: ImageMinimizerPlugin.sharpGenerate,
                        options: {
                            encodeOptions: {
                                webp: { quality: 85 }
                            }
                        }
                    }
                ]
            })
        ],
        
        // Advanced code splitting
        splitChunks: {
            chunks: 'all',
            minSize: 20000,
            minRemainingSize: 0,
            minChunks: 1,
            maxAsyncRequests: 30,
            maxInitialRequests: 30,
            enforceSizeThreshold: 50000,
            cacheGroups: {
                // Vendor libraries
                vendor: {
                    test: /[\\/]node_modules[\\/]/,
                    name: 'vendors',
                    priority: 10,
                    chunks: 'all',
                    reuseExistingChunk: true
                },
                
                // React framework
                react: {
                    test: /[\\/]node_modules[\\/](react|react-dom)[\\/]/,
                    name: 'react',
                    priority: 15,
                    chunks: 'all',
                    reuseExistingChunk: true
                },
                
                // Utility libraries
                utils: {
                    test: /[\\/]node_modules[\\/](lodash|axios|moment)[\\/]/,
                    name: 'utils',
                    priority: 12,
                    chunks: 'all',
                    reuseExistingChunk: true
                },
                
                // Common modules
                common: {
                    name: 'common',
                    minChunks: 2,
                    priority: 5,
                    chunks: 'all',
                    reuseExistingChunk: true
                },
                
                // CSS extraction
                styles: {
                    test: /\.css$/,
                    name: 'styles',
                    type: 'css/mini-extract',
                    chunks: 'all',
                    enforce: true
                }
            }
        },
        
        // Runtime chunk for better caching
        runtimeChunk: {
            name: 'runtime'
        },
        
        // Module concatenation (webpack 4+)
        concatenateModules: true,
        
        // Deterministic module IDs for better caching
        moduleIds: 'deterministic',
        chunkIds: 'deterministic'
    },
    
    resolve: {
        extensions: ['.js', '.jsx', '.ts', '.tsx', '.json'],
        alias: {
            '@': path.resolve(__dirname, 'src'),
            '@components': path.resolve(__dirname, 'src/components'),
            '@utils': path.resolve(__dirname, 'src/utils'),
            '@assets': path.resolve(__dirname, 'src/assets')
        },
        // Reduce resolve time
        modules: ['node_modules'],
        symlinks: false,
        cacheWithContext: false
    },
    
    module: {
        rules: [
            // JavaScript/TypeScript
            {
                test: /\.(js|jsx|ts|tsx)$/,
                exclude: /node_modules/,
                use: [
                    {
                        loader: 'babel-loader',
                        options: {
                            presets: [
                                ['@babel/preset-env', {
                                    targets: {
                                        browsers: ['> 1%', 'last 2 versions']
                                    },
                                    modules: false,
                                    useBuiltIns: 'usage',
                                    corejs: 3
                                }],
                                '@babel/preset-react'
                            ],
                            plugins: [
                                '@babel/plugin-proposal-class-properties',
                                '@babel/plugin-syntax-dynamic-import',
                                isProduction && ['babel-plugin-transform-react-remove-prop-types']
                            ].filter(Boolean),
                            cacheDirectory: true,
                            cacheCompression: false,
                            compact: isProduction
                        }
                    }
                ]
            },
            
            // CSS with optimization
            {
                test: /\.css$/,
                use: [
                    isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
                    {
                        loader: 'css-loader',
                        options: {
                            importLoaders: 1,
                            modules: {
                                auto: true,
                                localIdentName: isProduction 
                                    ? '[hash:base64:5]'
                                    : '[name]__[local]--[hash:base64:5]'
                            }
                        }
                    },
                    {
                        loader: 'postcss-loader',
                        options: {
                            postcssOptions: {
                                plugins: [
                                    ['autoprefixer'],
                                    ['cssnano', {
                                        preset: ['default', {
                                            discardComments: { removeAll: true }
                                        }]
                                    }]
                                ]
                            }
                        }
                    }
                ]
            },
            
            // Images with optimization
            {
                test: /\.(png|jpe?g|gif|svg|webp)$/i,
                type: 'asset',
                parser: {
                    dataUrlCondition: {
                        maxSize: 8 * 1024 // 8KB
                    }
                },
                generator: {
                    filename: 'images/[name].[contenthash:8][ext]'
                }
            },
            
            // Fonts
            {
                test: /\.(woff|woff2|eot|ttf|otf)$/i,
                type: 'asset/resource',
                generator: {
                    filename: 'fonts/[name].[contenthash:8][ext]'
                }
            }
        ]
    },
    
    plugins: [
        // HTML generation with optimization
        new HtmlWebpackPlugin({
            template: './public/index.html',
            filename: 'index.html',
            inject: 'body',
            minify: isProduction ? {
                removeComments: true,
                collapseWhitespace: true,
                removeRedundantAttributes: true,
                useShortDoctype: true,
                removeEmptyAttributes: true,
                removeStyleLinkTypeAttributes: true,
                keepClosingSlash: true,
                minifyJS: true,
                minifyCSS: true,
                minifyURLs: true
            } : false,
            scriptLoading: 'defer'
        }),
        
        // Preload important resources
        new PreloadWebpackPlugin({
            rel: 'preload',
            include: 'initial',
            fileBlacklist: [/\.map$/, /hot-update\.js$/]
        }),
        
        // Extract CSS
        isProduction && new MiniCssExtractPlugin({
            filename: 'css/[name].[contenthash:8].css',
            chunkFilename: 'css/[name].[contenthash:8].chunk.css',
            ignoreOrder: true
        }),
        
        // Gzip compression
        isProduction && new CompressionPlugin({
            algorithm: 'gzip',
            test: /\.(js|css|html|svg)$/,
            threshold: 8192,
            minRatio: 0.8,
            compressionOptions: { level: 9 }
        }),
        
        // Brotli compression
        isProduction && new CompressionPlugin({
            filename: '[path][base].br',
            algorithm: 'brotliCompress',
            test: /\.(js|css|html|svg)$/,
            compressionOptions: {
                params: {
                    [require('zlib').constants.BROTLI_PARAM_QUALITY]: 11
                }
            },
            threshold: 8192,
            minRatio: 0.8
        }),
        
        // Service Worker generation
        isProduction && new WorkboxPlugin.GenerateSW({
            clientsClaim: true,
            skipWaiting: true,
            swDest: 'sw.js',
            runtimeCaching: [
                {
                    urlPattern: /^https:\/\/fonts\.googleapis\.com/,
                    handler: 'StaleWhileRevalidate',
                    options: {
                        cacheName: 'google-fonts-stylesheets'
                    }
                },
                {
                    urlPattern: /^https:\/\/fonts\.gstatic\.com/,
                    handler: 'CacheFirst',
                    options: {
                        cacheName: 'google-fonts-webfonts',
                        expiration: {
                            maxEntries: 30,
                            maxAgeSeconds: 60 * 60 * 24 * 365 // 1 year
                        }
                    }
                },
                {
                    urlPattern: /\.(?:png|jpg|jpeg|svg|gif|webp)$/,
                    handler: 'CacheFirst',
                    options: {
                        cacheName: 'images',
                        expiration: {
                            maxEntries: 100,
                            maxAgeSeconds: 60 * 60 * 24 * 30 // 30 days
                        }
                    }
                }
            ]
        }),
        
        // Bundle analyzer (development)
        isDevelopment && process.env.ANALYZE && new BundleAnalyzerPlugin({
            analyzerMode: 'server',
            openAnalyzer: true
        }),
        
        // Define environment variables
        new webpack.DefinePlugin({
            'process.env.NODE_ENV': JSON.stringify(process.env.NODE_ENV),
            'process.env.BUILD_TIME': JSON.stringify(new Date().toISOString()),
            '__DEV__': isDevelopment,
            '__PROD__': isProduction
        }),
        
        // Module federation for micro-frontends (optional)
        new webpack.container.ModuleFederationPlugin({
            name: 'mediaServerDashboard',
            filename: 'remoteEntry.js',
            exposes: {
                './Dashboard': './src/components/Dashboard',
                './PerformanceMonitor': './src/components/PerformanceMonitor'
            },
            shared: {
                react: { singleton: true },
                'react-dom': { singleton: true }
            }
        }),
        
        // Progress plugin for build feedback
        new webpack.ProgressPlugin({
            activeModules: false,
            entries: true,
            modules: false,
            dependencies: false
        })
    ].filter(Boolean),
    
    // Performance budgets
    performance: {
        maxAssetSize: 250000, // 250KB
        maxEntrypointSize: 250000,
        hints: isProduction ? 'warning' : false,
        assetFilter(assetFilename) {
            return !assetFilename.endsWith('.map');
        }
    },
    
    // Cache configuration for faster rebuilds
    cache: {
        type: 'filesystem',
        buildDependencies: {
            config: [__filename]
        },
        cacheDirectory: path.resolve(__dirname, '.webpack-cache'),
        name: isProduction ? 'production' : 'development'
    },
    
    // Development server configuration
    devServer: isDevelopment ? {
        static: {
            directory: path.join(__dirname, 'public')
        },
        compress: true,
        port: 3000,
        hot: true,
        open: true,
        historyApiFallback: true,
        client: {
            overlay: {
                errors: true,
                warnings: false
            }
        },
        devMiddleware: {
            stats: 'minimal'
        }
    } : undefined,
    
    // Source maps
    devtool: isProduction 
        ? 'source-map' 
        : 'eval-cheap-module-source-map',
    
    // Resolve performance
    stats: {
        colors: true,
        hash: false,
        version: false,
        timings: true,
        assets: false,
        chunks: false,
        modules: false,
        reasons: false,
        children: false,
        source: false,
        errors: true,
        errorDetails: true,
        warnings: true,
        publicPath: false
    }
};

// Performance optimization tips
console.log(`
🚀 Webpack Performance Configuration Loaded
📊 Mode: ${isProduction ? 'Production' : 'Development'}
⚡ Key Optimizations:
   - Code splitting with smart chunking strategy
   - Tree shaking and dead code elimination
   - Image optimization with WebP generation
   - CSS extraction and minification
   - Gzip and Brotli compression
   - Service Worker with Workbox
   - Module concatenation and deterministic IDs
   - Performance budgets enforced

💡 Build Tips:
   - Run 'npm run analyze' to see bundle composition
   - Monitor performance budgets during builds
   - Use source maps for debugging
   - Enable caching for faster rebuilds
`);