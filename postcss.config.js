// PostCSS Performance Configuration
module.exports = {
  plugins: [
    // Autoprefixer for browser compatibility
    require('autoprefixer')({
      overrideBrowserslist: ['> 1%', 'last 2 versions', 'not ie <= 11'],
      grid: 'autoplace'
    }),
    
    // CSS optimization for production
    ...(process.env.NODE_ENV === 'production' ? [
      require('cssnano')({
        preset: [
          'default',
          {
            // Optimization options
            discardComments: { removeAll: true },
            normalizeUnicode: false,
            discardUnused: { fontFace: false },
            mergeIdents: false,
            reduceIdents: false,
            zindex: false,
            
            // Advanced optimizations
            calc: { precision: 5 },
            colormin: true,
            convertValues: { length: false },
            discardDuplicates: true,
            discardEmpty: true,
            mergeRules: true,
            mergeLonghand: true,
            minifyFontValues: true,
            minifyParams: true,
            minifySelectors: true,
            normalizeCharset: true,
            normalizeDisplayValues: true,
            normalizePositions: true,
            normalizeRepeatStyle: true,
            normalizeString: true,
            normalizeTimingFunctions: true,
            normalizeUrl: true,
            normalizeWhitespace: true,
            orderedValues: true,
            reduceInitial: true,
            reduceTransforms: true,
            svgo: {
              plugins: [
                {
                  name: 'preset-default',
                  params: {
                    overrides: {
                      removeViewBox: false
                    }
                  }
                }
              ]
            },
            uniqueSelectors: true
          }
        ]
      })
    ] : []),
    
    // CSS-in-JS support
    require('postcss-nested'),
    
    // Custom media queries
    require('postcss-custom-media'),
    
    // Import optimization
    require('postcss-import')({
      plugins: [
        require('stylelint')({
          // Performance-focused rules
          rules: {
            'selector-max-id': 0,
            'selector-max-universal': 1,
            'selector-max-type': 3,
            'max-nesting-depth': 4,
            'declaration-no-important': true,
            'shorthand-property-no-redundant-values': true
          }
        })
      ]
    })
  ]
};