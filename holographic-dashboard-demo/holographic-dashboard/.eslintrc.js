module.exports = {
    env: {
        browser: true,
        es2021: true,
        node: true
    },
    extends: [
        'eslint:recommended'
    ],
    parserOptions: {
        ecmaVersion: 2021,
        sourceType: 'module'
    },
    globals: {
        THREE: 'readonly',
        gsap: 'readonly',
        CONFIG: 'writable'
    },
    rules: {
        // Code style
        'indent': ['error', 4],
        'quotes': ['error', 'single'],
        'semi': ['error', 'always'],
        'no-trailing-spaces': 'error',
        'eol-last': 'error',
        
        // Best practices
        'no-console': 'warn',
        'no-debugger': 'error',
        'no-alert': 'warn',
        'no-eval': 'error',
        'no-implied-eval': 'error',
        'no-new-func': 'error',
        
        // Variables
        'no-unused-vars': 'warn',
        'no-undef': 'error',
        'no-redeclare': 'error',
        
        // ES6+
        'prefer-const': 'error',
        'no-var': 'error',
        'prefer-arrow-callback': 'warn',
        'arrow-spacing': 'error',
        
        // WebGL specific
        'no-magic-numbers': ['warn', { 
            ignore: [0, 1, 2, 3, 4, 5, -1],
            ignoreArrayIndexes: true
        }]
    },
    overrides: [
        {
            files: ['demo-server.js'],
            env: {
                node: true,
                browser: false
            }
        },
        {
            files: ['js/shaders.js'],
            rules: {
                'quotes': 'off', // GLSL strings use various quote styles
                'no-magic-numbers': 'off' // Shaders use many numeric constants
            }
        }
    ]
};