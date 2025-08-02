module.exports = {
  testEnvironment: 'node',
  testMatch: ['**/integration/**/*.test.js', '**/integration/**/*.test.ts'],
  setupFilesAfterEnv: ['./integration/setup.js'],
  testTimeout: 60000,
  coverageDirectory: './reports/coverage',
  collectCoverageFrom: [
    'integration/**/*.{js,ts}',
    '!integration/mocks/**',
    '!integration/setup.js'
  ],
  reporters: [
    'default',
    ['jest-html-reporter', {
      pageTitle: 'Integration Test Report',
      outputPath: './reports/integration-test-report.html',
      includeFailureMsg: true,
      includeConsoleLog: true
    }]
  ],
  globalSetup: './integration/globalSetup.js',
  globalTeardown: './integration/globalTeardown.js',
  verbose: true
};