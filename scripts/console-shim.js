// Console shim to route console.* calls to the project's structured logger
// This file is safe to require at process entrypoints to centralize logging
try {
  const logger = require('../middleware/logger');

  console.log = (...args) => logger.info(args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
  console.info = (...args) => logger.info(args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
  console.warn = (...args) => logger.warn(args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
  console.error = (...args) => logger.error(args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
  console.debug = (...args) => logger.debug(args.map(a => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
} catch (e) {
  // If logger isn't available for some reason, fall back to native console
}
