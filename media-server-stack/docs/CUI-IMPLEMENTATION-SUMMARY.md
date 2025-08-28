# CUI (CLI) Implementation Summary

The CLI tool (`setup-env.js`) provides an interactive way to configure the environment.

![CLI Setup](images/cli-setup.png)

## Capabilities
- Interactive prompts with validation (domain, email, timezone, etc.)
- Defaults from the system for PUID/PGID/TZ
- Non-interactive, validate-only, and dry-run modes
- Secure generation of secrets; backups created automatically

## Usage
```bash
node media-server-stack/setup-env.js                 # Interactive setup
node media-server-stack/setup-env.js --no-interactive # Use defaults for missing vars
node media-server-stack/setup-env.js --validate-only  # Validate existing .env
node media-server-stack/setup-env.js --dry-run        # Show changes only
```

## Output
- `.env` in the stack directory with only necessary variables
- `.env.backup.<timestamp>` created automatically

![CLI Validation](images/cli-validation.png)

