# Contributing

## Markdown formatting and linting
This project standardizes Markdown formatting using Prettier and markdownlint via pre-commit hooks.

Setup locally:
```bash
pip install pre-commit  # or: brew install pre-commit
pre-commit install
pre-commit run --all-files
```

What runs:
- Trailing whitespace and end-of-file fixes
- Prettier on `.md` and `.yml` files
- markdownlint with rules configured in `.markdownlint.json`

CI:
- `.github/workflows/markdown-quality.yml` runs the same checks on every push and pull request.

## Publishing the static site
GitHub Pages deployment is configured in `.github/workflows/deploy-pages.yml` to publish the repository root.

To enable:
1. Push to GitHub
2. Settings -> Pages -> Build and deployment: Source = GitHub Actions
3. Push to `main` or `master` to deploy
