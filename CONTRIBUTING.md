# Contributing to Telecom Anomaly Detection

Thank you for your interest in contributing! This is a portfolio project, but suggestions and improvements are welcome.

## How to Contribute

### Reporting Issues

If you find a bug or have a suggestion:

1. **Check existing issues** to avoid duplicates
2. **Open a new issue** with a clear title and description
3. **Include**: steps to reproduce, expected vs. actual behavior, environment details

### Pull Requests

PRs are welcome for bug fixes, documentation, test coverage, and code quality.

**Before submitting a PR:**

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Run tests: `uv run pytest tests/ -v`
4. Run linter: `uv run ruff check src/ tests/`
5. Commit with clear messages
6. Push and create PR

## Development Setup

```bash
git clone https://github.com/adityonugrohoid/telecom-anomaly-detection.git
cd telecom-anomaly-detection
uv sync --all-extras
```

## Code Style

This project uses **Ruff** for linting and formatting:

```bash
uv run ruff format src/ tests/
uv run ruff check src/ tests/
```

## Testing

```bash
uv run pytest tests/ -v
uv run pytest tests/ --cov=anomaly_detection
```

## Project Philosophy

1. **Domain expertise over code complexity**
2. **Clarity over cleverness**
3. **Practical over perfect**
4. **Minimal but complete**

---

Thank you for contributing!
