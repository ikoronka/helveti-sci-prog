# Developer Guidelines

Welcome to the **Zurich Commuting and Delay Analyzer** project.

These guidelines are written for students and first-time contributors. If you follow this page step by step, you will be productive quickly and avoid common setup and workflow problems.

## 1) Project Setup (uv Workflow)

### Clone and set up

```bash
git clone <your-repo-url>
cd helveti-sci-prog
uv sync
```

> [!IMPORTANT]
> **Do not use `pip` or `conda` in this project.**
> Everything (dependencies and environment management) is handled by `uv`.

### Launch the interactive notebook app

```bash
uv run marimo edit app.py
```

This opens the reactive notebook interface in your browser.

## 2) Working with Marimo

- `marimo` notebooks in this project are regular Python files (`.py`), not `.ipynb` notebooks.
- Save frequently in the marimo UI; your changes are written back to `app.py` automatically.

> [!IMPORTANT]
> Never rename `app.py` to `.ipynb`.
> This project is designed around marimo's reactive Python-file workflow.

## 3) Version Control (Simplified Conventional Commits)

Use commit messages in this format:

`type: short description`

### Allowed types (for now)

- `feat` → new behavior or functionality
- `fix` → bug fix or correctness improvement
- `docs` → documentation updates

### Examples

- `feat: add delay calculation for Winterthur routes`
- `fix: handle null values in API response`
- `docs: update setup instructions`

Why we do this:

- Keeps project history readable.
- Makes changes easier to review and understand.
- Builds professional habits used in real teams.

## 4) GitHub Issues & Daily Workflow

Before writing code:

1. Pick an existing issue, or create a new one.
2. Assign yourself to that issue.
3. Start implementation.

> [!TIP]
> Self-assignment prevents duplicate work and makes team coordination much easier.

## 5) Pull Requests (PRs) & Review Policy

### Branch and PR flow

1. Create a feature branch from `main`:

   ```bash
   git checkout -b feat/your-feature-name
   ```

2. Commit your changes with a Conventional Commit message.
3. Push your branch.
4. Open a Pull Request on GitHub.

### Project-specific merge policy

> [!IMPORTANT]
> For this project’s current scale, we use a **0 Reviewer policy**.
> After opening your PR and verifying your code runs correctly, you may merge your own PR.

Before merging, always check:

- The marimo app still launches with `uv run marimo edit app.py`.
- Your changes do not break existing analysis workflow.
- Your commit message and PR title are clear.

## 6) Quick Contributor Checklist

- [ ] I used `uv sync` (not pip/conda).
- [ ] I tested by running `uv run marimo edit app.py`.
- [ ] I linked my work to an issue and assigned myself.
- [ ] I used a Conventional Commit message.
- [ ] I confirmed the app still works before merging.

Thanks for contributing — clean process and clear communication are as valuable as code quality.
