# Repository Guidelines

## Project Structure & Module Organization
This repository is currently a clean starting point. Keep the layout consistent as code is added:
- `src/`: main application and training code.
- `tests/`: unit and integration tests mirroring `src/`.
- `configs/`: experiment and runtime config files (YAML/JSON).
- `scripts/`: reproducible CLI utilities (data prep, training, eval).
- `assets/` or `docs/`: static resources and supporting documentation.

Example: if adding `src/data/loader.py`, place tests in `tests/data/test_loader.py`.

## Build, Test, and Development Commands
Standardize on a small command surface and keep it documented in this file when tooling changes.
- Required for every new shell session in this repo: `conda activate qwen-omni`.
- `conda create -y -n qwen-omni python=3.11`: create the project environment (one-time setup).
- `pip install -r requirements.txt`: install dependencies in `qwen-omni` (once `requirements.txt` exists).
- `pytest -q`: run test suite.
- `ruff check .`: run lint checks.
- `ruff format .`: apply formatting.

If `Makefile` or task runners are introduced, add canonical commands such as `make test` and `make lint`.

## Codex Interaction Protocol
- For all user instructions written in English, Codex must first provide a corrected English version.
- After correction, Codex should execute the request in the same turn unless the user explicitly asks to stop after correction.
- Keep corrections minimal and intent-preserving; do not rewrite beyond what is needed for clarity and grammar.
- Suggested response shape:
  - `Corrected English: <corrected sentence>`
  - `Execution: <actions/results>`

## Coding Style & Naming Conventions
- Python: 4-space indentation, UTF-8, and type hints for public functions.
- Naming: `snake_case` for modules/functions, `PascalCase` for classes, `UPPER_SNAKE_CASE` for constants.
- Prefer small, single-purpose modules; avoid large utility files.
- Keep functions side-effect-light where possible; isolate I/O in boundary layers.

## Testing Guidelines
- Framework: `pytest`.
- Test files: `test_*.py`; test functions: `test_<behavior>()`.
- Mirror source paths under `tests/` for discoverability.
- Include at least one happy-path and one failure-path test for new logic.
- Add regression tests for every bug fix.

## Commit & Pull Request Guidelines
- Use Conventional Commit style: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Keep commits focused and atomic; avoid mixing refactors with behavior changes.
- PRs should include:
  - short summary of intent,
  - linked issue (if available),
  - testing evidence (command output or explanation),
  - migration/config notes if behavior changes.

## Documentation Sync Rule
- After every code, script, or config change, check if `docs/README.md` (or other docs) needs updating.
- If docs are affected, update them immediately in the same turn. Never leave docs out of sync with the codebase.

## Security & Configuration Tips
- Never commit secrets, API keys, or raw credentials.
- Use `.env` files locally and provide `.env.example` with placeholder values.
- Pin critical dependency versions and review updates before merging.
