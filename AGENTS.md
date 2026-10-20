# AGENTS.md -- rules for any contributor or coding agent working in this repository

## Project
privgate: a public, independent research project that (1) measures how well different
detectors find personal data (PII) in text and (2) implements a fail-closed gateway that screens
text before it may be sent to an external language model.

## Hard rules
1. Original code only. Never copy, recreate or reference code, configs, names, schemas, data or
   internal designs from any private or company project. Use public documentation, open-source
   libraries and textbook methods.
2. Never fabricate results. Every number in docs must come from a script in this repo that was
   actually run. If something was not run, write "NOT MEASURED".
3. Data: only public datasets whose license is recorded in docs/DATA.md, or synthetic data that
   is clearly labelled as synthetic. Never commit real personal data. Never commit downloaded
   datasets (they go in data/raw/, which is gitignored).
4. No network calls to external model APIs, no paid APIs. Local models only, and only when a task
   explicitly says so. Downloads of public datasets/models only when a task says so.
5. Every new behaviour gets pytest tests, including edge cases and failure cases.
6. Before finishing any task run `pytest -q` and `ruff check .`; both must pass. Fix code, not
   tests, unless the test is wrong (explain why).
7. Determinism: anything random takes an explicit seed; never use global random state.
8. Never commit secrets, .env files, virtual environments, datasets or large generated files.
9. Small commits with clear imperative messages; tests in the same commit as their code.
10. Do not start the next milestone on your own.

## Code style
Python >= 3.10, type hints, dataclasses, docstrings explaining the "why". Readable over clever.
Line length 100. Ruff must pass.

## Final report format (always)
1. Files created/modified. 2. Exact commands and full output of: pytest -q, ruff check .,
git log --oneline -8. 3. Problems, assumptions, deviations.
