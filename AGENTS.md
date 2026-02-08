## 0) Precedence & Interpretation

* `NOTE:` comments are **authoritative**. Obey them verbatim; do not remove.
* `TODO:` comments are **actionable**. Implement the specific gap and then remove the `TODO:` line.
* Follow existing file/project conventions first. When this doc and code comments conflict, **code comments win**. Your primary responsibility is to **match the current file’s patterns** (naming, logging, returns).

**Core principle:** Challenge every line of code. If a line is not required for correctness, clarity, or measurable performance, **remove it**. Agents tend to over-abstract: resist helper sprawl, avoid “future-proofing,” and keep the API surface minimal.
**Design lens:** Functional core / imperative shell, explicit dependency injection (Typer `ctx.obj` over globals), and separation of concerns (CLI parsing ≠ business logic ≠ I/O).

### UV + pytest conventions
- Keep pytest configuration in `pyproject.toml` under `[tool.pytest.ini_options]`; use `pythonpath = ["."]` instead of sys.path hacks.
- Run with dev dependencies using `uv run --group dev ...`.
- Do not add uv-only fields outside the documented schema; prefer `uv run python -m pytest` for test execution.

## 1) Project Structure

* Use `tree` to get oriented.
* Use appropriate search or package manager CLI for more details.

## 2) Build, Test, and Dev

* Run: `uv run` (if `UV_ENV_FILE` not set and cwd includes `.env`, set it)
* Type check: `pyright`
* Format: `black .`
* Tests: `pytest -q`
* Library usage rules: navigate to definition to study usage then use your python REPL to test and inspect before designing and writing code. 

## 3) Style & Naming

* Python ≥ 3.11. **Strict typing everywhere** (no untyped public functions).
* When you need structures, define idiomatic **Pydantic v2** models (no ad-hoc dicts, no dataclasses for runtime models).
* **Typer** for CLIs (typed arguments; treat commands as public functions, not as wrappers).
* **Loguru** for logging (configure once).
* Paths use `pathlib.Path`. Prefer Unicode output.
* Tools (configured in `pyproject.toml`): Black (100 cols), Pyright (strict).

## 4) Testing

* Unless requested, don't add tests, just spot test the CLI using your REPL.
* Framework: pytest. Tests live in `tests/`, named `test_*.py`.
* Treat type errors as failures: `pyright` must pass locally and in CI.

## 5) Errors & Contracts

* Fail **fast** with clear exceptions. Validate external input at the **edges** (CLI args, files, network).
* Use typed returns; avoid sentinel `None` where an exception or a result type is clearer.

## 6) Performance & Reliability

* Correctness and clarity first; only then optimize with measurements.
* Keep side effects local; make data flow explicit.
* Batch/stream only when it **materially** improves performance—justify in a short comment.

## 7) Terminal UX

* Use `rich` idiomatically for human-friendly console output.
* If prompting is necessary, prefer **Rich**’s prompt (`rich.prompt.Prompt`).

## 8) Do / Don’t

**Do**

* Use canonical library types and utilities.
* In private functions, always use canonical library types for your argument, never bespoke intermediate types.
* In public functions, use discernment, either use `Typer` idioms or canonical library types or bespoke pydantic v2 models depending the the use case.
* When data validation is necessary, never create bespoke helpers until you ruled out using pydantic validators.
* Never configure **Loguru** use default `from loguru import logger` directly.
* Document why non-obvious choices exist (1–3 lines, not essays).
* Only introduce wrappers/helpers when they unlock a concrete benefit (testability, composition, or reuse). State that benefit briefly in the helper’s docstring.
* Prefer functional core / imperative shell: keep side-effect–free logic isolated and push I/O to the edges. Mark side-effect boundaries clearly.
* Use explicit dependency injection for CLIs (options/envvars + `ctx.obj`) instead of implicit globals; keep `ctx.obj` minimal and per-invocation.
* Respect the types you already have—don’t re-wrap `Path` objects or recreate resources unnecessarily; lean on idiomatic library APIs directly.

**Don’t**

* Don’t wrap or duplicate canonical library APIs.
* Don't write bespoke code without first ruling mentally surveying all libraries in project stack to rule out canonical APIs.
* Don't define helpers before your can prove that inlining the code would add significant duplications and more lines of code.
* Don’t ship untyped functions or untyped public interfaces.
* Don’t add dead code, speculative hooks, or “future-proofing.”
* Don’t stash long-lived state in module globals for Typer CLIs; rely on `ctx.obj` and explicit options/envvars instead.
* Don’t wrap canonical library APIs unless composition or test seams demand it; avoid helper sprawl.
