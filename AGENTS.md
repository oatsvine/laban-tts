## 0) Precedence & Interpretation

* `NOTE:` comments are **authoritative**. Obey them verbatim; do not remove.
* `TODO:` comments are **authoritative and actionable**. They are authored by the user. Implement the specific gap and then remove the `TODO:` line. Do not create new `TODO:` lines as a substitute for implementation.
* Follow existing file/project conventions first. When this doc and code comments conflict, **code comments win**. Your primary responsibility is to **match the current file’s patterns** (naming, logging, returns).
* Canonical user code takes priority over docs; disambiguate your commits ("AI:" prefix) from user commits, anchor implementation cues in canonical user commits code.
* `ARCHITECTURE.md` defines intended invariants; apply only when consistent with canonical code.

## Implementation constraints

- Bespoke code minimalism: Challenge every line of code. If a line is not required for correctness, clarity, or measurable performance, **remove it**. Agents tend to over-abstract: resist helper sprawl, avoid “future-proofing,” and keep the API surface minimal.
- Idiomatic-first: use the language/framework/library facilities directly.
- Search-first: before creating any new helper, search for prior art and reuse or extract.
- Helper skepticism: Assume every helper is illegal until **proven** to uphold the constraints in this document and the canonical code.
- No thin wrappers: do not wrap a library call unless you enforce a domain invariant or combine a multi-step protocol used in ≥2 places.
- No speculative generality: no config knobs, factories, or “flexible” abstractions unless explicitly requested.
- Design lens: Functional core / imperative shell, and separation of concerns (CLI parsing ≠ business logic ≠ I/O).

## Anti Code Smells Rules (Canonical)

### Hard bans

* Never use `cast(...)`.
* Never use `getattr(...)`, `setattr(...)`, or `hasattr(...)`.
* Never use `# type: ignore`; use `# pyright: ignore[rule]` only under approved exception protocol.

### Compatibility Exception (approval-required)

* Applicable only to vendor libraries already in the stack (if any)
* Exception-only area: isolate interop in one narrowly scoped adapter function/module so type compromises do not spread.
* Docstring marker required on that isolated function:
  - `COMPAT(<library>, <version>): <constraint>.`
  - Ask user to review and approve
* Ignore rule notation allowed only inside approved exception blocks:
  - `# pyright: ignore[reportArgumentType]`
  - The scope of compat functions is THE ONLY EXCEPTION to ignore ban.

### Search-first error workflow (mandatory)

**Search-first means always proactively find the canonical syntax pattern**

1. Reproduce the error locally (`pyright`, CLI command, or REPL snippet).
2. Navigate to library source and type signatures first (installed package source, stubs, and official docs).
3. Use the library idiomatically in user code without dynamic bypasses.
4. If blocked by third-party typing defects, propose a scoped exception to the user before writing ignores.

### Enforcement stack

* Baseline: `pyright`.
* Required stricter companion: `basedpyright --project basedpyrightconfig.json` with `reportInvalidCast`, `reportAny`, `reportExplicitAny`, and `reportIgnoreCommentWithoutRule`.
* Required smell lint: `ruff` with flake8-bugbear rules `B009`/`B010` to catch dynamic attribute access patterns (`getattr`/`setattr` with constant attribute names).

## Repository expectations

* Use `tree` to get oriented.
* Use appropriate search or package manager CLI for more details.
* Explore `references/` for applicable curated corpus. 

### UV + pytest conventions
- Check `printenv UV_SYSTEM_PYTHON`, if set, you are in the deterministic container, assume all dependencies and editable project preinstalled.
- Keep pytest configuration in `pyproject.toml` using canonical "src/" layout.
- Do not add uv-only fields outside the documented schema; prefer `python -m` for execution the container.

## Build, Test, and Dev

* Run: `python -m laban_tts.cli --help` (explore subcommands and their usage as required) 
* Type check: `pyright`
* Companion type check: `basedpyright --project basedpyrightconfig.json`
* Smell lint: `ruff check .`
* Format: `black .`
* Library usage rules: navigate to definition to study usage then use your python REPL to test and inspect before designing and writing code. 

## Style & Naming

* Python ≥ 3.12. **Strict typing everywhere** (no untyped public functions).
* Project typing convention: use `typing` generics (`List`, `Dict`, `Tuple`, `Set`, `Optional`, etc.), not builtin generic forms (`list[...]`, `dict[...]`, ...).
* When you need structures, define idiomatic **Pydantic v2** models (no ad-hoc dicts, no dataclasses for runtime models).
* **Typer** for CLIs (typed arguments; treat commands as public functions, not as wrappers).
* **Loguru** for logging (configure once).
* Paths use `pathlib.Path`. Prefer Unicode output.
* Tools (configured in `pyproject.toml`): Black (100 cols), Pyright (strict).
