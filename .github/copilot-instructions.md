# Project Guidelines

## Code Style
- Format with Black and isort (project badges) [README.md](README.md#L5-L7).
- Configs use pydantic dataclasses via `configure`/`data` helpers (extra fields are forbidden) [src/flexrag/common/configure.py](src/flexrag/common/configure.py#L167-L173).

## Architecture
- Modular components are registered via `Register` in [src/flexrag/common/configure.py](src/flexrag/common/configure.py#L179).
- Entrypoints use Hydra `ConfigStore` + `@hydra.main` to assemble pipelines (see [src/flexrag/entrypoints/prepare_corpus.py](src/flexrag/entrypoints/prepare_corpus.py#L27-L92)).
- Interactive UI is a Gradio app that loads assistants from the registry in [src/flexrag/entrypoints/run_interactive.py](src/flexrag/entrypoints/run_interactive.py#L1-L120).

## Build and Test
- Requires Python >=3.11 [pyproject.toml](pyproject.toml#L5-L12).
- Build backend is Hatchling [pyproject.toml](pyproject.toml#L1-L3) (use `python -m build` or `hatch build`).
- Tests use pytest (+asyncio/mocks) [pyproject.toml](pyproject.toml#L83-L99); GPU marker exists [pyproject.toml](pyproject.toml#L116-L118). Commands: `pytest`, `pytest -m "not gpu"`.

## Project Conventions
- Constrained config fields use `Choices(...)` (Hydra-friendly alternative to `Literal`) [src/flexrag/common/configure.py](src/flexrag/common/configure.py#L115-L122).
- Config classes are registered in Hydra `ConfigStore` under `default` (example [src/flexrag/entrypoints/prepare_corpus.py](src/flexrag/entrypoints/prepare_corpus.py#L42-L43)).
- Environment-based defaults: cache/user module path in [src/flexrag/common/default_vars.py](src/flexrag/common/default_vars.py#L8-L11), log level in [src/flexrag/common/logging.py](src/flexrag/common/logging.py#L106-L114).

## Integration Points
- External LLM APIs read env keys in generator configs (OpenAI `OPENAI_API_KEY`, Anthropic `ANTHROPIC_API_KEY`) [src/flexrag/models/generators/openai_generator.py](src/flexrag/models/generators/openai_generator.py#L22-L49), [src/flexrag/models/generators/anthropic_generator.py](src/flexrag/models/generators/anthropic_generator.py#L13-L33).
- Gradio UI entrypoint for interactive RAG is [src/flexrag/entrypoints/run_interactive.py](src/flexrag/entrypoints/run_interactive.py#L37-L120).

## Security
- `run_interactive` loads user modules via `user_module=...` CLI arg; only use trusted paths [src/flexrag/entrypoints/run_interactive.py](src/flexrag/entrypoints/run_interactive.py#L15-L19).
- API keys are sourced from env vars in generator configs; avoid logging secrets [src/flexrag/models/generators/openai_generator.py](src/flexrag/models/generators/openai_generator.py#L44-L48), [src/flexrag/models/generators/anthropic_generator.py](src/flexrag/models/generators/anthropic_generator.py#L30-L33).

## Active Refactor Context (FlexRAG 1.0.0)
- FlexRAG is in a long-running refactor toward version 1.0.0.
- Main refactor goals:
  - Use Ray to manage resource-intensive components and improve local resource isolation and scheduling.
  - Rebuild the evaluation system around `Dataset` + `Task` so most datasets expose tasks consistently and are easier to evaluate.
  - Update Assistant-related code to support Task-oriented evaluation workflows, since Assistant is the main user-facing interface for evaluation.
  - After the evaluation refactor stabilizes, rewrite the code in `src/flexrag/entrypoints/` and replace the current multiple entrypoints with a unified entrypoint.
  - Improve configuration management. The current Hydra + dataclass approach is still active, but it is expected to be replaced by a more flexible design in the future; the exact replacement has not been finalized yet.
  - Replace the current user-code loading mechanism (path import + decorator registration) with a plugin system based on Pluggy.
  - Update documentation, packaging, and CI/CD as part of the 1.0.0 transition.

## Transitional Constraints
- Treat the current Hydra + dataclass configuration system as legacy-but-active: keep it working unless the task explicitly includes migration work.
- Treat the current path-import + decorator-registration plugin mechanism as legacy-but-active: do not remove it without a compatibility or migration plan.
- Existing code in `src/flexrag/entrypoints/` remains the active interface until the unified entrypoint is implemented.
- Assistant-related code is in transition: prefer Task-oriented abstractions for new work, but preserve current user-facing behavior during the migration.
- Legacy code may remain intentionally during the refactor. Verify whether an old path is still part of the active transition plan before removing it.
- When adding new code, prefer abstractions that can survive the Ray migration, Task-centric evaluation design, future config changes, and the planned Pluggy-based plugin system.
- Avoid deepening coupling to legacy entrypoints, current config internals, or ad-hoc plugin loading unless it is necessary for backward compatibility.
- If legacy code must remain temporarily, isolate it clearly and avoid using it as the foundation for new features.

## Known Legacy Areas
- Hydra `ConfigStore`-based configuration and the current `configure` / `data` helpers are part of the active legacy system, not necessarily the target end state.
- The current entrypoint layout is transitional and should not be treated as the final architecture.
- The current plugin loading approach based on path import and decorator registration is transitional.
- Some Assistant interfaces may still reflect pre-Task evaluation assumptions and may require adaptation during the refactor.
- Parts of the evaluation and task stack are still being reshaped; avoid baking new features into assumptions that only hold for the pre-Task design.
