# AGENTS.md

## Scope
- This file applies to the entire repository.
- If a subdirectory later adds its own `AGENTS.md`, the deeper file takes precedence for that subtree.

## Project Overview
- FlexRAG is a Python 3.11+ RAG framework with a modular architecture under `src/flexrag`.
- The package is built with Hatchling.
- The current public surface is still centered on the existing entrypoints in `src/flexrag/entrypoints`.
- The project is in an active transition toward FlexRAG `1.0.0`; treat several core systems as legacy-but-active rather than removable.

## Repository Layout
- `src/flexrag/common`: shared config, registry, logging, dataclasses, defaults.
- `src/flexrag/assistants`: user-facing assistant implementations.
- `src/flexrag/retrievers`: retrievers, indices, and web retrieval components.
- `src/flexrag/models`: generators, encoders, scorers, tokenizer, HF/LiteLLM integrations, and shared async client plus process-backed local runtime helpers.
- `src/flexrag/processors`: chunkers, parsers, refiners, rankers, text processors.
- `src/flexrag/datasets`: corpora, readers, benchmark datasets.
- `src/flexrag/tasks`: task abstractions and task-specific evaluation logic.
- `src/flexrag/metrics`: retrieval, generation, and judge-style metrics.
- `src/flexrag/entrypoints`: current Hydra-based executable flows such as corpus prep, evaluation, retriever prep, and interactive demo.
- `tests`: pytest-based coverage for assistants, retrievers, datasets, cache, chunkers, models, and benchmarks.
- `docs`: Sphinx documentation sources.

## Development Workflow
- Keep the development environment in sync with the project's declared dependencies and optional extras needed for the task.
- Run tests with `pytest -m "not gpu"`.
- Run a narrower test target when touching a focused area, then fall back to the full non-GPU suite if practical.
- Build docs with `sphinx-build docs/source docs/build/en -D language=en -W` or `.../zh_CN ...`.
- Build the package with `python -m build` if packaging changes are involved.

## Code Style
- Use Black-compatible formatting and keep imports isort-friendly.
- Follow existing typing style and dataclass-based configuration patterns.
- Keep changes minimal and local; preserve naming and structure already used by the surrounding module.
- Avoid introducing new framework layers unless the task clearly requires them.

## Configuration And Registry Conventions
- Configuration classes generally use `@configure` from `src/flexrag/common/configure.py`.
- `@configure` produces pydantic dataclasses with `extra="forbid"`; do not rely on undeclared config fields.
- When a field needs constrained string choices, prefer `Choices(...)` over `Literal` for Hydra compatibility.
- Modular components are commonly registered through `Register`; preserve the registry pattern instead of hardcoding implementations.
- Hydra entrypoints typically define a `Config`, register it with `ConfigStore`, then call `extract_config(...)` inside `main`.

## Entrypoint Conventions
- Existing entrypoints in `src/flexrag/entrypoints` remain the active interface until the unified entrypoint work lands.
- Keep Hydra-driven CLI behavior backward compatible unless the task explicitly changes the interface.
- `run_interactive.py` can load user modules via `user_module=...`; treat that path-loading flow as trusted-input-only.

## Testing Expectations
- Add or update pytest coverage when behavior changes.
- Prefer focused tests near the changed subsystem:
  - assistant changes: `tests/test_assistant.py`
  - retriever changes: `tests/test_retriever.py`, `tests/test_ranker.py`, retriever benchmark tests
  - model changes: `tests/test_model.py`, `tests/test_hf_generator.py`, `tests/test_litellm_model.py`, `tests/test_sentence_transformer_encoder.py`, `tests/test_scorer.py`, and relevant local-process coverage such as `tests/test_local_process_encoder.py`, `tests/test_local_process_generator.py`, `tests/test_local_process_scorer.py`, or `tests/test_process_worker.py`
  - chunking/database changes: matching unit tests under `tests/`
- If a change touches docs, configs, or entrypoints, consider whether a smoke test or doc build is needed.
- GPU-only behavior should remain guarded by the existing `gpu` marker.

## Active Refactor Context
- FlexRAG is moving toward a `1.0.0` architecture.
- Planned direction includes:
  - Process-backed handling for resource-intensive local components, especially in `src/flexrag/models`, with clear worker/runtime boundaries for resource management and future backend flexibility.
  - A `Dataset` + `Task` centered evaluation design.
  - Assistant code adapting to task-oriented evaluation workflows.
  - Replacement of multiple current entrypoints with a unified entrypoint later.
  - Eventual replacement of the current Hydra + dataclass config model.
  - Migration from path-import + decorator registration to a Pluggy-based plugin system.
  - Documentation, packaging, and CI/CD updates as part of the transition.

## Transitional Constraints
- Treat the current Hydra + dataclass configuration system as legacy-but-active. Keep it working unless migration work is explicitly requested.
- Treat the current path-import + decorator-registration mechanism as legacy-but-active. Do not remove it without a compatibility plan.
- Assistant-related code is in transition. Prefer task-oriented abstractions for new work, but preserve current user-facing behavior.
- Legacy code may be intentionally retained during the refactor. Confirm it is actually obsolete before deleting or bypassing it.
- When adding new code, prefer abstractions that can survive the ongoing process-backed runtime migration, task-centric evaluation design, config redesign, and plugin-system migration.
- For model work, prefer the established `AsyncClientMixin` + `LocalProcess*Base` + `ProcessWorkerPoolClient` patterns over ad-hoc threading or multiprocessing glue.
- Avoid deepening coupling to current entrypoints, current config internals, or ad-hoc plugin loading unless backward compatibility requires it.

## External Integrations And Secrets
- Model providers read credentials from environment variables such as `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.
- Do not hardcode secrets, log secret values, or add fixtures that require real external credentials.
- Web retrievers and remote model integrations should degrade cleanly in tests through mocks or opt-in execution.

## Change Guidance For Codex
- Before larger edits, inspect nearby modules for the existing pattern and follow it.
- Prefer incremental changes over broad rewrites.
- Do not remove transitional systems solely because a newer direction exists in project plans.
- When introducing new config or registry entries, wire them in the same style as adjacent implementations.
- When adding a local model backend, keep the registered public wrapper separate from the concrete `*Impl` worker implementation, and preserve current device remapping / worker-pool behavior unless the task explicitly changes scheduling semantics.
- If a request conflicts with the active refactor constraints, implement the smallest compatible change and note the tradeoff.
