# AGENTS.md

## Scope
- This file applies to the entire repository.
- If a subdirectory later adds its own `AGENTS.md`, the deeper file takes precedence for that subtree.

## Project Overview
- FlexRAG is a Python 3.11+ RAG framework with a modular architecture under `src/flexrag`.
- The package is built with Hatchling.
- The current public surface is still centered on the existing entrypoints in `src/flexrag/entrypoints`.
- The project is in an active transition toward FlexRAG `1.0.0`; treat several core systems as legacy-but-active rather than removable.
- Project planning and milestone direction belong in `ROADMAP.md`; this file focuses on implementation guidance for contributors and coding agents.

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
- Use Ruff for formatting and import sorting; keep changes compatible with the repository's Ruff configuration in `pyproject.toml`.
- Follow existing typing style and dataclass-based configuration patterns.
- Keep changes minimal and local; preserve naming and structure already used by the surrounding module.
- Avoid introducing new framework layers unless the task clearly requires them.

## Docstring Conventions
- Use reStructuredText-style docstrings for public documentation.
- Use Python annotations as the source of truth for types. Do not add `:type:` fields in new or updated docstrings.
- Public methods on base classes, protocols, and abstract interfaces must provide complete docstrings, including a concise behavior description, parameters, return values, and important side effects.
- Public methods on subclasses may rely on the base method documentation when they only implement the inherited contract. Add or update docstrings when the subclass introduces extra parameters, behavior differences, return-shape differences, side effects, cache/download behavior, or error-handling differences.
- Private methods have no mandatory docstring requirement. Add a concise one-line docstring when the helper's behavior is non-obvious or domain-specific.
- Benchmark, dataset, task, and metric integrations should document their source when available, including at least one relevant link such as the paper, official repository, Hugging Face dataset, download source, official metric, or evaluation protocol.
- Configuration class docstrings should be prioritized and kept complete, especially for user-facing options, defaults, available choices, download/cache behavior, and externally visible effects.

## Configuration And Registry Conventions
- Configuration classes generally use `@configure` from `src/flexrag/common/configure.py`.
- `@configure` produces pydantic dataclasses with `extra="forbid"`; do not rely on undeclared config fields.
- When a field needs constrained string choices, prefer `Choices(...)` over `Literal` for Hydra compatibility.
- Modular components are commonly registered through `Register`; preserve the registry pattern instead of hardcoding implementations.
- Hydra entrypoints typically define a `Config`, register it with `ConfigStore`, then call `extract_config(...)` inside `main`.

## Entrypoint Conventions
- Existing entrypoints in `src/flexrag/entrypoints` remain the active interface until the unified entrypoint work lands.
- Before `1.0.0`, backward compatibility is not a hard requirement for Hydra-driven CLI behavior. Preserve it when it is cheap, but prefer cleaner and more consistent interfaces when they materially improve the project.
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
- Avoid fossilization tests: do not add tests whose only purpose is to assert that an internal field, legacy option, helper class, or implementation detail no longer exists. Prefer smoke tests and behavior-level assertions through the public or intended interface. Add absence or negative-surface tests only when the absence itself is an explicit public contract, security boundary, lifecycle boundary, or compatibility guarantee, and keep the assertion narrowly tied to that contract.

## Pre-1.0 Compatibility Policy
- Before the `1.0.0` release, backward compatibility is not a project constraint by itself.
- If a breaking change materially improves logic consistency, performance, maintainability, or API ergonomics, prefer the cleaner design.
- Do not preserve awkward legacy behavior solely to avoid breaking pre-`1.0.0` users of the library, CLI, or configs.

## Transitional Constraints
- Treat the current Hydra + dataclass configuration system as legacy-but-active. Keep it working unless migration work is explicitly requested.
- Treat the current path-import + decorator-registration mechanism as legacy-but-active. Do not remove it casually, but pre-`1.0.0` breaking changes are acceptable when they clearly improve the design.
- Assistant-related code is in transition. Prefer task-oriented abstractions for new work, but preserve current user-facing behavior.
- Legacy code may be intentionally retained during the refactor. Confirm it is actually obsolete before deleting or bypassing it.
- When adding new code, prefer abstractions that can survive the ongoing `1.0.0` transition work around task-centric evaluation, config redesign, plugin-system migration, and isolated-process execution for resource-intensive components.
- For isolated-process implementations, prefer the established `AsyncClientMixin` + `LocalProcess*Base` + `ProcessWorkerPoolClient` patterns over ad-hoc threading or multiprocessing glue. Use the model-side implementation as the current reference when extending similar behavior to retrievers or rankers.
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
- If a request conflicts with the transition constraints in this file, implement the smallest compatible change and note the tradeoff.
