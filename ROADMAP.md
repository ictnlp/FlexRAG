# FlexRAG Roadmap

This document tracks the shared product and architecture direction of FlexRAG.
It is intentionally versioned in git because it describes project-level plans,
not personal notes. The roadmap is a planning document rather than a release
contract.

## Snapshot

FlexRAG is in an active transition toward `1.0.0`. The current codebase remains
usable and maintained, but several core systems are still being reshaped for a
cleaner long-term architecture.

| Area | Status | Notes |
| --- | --- | --- |
| Model isolated-process runtime | Completed | The model-side process-backed runtime is already in place and acts as the current reference implementation. |
| Retriever and ranker isolated-process runtime | Next | This is the clearest next implementation target after completing the model-side runtime. |
| Dataset + Task-centered evaluation | In Progress | Evaluation is still moving toward shared dataset/task abstractions. |
| Assistant adaptation to task-oriented flows | In Progress | Assistant code is still converging with the task-centered evaluation direction. |
| Unified entrypoint | Planned | Existing Hydra entrypoints remain the active interface for now. |
| Configuration system evolution | Planned | The current Hydra + dataclass config system remains active during the transition. |
| Plugin and extension system | Planned | The current path-import + decorator registration path remains active for now. |
| Documentation, packaging, and CI | In Progress | Transition-related support work is still ongoing. |

## Guiding Principles

- Prefer clearer design over preserving awkward pre-`1.0.0` behavior.
- Keep transitional systems working until replacements are ready.
- Make changes incremental, testable, and easy to review.
- Preserve the modular structure that makes FlexRAG extensible.

## Completed Foundations

- Model-side isolated-process execution is already complete.
- Existing model-side worker/runtime/client boundaries should be treated as the
  current reference when similar capabilities are extended elsewhere.
- Existing Hydra entrypoints, configuration, and registration systems remain
  usable while the larger `1.0.0` transition continues.

## Current Focus

### 1. Isolated Runtime Expansion

Status: `Next`

Done:
- Model-side isolated-process runtime is complete.

Next:
- Extend isolated-process execution to rankers.
- Extend isolated-process execution to retrievers.
- Reuse the existing worker/runtime/client split instead of adding new ad-hoc
  multiprocessing paths.
- Add focused tests around lifecycle, resource cleanup, and behavior parity for
  process-backed retrievers and rankers.

## Active Tracks

### 2. Dataset And Task-Centered Evaluation

Status: `In Progress`

Current direction:
- Continue moving evaluation flows toward `Dataset` + `Task` abstractions.
- Unify benchmark behavior and task-specific evaluation logic behind cleaner
  interfaces.
- Improve reproducibility and regression testing for end-to-end RAG workflows.

### 3. Assistant Architecture

Status: `In Progress`

Current direction:
- Adapt assistants to work more naturally with task-oriented evaluation flows.
- Preserve current user-facing assistant behavior where practical during the
  transition.
- Simplify the path from modular components to complete assistants.

### 4. Documentation, Packaging, And Release Engineering

Status: `In Progress`

Current direction:
- Expand user and developer documentation alongside architectural changes.
- Keep packaging and optional dependencies predictable across platforms.
- Improve CI and test coverage around transition-sensitive non-GPU paths.

## Planned Tracks

### 5. Unified Entrypoint Experience

Status: `Planned`

Planned direction:
- Gradually replace multiple current entrypoints with a more coherent unified
  interface.
- Keep the existing Hydra-driven entrypoints available until the replacement is
  ready.
- Improve discoverability for common workflows such as corpus preparation,
  retriever preparation, evaluation, and interactive use.

### 6. Configuration System Evolution

Status: `Planned`

Planned direction:
- Keep the current Hydra + dataclass configuration system working during the
  transition.
- Reduce sharp edges in configuration ergonomics where it can be done locally.
- Move toward a cleaner long-term configuration model without broad rewrites.

### 7. Plugin And Extension System

Status: `Planned`

Planned direction:
- Move from path-import + decorator registration toward a Pluggy-based plugin
  system.
- Keep current registration patterns functional until the new extension surface
  is stable.
- Make it easier for external integrations to add components cleanly.

## What Is Not Changing Immediately

- Existing Hydra-based entrypoints remain the active interface for now.
- The current configuration and registration systems remain supported during the
  transition.
- The completed model-side process runtime remains the reference implementation
  rather than something to replace again immediately.
- Pre-`1.0.0` backward compatibility is desirable when cheap, but it is not the
  main design constraint.

## How To Use This Document

- Update the status table first when a major track changes state.
- Move completed work into `Completed Foundations` when it becomes stable enough
  to serve as a reference for later work.
- Keep `Current Focus` limited to the next most important implementation target.
- Add or remove planned tracks when the `1.0.0` transition plan becomes more
  concrete.
- Keep detailed implementation plans, issue triage, and personal notes outside
  this file.
