from __future__ import annotations

import os
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

_DEVICE_REF_RE = re.compile(r"^(?P<kind>[A-Za-z][A-Za-z0-9_]*):(?P<index>[0-9]+)$")
_DEVICE_ENV_KEYS = {
    "cuda": "CUDA_VISIBLE_DEVICES",
    "xpu": "ZE_AFFINITY_MASK",
}


def worker_env_updates_from_device_groups(
    device_groups: object,
) -> tuple[dict[str, str], ...]:
    """Parse process runtime device groups into per-worker env updates.

    Only explicit accelerator kinds are mapped. ``cuda`` maps to
    ``CUDA_VISIBLE_DEVICES`` and ``xpu`` maps to ``ZE_AFFINITY_MASK``. The helper
    does not implement CPU isolation.

    :param device_groups: List of non-empty device-ref groups.
    :returns: Per-worker environment updates.
    :raises ValueError: If the device group shape or kind is invalid.
    """

    if not isinstance(device_groups, list) or not device_groups:
        raise ValueError("device_groups must be a non-empty list of device groups.")

    worker_updates: list[dict[str, str]] = []
    selected_kind: str | None = None
    for group in device_groups:
        kind, indices = _parse_device_group(group)
        if selected_kind is None:
            selected_kind = kind
        elif kind != selected_kind:
            raise ValueError("device_groups cannot mix device kinds.")
        worker_updates.append({_DEVICE_ENV_KEYS[kind]: ",".join(indices)})
    return tuple(worker_updates)


def _parse_device_group(group: object) -> tuple[str, list[str]]:
    """Parse one worker's homogeneous device group."""
    if not isinstance(group, list) or not group:
        raise ValueError("Each device group must be a non-empty list of device refs.")

    selected_kind: str | None = None
    indices: list[str] = []
    for ref in group:
        kind, index = _parse_device_ref(ref)
        if selected_kind is None:
            selected_kind = kind
        elif kind != selected_kind:
            raise ValueError("A device group cannot mix device kinds.")
        indices.append(index)

    if selected_kind is None:
        raise ValueError("Each device group must include at least one device ref.")
    return selected_kind, indices


def _parse_device_ref(ref: object) -> tuple[str, str]:
    """Parse one ``kind:index`` device reference."""
    if not isinstance(ref, str):
        raise ValueError("Device refs must be strings like 'cuda:0' or 'xpu:0'.")
    match = _DEVICE_REF_RE.match(ref)
    if match is None:
        raise ValueError(f"Invalid device ref: {ref!r}.")

    kind = match.group("kind").lower()
    if kind not in _DEVICE_ENV_KEYS:
        raise ValueError(f"Unsupported device kind: {kind}.")
    return kind, match.group("index")


@contextmanager
def temporary_env(updates: Mapping[str, str] | None) -> Iterator[None]:
    """Temporarily apply environment variable overrides.

    :param updates: Environment variables to set while the context is active.
    """

    if not updates:
        yield
        return

    previous = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return
