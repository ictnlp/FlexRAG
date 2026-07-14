from __future__ import annotations

from typing import Any

from flexrag.common.dataclasses import Context


def context_to_payload(context: Context) -> dict[str, Any]:
    """Convert a context into a serializer-friendly payload.

    :param context: Context to persist.
    :returns: Plain dictionary with context id, data, source, and metadata.
    """
    return {
        "context_id": context.context_id,
        "data": context.data,
        "source": context.source,
        "metadata": context.metadata,
    }


def payload_to_context(payload: dict[str, Any]) -> Context:
    """Restore a context from a persisted payload dictionary.

    :param payload: Payload produced by ``context_to_payload``.
    :returns: Restored context.
    """
    return Context(
        context_id=payload["context_id"],
        data=payload["data"],
        source=payload["source"],
        metadata=payload["metadata"],
    )
