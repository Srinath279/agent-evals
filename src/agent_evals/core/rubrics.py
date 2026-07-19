"""Rubric resolution — the trace store's prompt management as the rubric
source (master plan §7), with code constants as the seed/fallback.

Version continuity (note 09 §8): `evals push-rubrics` stores the code
constant's rubric_version string inside the stored prompt's config, so
moving storage does NOT change cache keys or create a fake metric epoch.
A rubric *edited in the platform UI* gets a new derived version
(`<name>/<store>-v<N>`) — which correctly invalidates caches, because the
text actually changed.

Fail-closed: if a config asks for prompt management and the store is
unreachable, we raise rather than silently falling back — silent fallback
would mix rubric epochs invisibly.
"""

from __future__ import annotations

from typing import Optional


def resolve_rubric(
    name: str,
    fallback_text: str,
    fallback_version: str,
    from_store: bool = False,
    store: str = "langfuse",
    client: Optional[object] = None,
) -> tuple[str, str]:
    """Returns (rubric_text, rubric_version)."""
    if not from_store:
        return fallback_text, fallback_version

    try:
        if client is None:
            from agent_evals.core.store import get_store

            client = get_store(store)
        text, remote_version, config = client.get_prompt(name)
    except Exception as e:
        raise RuntimeError(
            f"rubric '{name}' requested from the '{store}' store's prompt management "
            f"but could not be fetched ({e}); refusing to silently fall back to the "
            f"code constant — run `evals push-rubrics` first or set "
            f"rubric_from_store: false"
        ) from e

    pinned = (config or {}).get("rubric_version")
    # "lf" predates multi-store support — keep it so existing derived
    # versions (and their cache keys) survive the refactor
    prefix = "lf" if store == "langfuse" else store
    version = pinned or f"{name}/{prefix}-v{remote_version}"
    return text, version
