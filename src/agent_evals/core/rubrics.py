"""Rubric resolution — Langfuse Prompt Management as the rubric source
(master plan §7), with code constants as the seed/fallback.

Version continuity (note 09 §8): `evals push-rubrics` stores the code
constant's rubric_version string inside the Langfuse prompt's config, so
moving storage does NOT change cache keys or create a fake metric epoch.
A rubric *edited in the Langfuse UI* gets a new derived version
(`<name>/lf-v<N>`) — which correctly invalidates caches, because the text
actually changed.

Fail-closed: if a config asks for Prompt Management and Langfuse is
unreachable, we raise rather than silently falling back — silent fallback
would mix rubric epochs invisibly.
"""

from __future__ import annotations

from typing import Optional


def resolve_rubric(
    name: str,
    fallback_text: str,
    fallback_version: str,
    from_langfuse: bool = False,
    client: Optional[object] = None,
) -> tuple[str, str]:
    """Returns (rubric_text, rubric_version)."""
    if not from_langfuse:
        return fallback_text, fallback_version

    if client is None:
        from agent_evals.core.langfuse_client import LangfuseClient

        client = LangfuseClient()
    try:
        text, lf_version, config = client.get_prompt(name)
    except Exception as e:
        raise RuntimeError(
            f"rubric '{name}' requested from Langfuse Prompt Management but could not be "
            f"fetched ({e}); refusing to silently fall back to the code constant — "
            f"run `evals push-rubrics` first or set rubric_from_langfuse: false"
        ) from e

    pinned = (config or {}).get("rubric_version")
    version = pinned or f"{name}/lf-v{lf_version}"
    return text, version
