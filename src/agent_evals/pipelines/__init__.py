# The ONLY package allowed to import temporalio (note 08, rule 1).
# Everything else stays pure library code so `evals run` works with no cluster.
#
# Temporal is OPT-IN via config (pipeline.engine: temporal). These helpers let
# callers check/enforce availability WITHOUT importing temporalio — importing
# this package must stay cheap and dependency-free so the local engine and the
# test suite never require the 'temporal' extra.
from __future__ import annotations

import importlib.util


def is_available() -> bool:
    """True when the 'temporal' extra (temporalio) is importable. Does not
    actually import it — just checks the module spec."""
    return importlib.util.find_spec("temporalio") is not None


def require_temporal() -> None:
    """Raise a friendly, actionable error when the Temporal engine is selected
    but the extra isn't installed. Callers hit this only on the temporal path,
    so the local engine never pays for it."""
    if not is_available():
        raise RuntimeError(
            "pipeline.engine is 'temporal' but the 'temporal' extra is not "
            "installed.\n"
            "  install it:   pip install 'agent-evals[temporal]'\n"
            "  or:           uv sync --extra temporal\n"
            "  or run local: set pipeline.engine: local in the config."
        )
