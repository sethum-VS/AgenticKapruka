"""Local development utilities (not mounted in production)."""

from lib.dev.routing_simulator import (
    SCENARIOS,
    TONE_PROFILES,
    SimulatorResult,
    run_simulation,
)

__all__ = [
    "SCENARIOS",
    "TONE_PROFILES",
    "SimulatorResult",
    "run_simulation",
]
