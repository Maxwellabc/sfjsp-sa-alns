"""ILS outer loop wrapping ALNS local search (stub)."""

from __future__ import annotations

from sfjsp_alns.alns import ALNSConfig, ALNSResult
from sfjsp_alns.types import Instance, ScenarioBundle, Solution


def run_ils(
    instance: Instance,
    init: Solution,
    scenarios_search: ScenarioBundle,
    *,
    alns_config: ALNSConfig | None = None,
    outer_restarts: int = 10,
) -> ALNSResult:
    raise NotImplementedError("run_ils: optional outer layer after ALNS M1")
