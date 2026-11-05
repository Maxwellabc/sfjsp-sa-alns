"""Phase-2 LLM operator selection interface; default is classic roulette."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class SearchFeatures:
    mean_load: float
    load_std: float
    critical_len: float
    scenario_var: float
    segment_scores: dict[str, float]


class OperatorSelector(Protocol):
    def select_destroy(self, features: SearchFeatures, names: list[str], rng: np.random.Generator) -> str: ...

    def select_repair(self, features: SearchFeatures, names: list[str], rng: np.random.Generator) -> str: ...


class RouletteSelector:
    """Adaptive weight roulette (no LLM) — production default for strong baseline."""

    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = weights or {}

    def select_destroy(self, features: SearchFeatures, names: list[str], rng: np.random.Generator) -> str:
        return self._pick(names, rng)

    def select_repair(self, features: SearchFeatures, names: list[str], rng: np.random.Generator) -> str:
        return self._pick(names, rng)

    def _pick(self, names: list[str], rng: np.random.Generator) -> str:
        w = np.array([self.weights.get(n, 1.0) for n in names], dtype=np.float64)
        w = w / w.sum()
        return str(rng.choice(names, p=w))


class LLMSelector:
    """Optional phase-2: call an LLM with features; fall back to RouletteSelector."""

    def __init__(self, fallback: RouletteSelector | None = None):
        self.fallback = fallback or RouletteSelector()

    def select_destroy(self, features: SearchFeatures, names: list[str], rng: np.random.Generator) -> str:
        # TODO(M4): prompt + parse; on failure use fallback
        return self.fallback.select_destroy(features, names, rng)

    def select_repair(self, features: SearchFeatures, names: list[str], rng: np.random.Generator) -> str:
        return self.fallback.select_repair(features, names, rng)
