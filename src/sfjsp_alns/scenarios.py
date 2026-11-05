"""Processing-time scenarios.

``lognormal_random_cov`` copies Smit et al. ``create_stochastic_realizations_random_variances``
(CoV ~ U(0.1, 0.5), then lognormal). Incompatible machine entries stay 0.

The bank generator uses NumPy's global RNG so a single ``np.random.seed`` followed by
instances in sorted order tracks the official script. The K_eval harness uses a
``Generator`` so a smoke run does not touch global state and is not an SD3 bank.
"""

from __future__ import annotations

import numpy as np


def _draw(duration: np.ndarray, num_samples: int, uniform, lognormal) -> np.ndarray:
    duration = np.asarray(duration)
    mask = duration != 0
    mean = np.zeros(duration.shape, dtype=np.float64)
    sigma = np.zeros(duration.shape, dtype=np.float64)
    if np.any(mask):
        mean[mask] = np.log(duration[mask].astype(np.float64))
        cov = np.asarray(uniform(0.1, 0.5, size=duration.shape), dtype=np.float64)
        sigma[mask] = np.sqrt(
            np.log(1.0 + np.square(cov[mask]))
        )
    tile_mean = np.tile(mean, (num_samples, 1, 1))
    tile_sigma = np.tile(sigma, (num_samples, 1, 1))
    tile_mask = np.tile(mask, (num_samples, 1, 1))
    samples = np.zeros(tile_mean.shape, dtype=np.int64)
    if np.any(tile_mask):
        drawn = np.asarray(lognormal(tile_mean[tile_mask], tile_sigma[tile_mask]), dtype=np.float64)
        samples[tile_mask] = np.maximum(np.rint(drawn), 1).astype(np.int64)
    return samples


def lognormal_random_cov_bank(op_pt: np.ndarray, num_samples: int) -> np.ndarray:
    """Official recipe on the global ``np.random`` stream."""
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    return _draw(op_pt, num_samples, np.random.uniform, np.random.lognormal)


def lognormal_random_cov(op_pt: np.ndarray, num_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Same recipe on an isolated Generator (synthetic smoke / unit tests)."""
    if num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    return _draw(op_pt, num_samples, rng.uniform, rng.lognormal)
