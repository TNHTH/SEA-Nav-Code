# SPDX-License-Identifier: MIT
"""Evaluation statistics: seed aggregates and bootstrap stub (A8.2)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from evaluation import EpisodeOutcome

BOOTSTRAP_SEED = 20260916


def mean_std(values: Sequence[float], *, ddof: int = 1) -> Tuple[Optional[float], Optional[float]]:
    n = len(values)
    if n == 0:
        return None, None
    mu = sum(values) / n
    if n <= ddof:
        return mu, None
    var = sum((x - mu) ** 2 for x in values) / (n - ddof)
    return mu, math.sqrt(var)


@dataclass
class SeedRates:
    seed: int
    success_rate: float
    collision_rate: float
    timeout_rate: float
    n: int


def rates_for_outcomes(outcomes: Iterable[str]) -> Tuple[float, float, float, int]:
    items = list(outcomes)
    n = len(items)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    sr = sum(1 for o in items if o == EpisodeOutcome.SUCCESS.value) / n
    cr = sum(1 for o in items if o == EpisodeOutcome.COLLISION.value) / n
    tr = sum(1 for o in items if o == EpisodeOutcome.TIMEOUT.value) / n
    return sr, cr, tr, n


def aggregate_profile_seeds(
    by_seed: Dict[int, Sequence[str]],
) -> Dict[str, Optional[float]]:
    seed_rates: List[SeedRates] = []
    for seed, outcomes in sorted(by_seed.items()):
        sr, cr, tr, n = rates_for_outcomes(outcomes)
        seed_rates.append(SeedRates(seed, sr, cr, tr, n))
    srs = [r.success_rate for r in seed_rates]
    crs = [r.collision_rate for r in seed_rates]
    trs = [r.timeout_rate for r in seed_rates]
    sr_mu, sr_std = mean_std(srs, ddof=1)
    cr_mu, cr_std = mean_std(crs, ddof=1)
    tr_mu, tr_std = mean_std(trs, ddof=1)
    return {
        "success_mean": sr_mu,
        "success_std": sr_std,
        "collision_mean": cr_mu,
        "collision_std": cr_std,
        "timeout_mean": tr_mu,
        "timeout_std": tr_std,
        "seed_rates": seed_rates,
    }


@dataclass
class BootstrapResult:
    seed: int
    n_resamples: int
    indices: torch.Tensor
    note: str = "hierarchical paired bootstrap stub"


def hierarchical_paired_bootstrap_stub(
    profile_indices: Dict[str, List[int]],
    *,
    n_resamples: int = 1000,
    seed: int = BOOTSTRAP_SEED,
) -> BootstrapResult:
    """Fixed-seed stub returning shared resample indices per profile."""
    gen = torch.Generator()
    gen.manual_seed(seed)
    # One shared index draw per profile for pairing (stub API).
    max_len = max(len(v) for v in profile_indices.values()) if profile_indices else 0
    if max_len == 0:
        indices = torch.zeros((n_resamples, 0), dtype=torch.int64)
    else:
        indices = torch.randint(0, max_len, (n_resamples, max_len), generator=gen)
    return BootstrapResult(seed=seed, n_resamples=n_resamples, indices=indices)


def success_only_time_mean(times: Sequence[Optional[float]]) -> Tuple[Optional[float], int]:
    finite = [t for t in times if t is not None and math.isfinite(t)]
    if not finite:
        return None, 0
    return sum(finite) / len(finite), len(finite)


def safe_ratio(numerator: float, denominator: float) -> Optional[float]:
    if denominator == 0:
        return None
    return numerator / denominator
