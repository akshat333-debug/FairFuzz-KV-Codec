import random
from typing import List, Sequence, Tuple

import torch

from fairfuzzkv_codec.quantization.scales import Granularity, aggregate_calibration_range


def select_calibration_subset(samples: Sequence[torch.Tensor], num_samples: int, seed: int) -> List[torch.Tensor]:
    """Deterministic calibration-set selection: a fixed-seed random subset of
    the available samples, not the full population (calibration should
    generalize to unseen data, not memorize it)."""
    if num_samples > len(samples):
        raise ValueError(f"requested {num_samples} calibration samples but only {len(samples)} available")
    indices = list(range(len(samples)))
    random.Random(seed).shuffle(indices)
    return [samples[i] for i in indices[:num_samples]]


def calibrate_range(
    samples: Sequence[torch.Tensor], num_calibration_samples: int, seed: int, granularity: Granularity, group_size=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Select a calibration subset, then compute the reusable (min, max)
    range from it - the range that will later be applied to held-out data,
    not recomputed per-sample."""
    subset = select_calibration_subset(samples, num_calibration_samples, seed)
    return aggregate_calibration_range(subset, granularity, group_size)
