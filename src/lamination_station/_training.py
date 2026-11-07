"""Shared validation and unbiased minibatching for the model variants."""

import torch
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler


@dataclass(frozen=True)
class TrainingData:
    """Aligned tensors with a categorical identity vocabulary."""

    counts: torch.Tensor
    gradients: torch.Tensor
    type_vectors: torch.Tensor
    cosines: torch.Tensor
    identities: torch.Tensor


def validate_settings(
    *,
    num_structures: int,
    latent_dim: int,
    batch_size: int,
    num_epochs: int,
    lr_steps: Sequence[float],
    num_particles: int,
    loss_scale: float,
    location_scale: float,
    hidden_dim: int,
) -> None:
    """Reject settings that would skip training or create invalid distributions."""
    if num_structures < 2:
        raise ValueError("NUM_STRUCTURES must be at least 2 for top-two outputs")
    if min(latent_dim, batch_size, num_epochs, num_particles, hidden_dim) < 1:
        raise ValueError("Dimensions, batch_size, num_epochs, and num_particles must be positive")
    if not lr_steps or not np.isfinite(lr_steps).all() or min(lr_steps) <= 0:
        raise ValueError("lr_steps must contain finite positive learning rates")
    if not np.isfinite([loss_scale, location_scale]).all() or min(loss_scale, location_scale) <= 0:
        raise ValueError("LOSS_SCALE and STRUCT_LOC_PRIOR_SCALE must be finite and positive")


def prepare_data(
    *,
    df: pd.DataFrame,
    cell_types: pd.Series,
    counts: pd.DataFrame,
    gradients: NDArray[np.float64],
    type_vectors: pd.DataFrame,
    device: str | torch.device,
    ignore_identity: bool,
) -> TrainingData:
    """Validate aligned model inputs; never silently replace missing observations by zeros.

    Empty neighborhoods are missing composition information and must be filtered
    from every input together, or recomputed, before fitting.
    """
    if len(df) == 0 or not df.index.is_unique:
        raise ValueError("Input rows must be nonempty and have unique indices")
    for value in (cell_types, counts, type_vectors):
        if not value.index.equals(df.index):
            raise ValueError("All input indices must match df_grads in the same order")
    if not isinstance(cell_types.dtype, pd.CategoricalDtype) or cell_types.isna().any():
        raise ValueError("cell_types_series must be categorical with no missing labels")
    n, t = counts.shape
    if t == 0 or len(cell_types.cat.categories) != t:
        raise ValueError("Count columns must match the cell-type category vocabulary")
    expected_counts = [f"{label}_count" for label in cell_types.cat.categories]
    expected_vectors = [
        f"{label}_{axis}" for label in cell_types.cat.categories for axis in ("vec_x", "vec_y")
    ]
    if list(counts.columns) != expected_counts or list(type_vectors.columns) != expected_vectors:
        raise ValueError("Count and vector columns must follow the preprocessing category order")
    c = counts.to_numpy(dtype=float)
    g = np.asarray(gradients, dtype=float)
    v = type_vectors.to_numpy(dtype=float)
    if g.shape != (n, 2) or v.shape != (n, 2 * t):
        raise ValueError("Expected gradients shaped (N, 2) and type vectors shaped (N, 2*T)")
    if not all(np.isfinite(value).all() for value in (c, g, v)):
        raise ValueError("Counts, gradients, and type vectors must be finite")
    if (c < 0).any() or not np.equal(c, np.floor(c)).all():
        raise ValueError("Neighbor counts must be nonnegative integers")
    if (c.sum(axis=1) == 0).any():
        raise ValueError(
            "Empty neighborhoods: filter zero-count rows from every input together or recompute neighbors"
        )
    tensor_counts = torch.as_tensor(c, dtype=torch.float32, device=device)
    tensor_grads = torch.as_tensor(g, dtype=torch.float32, device=device)
    tensor_vectors = torch.as_tensor(v.reshape(n, t, 2), dtype=torch.float32, device=device)
    codes = torch.as_tensor(
        cell_types.cat.codes.to_numpy().copy(), dtype=torch.long, device=device
    )
    identities = torch.nn.functional.one_hot(codes, num_classes=t).float()
    if ignore_identity:
        identities.zero_()
    cosines = torch.nn.functional.cosine_similarity(tensor_grads[:, None], tensor_vectors, dim=-1)
    return TrainingData(tensor_counts, tensor_grads, tensor_vectors, cosines, identities)


def make_training_loader(
    *,
    data: TrainingData,
    cell_types: pd.Series,
    batch_size: int,
    normalize_sampling: bool,
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Return complete batches and inverse selection probabilities for local ELBO terms.

    Class balancing affects sampling efficiency only; importance weights preserve
    the likelihood of the observed population. Divide the returned weights by
    the actual batch size inside both model and guide.
    """
    labels = cell_types.cat.codes.to_numpy()
    frequencies = np.bincount(labels, minlength=len(cell_types.cat.categories)) + 200
    weights = 1.0 / frequencies[labels] if normalize_sampling else np.ones(len(labels))
    probabilities = torch.as_tensor(weights / weights.sum(), dtype=torch.float64)
    inverse_probabilities = probabilities.reciprocal().to(data.counts)
    dataset = TensorDataset(
        data.identities,
        data.gradients,
        data.type_vectors,
        data.counts,
        data.cosines,
        inverse_probabilities,
    )
    if normalize_sampling:
        sampler = WeightedRandomSampler(
            probabilities.tolist(), num_samples=len(labels), replacement=True
        )
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
