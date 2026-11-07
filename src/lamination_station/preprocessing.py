import torch  # noqa: F401
from enum import Enum
from numpy.typing import NDArray
import numpy as np
import pandas as pd


class CoordinateScaling(str, Enum):
    """Coordinate metric used for spatial neighborhoods."""

    ISOTROPIC = "isotropic"
    AXISWISE = "axiswise"
    NONE = "none"


def calculate_comp_grads(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    celltype_col: str,
    near_k: int = 25,
    far_k: int = 100,
    near_quantile: float | None = None,
    far_quantile: float | None = None,
    eps: float = 1e-2,
    n_bins: int = 36,
    absgrad: bool = True,
    use_cuda: bool = False,
    grad_clip: float | None = 0.9,
    elev_weight: float = 1.0,
    *,
    sample_col: str | None = None,
    coordinate_scaling: CoordinateScaling | str = CoordinateScaling.ISOTROPIC,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Compute local compositions and directional descriptors within physical samples.

    Isotropic scaling preserves angles and neighbor order under rigid rotation.
    Axiswise scaling reproduces the legacy metric. Set sample_col whenever rows
    combine sections: neighborhoods never cross its groups. Gradients are heuristic
    directional descriptors, not signed derivatives. Legacy n_bins and absgrad
    arguments remain accepted but do not affect the computation.
    """
    df = df.copy()
    if len(df) < 2 or not df.index.is_unique:
        raise ValueError("At least two cells with unique indices are required")
    if not 1 <= near_k <= far_k:
        raise ValueError("Require 1 <= near_k <= far_k")
    for value in (near_quantile, far_quantile, grad_clip):
        if value is not None and not 0 <= value <= 1:
            raise ValueError("Quantiles must be in [0, 1]")
    if not np.isfinite([eps, elev_weight]).all() or eps <= 0 or elev_weight < 0:
        raise ValueError("eps must be positive and elev_weight nonnegative and finite")
    if df[celltype_col].isna().any():
        raise ValueError("Cell types must not be missing")
    coordinates = df[[x_col, y_col]].to_numpy(dtype=float)
    if not np.isfinite(coordinates).all():
        raise ValueError("Coordinates must be finite")
    if sample_col is None:
        groups = [np.arange(len(df))]
    else:
        if df[sample_col].isna().any():
            raise ValueError("Sample IDs must not be missing")
        groups = [
            np.asarray(rows)
            for rows in df.groupby(sample_col, observed=True, sort=False).indices.values()
        ]
    if any(len(rows) <= far_k for rows in groups):
        raise ValueError("far_k must be smaller than the number of cells in every sample")
    centered = np.empty_like(coordinates)
    for rows in groups:
        centered[rows] = coordinates[rows] - coordinates[rows].mean(axis=0)
    scaling = CoordinateScaling(coordinate_scaling)
    if scaling is CoordinateScaling.ISOTROPIC:
        scales: NDArray[np.float64] = np.full(2, np.sqrt(np.mean(centered**2)))
    elif scaling is CoordinateScaling.AXISWISE:
        scales = np.sqrt(np.mean(centered**2, axis=0))
    else:
        scales = np.ones(2)
    scales = np.where(scales > 0, scales, 1.0)
    X_scaled = centered / scales
    df.attrs["lamination_station_coordinate_scale"] = scales.tolist()
    n_cells = len(df)
    distances_far = np.empty((n_cells, far_k), dtype=float)
    indices_far = np.empty((n_cells, far_k), dtype=np.int64)
    for rows in groups:
        if use_cuda:
            import cupy as cp
            from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

            coords_gpu = cp.asarray(X_scaled[rows])
            nearest = cuNearestNeighbors(n_neighbors=far_k + 1).fit(coords_gpu)
            distances_gpu, indices_gpu = nearest.kneighbors(coords_gpu)
            distances = cp.asnumpy(distances_gpu)
            indices = cp.asnumpy(indices_gpu)
        else:
            from sklearn.neighbors import NearestNeighbors

            nearest = NearestNeighbors(n_neighbors=far_k + 1).fit(X_scaled[rows])
            distances, indices = nearest.kneighbors(X_scaled[rows])
        for local_i, global_i in enumerate(rows):
            # Coordinate ties need identity-based exclusion, not dropping column 0.
            keep = np.flatnonzero(indices[local_i] != local_i)[:far_k]
            distances_far[global_i] = distances[local_i, keep]
            indices_far[global_i] = rows[indices[local_i, keep]]

    # 1a) far_quantile thresholding ----------------------------------
    if far_quantile is not None:
        global_far_thresh = np.quantile(distances_far.ravel(), far_quantile)
        neighbor_indices_far = [
            indices_far[i][distances_far[i] <= global_far_thresh] for i in range(n_cells)
        ]
        neighbor_distances_far = [
            distances_far[i][distances_far[i] <= global_far_thresh] for i in range(n_cells)
        ]
    else:
        neighbor_indices_far = [indices_far[i] for i in range(n_cells)]
        neighbor_distances_far = [distances_far[i] for i in range(n_cells)]

    # 2) SETUP --------------------------------------------------------
    df[celltype_col] = df[celltype_col].astype("category")
    cell_types = df[celltype_col].values
    unique_types = df[celltype_col].cat.categories
    n_types = unique_types.size
    type_to_idx = {t: i for i, t in enumerate(unique_types)}

    # 3) COMPOSITION VECTORS + PCA ------------------------------------
    one_hot = np.zeros((n_cells, n_types), float)
    for i, t in enumerate(cell_types):
        one_hot[i, type_to_idx[t]] = 1.0

    comp_vectors = np.zeros((n_cells, n_types), float)
    for i in range(n_cells):
        neigh = neighbor_indices_far[i]
        if len(neigh) > 0:
            comp_vectors[i] = one_hot[neigh].mean(axis=0)

    if use_cuda:
        from cuml.decomposition import PCA as cuPCA
        import cupy as cp

        comp_cp = cp.asarray(comp_vectors)
        vp = cuPCA(n_components=min(10, n_types, n_cells)).fit_transform(comp_cp)
        comp_pca = cp.asnumpy(vp)
    else:
        from sklearn.decomposition import PCA

        comp_pca = PCA(n_components=min(10, n_types, n_cells)).fit_transform(comp_vectors)

    # 4) GRADIENTS -----------------------------------------------------
    gradients: NDArray[np.float64] = np.zeros((n_cells, 2), float)
    for i in range(n_cells):
        neigh = neighbor_indices_far[i]
        if not len(neigh):
            continue

        comp_i = comp_vectors[i]
        comp_n = comp_vectors[neigh]
        diffs = np.linalg.norm(comp_n - comp_i, axis=1)

        dx = X_scaled[neigh, 0] - X_scaled[i, 0]
        dy = X_scaled[neigh, 1] - X_scaled[i, 1]
        dist = np.hypot(dx, dy) + eps

        uv = np.stack((dx, dy), axis=1) / dist[:, None]
        w = diffs / dist
        gv = (w[:, None] * uv).sum(axis=0)
        gradients[i] = gv / (w.sum() + eps)

    # gradient clipping & norms in df_grads
    df_grads = df.copy()
    norms = np.linalg.norm(gradients, axis=1)
    df_grads["gradient_norm"] = norms
    if grad_clip is not None:
        thresh = np.quantile(norms, grad_clip)
        scales = np.minimum(
            1.0, np.divide(thresh, norms, out=np.ones_like(norms), where=norms > 0)
        )
        gradients = gradients * scales[:, None]
        df_grads["gradient_norm"] = np.linalg.norm(gradients, axis=1)

    # prepare for penalization
    grad_norms = np.linalg.norm(gradients, axis=1)
    unit_grads = gradients / (grad_norms[:, None] + eps)

    # 5) RECOMPUTE near_k WITH PENALIZED DISTANCES --------------------
    # collect all modified d_mod if we need a global quantile
    all_dmod = []
    tmp_dmod: list[NDArray[np.float64]] = [np.empty(0)] * n_cells

    for i in range(n_cells):
        neigh_far = neighbor_indices_far[i]
        dist_far_i = neighbor_distances_far[i]
        g_u = unit_grads[i]
        g_norm = grad_norms[i]

        # unit vectors toward each far-neighbor
        dx = X_scaled[neigh_far, 0] - X_scaled[i, 0]
        dy = X_scaled[neigh_far, 1] - X_scaled[i, 1]
        uv = np.stack((dx, dy), axis=1) / (dist_far_i[:, None] + eps)

        # positive alignment penalty
        align = uv.dot(g_u)  # [-1 .. 1]
        penalty = np.maximum(0, np.abs(align)) * g_norm
        d_mod = dist_far_i + elev_weight * penalty

        tmp_dmod[i] = d_mod
        all_dmod.append(d_mod)

    if near_quantile is not None:
        global_near_thresh = (
            float(np.quantile(np.hstack(all_dmod), near_quantile))
            if any(len(d) for d in all_dmod)
            else None
        )
    else:
        global_near_thresh = None

    neighbor_indices_near: list[NDArray[np.int64]] = [np.empty(0, dtype=np.int64)] * n_cells
    neighbor_distances_near: list[NDArray[np.float64]] = [np.empty(0)] * n_cells

    for i in range(n_cells):
        neigh_far = neighbor_indices_far[i]
        d_mod = tmp_dmod[i]

        # apply global threshold if requested
        if global_near_thresh is not None:
            mask = d_mod <= global_near_thresh
            idxs = np.where(mask)[0]
            if idxs.size > near_k:
                order = np.argsort(d_mod[mask])[:near_k]
                sel = idxs[order]
            else:
                sel = idxs
        else:
            sel = np.argsort(d_mod)[:near_k]

        neighbor_indices_near[i] = neigh_far[sel]
        neighbor_distances_near[i] = d_mod[sel]

    # 6) NEIGHBOR‐TYPE COUNTS & VECTORS -------------------------------
    counts = np.zeros((n_cells, n_types), dtype=int)
    type_vecs = np.zeros((n_cells, n_types, 2), dtype=float)
    type_codes = df[celltype_col].cat.codes.to_numpy()

    for i in range(n_cells):
        near = neighbor_indices_near[i]
        far = neighbor_indices_far[i]
        dfar = neighbor_distances_far[i]
        counts[i] = np.bincount(type_codes[near], minlength=n_types)
        codes = type_codes[far]
        frequency = np.bincount(codes, minlength=n_types)
        mean_distance = np.divide(
            np.bincount(codes, weights=dfar, minlength=n_types),
            frequency,
            out=np.zeros(n_types),
            where=frequency > 0,
        )
        displacement = X_scaled[far] - X_scaled[i]
        unit_vectors = np.divide(
            displacement,
            dfar[:, None],
            out=np.zeros_like(displacement),
            where=dfar[:, None] > 0,
        )
        direction_sum = np.column_stack(
            [
                np.bincount(codes, weights=unit_vectors[:, axis], minlength=n_types)
                for axis in range(2)
            ]
        )
        mean_direction = np.divide(
            direction_sum,
            frequency[:, None],
            out=np.zeros_like(direction_sum),
            where=frequency[:, None] > 0,
        )
        norms_by_type = np.linalg.norm(mean_direction, axis=1)
        type_vecs[i] = (
            np.divide(
                mean_direction,
                norms_by_type[:, None],
                out=np.zeros_like(mean_direction),
                where=norms_by_type[:, None] > eps,
            )
            * mean_distance[:, None]
        )

    neighbor_counts = pd.DataFrame(
        counts,
        columns=[f"{t}_count" for t in unique_types],
        index=df.index,
    )
    flat = type_vecs.reshape(n_cells, -1)
    cols = [f"{t}_{ax}" for t in unique_types for ax in ("vec_x", "vec_y")]
    neighbor_type_vecs = pd.DataFrame(flat, columns=cols, index=df.index)

    return (
        df_grads,
        df[celltype_col],
        gradients,
        comp_pca,
        comp_vectors,
        neighbor_counts,
        neighbor_type_vecs,
    )
