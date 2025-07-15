import sklearn
import numpy as np
import pandas as pd

def calculate_comp_grads(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    celltype_col: str,
    near_k: int = 25,
    far_k: int = 100,
    near_quantile: float = None,   # new: between 0 and 1, or None
    far_quantile: float  = None,    # new: between 0 and 1, or None
    eps: float = 1e-2,
    n_bins: int = 36,
    absgrad: bool = True,
    use_cuda: bool = False,
    grad_clip: float = 0.9
):
    """
    If use_cuda=True, uses cuML StandardScaler, NearestNeighbors, PCA on GPU.
    If near_quantile/far_quantile are set, neighbors beyond that per-cell
    distance quantile are discarded.
    """

    # 1) SCALE + KNN ---------------------------------------------------
    if use_cuda:
        import cupy as cp
        from cuml.preprocessing import StandardScaler as cuStandardScaler
        from cuml.neighbors    import NearestNeighbors as cuNearestNeighbors

        X_cpu = df[[x_col, y_col]].values
        X_cp  = cp.asarray(X_cpu)
        X_scaled_cp = cuStandardScaler().fit_transform(X_cp)

        knn_near = cuNearestNeighbors(n_neighbors=near_k+1)
        knn_far  = cuNearestNeighbors(n_neighbors=far_k +1)

        knn_near.fit(X_scaled_cp)
        d_near_cp, i_near_cp = knn_near.kneighbors(X_scaled_cp)

        knn_far.fit(X_scaled_cp)
        d_far_cp,  i_far_cp  = knn_far.kneighbors(X_scaled_cp)

        X_scaled    = cp.asnumpy(X_scaled_cp)
        d_near, i_near = cp.asnumpy(d_near_cp), cp.asnumpy(i_near_cp)
        d_far,  i_far  = cp.asnumpy(d_far_cp),  cp.asnumpy(i_far_cp)

    else:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors    import NearestNeighbors

        X_cpu = df[[x_col, y_col]].values
        X_scaled = StandardScaler().fit_transform(X_cpu)

        knn_near = NearestNeighbors(n_neighbors=near_k+1)
        knn_far  = NearestNeighbors(n_neighbors=far_k +1)

        knn_near.fit(X_scaled)
        d_near, i_near = knn_near.kneighbors(X_scaled)

        knn_far.fit(X_scaled)
        d_far,  i_far  = knn_far.kneighbors(X_scaled)

    # drop self
    distances_near, indices_near = d_near[:,1:], i_near[:,1:]
    distances_far,  indices_far  = d_far[:,1:],  i_far[:,1:]

    n_cells = X_scaled.shape[0]

    # 1a) apply quantile‐thresholding if requested --------------------
    # build per-cell lists of neighbor indices & (for far) distances
    if near_quantile is not None:
        thresh_near = np.quantile(distances_near, near_quantile, axis=1)
        neighbor_indices_near = [
            indices_near[i][distances_near[i] <= thresh_near[i]]
            for i in range(n_cells)
        ]
    else:
        neighbor_indices_near = [indices_near[i] for i in range(n_cells)]

    if far_quantile is not None:
        thresh_far = np.quantile(distances_far, far_quantile, axis=1)
        neighbor_indices_far = [
            indices_far[i][distances_far[i] <= thresh_far[i]]
            for i in range(n_cells)
        ]
        neighbor_distances_far = [
            distances_far[i][distances_far[i] <= thresh_far[i]]
            for i in range(n_cells)
        ]
    else:
        neighbor_indices_far = [indices_far[i] for i in range(n_cells)]
        neighbor_distances_far = [distances_far[i] for i in range(n_cells)]

    # 2) SETUP ----------------------------------------------------------
    df[celltype_col] = df[celltype_col].astype('category')
    cell_types   = df[celltype_col].values
    unique_types = df[celltype_col].cat.categories
    n_types      = unique_types.size
    type_to_idx  = {t: i for i, t in enumerate(unique_types)}

    # 3) NEIGHBOR‐TYPE VECTORS & COUNTS ---------------------------------
    counts    = np.zeros((n_cells, n_types), dtype=int)
    type_vecs = np.zeros((n_cells, n_types, 2), dtype=float)

    for i in range(n_cells):
        neigh_near   = neighbor_indices_near[i]
        neigh_far    = neighbor_indices_far[i]
        dist_far_i   = neighbor_distances_far[i]
        neigh_t_near = cell_types[neigh_near]
        neigh_t_far  = cell_types[neigh_far]

        dx_far = X_scaled[neigh_far,0] - X_scaled[i,0]
        dy_far = X_scaled[neigh_far,1] - X_scaled[i,1]

        for t, j in type_to_idx.items():
            # counts
            mask_near = (neigh_t_near == t)
            counts[i,j] = mask_near.sum()

            # mean‐neighbor‐vector
            mask_far = (neigh_t_far == t)
            c_far = mask_far.sum()
            if c_far > 0:
                md = dist_far_i[mask_far].mean()
                uv = np.vstack((dx_far[mask_far], dy_far[mask_far])).T \
                     / dist_far_i[mask_far][:,None]
                muv = uv.mean(axis=0)
                norm = np.linalg.norm(muv)
                type_vecs[i,j] = (muv/norm)*md if norm > eps else 0.0
            else:
                type_vecs[i,j] = 0.0

    neighbor_counts = pd.DataFrame(
        counts,
        columns=[f"{t}_count" for t in unique_types],
        index=df.index,
    )
    flat = type_vecs.reshape(n_cells, -1)
    cols = [f"{t}_{axis}" for t in unique_types for axis in ("vec_x","vec_y")]
    neighbor_type_vecs = pd.DataFrame(flat, columns=cols, index=df.index)

    # 4) COMPOSITION VECTORS + PCA -------------------------------------
    one_hot = np.zeros((n_cells, n_types), float)
    for i, t in enumerate(cell_types):
        one_hot[i, type_to_idx[t]] = 1.0

    comp_vectors = np.zeros((n_cells, n_types), float)
    for i in range(n_cells):
        neigh = neighbor_indices_far[i]
        if len(neigh) > 0:
            comp_vectors[i] = one_hot[neigh].mean(axis=0)
        # else leave as zeros

    if use_cuda:
        from cuml.decomposition import PCA as cuPCA
        import cupy as cp
        comp_cp = cp.asarray(comp_vectors)
        vp = cuPCA(n_components=min(10, n_types)).fit_transform(comp_cp)
        comp_vectors_pca = cp.asnumpy(vp)
    else:
        from sklearn.decomposition import PCA
        comp_vectors_pca = PCA(n_components=min(10, n_types))\
                             .fit_transform(comp_vectors)

    # 5) GRADIENTS ------------------------------------------------------
    gradients = np.zeros((n_cells, 2), float)
    for i in range(n_cells):
        neigh = neighbor_indices_far[i]
        if len(neigh) == 0:
            continue
        neigh_cv = comp_vectors[neigh]
        dx = X_scaled[neigh,0] - X_scaled[i,0]
        dy = X_scaled[neigh,1] - X_scaled[i,1]

        ang  = np.arctan2(dy, dx)
        bins = np.linspace(-np.pi, np.pi, n_bins+1)
        dig  = np.digitize(ang, bins) - 1
        bc   = np.bincount(dig, minlength=n_bins)
        mn   = np.mean(np.hypot(dx, dy))
        pseudo = np.full((n_bins, n_types), mn)

        for b in range(n_bins):
            if bc[b] == 0:
                neigh_cv = np.vstack([neigh_cv, pseudo[b]])
                dx  = np.append(dx, mn)
                dy  = np.append(dy, mn)
                dig = np.append(dig, b)
                bc[b] += 1

        w = 1.0 / (bc[dig] + eps)
        w /= w.sum()
        dx[dx == 0] = eps * np.sign(dx[dx == 0])
        dy[dy == 0] = eps * np.sign(dy[dy == 0])

        dV = neigh_cv - comp_vectors[[i]]
        gx = (np.abs(dV) * w[:,None] / dx[:,None]).sum()
        gy = (np.abs(dV) * w[:,None] / dy[:,None]).sum()
        gradients[i,0] = gx
        gradients[i,1] = np.abs(gy) if absgrad else gy

    df_grads = df.copy()
    norms = np.linalg.norm(gradients, axis=1)
    df_grads['gradient_norm'] = norms

    if grad_clip is not None:
        threshold = np.quantile(norms, grad_clip)
        scales = np.where(norms>0,
                          np.minimum(1.0, threshold/norms),
                          1.0)
        gradients = gradients * scales[:,None]
        df_grads['gradient_norm'] = np.linalg.norm(gradients, axis=1)

    return (
        df_grads,
        df[celltype_col],
        gradients,
        comp_vectors_pca,
        comp_vectors,
        neighbor_counts,
        neighbor_type_vecs,
    )

