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
    near_quantile: float = None,
    far_quantile: float = None,
    eps: float = 1e-2,
    n_bins: int = 36,
    absgrad: bool = True,
    use_cuda: bool = False,
    grad_clip: float = 0.9,
    elev_weight: float = 1.0,      # how strongly to penalize “uphill”
):
    """
    ...same docstring...
    """
    import numpy as np
    # 1) SCALE + KNN (only far_k) -----------------------------------
    if use_cuda:
        import cupy as cp
        from cuml.preprocessing import StandardScaler as cuStandardScaler
        from cuml.neighbors    import NearestNeighbors as cuNearestNeighbors

        X_cpu       = df[[x_col, y_col]].values
        X_cp        = cp.asarray(X_cpu)
        X_scaled_cp = cuStandardScaler().fit_transform(X_cp)

        knn_far = cuNearestNeighbors(n_neighbors=far_k + 1)
        knn_far.fit(X_scaled_cp)
        d_far_cp, i_far_cp = knn_far.kneighbors(X_scaled_cp)

        # drop self
        distances_far = cp.asnumpy(d_far_cp)[:,1:]
        indices_far   = cp.asnumpy(i_far_cp)[:,1:]
        X_scaled      = cp.asnumpy(X_scaled_cp)

    else:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors    import NearestNeighbors

        X_cpu     = df[[x_col, y_col]].values
        X_scaled  = StandardScaler().fit_transform(X_cpu)

        knn_far = NearestNeighbors(n_neighbors=far_k + 1)
        knn_far.fit(X_scaled)
        d_far_all, i_far_all = knn_far.kneighbors(X_scaled)

        distances_far = d_far_all[:,1:]
        indices_far   = i_far_all[:,1:]

    n_cells = X_scaled.shape[0]

    # 1a) far_quantile thresholding ----------------------------------
    if far_quantile is not None:
        global_far_thresh = np.quantile(distances_far.ravel(), far_quantile)
        neighbor_indices_far = [
            indices_far[i][distances_far[i] <= global_far_thresh]
            for i in range(n_cells)
        ]
        neighbor_distances_far = [
            distances_far[i][distances_far[i] <= global_far_thresh]
            for i in range(n_cells)
        ]
    else:
        neighbor_indices_far    = [indices_far[i]    for i in range(n_cells)]
        neighbor_distances_far  = [distances_far[i]  for i in range(n_cells)]

    # 2) SETUP --------------------------------------------------------
    df[celltype_col] = df[celltype_col].astype('category')
    cell_types   = df[celltype_col].values
    unique_types = df[celltype_col].cat.categories
    n_types      = unique_types.size
    type_to_idx  = {t: i for i, t in enumerate(unique_types)}

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
        vp      = cuPCA(n_components=min(10, n_types)).fit_transform(comp_cp)
        comp_pca = cp.asnumpy(vp)
    else:
        from sklearn.decomposition import PCA
        comp_pca = PCA(n_components=min(10, n_types))\
                    .fit_transform(comp_vectors)

    # 4) GRADIENTS -----------------------------------------------------
    gradients = np.zeros((n_cells, 2), float)
    for i in range(n_cells):
        neigh = neighbor_indices_far[i]
        if not len(neigh):
            continue

        comp_i = comp_vectors[i]
        comp_n = comp_vectors[neigh]
        diffs  = np.linalg.norm(comp_n - comp_i, axis=1)

        dx   = X_scaled[neigh,0] - X_scaled[i,0]
        dy   = X_scaled[neigh,1] - X_scaled[i,1]
        dist = np.hypot(dx, dy) + eps

        uv = np.stack((dx, dy), axis=1) / dist[:,None]
        w  = diffs / dist
        gv = (w[:,None] * uv).sum(axis=0)
        gradients[i] = gv / (w.sum() + eps)

    # gradient clipping & norms in df_grads
    df_grads = df.copy()
    norms = np.linalg.norm(gradients, axis=1)
    df_grads['gradient_norm'] = norms
    if grad_clip is not None:
        thresh = np.quantile(norms, grad_clip)
        scales = np.where(norms>0, np.minimum(1.0, thresh/norms), 1.0)
        gradients = gradients * scales[:,None]
        df_grads['gradient_norm'] = np.linalg.norm(gradients, axis=1)

    # prepare for penalization
    grad_norms = np.linalg.norm(gradients, axis=1)
    unit_grads = gradients / (grad_norms[:,None] + eps)

    # 5) RECOMPUTE near_k WITH PENALIZED DISTANCES --------------------
    # collect all modified d_mod if we need a global quantile
    all_dmod = []
    tmp_dmod = [None]*n_cells

    for i in range(n_cells):
        neigh_far    = neighbor_indices_far[i]
        dist_far_i   = neighbor_distances_far[i]
        g_u          = unit_grads[i]
        g_norm       = grad_norms[i]

        # unit vectors toward each far-neighbor
        dx = X_scaled[neigh_far,0] - X_scaled[i,0]
        dy = X_scaled[neigh_far,1] - X_scaled[i,1]
        uv = np.stack((dx, dy), axis=1) / (dist_far_i[:,None] + eps)

        # positive alignment penalty
        align   = uv.dot(g_u)                # [-1 .. 1]
        penalty = np.maximum(0, np.abs(align)) * g_norm
        d_mod   = dist_far_i + elev_weight * penalty

        tmp_dmod[i] = d_mod
        all_dmod.append(d_mod)

    if near_quantile is not None:
        global_near_thresh = np.quantile(np.hstack(all_dmod), near_quantile)
    else:
        global_near_thresh = None

    neighbor_indices_near = [None]*n_cells
    neighbor_distances_near = [None]*n_cells

    for i in range(n_cells):
        neigh_far = neighbor_indices_far[i]
        d_mod     = tmp_dmod[i]

        # apply global threshold if requested
        if global_near_thresh is not None:
            mask = (d_mod <= global_near_thresh)
            idxs = np.where(mask)[0]
            if idxs.size > near_k:
                order = np.argsort(d_mod[mask])[:near_k]
                sel   = idxs[order]
            else:
                sel   = idxs
        else:
            sel = np.argsort(d_mod)[:near_k]

        neighbor_indices_near[i]   = neigh_far[sel]
        neighbor_distances_near[i] = d_mod[sel]

    # 6) NEIGHBOR‐TYPE COUNTS & VECTORS -------------------------------
    counts    = np.zeros((n_cells, n_types), dtype=int)
    type_vecs = np.zeros((n_cells, n_types, 2), dtype=float)

    for i in range(n_cells):
        near = neighbor_indices_near[i]
        far  = neighbor_indices_far[i]
        dfar = neighbor_distances_far[i]
        t_near = cell_types[near]
        t_far  = cell_types[far]

        dx_far = X_scaled[far,0] - X_scaled[i,0]
        dy_far = X_scaled[far,1] - X_scaled[i,1]

        for t, j in type_to_idx.items():
            mask_n = (t_near == t)
            counts[i,j] = mask_n.sum()

            mask_f = (t_far == t)
            c_f = mask_f.sum()
            if c_f > 0:
                md = dfar[mask_f].mean()
                uv = np.stack((dx_far[mask_f], dy_far[mask_f]), axis=1) \
                     / dfar[mask_f][:,None]
                muv = uv.mean(axis=0)
                norm = np.linalg.norm(muv)
                type_vecs[i,j] = (muv/norm)*md if norm > eps else 0.0

    neighbor_counts = pd.DataFrame(
        counts,
        columns=[f"{t}_count" for t in unique_types],
        index=df.index,
    )
    flat = type_vecs.reshape(n_cells, -1)
    cols = [f"{t}_{ax}" for t in unique_types for ax in ("vec_x","vec_y")]
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


