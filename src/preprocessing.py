import sklearn
import numpy as np
import pandas as pd

def calculate_comp_grads(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    celltype_col: str,
    k: int = 5,
    eps: float = 1e-2,
    n_bins: int = 36,
    absgrad: bool = True,
    use_cuda: bool = False,
    grad_clip: float = 0.9
):
    """
    If use_cuda=True, uses cuML StandardScaler, NearestNeighbors, PCA on GPU,
    then brings X_scaled, dists, idx back as NumPy arrays for the rest of the work.
    """

    # 1) SCALE + KNN ---------------------------------------------------
    if use_cuda:
        import cupy as cp
        from cuml.preprocessing import StandardScaler as cuStandardScaler
        from cuml.neighbors import NearestNeighbors as cuNearestNeighbors

        # raw coords → GPU
        X_cpu = df[[x_col, y_col]].values
        X_cp  = cp.asarray(X_cpu)

        # GPU scale
        scaler = cuStandardScaler()
        X_scaled_cp = scaler.fit_transform(X_cp)

        # GPU KNN
        knn = cuNearestNeighbors(n_neighbors=k)
        knn.fit(X_scaled_cp)
        dists_cp, idx_cp = knn.kneighbors(X_scaled_cp)

        # back to CPU NumPy
        X_scaled = cp.asnumpy(X_scaled_cp)
        dists    = cp.asnumpy(dists_cp)
        idx      = cp.asnumpy(idx_cp)

    else:
        from sklearn.preprocessing import StandardScaler
        from sklearn.neighbors    import NearestNeighbors

        X_cpu = df[[x_col, y_col]].values
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_cpu)

        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(X_scaled)
        dists, idx = knn.kneighbors(X_scaled)

    # drop self
    distances = dists[:, 1:]
    indices   = idx[:, 1:]

    # 2) SETUP ----------------------------------------------------------
    df[celltype_col] = df[celltype_col].astype('category')
    cell_types   = df[celltype_col].values
    unique_types = df[celltype_col].cat.categories
    n_cells      = X_scaled.shape[0]
    n_types      = unique_types.size
    type_to_idx  = {t: i for i, t in enumerate(unique_types)}

    # 3) NEIGHBOR‐TYPE VECTORS & COUNTS ---------------------------------
    counts    = np.zeros((n_cells, n_types), dtype=int)
    type_vecs = np.full((n_cells, n_types, 2), np.nan, dtype=float)

    for i in range(n_cells):
        neigh      = indices[i]
        neigh_t    = cell_types[neigh]
        neigh_dist = distances[i]
        dx_all     = X_scaled[neigh, 0] - X_scaled[i, 0]
        dy_all     = X_scaled[neigh, 1] - X_scaled[i, 1]

        for t, j in type_to_idx.items():
            mask = (neigh_t == t)
            c = mask.sum()
            counts[i, j] = c
            if c > 0:
                md = neigh_dist[mask].mean()
                uv = np.column_stack((dx_all[mask], dy_all[mask])) / neigh_dist[mask][:, None]
                muv = uv.mean(axis=0)
                nv  = np.linalg.norm(muv)
                if nv > eps:
                    muv /= nv
                    type_vecs[i, j] = muv * md
                else:
                    type_vecs[i, j] = 0.0

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
        comp_vectors[i] = one_hot[indices[i]].mean(axis=0)

    if use_cuda:
        from cuml.decomposition import PCA as cuPCA
        import cupy as cp

        comp_cp = cp.asarray(comp_vectors)
        pca = cuPCA(n_components=min(10, n_types))
        vp = pca.fit_transform(comp_cp)
        comp_vectors_pca = cp.asnumpy(vp)

    else:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(10, n_types))
        comp_vectors_pca = pca.fit_transform(comp_vectors)

    # 5) GRADIENTS ------------------------------------------------------
    gradients = np.zeros((n_cells, 2), float)
    for i in range(n_cells):
        neigh     = indices[i]
        neigh_cv  = comp_vectors[neigh]
        dx        = X_scaled[neigh, 0] - X_scaled[i, 0]
        dy        = X_scaled[neigh, 1] - X_scaled[i, 1]

        ang  = np.arctan2(dy, dx)
        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
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
        gx = (np.abs(dV) * w[:, None] / dx[:, None]).sum()
        gy = (np.abs(dV) * w[:, None] / dy[:, None]).sum()
        gradients[i, 0] = gx
        gradients[i, 1] = np.abs(gy) if absgrad else gy

    df_grads = df.copy()
    norms = np.nan_to_num(np.linalg.norm(gradients, axis=1),0)
    df_grads['gradient_norm'] = norms
    if grad_clip is not None:    
        # 2) find the 90th‐percentile threshold
        threshold = np.quantile(norms, grad_clip)
        
        # 3) compute a per‐vector scaling factor ≤1
        #    (if norm==0, we leave it at 1 to avoid division by zero)
        scales = np.where(norms > 0,
                          np.minimum(1.0, threshold / norms),
                          1.0)
        
        # 4) apply the scaling
        gradients = np.nan_to_num(gradients * scales[:, None],0)
        
        # 5) (optional) update your df_grads gradient_norm
        clipped_norms = np.linalg.norm(gradients, axis=1)
        df_grads['gradient_norm'] = clipped_norms
        
    return (
        df_grads,
        df[celltype_col],
        gradients,
        comp_vectors_pca,
        comp_vectors,
        neighbor_counts,
        neighbor_type_vecs,
    )

