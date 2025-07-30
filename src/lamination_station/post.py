import numpy as np
import networkx as nx
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata
from sklearn.manifold import SpectralEmbedding

def build_structure_graph(out):
    '''
        build a simple structure graph from a lamination_station output dataframe 
        which has columns structure_pred1/2 where weights are the overlap coefficient between two structures
    '''
    top2_arr = np.stack([out['structure_pred1'].astype(int),out['structure_pred2'].astype(int)],axis=-1)
    structs = set(out['structure_pred1'])|set(out['structure_pred2'])
    N, S = top2_arr.shape[0], top2_arr.max()+1
    sets = {s: set(np.where((top2_arr == s).any(axis=1))[0]) for s in range(S)}

    G = nx.Graph()
    G.add_nodes_from(range(S))
    for u in range(S):
        for v in range(u+1, S):
            A, B = sets[u], sets[v]
            if not A or not B:
                continue
            # overlap coefficient
            w = len(A & B) / min(len(A), len(B)) # len(A | B)#
            if w > 0:
                G.add_edge(u, v, weight=w)
    return G

def _node_diffusion_shortest_path(G, source, weight='weight'):
    # 1) get weighted shortest‐path lengths
    dists = nx.single_source_dijkstra_path_length(G, source, weight=weight)
    if not dists:
        # only source exists
        return {n: (1.0 if n == source else 0.0) for n in G.nodes()}
    max_d = max(dists.values())
    # 2) invert+normalize to [0,1]
    return {
        n: (1.0 - dists.get(n, np.inf) / max_d) if np.isfinite(dists.get(n, np.inf)) else 0.0
        for n in G.nodes()
    }

def _node_diffusion_pagerank(G, source, alpha=0.85, weight='weight'):
    # 1) set up personalization vector
    pers = {n: 1.0 if n == source else 0.0 for n in G.nodes()}
    # 2) run PPR
    pr = nx.pagerank(G, alpha=alpha, personalization=pers, weight=weight, max_iter=10000,tol=1.0e-5)
    vals = np.fromiter(pr.values(), float)
    lo, hi = vals.min(), vals.max()
    # 3) normalize into [0,1]
    if hi > lo:
        return {n: ((v - lo) / (hi - lo))**0.5 for n, v in pr.items()} #Added 0.5 since decay seems square/exp
    else:
        return {n: 0.0 for n in pr}

def compute_cell_diffusion(
    out: pd.DataFrame,
    G: nx.Graph,
    source_node: int,
    method: str = 'pagerank',
    **diff_kwargs
) -> pd.DataFrame:
    """
    Returns a copy of `out` with a new column `diffusion` in [0,1].

    Parameters
    ----------
    out : DataFrame
        Must have 'structure_pred1', 'structure_pred2', 'phi' columns.
    G : Graph
        Full structure‐graph with numeric node IDs.
    source_node : int
        Node in G from which to diffuse.
    method : {'shortest_path','pagerank'}
        Which node‐diffusion to use.
    diff_kwargs :
        Passed to the underlying diffusion function:
          - for shortest_path: weight='weight'
          - for pagerank: alpha=0.85, weight='weight'
    """
    # 1) compute diffusion on graph nodes
    if method == 'shortest_path':
        node_scores = _node_diffusion_shortest_path(G, source_node, **diff_kwargs)
    elif method == 'pagerank':
        node_scores = _node_diffusion_pagerank(G, source_node, **diff_kwargs)
    else:
        raise ValueError(f"Unknown method {method!r}")

    # 2) map scores to each end of the predicted‐edge
    print(node_scores)
    d1 = out['structure_pred1'].astype(int).replace(node_scores).fillna(0.0)
    d2 = out['structure_pred2'].astype(int).replace(node_scores).fillna(0.0)

    # 3) interpolate along phi
    df = out.copy()
    df['diffusion'] = df['phi'] * d1 + (1 - df['phi']) * d2
    return df


def compute_diffusion_time(
    X,
    n_neighbors: int = 50,
    use_cuda: bool = False
):
    """
    Compute diffusion pseudotime via 1D spectral embedding of a kNN graph,
    and return both the [0,1]-scaled diffusion time and the clipped, normalized ranks.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
    n_neighbors : int
        Number of nearest neighbors to use for the graph.
    use_cuda : bool
        If True, uses cuML + CuPy for kNN; otherwise uses scikit-learn on CPU.

    Returns
    -------
    diffusion_time : ndarray, shape (n_samples,)
        Values in [0,1] from the 1D spectral embedding.
    normalized_ranks : ndarray, shape (n_samples,)
        Ranks of diffusion_time, clipped at the 1st/99th percentiles and re‑scaled to [0,1].
    """
    X = np.asarray(X, dtype=np.float32)
    if use_cuda:
        import cupy as cp
        from cuml.neighbors import NearestNeighbors as cuNN
        X_gpu = cp.asarray(X)
        nn = cuNN(n_neighbors=n_neighbors, algorithm='brute')
        nn.fit(X_gpu)
        _, idx_gpu = nn.kneighbors(X_gpu)
        idx = cp.asnumpy(idx_gpu)
    else:
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto')
        nn.fit(X)
        _, idx = nn.kneighbors(X)

    n = X.shape[0]
    rows = np.repeat(np.arange(n), n_neighbors)
    cols = idx.ravel()
    data = np.ones_like(rows, dtype=np.float32)
    A = sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = (A + A.T) * 0.5

    se = SpectralEmbedding(n_components=1, affinity='precomputed')
    phi = se.fit_transform(A).ravel()

    diffusion_time = (phi - phi.min()) / (phi.max() - phi.min())

    ranks = rankdata(diffusion_time, method='average')
    normalized_ranks = (ranks - ranks.min()) / (ranks.max() - ranks.min())

    return diffusion_time, normalized_ranks

