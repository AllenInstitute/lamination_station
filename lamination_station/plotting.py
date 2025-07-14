import networkx as nx
import seaborn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def clip_latent_dimensions(matrix, x):
    """
    Clips each latent dimension of the matrix at the 0+x and 100-x percentile.

    Parameters:
    - matrix: A 2D NumPy array of shape [number of observations, latent dimensions].
    - x: The percentage for the lower and upper bounds (0 < x < 50).

    Returns:
    - A 2D NumPy array with the same shape as the input matrix, with values clipped.
    """
    # Ensure x is within the valid range
    if x < 0 or x > 50:
        raise ValueError("x must be between 0 and 50")

    # Initialize a clipped matrix with the same shape as the input matrix
    clipped_matrix = np.zeros_like(matrix)

    # Iterate over each column (latent dimension) to apply clipping
    for col_idx in range(matrix.shape[1]):
        # Calculate the percentiles for the current column
        lower_percentile = np.percentile(matrix[:, col_idx], x)
        upper_percentile = np.percentile(matrix[:, col_idx], 100-x)
        
        # Clip the values in the current column based on the calculated percentiles
        clipped_matrix[:, col_idx] = np.clip(matrix[:, col_idx], lower_percentile, upper_percentile)

    return clipped_matrix

def plot_cells_on_structure_graph(out,
                                  G,
                                  jitter=0.05,
                                  cell_size=20,
                                  cell_alpha=0.6,
                                  color_col='Group',
                                  figsize=(8,8),
                                  spring_k=None,
                                  edge_scale=5.0,
                                  seed=0):
    """
    Plots a force‐directed layout of the subset of structures
    that occur in out[['structure_pred1','structure_pred2']],
    then scatters each cell along its predicted‐edge position,
    colored by out[color_col].  Edges are drawn with width and
    alpha proportional to their G.edges[u,v]['weight'].

    Parameters
    ----------
    out : pandas.DataFrame
        Must contain columns
          - 'structure_pred1', 'structure_pred2', 'phi'
          - color_col (categorical or string)
    G : networkx.Graph
        Full graph of all structures; we'll induce the subgraph.
    jitter : float
        Std-dev of Gaussian noise added to each point (in layout coords).
    cell_size : int
        Marker size for cell points.
    cell_alpha : float
        Alpha for cell points.
    color_col : str
        Column in `out` to use for point hue.
    figsize : tuple
        Figure size passed to plt.subplots.
    spring_k : float or None
        Optimal node spacing for nx.spring_layout; if None, auto.
    edge_scale : float
        Multiplier for edge line‐width.
    seed : int
        Random seed for deterministic layout.
    """
    # 1) Induce subgraph on only used structure‐nodes
    used = set(out['structure_pred1']).union(out['structure_pred2'])
    G_sub = G.subgraph(used).copy()

    # 2) Compute a force‐directed (spring) layout
    pos = nx.spring_layout(G_sub, k=spring_k, seed=seed)

    fig, ax = plt.subplots(figsize=figsize)

    # 3) Draw edges individually, scaled by weight
    for u, v, data in G_sub.edges(data=True):
        w = data.get('weight', 1.0)
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot(
            [x1, x2], [y1, y2],
            color='gray',
            linewidth=w * edge_scale,
            alpha=w
        )

    # 4) Draw nodes & labels
    nx.draw_networkx_nodes(
        G_sub, pos, ax=ax,
        node_color='lightblue',
        node_size=300
    )
    nx.draw_networkx_labels(G_sub, pos, ax=ax)

    # 5) Compute each cell's (x,y) along its top‐2 edge
    coords1 = np.array([pos[n] for n in out['structure_pred1']])
    coords2 = np.array([pos[n] for n in out['structure_pred2']])
    phi     = out['phi'].values[:, None]
    pts     = phi * coords1 + (1 - phi) * coords2

    # 6) Add jitter
    pts += np.random.randn(*pts.shape) * jitter

    # 7) Scatter, coloring by `color_col`
    cell_df = out.copy()
    cell_df['x_graph'], cell_df['y_graph'] = pts[:,0], pts[:,1]

    seaborn.scatterplot(
        ax=ax,
        data=cell_df,
        x='x_graph',
        y='y_graph',
        hue=color_col,
        s=cell_size,
        alpha=cell_alpha,
        palette='tab20',
        legend='full'
    )

    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def plot_loss(loss_tracker):
    '''Plots vector of values along with moving average'''
    seaborn.scatterplot(x=list(range(len(loss_tracker))),y=loss_tracker,alpha=0.5,s=2)
    w=300
    mvavg=moving_average(np.pad(loss_tracker,int(w/2),mode='edge'),w)
    seaborn.lineplot(x=list(range(len(mvavg))),y=mvavg,color='coral')
    plt.show()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_gradient_arrows(df_grads,gradients,taxon,x_col,y_col,xlim,ylim):
    mask = np.array([True for x in df_grads.index])
    x = df_grads.loc[mask, x_col]
    y = df_grads.loc[mask, y_col]
    
    # 3) compute alphas from your gradient_norm
    norms = df_grads['gradient_norm']
    alphas = norms / norms.max()
    
    # 4) build your color mapping off of df_grads
    unique_types = df_grads.loc[mask, taxon].unique()
    cmap = list(sc.pl.palettes.godsnot_102) + list(sc.pl.palettes.godsnot_102)
    color_map = {ctype: cmap[i] for i, ctype in enumerate(unique_types)}
    colors = df_grads.loc[mask, taxon].map(color_map)
    
    # 5) now plot
    plt.figure(figsize=(6,6))
    plt.scatter(x, y,
                c=colors,
                s=0.1,
                rasterized=True)
    
    # use the returned `gradients` array directly
    gx = gradients[mask, 0]
    gy = gradients[mask, 1]
    
    plt.quiver(x, y,
               gx, gy,
               color=colors,
               alpha=alphas,
               angles='xy',
               scale_units='xy',
               scale=2.)
    
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(x_col)
    plt.ylabel(y_col)


