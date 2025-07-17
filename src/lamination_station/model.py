import pandas as pd
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
import pyro.infer as infer
import pyro.optim as optim
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam

def safe_softmax(x,dim=-1,eps=1e-10):
    x=torch.softmax(x,dim)
    x=x+eps
    return (x/x.sum(dim,keepdim=True))

class Encoder(PyroModule):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.loc   = nn.Linear(hidden_dim, latent_dim)
        self.scale = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        μ = self.loc(h)
        σ = F.softplus(self.scale(h)) + 1e-3
        return μ, σ

class Classifier(PyroModule):
    def __init__(self, latent_dim, hidden_dim, num_structures):
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_structures)
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return self.out(h)

class Decoder(PyroModule):
    def __init__(self, latent_dim, comp_dim, id_dim, hidden_dim, out_dim):
        super().__init__()
        inp = latent_dim + comp_dim + id_dim + comp_dim + 1
        self.fc1 = nn.Linear(inp, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)
    def forward(self, z, comp, cid, cos_grad, mean_dist):
        x = torch.cat([z, comp, cid, cos_grad, mean_dist], dim=-1)
        h = F.relu(self.fc1(x))
        return self.out(h)

def run_model(
    df_grads, #output from calculate_comp_grads
    cell_types_series, #pandas categorical series
    neighbor_counts,
    gradients, #output from calculate_comp_grads
    neighbor_type_vecs,
    LATENT_DIM     = 2,
    NUM_STRUCTURES = 25,
    num_epochs=2000,
    lr_steps = [1e-3,1e-4,1e-5],
    batch_size     = 1024,
    STRUCT_SCALE_SCALE = 0.1, #Strength of regularization on latent space dispersion
    OBS_FAMILY = "nb", #must be 'nb','poisson' or 'multinomial'
    STRUCT_LOC_PRIOR_SCALE = 1., #Strength of regularization on more structures
    LOSS_SCALE = 1., #Multiply all ELBO by this factor (can increase stability if too high or too low)
    HIDDEN_DIM = 512,
    HIDDEN_DIM_CLASSIFIER = 128,
    device = 'cpu',
    clear_params = True,
    normalize_sampling = True
):
    """
    Train a Pyro-based variational model to infer latent spatial structures
    from per-cell gradient features and categorical cell identities.

    This function takes outputs from  builds the dataloader, fits the model and retrieves the outputs in a single step.

    Parameters
    ----------
    df_grads : pandas.DataFrame
        DataFrame returned by `calculate_comp_grads`, containing per-cell
        component gradients and related metadata.
    cell_types_series : pandas.Categorical
        Series of categorical cell-type labels corresponding to rows of
        `df_grads`.
    gradients : array-like, shape (N, D)
        Gradient vectors per cell (also returned by `calculate_comp_grads`).
    neighbor_counts : array-like, shape (N, T)
        Composition of neighbors per cell (also returned by `calculate_comp_grads`).
    neighbor_type_vecs : array-like, shape (N, T, 2)
        Mean vector from each cell to neighbor types (also returned by `calculate_comp_grads`).
    LATENT_DIM : int, default=2
        Dimensionality of the latent embedding space (z).
    NUM_STRUCTURES : int, default=25
        Number of latent structures (clusters) to infer.
    num_epochs : int, default=2000
        Number of training epochs per entry in `lr_steps`.
    lr_steps : list of float, default=[1e-3, 1e-4, 1e-5]
        Sequence of learning rates for staged training.
    batch_size : int, default=1024
        Number of samples per mini-batch for SVI.
    STRUCT_SCALE_SCALE : float, default=0.1
        Scale parameter for the HalfCauchy prior on structure dispersion.
    OBS_FAMILY : {'nb', 'poisson', 'multinomial'}, default='nb'
        Observation likelihood for count data.
    STRUCT_LOC_PRIOR_SCALE : float, default=1.0
        Scale parameter for the Laplace prior on structure locations.
    LOSS_SCALE : float, default=1.0
        Factor to scale the ELBO loss for numerical stability.
    HIDDEN_DIM : int, default=512
        Hidden layer size for encoder and decoder networks.
    HIDDEN_DIM_CLASSIFIER : int, default=128
        Hidden layer size for the classifier network.
    device : {'cpu', 'cuda'}, default='cpu'
        Compute device for tensors and model parameters.
    clear_params : bool, default=True
        Whether to clear the Pyro parameter store before fitting.
    normalize_sampling : bool, default=True
        Whether to sample each cell type equally or in the existing proportion.

    Returns
    -------
    out : pandas.DataFrame
        Copy of `df_grads` augmented with:
        - `"z1"`, …, `"z{LATENT_DIM}"`: posterior means of latent embeddings,
        - `"structure_pred1"`, `"structure_pred2"`: top-2 inferred structures,
        - `"phi"`: mixture weight for the top structure,
        - `"structure_pred_nearest"`: assignment by nearest structure center.
    logits_s : numpy.ndarray, shape (N, NUM_STRUCTURES)
        Raw classifier logits for each cell over all structures.
    struct_comp_probs : numpy.ndarray, shape (NUM_STRUCTURES, T)
        Learned, softmax-normalized structure composition probabilities
        (from `pyro.param('struct_comp')`).

    """

    # ──────────────── Prepare Data & One-Hot IDs ────────────────
    # comp_vectors, gradients, neighbor_type_vecs (DataFrame),
    # neighbor_counts (DataFrame), and df already in scope.
    
    counts = torch.tensor(neighbor_counts.values, dtype=torch.float, device=device,requires_grad=False).nan_to_num(0.)
    comp = counts / counts.sum(-1,keepdims=True)
    grads = torch.tensor(gradients, dtype=torch.float, device=device,requires_grad=False).nan_to_num(0.)
    
    # reshape neighbor_type_vecs → (N, T, 2)
    N, twoT = neighbor_type_vecs.shape
    T       = counts.shape[1]
    vecs_np = neighbor_type_vecs.values.reshape(N, T, 2)
    type_vecs = torch.tensor(vecs_np, dtype=torch.float, device=device,requires_grad=False).nan_to_num(0.)
    
    # precompute cosine between grad & type vectors
    grad_type_cos = F.cosine_similarity(
        grads.unsqueeze(1),
        type_vecs,
        dim=-1
    ).to(device)
    
    
    # one-hot cell identity
    cell_types   = cell_types_series.values
    unique_types = cell_types_series.cat.categories
    id_dim       = len(unique_types)
    id_arr       = np.zeros((N, id_dim), dtype=float)
    type_to_ix   = {t:i for i,t in enumerate(unique_types)}
    for i,t in enumerate(cell_types):
        id_arr[i, type_to_ix[t]] = 1.0
    id_b = torch.tensor(id_arr, dtype=torch.float, device=device,requires_grad=False)
    
    # ──────────────── DataLoader ────────────────
    # dataset = TensorDataset(id_b, grads, type_vecs, counts, grad_type_cos)
    # loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    
    labels       = np.array([type_to_ix[t] for t in cell_types])  # shape (N,)
    class_counts = np.bincount(labels)+200                  #Add 200 so super rare don't get way oversampled
    # weight for sample i = 1 / count_of_class(labels[i])
    sample_weights = 1.0 / class_counts[labels] if normalize_sampling else  class_counts[labels]/class_counts[labels]
    sample_weights = torch.from_numpy(sample_weights).double().to(device)
    
    # 3) make a WeightedRandomSampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,   # a Tensor of length N
        num_samples=len(sample_weights),  
        replacement=True          # allow resampling the same index in an epoch
    )
    
    # 4) DataLoader with sampler instead of shuffle
    dataset = TensorDataset(id_b, grads, type_vecs, counts, grad_type_cos)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        drop_last=True
    )
    
    
    # ──────────────── Build & Register Modules ────────────────
    input_dim = T + id_dim + id_dim + 2 + 1
    if clear_params:
        pyro.clear_param_store()
    encoder    = Encoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    classifier = Classifier(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM_CLASSIFIER, num_structures=NUM_STRUCTURES).to(device)
    decoder    = Decoder(latent_dim=LATENT_DIM, comp_dim=T, id_dim=id_dim,
                         hidden_dim=HIDDEN_DIM, out_dim=T).to(device)
    
    pyro.module("encoder",    encoder)
    pyro.module("classifier", classifier)
    pyro.module("decoder",    decoder)
    
    def unified_model(id_b, grad_b, vecs_b, counts_b, cos_b):
        '''gpgmm style NB'''
        with pyro.poutine.scale(scale=LOSS_SCALE):
            comp_b = (counts_b / counts_b.sum(-1,keepdims=True)).detach()
            B, T = comp_b.shape
            S, D = NUM_STRUCTURES, LATENT_DIM
        
            # 1) struct_loc & struct_scale as before
            struct_loc   = pyro.param("struct_loc",   0.1*torch.randn(S, D, device=device))
            struct_loc = pyro.sample("struct_loc_sample",
                                       dist.Laplace(torch.zeros(S, D, device=device),STRUCT_LOC_PRIOR_SCALE*torch.ones(S, D, device=device))
                                           .to_event(2))
            struct_scale = pyro.sample("struct_scale_sample",
                                       dist.HalfCauchy(STRUCT_SCALE_SCALE*torch.ones(S, D, device=device))
                                           .to_event(2))
            struct_comp  = pyro.param("struct_comp", 0.1*torch.randn(S, T, device=device))
            theta_a = pyro.param('theta_a',torch.zeros(1,device=device))
            theta_b = pyro.param('theta_b',100*torch.ones((1,T),device=device))
            
            with pyro.plate("cells", B):
                # — sample structure & z as before —
                s  = pyro.sample("structure",
                                 dist.Categorical(torch.ones(B, S, device=device)),
                                 infer={"enumerate":"parallel"})
                μ_s = struct_loc[s]
                σ_s = struct_scale[s]
                z   = pyro.sample("z",
                                  dist.Normal(μ_s, σ_s+1e-6).to_event(1))
        
                mean_dist = vecs_b.norm(dim=-1).mean(dim=-1, keepdim=True)
                rate      = decoder(z, comp_b, id_b, cos_b, mean_dist)
                theta = F.softplus(theta_b + theta_a*(mean_dist)) + 1e-6
        
                if OBS_FAMILY=='nb':
                    out_dist = dist.NegativeBinomial(total_count=theta,logits=rate - torch.logsumexp(rate,-1,keepdim=True) + counts_b.sum(-1).unsqueeze(-1).log() - theta.log()).to_event(1)
                elif OBS_FAMILY=='poisson':
                    out_dist = dist.Poisson(rate=safe_softmax(rate, dim=-1)*counts_b.sum(-1).unsqueeze(-1)).to_event(1)
                elif OBS_FAMILY=='multinomial':
                    out_dist = dist.Multinomial(total_count=int(counts_b.sum(-1).max().squeeze()), logits=rate)
                else:
                    raise ValueError(f"Unsupported OBS_FAMILY: {OBS_FAMILY!r}")
                pyro.sample("obs",
                            out_dist,
                            obs=counts_b)
        
                phi = pyro.sample("phi",
                    dist.Dirichlet(torch.ones(B, S, device=device)))
        
                # 2) pick top-2 probs and their indices
                top2_p, top2_idx = phi.topk(2, dim=1)  # both shape (B, 2)
        
                # 3) normalize to mixture weights
                mix_w = top2_p / top2_p.sum(dim=1, keepdim=True)  # (B, 2)
        
                # 4) gather the corresponding struct_comp logits → (B, 2, T)
                comp_logits = (struct_comp[top2_idx] * mix_w.unsqueeze(-1)).sum(1)
                
                if OBS_FAMILY=='nb':
                    out_dist2 = dist.NegativeBinomial(total_count=theta,logits=comp_logits - torch.logsumexp(comp_logits,-1,keepdim=True) + counts_b.sum(-1).unsqueeze(-1).log() - theta.log()).to_event(1)
                elif OBS_FAMILY=='poisson':
                    out_dist2 = dist.Poisson(rate=safe_softmax(comp_logits,dim=-1)*counts_b.sum(-1).unsqueeze(-1)).to_event(1)
                elif OBS_FAMILY=='multinomial':
                    out_dist2 = dist.Multinomial(total_count=int(counts_b.sum(-1).max().squeeze()), logits=comp_logits)
                else:
                    raise ValueError(f"Unsupported OBS_FAMILY: {OBS_FAMILY!r}")
                pyro.sample("obs_2",
                            out_dist2,
                            obs=counts_b)
    
    
    def guide(id_b, grad_b, vecs_b, counts_b, cos_b): 
        with pyro.poutine.scale(scale=LOSS_SCALE):
            comp_b = counts_b / counts_b.sum(-1,keepdims=True)
            B, T = comp_b.shape
            S, D = NUM_STRUCTURES, LATENT_DIM
            struct_loc   = pyro.param("struct_loc",   0.1*torch.randn(S, D, device=device))
            struct_loc = pyro.sample("struct_loc_sample",
                                       dist.Delta(struct_loc)
                                           .to_event(2))
            struct_scale_param = pyro.param("struct_scale",
                                            torch.ones(S, D, device=device),
                                            constraint=dist.constraints.positive)
            pyro.sample("struct_scale_sample",
                        dist.Delta(struct_scale_param).to_event(2))
        
            with pyro.plate("cells", B):
                mean_dist = vecs_b.norm(dim=-1).mean(dim=-1, keepdim=True)
                x_enc     = torch.cat([comp_b, id_b, grad_b, cos_b, mean_dist], dim=-1)
                μ_z, σ_z  = encoder(x_enc)
                z = pyro.sample("z", dist.Normal(μ_z, σ_z+1e-6).to_event(1))
        
                logits_s  = classifier(z)  
                phi_q = safe_softmax(logits_s, dim=-1)
                phi_q = phi_q+1e-6
                phi_q = (phi_q/phi_q.sum(-1)[...,None]).squeeze()
                pyro.sample("phi",
                    dist.Delta(phi_q).to_event(1))
                pyro.sample("structure",
                            dist.Categorical(logits=logits_s),
                            infer={"enumerate":"parallel"})
    
    
    if clear_params:
        pyro.clear_param_store()
    losses = []

    for lr_step in lr_steps:
        optimizer = Adam({"lr": lr_step})
        svi       = SVI(unified_model, guide, optimizer, loss=TraceEnum_ELBO())
        
        for epoch in tqdm.tqdm(range(1, num_epochs+1), desc=f"Epoch"):
            total_loss = 0.0
            for id_b_, grad_b, vecs_b, counts_b, cos_b in loader:
                # move each batch to device (they’re already on device, but safe):
                id_b_    = id_b_.to(device)
                grad_b   = grad_b.to(device)
                vecs_b   = vecs_b.to(device)
                counts_b = counts_b.to(device)
                cos_b    = cos_b.to(device)
                loss = svi.step(id_b_, grad_b, vecs_b, counts_b, cos_b)
                total_loss += loss
                losses.append(loss)
        
            if epoch % 100 ==0:
                avg_loss = total_loss / len(loader)
                print(f"[Epoch {epoch:02d}] avg ELBO loss: {avg_loss:.2f}")
    
    # ─────────────── 4) Posterior Extraction ───────────────
    # a) Compute the amortized posterior means for z via the encoder
    #    (guide assumes z ~ Normal(loc, scale), so loc is the mean)
    mean_dist_full = type_vecs.norm(dim=-1).mean(dim=-1, keepdim=True)
    X_full = torch.cat([comp, id_b, grads, mean_dist_full, grad_type_cos], dim=1)
    loc_full, scale_full = encoder(X_full)
    z_pred = loc_full.detach()                        # (N, latent_dim)

    # b) compute classifier probs and take top-2
    logits_s = classifier(z_pred)                     # (N, S)
    probs_s = torch.softmax(logits_s, dim=1)
    top2_p, top2_idx = probs_s.topk(2, dim=1)         # each is (N,2)
    # phi = p1/(p1+p2)
    phi = (top2_p[:,0] / top2_p.sum(dim=1)).cpu().detach().numpy()
    pred1 = top2_idx[:,0].cpu().detach().numpy()
    pred2 = top2_idx[:,1].cpu().detach().numpy()

    # c) nearest‐mean assignment (unchanged)
    struct_loc = pyro.param("struct_loc").detach()
    dists = torch.cdist(z_pred, struct_loc)
    struct_nearest = torch.argmin(dists, dim=1).cpu().detach().numpy()

    z_pred = loc_full.cpu().detach().numpy()                      # (N, latent_dim)
    out = df_grads.copy()
    for d in range(LATENT_DIM):
        out[f"z{d+1}"] = z_pred[:,d]
    # replace old single‐best
    # out["structure_pred"] = struct_pred
    out["structure_pred1"] = pred1#[str(x) for x in pred1]
    out["structure_pred2"] = pred2#[str(x) for x in pred2]
    out["phi"]             = phi
    out["structure_pred_nearest"] = struct_nearest#[str(x) for x in struct_nearest]
    out["structure_pred1"] = out["structure_pred1"].astype('category')
    out["structure_pred2"] = out["structure_pred2"].astype('category')
    out["structure_pred_nearest"] = out["structure_pred_nearest"].astype('category')
    return out,logits_s.cpu().detach().numpy(),torch.softmax(pyro.param('struct_comp').cpu().detach(),-1).numpy(),losses