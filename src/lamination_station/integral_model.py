import pandas as pd
import numpy as np
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam
from torch.utils.data import WeightedRandomSampler, DataLoader, TensorDataset
from torch.distributions import constraints

def safe_softmax(x,dim=-1,eps=1e-10):
    x=torch.softmax(x,dim)
    x=x+eps
    return (x/x.sum(dim,keepdim=True))

class Encoder(PyroModule):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.loc   = nn.Linear(hidden_dim, latent_dim)
        self.scale = nn.Linear(hidden_dim, latent_dim)
    def forward(self, x):
        h  = F.relu(self.fc1(x))
        h  = F.relu(self.fc2(h))
        mu = self.loc(h)
        sd = F.softplus(self.scale(h)) + 1e-4
        return mu, sd

def gaussian_log_overlap_full(mu_q, var_q_diag, mus, scale_trils, jitter=1e-6):
    """
    mu_q:       (B,D)
    var_q_diag: (B,D)
    mus:        (S,D)
    scale_trils:(S,D,D) with Sigma_k = L_k L_k^T
    returns:    (B,S) log inner-products N(mu_q; mu_k, Sigma_k + diag(var_q))
    """
    B, D = mu_q.shape
    S = mus.size(0)
    c = D * torch.log(torch.tensor(2.0 * torch.pi, device=mu_q.device))
    diagQ = torch.diag_embed(var_q_diag)  # (B,D,D)
    logw = []
    I = torch.eye(D, device=mu_q.device).expand(B, D, D)
    for k in range(S):
        Sigma_k = scale_trils[k] @ scale_trils[k].transpose(-1, -2)     # (D,D)
        S_b = Sigma_k.expand(B, D, D) + diagQ                           # (B,D,D)
        S_b = 0.5*(S_b + S_b.transpose(-1, -2)) + jitter*I
        L = torch.linalg.cholesky(S_b)                                  # (B,D,D)
        logdet = 2.0*torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)  # (B,)
        d = (mu_q - mus[k].expand(B, D)).unsqueeze(-1)                  # (B,D,1)
        y = torch.linalg.solve_triangular(L, d, upper=False)
        z = torch.linalg.solve_triangular(L.transpose(-1, -2), y, upper=True)
        quad = (d.squeeze(-1) * z.squeeze(-1)).sum(-1)                  # (B,)
        logw.append(-0.5*(c + logdet + quad))
    return torch.stack(logw, dim=1)  # (B,S)

def run_model(
    df_grads,
    cell_types_series,
    neighbor_counts,
    gradients,
    neighbor_type_vecs,
    LATENT_DIM=2,
    NUM_STRUCTURES=25,
    num_epochs=2000,
    lr_steps=(1e-3, 1e-4, 1e-5),
    batch_size=1024,
    OBS_FAMILY="nb",
    STRUCT_LOC_PRIOR_SCALE=1.0,
    LOSS_SCALE=1.0,
    HIDDEN_DIM=512,
    device="cpu",
    clear_params=True,
    normalize_sampling=True,
    num_particles=1,
    ignore_cur_type=False,
):
    if clear_params:
        pyro.clear_param_store()
    pyro.set_rng_seed(0)

    counts = torch.tensor(neighbor_counts.values, dtype=torch.float, device=device).nan_to_num_(0.0)
    comp   = counts / counts.sum(-1, keepdim=True).clamp_min(1.0)
    grads  = torch.tensor(gradients, dtype=torch.float, device=device).nan_to_num_(0.0)

    N, twoT = neighbor_type_vecs.shape
    T = counts.shape[1]
    vecs_np = neighbor_type_vecs.values.reshape(N, T, 2)
    type_vecs = torch.tensor(vecs_np, dtype=torch.float, device=device).nan_to_num_(0.0)
    grad_type_cos = F.cosine_similarity(grads.unsqueeze(1), type_vecs, dim=-1)

    cell_types   = cell_types_series.values
    unique_types = cell_types_series.cat.categories
    id_dim       = len(unique_types)
    id_arr       = np.zeros((N, id_dim), dtype=float)
    type_to_ix   = {t:i for i,t in enumerate(unique_types)}
    for i,t in enumerate(cell_types):
        id_arr[i, type_to_ix[t]] = 1.0 if not ignore_cur_type else 0.0
    id_b = torch.tensor(id_arr, dtype=torch.float, device=device)

    labels       = np.array([type_to_ix[t] for t in cell_types])
    class_counts = np.bincount(labels) + 200
    sample_weights = (1.0 / class_counts[labels]) if normalize_sampling else np.ones_like(labels, dtype=float)
    sample_weights = torch.from_numpy(sample_weights).double().to(device)

    dataset = TensorDataset(id_b, grads, type_vecs, counts, grad_type_cos)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    loader  = DataLoader(dataset, batch_size=batch_size, sampler=sampler, drop_last=True)

    input_dim = T + id_dim + grads.shape[1] + T + 1
    encoder = Encoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    pyro.module("encoder", encoder)

    def model(id_b, grad_b, vecs_b, counts_b, cos_b):
        with pyro.poutine.scale(scale=LOSS_SCALE):
            B = counts_b.size(0)
            D, S, Tloc = LATENT_DIM, NUM_STRUCTURES, counts_b.size(1)

            struct_loc = pyro.sample(
                "struct_loc",
                dist.Laplace(torch.zeros(S, D, device=device),
                             STRUCT_LOC_PRIOR_SCALE*torch.ones(S, D, device=device)).to_event(2)
            )
            # full covariances via lower-triangular factors (MAP in guide)
            struct_L = pyro.sample(
                "struct_L",
                dist.LKJCholesky(D, 2.0*torch.ones(1,device=device)).expand([S]).to_event(1)
            )
            struct_scales = pyro.sample(
                "struct_scales",
                dist.HalfCauchy(0.5*torch.ones(S, D, device=device)).to_event(2)
            )

            struct_comp_logits = pyro.param("struct_comp_logits",
                                            0.01*torch.randn(S, Tloc, device=device))
            theta_a = pyro.param('theta_a', torch.zeros(1, device=device))
            theta_b = pyro.param('theta_b', 100*torch.ones((1, Tloc), device=device))

            with pyro.plate("cells", B):
                phi = pyro.sample("phi", dist.Dirichlet(torch.ones(S, device=device)))
                comp_probs = safe_softmax(struct_comp_logits, dim=-1)    # (S,T)
                mix_probs  = phi @ comp_probs                             # (T,)
                tot = counts_b.sum(-1, keepdim=True).clamp_min(1.0)
                if OBS_FAMILY == "nb":
                    theta = F.softplus(theta_b + theta_a*vecs_b.norm(dim=-1).mean(dim=-1, keepdim=True)) + 1e-6
                    logits = torch.log(mix_probs.clamp_min(1e-12)) + torch.log(tot) - torch.log(theta)
                    pyro.sample("obs", dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1), obs=counts_b)
                elif OBS_FAMILY == "poisson":
                    rate = mix_probs * tot
                    pyro.sample("obs", dist.Poisson(rate=rate).to_event(1), obs=counts_b)
                elif OBS_FAMILY == "multinomial":
                    pyro.sample("obs", dist.Multinomial(total_count=int(tot.max().item()), probs=mix_probs), obs=counts_b)
                else:
                    raise ValueError(OBS_FAMILY)

    def guide(id_b, grad_b, vecs_b, counts_b, cos_b):
        with pyro.poutine.scale(scale=LOSS_SCALE):
            B = counts_b.size(0)
            D, S, Tloc = LATENT_DIM, NUM_STRUCTURES, counts_b.size(1)

            struct_loc_param = pyro.param("struct_loc_param", 0.1*torch.randn(S, D, device=device))
            pyro.sample("struct_loc", dist.Delta(struct_loc_param).to_event(2))

            struct_L_param = pyro.param("struct_L_param",
                                        torch.eye(D, device=device).expand(S, D, D).clone(),
                                        constraint=constraints.lower_cholesky)
            pyro.sample("struct_L", dist.Delta(struct_L_param).to_event(3))

            struct_scales_param = pyro.param("struct_scales_param",
                                             0.5*torch.ones(S, D, device=device),
                                             constraint=constraints.positive)
            pyro.sample("struct_scales", dist.Delta(struct_scales_param).to_event(2))

            comp_b = counts_b / counts_b.sum(-1, keepdim=True).clamp_min(1.0)
            mean_dist = vecs_b.norm(dim=-1).mean(dim=-1, keepdim=True)
            x_enc = torch.cat([comp_b, id_b, grad_b, cos_b, mean_dist], dim=-1)
            mu_q, sd_q = encoder(x_enc)
            var_q = sd_q**2

            scale_tril = torch.matmul(torch.diag_embed(struct_scales_param), struct_L_param)  # (S,D,D)
            logw = gaussian_log_overlap_full(mu_q, var_q, struct_loc_param, scale_tril)       # (B,S)
            phi = safe_softmax(logw, dim=-1)
            with pyro.plate("cells", B):
                pyro.sample("phi", dist.Delta(phi).to_event(1))

    if clear_params:
        pyro.clear_param_store()

    losses = []
    for lr in lr_steps:
        svi = SVI(model, guide, Adam({"lr": lr}), loss=JitTrace_ELBO(num_particles=num_particles))
        for _ in tqdm.tqdm(range(num_epochs), desc=f"lr={lr}"):
            for id_b_, grad_b, vecs_b, counts_b, cos_b in loader:
                loss = svi.step(id_b_, grad_b, vecs_b, counts_b, cos_b)
                losses.append(loss)

    # ── Minibatch inference over all N
    encoder.eval()
    with torch.no_grad():
        N = counts.size(0)
        mean_dist_full = type_vecs.norm(dim=-1).mean(dim=-1, keepdim=True)

        idx_all = torch.arange(N, device=device)
        infer_set = TensorDataset(idx_all, id_b, grads, type_vecs, counts, grad_type_cos, mean_dist_full)
        infer_loader = DataLoader(infer_set, batch_size=batch_size, shuffle=False, drop_last=False)

        S = NUM_STRUCTURES
        Tloc = counts.shape[1]
        z_mu_np   = np.zeros((N, LATENT_DIM), dtype=np.float32)
        z_sd_np   = np.zeros((N, LATENT_DIM), dtype=np.float32)
        phi_np    = np.zeros((N, S), dtype=np.float32)
        mix_np    = np.zeros((N, Tloc), dtype=np.float32)
        pred1_np  = np.zeros(N, dtype=np.int64)
        pred2_np  = np.zeros(N, dtype=np.int64)
        phi1_np   = np.zeros(N, dtype=np.float32)

        struct_loc = pyro.param("struct_loc_param")
        struct_L   = pyro.param("struct_L_param")
        struct_sc  = pyro.param("struct_scales_param")
        scale_tril = torch.matmul(torch.diag_embed(struct_sc), struct_L)  # (S,D,D)
        comp_probs = torch.softmax(pyro.param("struct_comp_logits"), dim=-1)  # (S,T)

        for idx_b, id_b_, grad_b, vecs_b, counts_b, cos_b, md_b in infer_loader:
            comp_b = (counts_b / counts_b.sum(-1, keepdims=True).clamp_min(1.0)).nan_to_num(0.0)
            X_b    = torch.cat([comp_b, id_b_, grad_b, cos_b, md_b], dim=1)
            mu_q, sd_q = encoder(X_b)
            var_q = sd_q**2

            logw = gaussian_log_overlap_full(mu_q, var_q, struct_loc, scale_tril)  # (B,S)
            phi_b = torch.softmax(logw, dim=-1)                                     # (B,S)
            mix_b = phi_b @ comp_probs                                               # (B,T)

            top2_p_b, top2_idx_b = phi_b.topk(2, dim=1)
            i = idx_b.detach().cpu().numpy()
            z_mu_np[i]  = mu_q.detach().cpu().numpy()
            z_sd_np[i]  = var_q.sqrt().detach().cpu().numpy()
            phi_np[i]   = phi_b.detach().cpu().numpy()
            mix_np[i]   = mix_b.detach().cpu().numpy()
            pred1_np[i] = top2_idx_b[:, 0].detach().cpu().numpy()
            pred2_np[i] = top2_idx_b[:, 1].detach().cpu().numpy()
            phi1_np[i]  = (top2_p_b[:, 0] / top2_p_b.sum(dim=1)).detach().cpu().numpy()

    out = df_grads.copy()
    for d in range(LATENT_DIM):
        out[f"z{d+1}"] = z_mu_np[:, d]
        out[f"zsd{d+1}"] = z_sd_np[:, d]
    out["structure_pred1"] = pd.Categorical(pred1_np)
    out["structure_pred2"] = pd.Categorical(pred2_np)
    out["phi"] = phi1_np

    return (
        out,                 # dataframe with z and top-2
        phi_np,              # (N,S) overlap-simplex per cell
        torch.softmax(pyro.param("struct_comp_logits"), dim=-1).detach().cpu().numpy(),  # (S,T) cluster→type probs
        mix_np,              # (N,T) predicted composition
        losses,
    )
