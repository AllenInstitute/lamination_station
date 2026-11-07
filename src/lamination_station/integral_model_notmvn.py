import torch
from typing import cast
from collections.abc import Sequence
from numpy.typing import NDArray
import pandas as pd
import numpy as np
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO
from pyro.optim import Adam
from ._training import prepare_data, make_training_loader, validate_settings
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import constraints


def safe_softmax(x: torch.Tensor, dim: int = -1, eps: float = 1e-10) -> torch.Tensor:
    """Normalize probabilities with a strictly positive floor."""
    x = torch.softmax(x, dim)
    x = x + eps
    return x / x.sum(dim, keepdim=True)


class Encoder(nn.Module):
    def __init__(self, *, input_dim: int, hidden_dim: int, latent_dim: int) -> None:
        """Initialize the network dimensions."""
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.loc = nn.Linear(hidden_dim, latent_dim)
        self.scale = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate the network."""
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.loc(h)
        sd = F.softplus(self.scale(h)) + 1e-4
        return mu, sd


def rbf_log_overlap(
    mu_q: torch.Tensor,
    var_q_diag: torch.Tensor,
    struct_loc: torch.Tensor,
    lengthscales: torch.Tensor,
    eps: float = 1e-06,
) -> torch.Tensor:
    """
    Diagonal-Gaussian overlap (ARD RBF) in log-space.
    Equivalent to MVN inner product with Σ_k = diag(lengthscales_k^2),
    Σ_q = diag(var_q_diag), up to an additive constant.
    """
    # mu_q: (B,D), var_q_diag: (B,D), struct_loc: (S,D), lengthscales: (S,D)
    diff = mu_q[:, None, :] - struct_loc[None, :, :]  # (B,S,D)
    eff_var = (lengthscales**2)[None, :, :] + var_q_diag[:, None, :]  # (B,S,D)
    eff_var = eff_var.clamp_min(eps)
    quad = (diff**2 / eff_var).sum(-1)  # (B,S)
    log_norm = 0.5 * eff_var.log().sum(-1)  # (B,S)
    return cast(torch.Tensor, -0.5 * quad - log_norm)  # (B,S)


def run_model(
    df_grads: pd.DataFrame,
    cell_types_series: pd.Series,
    neighbor_counts: pd.DataFrame,
    gradients: NDArray[np.float64],
    neighbor_type_vecs: pd.DataFrame,
    LATENT_DIM: int = 2,
    NUM_STRUCTURES: int = 25,
    num_epochs: int = 2000,
    lr_steps: Sequence[float] = (0.001, 0.0001, 1e-05),
    batch_size: int = 1024,
    OBS_FAMILY: str = "nb",
    STRUCT_LOC_PRIOR_SCALE: float = 1.0,
    LOSS_SCALE: float = 1.0,
    HIDDEN_DIM: int = 512,
    device: str | torch.device = "cpu",
    clear_params: bool = True,
    normalize_sampling: bool = True,
    num_particles: int = 1,
    ignore_cur_type: bool = False,
    seed: int | None = None,
) -> tuple[
    pd.DataFrame, NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], list[float]
]:
    """Fit spatial composition structures and return per-cell summaries."""
    validate_settings(
        num_structures=NUM_STRUCTURES,
        latent_dim=LATENT_DIM,
        batch_size=batch_size,
        num_epochs=num_epochs,
        lr_steps=lr_steps,
        num_particles=num_particles,
        loss_scale=LOSS_SCALE,
        location_scale=STRUCT_LOC_PRIOR_SCALE,
        hidden_dim=HIDDEN_DIM,
    )
    data = prepare_data(
        df=df_grads,
        cell_types=cell_types_series,
        counts=neighbor_counts,
        gradients=gradients,
        type_vectors=neighbor_type_vecs,
        device=device,
        ignore_identity=ignore_cur_type,
    )
    counts, grads, type_vecs = data.counts, data.gradients, data.type_vectors
    id_b, grad_type_cos = data.identities, data.cosines
    N, T = counts.shape
    id_dim = id_b.shape[1]
    loader = make_training_loader(
        data=data,
        cell_types=cell_types_series,
        batch_size=batch_size,
        normalize_sampling=normalize_sampling,
    )
    if clear_params:
        pyro.clear_param_store()
    if seed is not None:
        pyro.set_rng_seed(seed)
    input_dim = 2 * T + id_dim + grads.shape[1] + 1
    encoder = Encoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)

    def model(
        id_b: torch.Tensor,
        grad_b: torch.Tensor,
        vecs_b: torch.Tensor,
        counts_b: torch.Tensor,
        cos_b: torch.Tensor,
        inverse_prob_b: torch.Tensor,
    ) -> None:
        """Evaluate global priors and population-scaled local terms."""
        with pyro.poutine.scale(scale=LOSS_SCALE):
            B = counts_b.size(0)
            D, S, Tloc = LATENT_DIM, NUM_STRUCTURES, counts_b.size(1)

            pyro.sample(
                "struct_loc",
                dist.Normal(
                    torch.zeros(S, D, device=device),
                    STRUCT_LOC_PRIOR_SCALE * torch.ones(S, D, device=device),
                ).to_event(2),
            )
            pyro.sample(  # log-lengthscales for positivity & heavy-tail prior
                "struct_log_ls",
                dist.Laplace(
                    torch.zeros(S, D, device=device), 0.5 * torch.ones(S, D, device=device)
                ).to_event(2),
            )

            struct_comp_logits = pyro.param(
                "struct_comp_logits", 0.01 * torch.randn(S, Tloc, device=device)
            )
            theta_a = pyro.param("theta_a", torch.zeros(1, device=device))
            theta_b = pyro.param("theta_b", 100 * torch.ones((1, Tloc), device=device))

            with pyro.plate("cells", B), pyro.poutine.scale(scale=inverse_prob_b / B):
                phi = pyro.sample("phi", dist.Dirichlet(torch.ones(S, device=device)))
                pyro.sample(
                    "mu_loc",
                    dist.Laplace(
                        torch.zeros(D, device=device),
                        STRUCT_LOC_PRIOR_SCALE * torch.ones(D, device=device),
                    ).to_event(1),
                )
                comp_probs = safe_softmax(struct_comp_logits, dim=-1)  # (S,T)
                mix_probs = phi @ comp_probs  # (T,)
                tot = counts_b.sum(-1, keepdim=True).clamp_min(1.0)
                if OBS_FAMILY == "nb":
                    theta = (
                        F.softplus(
                            theta_b + theta_a * vecs_b.norm(dim=-1).mean(dim=-1, keepdim=True)
                        )
                        + 1e-6
                    )
                    logits = (
                        torch.log(mix_probs.clamp_min(1e-12)) + torch.log(tot) - torch.log(theta)
                    )
                    pyro.sample(
                        "obs",
                        dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1),
                        obs=counts_b,
                    )
                elif OBS_FAMILY == "poisson":
                    rate = mix_probs * tot
                    pyro.sample("obs", dist.Poisson(rate=rate).to_event(1), obs=counts_b)
                elif OBS_FAMILY == "multinomial":
                    pyro.sample(
                        "obs",
                        dist.Multinomial(total_count=int(tot.max().item()), probs=mix_probs),
                        obs=counts_b,
                    )
                else:
                    raise ValueError(OBS_FAMILY)

    def guide(
        id_b: torch.Tensor,
        grad_b: torch.Tensor,
        vecs_b: torch.Tensor,
        counts_b: torch.Tensor,
        cos_b: torch.Tensor,
        inverse_prob_b: torch.Tensor,
    ) -> None:
        """Construct the variational guide with correctly weighted local terms."""
        with pyro.poutine.scale(scale=LOSS_SCALE):
            B = counts_b.size(0)
            D, S, _Tloc = LATENT_DIM, NUM_STRUCTURES, counts_b.size(1)
            struct_loc_param = pyro.param(
                "struct_loc_param", 0.1 * torch.randn(S, D, device=device)
            )
            pyro.sample("struct_loc", dist.Delta(struct_loc_param).to_event(2))

            struct_log_ls_param = pyro.param(
                "struct_log_ls_param",
                torch.zeros(S, D, device=device),
                constraint=constraints.real,
            )
            pyro.sample("struct_log_ls", dist.Delta(struct_log_ls_param).to_event(2))

            pyro.param(
                "struct_scales_param",
                0.5 * torch.ones(S, D, device=device),
                constraint=constraints.positive,
            )

            comp_b = counts_b / counts_b.sum(-1, keepdim=True).clamp_min(1.0)
            mean_dist = vecs_b.norm(dim=-1).mean(dim=-1, keepdim=True)
            x_enc = torch.cat([comp_b, id_b, grad_b, cos_b, mean_dist], dim=-1)
            pyro.module("encoder", encoder, update_module_params=True)
            mu_q, sd_q = encoder(x_enc)
            var_q = sd_q**2

            # kernel overlap
            lengthscales = torch.exp(struct_log_ls_param)  # (S,D), positive
            logw = rbf_log_overlap(mu_q, var_q, struct_loc_param, lengthscales)  # (B,S)

            phi = safe_softmax(logw, dim=-1)
            with pyro.plate("cells", B), pyro.poutine.scale(scale=inverse_prob_b / B):
                pyro.sample("phi", dist.Delta(phi).to_event(1))
                pyro.sample("mu_loc", dist.Delta(mu_q).to_event(1))

    losses = []
    for lr in lr_steps:
        svi = SVI(model, guide, Adam({"lr": lr}), loss=JitTrace_ELBO(num_particles=num_particles))
        for _ in tqdm.tqdm(range(num_epochs), desc=f"lr={lr}"):
            for id_b_, grad_b, vecs_b, counts_b, cos_b, inverse_prob_b in loader:
                loss = svi.step(id_b_, grad_b, vecs_b, counts_b, cos_b, inverse_prob_b)
                losses.append(loss)

    # ── Minibatch inference over all N
    encoder.eval()
    with torch.no_grad():
        N = counts.size(0)
        mean_dist_full = type_vecs.norm(dim=-1).mean(dim=-1, keepdim=True)

        idx_all = torch.arange(N, device=device)
        infer_set = TensorDataset(
            idx_all, id_b, grads, type_vecs, counts, grad_type_cos, mean_dist_full
        )
        infer_loader = DataLoader(infer_set, batch_size=batch_size, shuffle=False, drop_last=False)

        S = NUM_STRUCTURES
        Tloc = counts.shape[1]
        z_mu_np = np.zeros((N, LATENT_DIM), dtype=np.float32)
        z_sd_np = np.zeros((N, LATENT_DIM), dtype=np.float32)
        phi_np = np.zeros((N, S), dtype=np.float32)
        mix_np = np.zeros((N, Tloc), dtype=np.float32)
        pred1_np = np.zeros(N, dtype=np.int64)
        pred2_np = np.zeros(N, dtype=np.int64)
        phi1_np = np.zeros(N, dtype=np.float32)
        lengthscales = torch.exp(pyro.param("struct_log_ls_param"))

        struct_loc = pyro.param("struct_loc_param")
        pyro.param("struct_scales_param")
        comp_probs = torch.softmax(pyro.param("struct_comp_logits"), dim=-1)  # (S,T)

        for idx_b, id_b_, grad_b, vecs_b, counts_b, cos_b, md_b in infer_loader:
            comp_b = (counts_b / counts_b.sum(-1, keepdim=True).clamp_min(1.0)).nan_to_num(0.0)
            X_b = torch.cat([comp_b, id_b_, grad_b, cos_b, md_b], dim=1)
            pyro.module("encoder", encoder, update_module_params=True)
            mu_q, sd_q = encoder(X_b)
            var_q = sd_q**2

            logw = rbf_log_overlap(mu_q, var_q, struct_loc, lengthscales)  # (B,S)
            phi_b = torch.softmax(logw, dim=-1)  # (B,S)
            mix_b = phi_b @ comp_probs  # (B,T)

            top2_p_b, top2_idx_b = phi_b.topk(2, dim=1)
            i = idx_b.detach().cpu().numpy()
            z_mu_np[i] = mu_q.detach().cpu().numpy()
            z_sd_np[i] = var_q.sqrt().detach().cpu().numpy()
            phi_np[i] = phi_b.detach().cpu().numpy()
            mix_np[i] = mix_b.detach().cpu().numpy()
            pred1_np[i] = top2_idx_b[:, 0].detach().cpu().numpy()
            pred2_np[i] = top2_idx_b[:, 1].detach().cpu().numpy()
            phi1_np[i] = (top2_p_b[:, 0] / top2_p_b.sum(dim=1)).detach().cpu().numpy()

    out = df_grads.copy()
    for d in range(LATENT_DIM):
        out[f"z{d + 1}"] = z_mu_np[:, d]
        out[f"zsd{d + 1}"] = z_sd_np[:, d]
    out["structure_pred1"] = pd.Categorical(pred1_np)
    out["structure_pred2"] = pd.Categorical(pred2_np)
    out["phi"] = phi1_np

    return (
        out,  # dataframe with z and top-2
        phi_np,  # (N,S) overlap-simplex per cell
        torch.softmax(pyro.param("struct_comp_logits"), dim=-1)
        .detach()
        .cpu()
        .numpy(),  # (S,T) cluster→type probs
        mix_np,  # (N,T) predicted composition
        losses,
    )
