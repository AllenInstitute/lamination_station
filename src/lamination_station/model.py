import torch
from collections.abc import Sequence
from numpy.typing import NDArray
from typing import cast
import pandas as pd
import numpy as np
import tqdm
import torch.nn as nn
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from torch.utils.data import DataLoader, TensorDataset
from pyro.infer import SVI, TraceEnum_ELBO
from pyro.optim import Adam
from ._training import prepare_data, make_training_loader, validate_settings


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
        μ = self.loc(h)
        σ = F.softplus(self.scale(h)) + 1e-3
        return μ, σ


class Classifier(nn.Module):
    def __init__(self, *, latent_dim: int, hidden_dim: int, num_structures: int) -> None:
        """Initialize the network dimensions."""
        super().__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, num_structures)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Evaluate the network."""
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return cast(torch.Tensor, self.out(h))


class Decoder(nn.Module):
    def __init__(self, *, latent_dim: int, hidden_dim: int, out_dim: int) -> None:
        """Initialize the network dimensions."""
        super().__init__()
        inp = latent_dim + 1
        self.fc1 = nn.Linear(inp, hidden_dim)
        self.out = nn.Linear(hidden_dim, out_dim)

    def forward(self, z: torch.Tensor, mean_dist: torch.Tensor) -> torch.Tensor:
        """Evaluate the network."""
        x = torch.cat([z, mean_dist], dim=-1)
        h = F.relu(self.fc1(x))
        return cast(torch.Tensor, self.out(h))


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
    STRUCT_SCALE_SCALE: float = 0.1,
    OBS_FAMILY: str = "nb",
    STRUCT_LOC_PRIOR_SCALE: float = 1.0,
    LOSS_SCALE: float = 1.0,
    HIDDEN_DIM: int = 512,
    HIDDEN_DIM_CLASSIFIER: int = 128,
    device: str | torch.device = "cpu",
    clear_params: bool = True,
    normalize_sampling: bool = True,
    num_particles: int = 3,
    ignore_cur_type: bool = False,
    seed: int | None = None,
) -> tuple[pd.DataFrame, NDArray[np.float32], NDArray[np.float32], list[float]]:
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
        Whether to balance sampling by cell type; importance weights preserve the population objective.

    Returns
    -------
    out : pandas.DataFrame
        Copy of `df_grads` augmented with:
        - `"z1"`, …, `"z{LATENT_DIM}"`: posterior means of latent embeddings,
        - `"structure_pred1"`, `"structure_pred2"`: top-2 inferred structures,
        - `"phi"`: mixture weight for the top structure,
    logits_s : numpy.ndarray, shape (N, NUM_STRUCTURES)
        Raw classifier logits for each cell over all structures.
    struct_comp_probs : numpy.ndarray, shape (NUM_STRUCTURES, T)
        Learned, softmax-normalized structure composition probabilities
        (from `pyro.param('struct_comp')`).

    """

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
    if not np.isfinite(STRUCT_SCALE_SCALE) or STRUCT_SCALE_SCALE <= 0:
        raise ValueError("STRUCT_SCALE_SCALE must be finite and positive")
    encoder = Encoder(input_dim=input_dim, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM).to(device)
    classifier = Classifier(
        latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM_CLASSIFIER, num_structures=NUM_STRUCTURES
    ).to(device)
    decoder = Decoder(latent_dim=LATENT_DIM, hidden_dim=HIDDEN_DIM, out_dim=T).to(device)

    def _mask_out(logits: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Exclude a selected structure from categorical logits."""
        neg_inf = torch.finfo(logits.dtype).min
        mask = F.one_hot(idx, logits.size(-1)).to(logits.dtype)
        return logits + mask * neg_inf

    def left_expand(t: torch.Tensor, target_ndim: int) -> torch.Tensor:
        """Add leading singleton dimensions for broadcasting."""
        return t[(None,) * (target_ndim - t.dim()) + (...,)] if t.dim() < target_ndim else t

    def unified_model(
        id_b: torch.Tensor,
        grad_b: torch.Tensor,
        vecs_b: torch.Tensor,
        counts_b: torch.Tensor,
        cos_b: torch.Tensor,
        inverse_prob_b: torch.Tensor,
    ) -> None:
        """Evaluate global priors and population-scaled local terms."""
        with pyro.poutine.scale(scale=LOSS_SCALE):
            comp_b = (counts_b / counts_b.sum(-1, keepdim=True)).detach()
            B, T = comp_b.shape
            S, D = NUM_STRUCTURES, LATENT_DIM

            struct_loc = pyro.param("struct_loc", 0.1 * torch.randn(S, D, device=device))
            struct_loc = pyro.sample(
                "struct_loc_sample",
                dist.Laplace(
                    torch.zeros(S, D, device=device),
                    STRUCT_LOC_PRIOR_SCALE * torch.ones(S, D, device=device),
                ).to_event(2),
            )
            struct_scale = pyro.sample(
                "struct_scale_sample",
                dist.HalfCauchy(STRUCT_SCALE_SCALE * torch.ones(S, D, device=device)).to_event(2),
            )
            struct_comp = pyro.param("struct_comp", 0.1 * torch.randn(S, T, device=device))
            theta_a = pyro.param("theta_a", torch.zeros(1, device=device))
            theta_b = pyro.param("theta_b", 100 * torch.ones((1, T), device=device))

            with pyro.plate("cells", B), pyro.poutine.scale(scale=inverse_prob_b / B):
                # Prior over mixture weights used to set the 2-component mixing
                phi = pyro.sample("phi", dist.Dirichlet(torch.ones(B, S, device=device)))

                # First categorical (uniform prior here; change if you prefer)
                logits_prior = torch.zeros(B, S, device=device)
                w1 = pyro.sample(
                    "structure_1",
                    dist.OneHotCategorical(logits=logits_prior),
                    infer={"enumerate": "parallel"},
                )
                s1 = w1.argmax(-1)
                # Second categorical: mask out s1
                logits2 = _mask_out(logits_prior, s1)
                w2 = pyro.sample(
                    "structure_2",
                    dist.OneHotCategorical(logits=logits2),
                    infer={"enumerate": "parallel"},
                )
                w2.argmax(-1)

                a1 = (phi * w1).sum(-1)  # phi[s1]
                a2 = (phi * w2).sum(-1)  # phi[s2]
                den = a1 + a2 + 1e-8
                w_mix1 = a1 / den
                w_mix2 = a2 / den

                # select rows via one-hot matmul
                loc1 = w1 @ struct_loc  # [enum..., B, D]
                loc2 = w2 @ struct_loc
                var1 = w1 @ struct_scale.pow(2)
                var2 = w2 @ struct_scale.pow(2)

                # mixture moment-matching
                μ_s = w_mix1.unsqueeze(-1) * loc1 + w_mix2.unsqueeze(-1) * loc2
                # this is for true mixture of gaussians, which I don't think is correct here
                # var_s = (w_mix1.unsqueeze(-1)*(var1 + (loc1 - μ_s).pow(2)) +
                #          w_mix2.unsqueeze(-1)*(var2 + (loc2 - μ_s).pow(2)))
                var_s = w_mix1.unsqueeze(-1) * (var1) + w_mix2.unsqueeze(-1) * (var2)
                σ_s = (var_s + 1e-6).sqrt()

                # discrete head: mix in probability space, not in logit space
                p1 = F.softmax(w1 @ struct_comp, dim=-1)
                p2 = F.softmax(w2 @ struct_comp, dim=-1)
                p_mix = w_mix1.unsqueeze(-1) * p1 + w_mix2.unsqueeze(-1) * p2
                comp_logits = (
                    p_mix + 1e-12
                ).log()  # log-probs for your downstream NB/Poisson logic

                z = pyro.sample("z", dist.Normal(μ_s, σ_s + 1e-6).to_event(1))

                mean_dist = vecs_b.norm(dim=-1).mean(dim=-1, keepdim=True)
                pyro.module("decoder", decoder, update_module_params=True)
                rate = decoder(z, mean_dist)
                theta = F.softplus(theta_b + theta_a * (mean_dist)) + 1e-6

                if OBS_FAMILY == "nb":
                    logits = (
                        rate
                        - torch.logsumexp(rate, -1, keepdim=True)
                        + counts_b.sum(-1).unsqueeze(-1).log()
                        - theta.log()
                    )
                    out_dist = dist.NegativeBinomial(total_count=theta, logits=logits).to_event(1)
                elif OBS_FAMILY == "poisson":
                    out_dist = dist.Poisson(
                        rate=safe_softmax(rate, dim=-1) * counts_b.sum(-1).unsqueeze(-1)
                    ).to_event(1)
                elif OBS_FAMILY == "multinomial":
                    out_dist = dist.Multinomial(
                        total_count=int(counts_b.sum(-1).max().squeeze()), logits=rate
                    )
                else:
                    raise ValueError(f"Unsupported OBS_FAMILY: {OBS_FAMILY!r}")
                pyro.sample("obs", out_dist, obs=counts_b)

                # comp_logits = (torch.stack([Vindex(struct_comp)[s1, :],
                #                             Vindex(struct_comp)[s2, :]], dim=1)
                #                * mix_w.unsqueeze(-1)).sum(1)

                if OBS_FAMILY == "nb":
                    logits2 = (
                        comp_logits
                        - torch.logsumexp(comp_logits, -1, keepdim=True)
                        + counts_b.sum(-1).unsqueeze(-1).log()
                        - theta.log()
                    )
                    out_dist2 = dist.NegativeBinomial(total_count=theta, logits=logits2).to_event(
                        1
                    )
                elif OBS_FAMILY == "poisson":
                    out_dist2 = dist.Poisson(
                        rate=safe_softmax(comp_logits, dim=-1) * counts_b.sum(-1).unsqueeze(-1)
                    ).to_event(1)
                elif OBS_FAMILY == "multinomial":
                    out_dist2 = dist.Multinomial(
                        total_count=int(counts_b.sum(-1).max().squeeze()), logits=comp_logits
                    )
                else:
                    raise ValueError(f"Unsupported OBS_FAMILY: {OBS_FAMILY!r}")
                pyro.sample("obs_2", out_dist2, obs=counts_b)

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
            comp_b = counts_b / counts_b.sum(-1, keepdim=True)
            B, T = comp_b.shape
            S, D = NUM_STRUCTURES, LATENT_DIM

            struct_loc = pyro.param("struct_loc", 0.1 * torch.randn(S, D, device=device))
            pyro.sample("struct_loc_sample", dist.Delta(struct_loc).to_event(2))

            struct_scale_param = pyro.param(
                "struct_scale",
                torch.ones(S, D, device=device),
                constraint=dist.constraints.positive,
            )
            pyro.sample("struct_scale_sample", dist.Delta(struct_scale_param).to_event(2))

            with pyro.plate("cells", B), pyro.poutine.scale(scale=inverse_prob_b / B):
                mean_dist = vecs_b.norm(dim=-1).mean(dim=-1, keepdim=True)
                x_enc = torch.cat([comp_b, id_b, grad_b, cos_b, mean_dist], dim=-1)
                pyro.module("encoder", encoder, update_module_params=True)
                μ_z, σ_z = encoder(x_enc)
                z = pyro.sample("z", dist.Normal(μ_z, σ_z + 1e-6).to_event(1))

                pyro.module("classifier", classifier, update_module_params=True)
                logits_s = classifier(z)  # (B, S)

                # q(phi) as a point mass at softmax(classifier)
                phi_q = safe_softmax(logits_s, dim=-1)
                pyro.sample("phi", dist.Delta(phi_q).to_event(1))

                # First enumerated pick from classifier
                w1 = pyro.sample(
                    "structure_1",
                    dist.OneHotCategorical(logits=logits_s),
                    infer={"enumerate": "parallel"},
                )
                s1 = w1.argmax(-1)
                # Second pick with s1 masked out, then re-normalized by Categorical
                logits_s2 = _mask_out(logits_s, s1)
                pyro.sample(
                    "structure_2",
                    dist.OneHotCategorical(logits=logits_s2),
                    infer={"enumerate": "parallel"},
                )

    losses = []

    for lr_step in lr_steps:
        optimizer = Adam({"lr": lr_step})
        svi = SVI(
            unified_model, guide, optimizer, loss=TraceEnum_ELBO(num_particles=num_particles)
        )

        for epoch in tqdm.tqdm(range(1, num_epochs + 1), desc="Epoch"):
            total_loss = 0.0
            for id_b_, grad_b, vecs_b, counts_b, cos_b, inverse_prob_b in loader:
                # move each batch to device (they’re already on device, but safe):
                id_b_ = id_b_.to(device)
                grad_b = grad_b.to(device)
                vecs_b = vecs_b.to(device)
                counts_b = counts_b.to(device)
                cos_b = cos_b.to(device)
                loss = svi.step(id_b_, grad_b, vecs_b, counts_b, cos_b, inverse_prob_b)
                total_loss += loss
                losses.append(loss)

            if epoch % 100 == 0:
                avg_loss = total_loss / len(loader)
                print(f"[Epoch {epoch:02d}] avg ELBO loss: {avg_loss:.2f}")

    encoder.eval()
    classifier.eval()
    with torch.no_grad():
        N = counts.size(0)
        mean_dist_full = type_vecs.norm(dim=-1).mean(dim=-1, keepdim=True)

        z_pred_np = np.zeros((N, LATENT_DIM), dtype=np.float32)
        logits_s_np = np.zeros((N, NUM_STRUCTURES), dtype=np.float32)
        pred1_np = np.zeros(N, dtype=np.int64)
        pred2_np = np.zeros(N, dtype=np.int64)
        phi_np = np.zeros(N, dtype=np.float32)

        idx_all = torch.arange(N)
        infer_set = TensorDataset(
            idx_all, id_b, grads, type_vecs, counts, grad_type_cos, mean_dist_full
        )
        infer_loader = DataLoader(infer_set, batch_size=batch_size, shuffle=False, drop_last=False)

        for idx_b, id_b_, grad_b, vecs_b, counts_b, cos_b, md_b in infer_loader:
            comp_b = (counts_b / counts_b.sum(-1, keepdim=True)).nan_to_num(0.0)
            X_b = torch.cat([comp_b, id_b_, grad_b, cos_b, md_b], dim=1).to(device)

            pyro.module("encoder", encoder, update_module_params=True)
            loc_b, _ = encoder(X_b)
            z_b = loc_b
            pyro.module("classifier", classifier, update_module_params=True)
            logits_b = classifier(z_b)
            probs_b = torch.softmax(logits_b, dim=1)
            top2_p_b, top2_idx_b = probs_b.topk(2, dim=1)

            i = idx_b.cpu().numpy()
            z_pred_np[i] = z_b.cpu().numpy()
            logits_s_np[i] = logits_b.cpu().numpy()
            pred1_np[i] = top2_idx_b[:, 0].cpu().numpy()
            pred2_np[i] = top2_idx_b[:, 1].cpu().numpy()
            phi_np[i] = (top2_p_b[:, 0] / top2_p_b.sum(dim=1)).cpu().numpy()

    out = df_grads.copy()
    for d in range(LATENT_DIM):
        out[f"z{d + 1}"] = z_pred_np[:, d]
    out["structure_pred1"] = pd.Categorical(pred1_np)
    out["structure_pred2"] = pd.Categorical(pred2_np)
    out["phi"] = phi_np

    return (
        out,
        logits_s_np,
        torch.softmax(pyro.param("struct_comp").cpu().detach(), -1).numpy(),
        losses,
    )
