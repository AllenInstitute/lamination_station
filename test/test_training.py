"""Regression tests for parameter ownership, sample scaling, and input contracts."""

import torch
import importlib
from collections.abc import Callable
from typing import Any

import numpy as np
import pyro
import pytest
from pyro.infer import Trace_ELBO

from conftest import Inputs
from lamination_station._training import make_training_loader, prepare_data

VARIANTS = [
    "model",
    "single_model",
    "hybrid_model",
    "identity_model",
    "integral_model",
    "integral_model_classify",
    "integral_model_notmvn",
]


def fit(*, name: str, inputs: Inputs, **overrides: Any) -> Any:
    """Run a tiny fit with shared settings and optional targeted overrides."""
    module = importlib.import_module("lamination_station." + name)
    settings = dict(
        NUM_STRUCTURES=3,
        LATENT_DIM=2,
        num_epochs=1,
        lr_steps=(0.001,),
        batch_size=4,
        HIDDEN_DIM=12,
        num_particles=1,
        seed=17,
    )
    if hasattr(module, "Classifier"):
        settings["HIDDEN_DIM_CLASSIFIER"] = 10
    settings.update(overrides)
    return module.run_model(*inputs.arguments, **settings)


@pytest.mark.parametrize("name", VARIANTS)
@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("normalize_sampling", [False, True])
def test_all_variants_train_partial_and_oversized_batches(
    inputs: Inputs,
    name: str,
    batch_size: int,
    normalize_sampling: bool,
) -> None:
    """Every cell is processed, even when a batch is larger than the dataset."""
    output = fit(
        name=name, inputs=inputs, batch_size=batch_size, normalize_sampling=normalize_sampling
    )
    assert len(output[-1]) == (len(inputs.frame) + batch_size - 1) // batch_size
    assert np.isfinite(output[-1]).all()
    assert output[0].index.equals(inputs.frame.index)
    assert np.isfinite(output[0][["z1", "z2", "phi"]].to_numpy()).all()


@pytest.mark.parametrize(
    "name", ["model", "single_model", "hybrid_model", "identity_model", "integral_model_classify"]
)
def test_every_network_tensor_updates(
    inputs: Inputs, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catch frozen networks even when optimization returns finite losses."""
    module = importlib.import_module("lamination_station." + name)
    snapshots: dict[str, tuple[torch.nn.Module, dict[str, torch.Tensor]]] = {}

    def factory(*, original: type[torch.nn.Module], role: str) -> Callable[..., torch.nn.Module]:
        """Capture initial weights without modifying network behavior."""

        def construct(**kwargs: Any) -> torch.nn.Module:
            """Construct and snapshot one network."""
            network = original(**kwargs)
            snapshots[role] = (
                network,
                {k: v.detach().clone() for k, v in network.named_parameters()},
            )
            return network

        return construct

    for role in ("Encoder", "Classifier", "Decoder"):
        if hasattr(module, role):
            monkeypatch.setattr(module, role, factory(original=getattr(module, role), role=role))
    fit(name=name, inputs=inputs, num_epochs=2)
    registered = {id(v) for _, v in pyro.get_param_store().named_parameters()}
    for role, (network, before) in snapshots.items():
        for key, value in network.named_parameters():
            assert id(value) in registered, (role, key)
            assert not torch.equal(value.detach(), before[key]), (role, key)


@pytest.mark.parametrize("name", VARIANTS)
def test_invalid_or_unaligned_data_fail_early(inputs: Inputs, name: str) -> None:
    """Reject missing information before any NaN can reach an encoder."""
    inputs.counts.iloc[0] = 0
    with pytest.raises(ValueError, match="Empty neighborhoods"):
        fit(name=name, inputs=inputs)
    inputs.counts.iloc[0] = [1, 2, 3]
    inputs.counts = inputs.counts.iloc[::-1]
    with pytest.raises(ValueError, match="indices"):
        fit(name=name, inputs=inputs)


def test_identity_mask_does_not_destroy_observed_labels(inputs: Inputs) -> None:
    """The identity likelihood must still receive valid one-hot observations."""
    output = fit(name="identity_model", inputs=inputs, ignore_cur_type=True)
    assert np.isfinite(output[-1]).all()


@pytest.mark.parametrize("family", ["poisson", "multinomial"])
@pytest.mark.parametrize("name", [n for n in VARIANTS if n != "identity_model"])
def test_count_likelihoods(inputs: Inputs, name: str, family: str) -> None:
    """Count families support varying observed totals and a partial final batch."""
    inputs.counts.iloc[0] = [1, 2, 0]
    output = fit(name=name, inputs=inputs, OBS_FAMILY=family)
    assert np.isfinite(output[-1]).all()


@pytest.mark.parametrize("balanced", [False, True])
def test_minibatch_gradients_match_population(
    inputs: Inputs,
    balanced: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check complete model/guide gradients, including priors and importance weights."""
    module = importlib.import_module("lamination_station.integral_model_notmvn")
    captured: list[Any] = []
    original = module.SVI

    def capture(*args: Any, **kwargs: Any) -> Any:
        """Retain the fitted closures to compare exact gradient estimators."""
        svi = original(*args, **kwargs)
        captured.append(svi)
        return svi

    monkeypatch.setattr(module, "SVI", capture)
    fit(name="integral_model_notmvn", inputs=inputs)
    data = prepare_data(
        df=inputs.frame,
        cell_types=inputs.frame["type"],
        counts=inputs.counts,
        gradients=inputs.gradients,
        type_vectors=inputs.vectors,
        device="cpu",
        ignore_identity=False,
    )
    loader = make_training_loader(
        data=data, cell_types=inputs.frame["type"], batch_size=4, normalize_sampling=balanced
    )
    assert isinstance(loader.dataset, torch.utils.data.TensorDataset)
    arrays = loader.dataset.tensors
    params = [p for _, p in pyro.get_param_store().named_parameters()]
    svi = captured[0]
    elbo = Trace_ELBO()

    def gradient(*, indices: torch.Tensor, inverse: torch.Tensor) -> torch.Tensor:
        """Flatten exact gradients of the deterministic Delta-guide objective."""
        args = [a[indices] for a in arrays[:5]] + [inverse]
        loss = elbo.differentiable_loss(svi.model, svi.guide, *args)
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return torch.cat(
            [(torch.zeros_like(p) if g is None else g).reshape(-1) for p, g in zip(params, grads)]
        )

    n = len(inputs.frame)
    full = gradient(indices=torch.arange(n), inverse=torch.full((n,), float(n)))
    estimate = torch.zeros_like(full)
    if balanced:
        inverse = arrays[-1]
        for i in range(n):
            estimate += (
                gradient(indices=torch.tensor([i]), inverse=inverse[i : i + 1]) / inverse[i]
            )
    else:
        for start in range(0, n, 4):
            idx = torch.arange(start, min(start + 4, n))
            estimate += (
                gradient(indices=idx, inverse=torch.full((len(idx),), float(n))) * len(idx) / n
            )
    torch.testing.assert_close(estimate, full, atol=2e-4, rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device not available")
@pytest.mark.parametrize("name", VARIANTS)
def test_cuda_partial_batches(inputs: Inputs, name: str) -> None:
    """GPU fitting uses correctly placed importance weights and inference indices."""
    output = fit(name=name, inputs=inputs, device="cuda")
    assert np.isfinite(output[-1]).all()
    assert np.isfinite(output[0][["z1", "z2", "phi"]].to_numpy()).all()


@pytest.mark.parametrize("name", ["model", "integral_model_notmvn"])
def test_explicit_seed_reproduces_fit(inputs: Inputs, name: str) -> None:
    """Caller-selected seeds reproduce initialization, sampling, and predictions."""
    first = fit(name=name, inputs=inputs)
    second = fit(name=name, inputs=inputs)
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(first[-1], second[-1])


@pytest.mark.parametrize("name", ["integral_model", "integral_model_notmvn"])
def test_jit_objective_matches_eager_for_partial_batches(
    inputs: Inputs,
    name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing batch sizes must not freeze the population scaling in a JIT trace."""
    module = importlib.import_module("lamination_station." + name)
    captured: list[tuple[Any, Any]] = []
    original = module.SVI

    def capture(*args: Any, **kwargs: Any) -> Any:
        """Retain both the objective object and fitted model/guide closures."""
        svi = original(*args, **kwargs)
        captured.append((svi, kwargs["loss"]))
        return svi

    monkeypatch.setattr(module, "SVI", capture)
    fit(name=name, inputs=inputs)
    data = prepare_data(
        df=inputs.frame,
        cell_types=inputs.frame["type"],
        counts=inputs.counts,
        gradients=inputs.gradients,
        type_vectors=inputs.vectors,
        device="cpu",
        ignore_identity=False,
    )
    tensors = (data.identities, data.gradients, data.type_vectors, data.counts, data.cosines)
    svi, objective = captured[0]
    for size in (4, 1, len(inputs.frame)):
        args = [a[:size] for a in tensors] + [torch.full((size,), float(len(inputs.frame)))]
        eager = Trace_ELBO().loss(svi.model, svi.guide, *args)
        jitted = objective.loss(svi.model, svi.guide, *args)
        np.testing.assert_allclose(jitted, eager, atol=1e-4, rtol=1e-5)
