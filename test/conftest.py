"""Small deterministic inputs for model and graph regressions."""

import torch
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pyro
import pytest
from numpy.typing import NDArray


@dataclass
class Inputs:
    """Aligned neighborhood observations in the public preprocessing format."""

    frame: pd.DataFrame
    counts: pd.DataFrame
    gradients: NDArray[np.float64]
    vectors: pd.DataFrame

    @property
    def arguments(
        self,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, NDArray[np.float64], pd.DataFrame]:
        """Return the positional data arguments used by all seven runners."""
        return self.frame, self.frame["type"], self.counts, self.gradients, self.vectors


@pytest.fixture
def inputs() -> Inputs:
    """Generate uneven type frequencies and a final partial minibatch."""
    torch.set_num_threads(1)
    pyro.clear_param_store()
    pyro.set_rng_seed(17)
    rng = np.random.default_rng(41)
    n = 9
    frame = pd.DataFrame({"type": pd.Categorical(["A"] * 5 + ["B"] * 3 + ["C"])})
    counts = pd.DataFrame(
        rng.multinomial(10, [0.3, 0.3, 0.4], size=n), columns=["A_count", "B_count", "C_count"]
    )
    vectors = pd.DataFrame(
        rng.normal(size=(n, 6)), columns=[f"{t}_{ax}" for t in "ABC" for ax in ("vec_x", "vec_y")]
    )
    return Inputs(frame, counts, rng.normal(size=(n, 2)), vectors)
