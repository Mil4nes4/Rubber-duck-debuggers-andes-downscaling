"""Comprueba equivalencia del ajuste ridge y las separaciones temporales."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import cnn
import ridge


def verify_ridge():
    torch.manual_seed(927)
    hours = torch.arange(180, dtype=torch.int64) % 24
    shared = torch.randn(3, len(hours), 8, dtype=torch.float64)
    local = shared[..., :4]
    target = torch.randn(3, len(hours), 4, dtype=torch.float64) + 100
    m = ridge.fit_cell_ridge(shared, local, target, hours)
    group = F.one_hot(hours, 24)[None, :, :, None]
    local_group = torch.cat((torch.ones_like(local[..., :1]), local - m["local_mean"]), dim=-1)
    extra = (group * local_group[:, :, None, :]).flatten(2)
    scaled = extra / m["hourly_sd"].flatten(1)[:, None]
    z = (shared - m["mean"]) / m["sd"]
    full = torch.cat([torch.ones_like(z[..., :1]), z, scaled], dim=-1)
    gram = full.mT @ full
    gram[:, 1:, 1:].diagonal(dim1=-2, dim2=-1).add_(ridge.LAM * len(hours))
    beta = torch.linalg.solve(gram, full.mT @ target)
    expected = full @ beta
    actual = m["intercept"] + z @ m["beta"] + scaled @ m["hourly"].flatten(1, 2)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0)
    torch.testing.assert_close((actual - target).mean(1), torch.zeros_like(actual[:, 0]),
                               atol=1e-9, rtol=0)
    max_error = (actual - expected).abs().max().item()
    print(f"Ridge equivalente al sistema denso. Error absoluto máximo={max_error:.3e}")


def verify_dataset():
    data = ridge.Dataset(Path(__file__).resolve().parent / "data")
    for cut in ridge.CUTS:
        train, dev, internal, val = ridge.split(data, cut)
        assert train[-1] < val[0]
        assert data.clock[internal[0]] - data.clock[dev[-1]] > np.timedelta64(1, "D")
        assert data.clock[val[0]] > data.clock[internal[-1]]
        assert torch.all(data.lags <= torch.arange(len(data.times))[:, None])
        print(f"{cut} train={len(train)} desarrollo={len(dev)} interna={len(internal)} val={len(val)}")
    model = cnn.Correction()
    result = model(torch.randn(2, 18, 70, 35), torch.tensor([0, 23]))
    assert result.shape == (2, 4, 70, 35)
    assert torch.count_nonzero(result) == 0
    print("Forma de salida e inicialización nula verificadas")


if __name__ == "__main__":
    torch.set_num_threads(1)
    verify_ridge()
    verify_dataset()
