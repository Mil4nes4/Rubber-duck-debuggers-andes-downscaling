"""Contrasta la implementación del starter con las celdas originales."""
import json

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import starter as s


def verify():
    torch.set_num_threads(1)
    file = s.ROOT.parent / "Rubber-duck-debuggers-andes-downscaling" / "train" / "starter_notebook.ipynb"
    notebook = json.loads(file.read_text(encoding="utf-8"))
    data = s.Data(s.ROOT / "data")
    train, val = data.split("may16")
    stats = data.stats(train[:240], {})
    sample = val[:32]
    code = next("".join(cell["source"]) for cell in notebook["cells"]
                if cell["cell_type"] == "code" and "class Net(nn.Module)" in "".join(cell["source"]))
    code = code[code.index("class Net(nn.Module)"):code.index("def to_t(a)")]
    grid = data.grid["interpolation"]
    dem = ((data.grid["dem"] - stats["dem_mean"]) / stats["dem_sd"])[None, None]
    scope = {"torch": torch, "nn": nn, "np": np, "F": F, "GRID": grid, "DEM": dem}
    exec(code, scope)
    torch.manual_seed(0)
    original = scope["Net"]().eval()
    torch.manual_seed(0)
    replica = s.build(data.grid, stats, {}).eval()
    for key in original.state_dict():
        torch.testing.assert_close(original.state_dict()[key], replica.state_dict()[key], atol=0, rtol=0)
    x, h, _ = data.inputs(stats, sample, {})
    with torch.no_grad():
        old = original(x, h.float())
        new = replica(x, h)
    torch.testing.assert_close(old, new, atol=2e-6, rtol=1e-5)
    print(f"Starter. Los pesos iniciales coinciden y la diferencia maxima de salida es {(old-new).abs().max().item():.3e}.")
    for name, options in s.VARIANTS.items():
        stats = data.stats(train[:240], options)
        model = s.build(data.grid, stats, options)
        base_inputs = data.inputs(stats, sample[:2], options)
        perturbed = data.inputs(stats, sample[:2], options, shifted=True)
        a = s.predict(model, base_inputs, stats)
        b = s.predict(model, perturbed, stats)
        assert a.shape == b.shape == (2, 70, 35, 4), name
        assert (a - b).abs().max() > 0, name
        torch.testing.assert_close(base_inputs[0][..., 2:], perturbed[0][..., 2:])
        print(f"{name}. Formas, inferencia y perturbacion verificadas.")


if __name__ == "__main__":
    verify()
