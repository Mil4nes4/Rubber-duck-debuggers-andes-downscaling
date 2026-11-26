"""Mide el tiempo por época del starter (corte 16 jun) en CPU y GPU, sin tocar rdd-codex-cnn."""
import sys, time
sys.dont_write_bytecode = True
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
import starter as S

DATA = Path(__file__).resolve().parent / "data"
data = S.Data(DATA)
train, val = data.split("2024061600")  if "2024061600" in S.CUTS else data.split(list(S.CUTS)[-1])
print("cortes:", list(S.CUTS), "train", len(train), flush=True)
opts = {}
stats = data.stats(train, opts)
x, hours, _ = data.inputs(stats, train, opts)
target = (torch.as_tensor(data.y[train]) - stats["clim"][hours]) / stats["anomaly_sd"]

def bench(dev, epochs):
    torch.manual_seed(0)
    model = S.build(data.grid, stats, opts).to(dev)
    xs, hs, ts = x.to(dev), hours.to(dev), target.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    idx = np.arange(len(train)); rng = np.random.RandomState(0)
    times = []
    for e in range(epochs):
        if dev == "cuda": torch.cuda.synchronize()
        t = time.monotonic(); model.train(); rng.shuffle(idx); ls = []
        for s in range(0, len(train), S.BATCH):
            b = torch.as_tensor(idx[s:s+S.BATCH], device=dev)
            loss = F.mse_loss(model(xs[b], hs[b]), ts[b])
            opt.zero_grad(); loss.backward(); opt.step(); ls.append(loss.detach())
        if dev == "cuda": torch.cuda.synchronize()
        times.append(time.monotonic() - t)
        print(f"{dev} época {e+1} {times[-1]:.1f}s pérdida {torch.stack(ls).mean().item():.4f}", flush=True)
    return times

if __name__ == "__main__":
    g = bench("cuda", 3)
    torch.set_num_threads(4)
    c = bench("cpu", 2)
    print(f"GPU (sin la 1ª): {np.mean(g[1:]):.1f}s/época | CPU: {np.mean(c):.1f}s/época")
