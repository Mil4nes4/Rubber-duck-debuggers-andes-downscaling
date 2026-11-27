"""Corrección convolucional en la grilla WRF."""
import math

import torch
from torch import nn
import torch.nn.functional as F

from ridge import REF


ARCH = {"width": 16, "groups": 4, "smoothing": 9}
MIN_EPOCHS = 40
MAX_EPOCHS = 64
PATIENCE = 10
BATCH = 24
WEIGHT_DECAY = 0.01
EMA = 0.99


class Correction(nn.Module):
    def __init__(self, width=16, groups=4):
        super().__init__()
        self.project = nn.Conv2d(18, width, 1)
        self.hour = nn.Embedding(24, width)
        self.near = nn.Conv2d(width, width, 3, padding=1)
        self.far = nn.Conv2d(width, width, 3, padding=2, dilation=2, groups=groups)
        self.output = nn.Conv2d(width, 4, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, maps, hours):
        # Equivale a proyectar los 24 canales one-hot constantes.
        x = F.silu(self.project(maps) + self.hour(hours)[:, :, None, None])
        x = x + F.silu(self.near(x))
        x = x + F.silu(self.far(x))
        return self.output(x)


def static_maps(dem):
    dem = dem[None, None]
    k = ARCH["smoothing"]
    pad = k // 2
    smooth = F.avg_pool2d(F.pad(dem, (pad,) * 4, mode="replicate"), k, stride=1)
    return torch.cat([dem, dem - smooth], dim=1)


def maps(data, rows, base, prepared=None):
    p = data.prepared if prepared is None else prepared
    shape = data.grid["dem"].shape
    fields = [base, *(p["local"][data.lags[rows, i]] for i in range(3))]
    fields = torch.cat(fields, dim=-1).reshape(len(rows), *shape, 16).permute(0, 3, 1, 2)
    topo = static_maps(data.grid["dem"]).expand(len(rows), -1, -1, -1)
    return torch.cat([fields, topo], dim=1).contiguous(memory_format=torch.channels_last)


def stats(x):
    sd, mean = torch.std_mean(x, dim=(0, 2, 3), correction=0, keepdim=True)
    return {"mean": mean, "sd": sd.clamp_min(1e-6)}


def normalize(x, scales):
    return x.sub_(scales["mean"]).div_(scales["sd"])


def tensor_maps(data, flat):
    return flat.reshape(len(flat), *data.grid["dem"].shape, 4).permute(0, 3, 1, 2)


def clip_rh(prediction):
    prediction = prediction.clone()
    prediction[..., 0].clamp_(0, 100)
    return prediction


@torch.inference_mode()
def correction(model, inputs, hours):
    pieces = []
    for xb, hb in zip(inputs.split(BATCH), hours.split(BATCH)):
        pieces.append(model(xb, hb).permute(0, 2, 3, 1).flatten(1, 2))
    return torch.cat(pieces) * REF


def train(data, development, internal, dev_base, inner_base, loss_name, recent,
          seed, deadline, record):
    torch.manual_seed(seed)
    width, groups = ARCH["width"], ARCH["groups"]
    model = Correction(width, groups).to(memory_format=torch.channels_last)
    average = Correction(width, groups).to(memory_format=torch.channels_last)
    average.load_state_dict(model.state_dict())
    average.requires_grad_(False).eval()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=WEIGHT_DECAY)
    x = maps(data, development, dev_base)
    scales = stats(x)
    normalize(x, scales)
    h = data.hours[development]
    y = tensor_maps(data, (data.y[development] - dev_base) / REF)
    x_inner = normalize(maps(data, internal, inner_base), scales)
    h_inner = data.hours[internal]
    y_inner = data.y[internal]
    weights = torch.ones(len(development))
    if recent:
        weights[data.times[development] >= "2024050100"] = 2.0
    best_score = -math.inf
    epsilon = 1e-6 / REF.square()

    for epoch in range(1, MAX_EPOCHS + 1):
        deadline()
        opt.param_groups[0]["lr"] = 0.0001 + 0.0009 * (1 + math.cos(
            math.pi * (epoch - 1) / (MAX_EPOCHS - 1))) / 2
        losses = []
        for step, indices in enumerate(torch.randperm(len(x)).split(BATCH)):
            if step % 16 == 0:
                deadline()
            opt.zero_grad(set_to_none=True)
            prediction = model(x[indices].contiguous(memory_format=torch.channels_last), h[indices])
            squared = (prediction - y[indices]).square().mean(dim=(2, 3))
            w = weights[indices]
            mse = (squared * w[:, None]).sum(0) / w.sum()
            loss = (mse + epsilon).sqrt().mean() if loss_name == "rmse" else mse.mean()
            loss.backward()
            opt.step()
            with torch.no_grad():
                torch._foreach_lerp_(tuple(average.parameters()), tuple(model.parameters()), 1 - EMA)
            losses.append(loss.detach())
        prediction = clip_rh(inner_base + correction(average, x_inner, h_inner))
        rmse = (prediction - y_inner).square().mean(dim=(0, 1)).sqrt()
        score = float(1 - (rmse / REF).mean())
        if not math.isfinite(score):
            raise RuntimeError("Score interno no finito")
        if score > best_score:
            best_score, best_epoch = score, epoch
            best = {key: value.clone() for key, value in average.state_dict().items()}
        record(epoch, float(torch.stack(losses).mean()), score, best_epoch,
               opt.param_groups[0]["lr"])
        if epoch >= MIN_EPOCHS and epoch - best_epoch >= PATIENCE:
            break
    average.load_state_dict(best)
    return average, scales, {"epochs": epoch, "best_epoch": best_epoch, "internal_score": best_score}
