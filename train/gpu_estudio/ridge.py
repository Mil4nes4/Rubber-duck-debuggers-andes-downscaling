"""Ridge por celda y validación bloqueada."""
import datetime as dt

import numpy as np
import torch
import torch.nn.functional as F

REF = torch.tensor([25.360, 2.0652, 2.6776, 2.0876])
CUTS = {
    "may16": ("2024051600", "2024060100"),
    "jun01": ("2024060100", "2024061600"),
    "jun16": ("2024061600", "2024070100"),
}
LAM = 0.01


class Dataset:
    def __init__(self, path):
        with np.load(path / "train.npz") as data:
            self.x = torch.from_numpy(data["X"]).float()
            self.y = torch.from_numpy(data["Y"]).float().flatten(1, 2)
            self.times = data["times"].astype(str)
        with np.load(path / "static.npz") as data:
            self.grid = {key: torch.from_numpy(data[key]).float() for key in
                         ("dem", "gfs_lats", "gfs_lons", "wrf_lats", "wrf_lons")}
        dates = [dt.datetime.strptime(stamp, "%Y%m%d%H") for stamp in self.times]
        self.clock = np.array(dates, dtype="datetime64[h]")
        self.hours = torch.tensor([date.hour for date in dates])
        positions = {date: i for i, date in enumerate(dates)}
        first = [positions.get(date - dt.timedelta(hours=1), i)
                 for i, date in enumerate(dates)]
        second = [positions.get(date - dt.timedelta(hours=2), first[i])
                  for i, date in enumerate(dates)]
        self.lags = torch.tensor(np.column_stack([np.arange(len(dates)), first, second]))
        self.grid["interpolation"] = interpolation_grid(self.grid)
        self.prepared = self.prepare(self.x)

    def prepare(self, x, gradients=True):
        grid = self.grid["interpolation"]
        result = {"field": x.flatten(1), "local": interpolate(x, grid)}
        if gradients:
            for key, axis in (("gy", 1), ("gx", 2)):
                result[key] = interpolate(torch.from_numpy(np.gradient(x.numpy(), axis=axis)), grid)
        return result


def interpolation_grid(grid):
    lat, lon = grid["gfs_lats"].numpy(), grid["gfs_lons"].numpy()
    fi = np.interp(grid["wrf_lats"].numpy(), lat[::-1], np.arange(len(lat))[::-1])
    fj = np.interp(grid["wrf_lons"].numpy(), lon, np.arange(len(lon)))
    x, y = np.meshgrid(2 * fj / (len(lon) - 1) - 1, 2 * fi / (len(lat) - 1) - 1)
    return torch.tensor(np.stack([x, y], axis=-1), dtype=torch.float32)[None]


def interpolate(x, grid):
    maps = F.grid_sample(x.permute(0, 3, 1, 2), grid.expand(len(x), -1, -1, -1),
                         align_corners=True, padding_mode="border")
    return maps.flatten(2).transpose(1, 2).contiguous()


def fit_pca(field):
    field = field.double()
    mean = field.mean(0)
    sd = field.std(0, correction=0) + 1e-6
    z = (field - mean) / sd
    _, vectors = torch.linalg.eigh(z.T @ z)
    return {"mean": mean.float(), "sd": sd.float(), "axes": vectors[:, -30:].T.float()}


def project(field, pca):
    return ((field - pca["mean"]) / pca["sd"]) @ pca["axes"].T


def features(data, p, rows, cells, projection):
    local = p["local"][:, cells]
    history = [local[data.lags[rows, i]].transpose(0, 1) for i in range(3)]
    gy = p["gy"][rows, cells].transpose(0, 1)
    gx = p["gx"][rows, cells].transpose(0, 1)
    return torch.cat([*history, gy, gx, projection[rows].expand(len(history[0]), -1, -1)], -1)


def fit_cell_ridge(shared, local, target, hours):
    # Complemento de Schur sobre los bloques de indicador e interacciones horarias.
    # El intercepto explícito queda fuera de la regularización.
    cells, n, width = shared.shape
    mean = shared.mean(1, keepdim=True)
    sd = shared.std(1, correction=0, keepdim=True) + 1e-6
    z = (shared - mean) / sd
    z = torch.cat([torch.ones_like(z[..., :1]), z], dim=-1)
    target_mean = target.mean(1, keepdim=True)
    target = target - target_mean
    local_mean = local.mean(1, keepdim=True)
    local = local - local_mean
    gram = z.mT @ z
    gram[:, 1:, 1:].diagonal(dim1=-2, dim2=-1).add_(LAM * n)
    rhs = z.mT @ target
    blocks = []
    hourly_sd = local.new_ones((cells, 24, 5))
    for hour in range(24):
        selected = hours == hour
        if not selected.any():
            blocks.append(None)
            continue
        x = local[:, selected]
        x = torch.cat([torch.ones_like(x[..., :1]), x], dim=-1)
        scale = (x.square().sum(1) / n - (x.sum(1) / n).square()).clamp_min(0).sqrt() + 1e-6
        hourly_sd[:, hour] = scale
        x = x / scale[:, None]
        cross = x.mT @ z[:, selected]
        system = x.mT @ x
        system.diagonal(dim1=-2, dim2=-1).add_(LAM * n)
        hour_rhs = x.mT @ target[:, selected]
        solution = torch.linalg.solve(system, torch.cat([cross, hour_rhs], dim=-1))
        effect, direct = solution.split([width + 1, 4], dim=-1)
        gram -= cross.mT @ effect
        rhs -= cross.mT @ direct
        blocks.append((effect, direct))
    beta = torch.linalg.solve(gram, rhs)
    hourly = local.new_zeros((cells, 24, 5, 4))
    for hour, block in enumerate(blocks):
        if block is not None:
            effect, direct = block
            hourly[:, hour] = direct - effect @ beta
    return {"mean": mean, "sd": sd, "local_mean": local_mean, "hourly_sd": hourly_sd,
            "intercept": target_mean + beta[:, :1], "beta": beta[:, 1:], "hourly": hourly}


def fit(data, rows):
    pca = fit_pca(data.prepared["field"][rows])
    projection = project(data.prepared["field"], pca)
    fitted = []
    for start in range(0, data.y.shape[1], 64):
        cells = slice(start, start + 64)
        shared = features(data, data.prepared, rows, cells, projection)
        local = shared[..., :4]
        target = data.y[rows, cells].transpose(0, 1) - local
        fitted.append(fit_cell_ridge(shared, local, target, data.hours[rows]))
    return {"pca": pca, "ridge": {key: torch.cat([item[key] for item in fitted])
                                   for key in fitted[0]}, "lambda": LAM}


def predict(data, model, rows, prepared=None):
    p = data.prepared if prepared is None else prepared
    projection = project(p["field"], model["pca"])
    outputs = []
    for start in range(0, data.y.shape[1], 128):
        cells = slice(start, start + 128)
        s = {key: value[cells] for key, value in model["ridge"].items()}
        shared = features(data, p, rows, cells, projection)
        local = shared[..., :4]
        prediction = local + s["intercept"] + ((shared - s["mean"]) / s["sd"]) @ s["beta"]
        for hour in range(24):
            selected = data.hours[rows] == hour
            x = local[:, selected] - s["local_mean"]
            x = torch.cat([torch.ones_like(x[..., :1]), x], dim=-1)
            prediction[:, selected] += (x / s["hourly_sd"][:, hour, None]) @ s["hourly"][:, hour]
        outputs.append(prediction.transpose(0, 1))
    return torch.cat(outputs, dim=1)


def split(data, cut):
    start, end = CUTS[cut]
    train = np.flatnonzero(data.times < start)
    validation = np.flatnonzero((data.times >= start) & (data.times < end))
    cutoff = np.datetime64(dt.datetime.strptime(start, "%Y%m%d%H"), "h")
    end_development = cutoff - np.timedelta64(8, "D")
    start_internal = end_development + np.timedelta64(1, "D")
    development = train[data.clock[train] < end_development]
    internal = train[data.clock[train] >= start_internal]
    return train, development, internal, validation


def blocked_predictions(data, development, deadline, log):
    output = torch.empty_like(data.y[development])
    for i, block in enumerate(np.array_split(np.arange(len(development)), 5), 1):
        held = development[block]
        gap = np.timedelta64(1, "D")
        clock = data.clock[development]
        eligible = development[(clock < data.clock[held[0]] - gap) |
                               (clock > data.clock[held[-1]] + gap)]
        deadline()
        output[block] = predict(data, fit(data, eligible), held)
        log(f"Bloque fuera de muestra {i}/5. Ajuste={len(eligible)}, objetivo={len(held)}")
    return output


def shifted(data, validation):
    x = data.x.clone()
    affected = torch.unique(data.lags[validation])
    x[affected, ..., 0] -= 8
    x[affected, ..., 1] -= 1.5
    result = data.prepare(x, gradients=False)
    # Un desplazamiento constante en el campo no altera sus gradientes.
    for key in ("gx", "gy"):
        result[key] = data.prepared[key]
    return result
