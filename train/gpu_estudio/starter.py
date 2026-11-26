"""Starter completo con cortes temporales y evaluación de la última época."""
import argparse
import csv
from copy import deepcopy
import datetime as dt
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from ridge import CUTS, REF


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "starter"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = (0, 37, 73)
EPOCHS = 25
BATCH = 32
VARIANTS = {
    "starter": {},
    "hora_onehot": {"onehot": True},
    "t6": {"lags": (6, 2, 1, 0)},
    "t12": {"lags": (12, 2, 1, 0)},
    "t24": {"lags": (24, 2, 1, 0)},
    "gfs_derivados": {"derived": True},
    "perdida_rmse": {"rmse": True},
    "rh_recorte": {"clip": True},
    "ema": {"ema": True},
    "peso_mayjun": {"recent": True},
}


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (RESULTS / "corrida.log").open("a", encoding="utf-8") as file:
        print(line, file=file)


def save(obj, path):
    temporary = path.with_suffix(".tmp")
    torch.save(obj, temporary)
    temporary.replace(path)


def read(path):
    return torch.load(path, weights_only=True, map_location="cpu")


def write_json(obj, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    temporary.replace(path)


class Data:
    def __init__(self, path):
        with np.load(path / "train.npz") as train:
            self.x, self.y = train["X"], train["Y"]
            self.times = train["times"].astype(str)
        dates = [dt.datetime.strptime(stamp, "%Y%m%d%H") for stamp in self.times]
        positions = {date: i for i, date in enumerate(dates)}
        self.lags = {lag: np.array([positions.get(date - dt.timedelta(hours=lag), i)
                                    for i, date in enumerate(dates)])
                     for lag in (0, 1, 2, 6, 12, 24)}
        self.hours = np.array([date.hour for date in dates], dtype=np.int64)
        with np.load(path / "static.npz") as static:
            self.grid = {key: torch.from_numpy(static[key].astype(np.float32)) for key in
                         ("dem", "gfs_lats", "gfs_lons", "wrf_lats", "wrf_lons")}
            glat, glon = static["gfs_lats"], static["gfs_lons"]
            lat, lon = np.meshgrid(static["wrf_lats"], static["wrf_lons"], indexing="ij")
            gy = (lat - glat[0]) / (glat[-1] - glat[0]) * 2 - 1
            gx = (lon - glon[0]) / (glon[-1] - glon[0]) * 2 - 1
            self.grid["interpolation"] = torch.tensor(np.stack([gx, gy], -1)[None], dtype=torch.float32)

    def split(self, cut):
        start, end = CUTS[cut]
        return np.flatnonzero(self.times < start), np.flatnonzero((self.times >= start) & (self.times < end))

    def derived(self, rows):
        current = self.x[rows]
        u, v = current[..., 2], current[..., 3]
        speed = np.hypot(u, v)[..., None]
        return np.concatenate([np.gradient(current, axis=1), np.gradient(current, axis=2),
                               speed], axis=-1)

    def stats(self, train, options):
        fields, targets = self.x[train], self.y[train]
        h = self.hours[train]
        climate = np.stack([targets[h == k].mean(0) for k in range(24)])
        anomaly = (targets - climate[h]).std((0, 1, 2))
        dem = self.grid["dem"].numpy()
        values = {"mu_x": fields.mean((0, 1, 2)), "sd_x": fields.std((0, 1, 2)),
                  "clim": climate, "anomaly_sd": anomaly, "dem_mean": dem.mean(), "dem_sd": dem.std()}
        if options.get("derived"):
            extra = self.derived(train)
            values["derived_mean"] = extra.mean((0, 1, 2))
            values["derived_sd"] = np.maximum(extra.std((0, 1, 2)), 1e-6)
        return {key: torch.as_tensor(value, dtype=torch.float32) for key, value in values.items()}

    def inputs(self, normalization, rows, options, shifted=False):
        lags = options.get("lags", (2, 1, 0))
        x = np.stack([self.x[self.lags[lag][rows]] for lag in lags], axis=1)
        if shifted:
            x[..., 0] -= 8
            x[..., 1] -= 1.5
        x = torch.as_tensor(x, dtype=torch.float32)
        x = (x - normalization["mu_x"]) / normalization["sd_x"]
        extra = None
        if options.get("derived"):
            # Una traslación uniforme de RH y T deja invariantes los gradientes.
            extra = torch.from_numpy(self.derived(rows)) - normalization["derived_mean"]
            extra = (extra / normalization["derived_sd"]).permute(0, 3, 1, 2)
        return x, torch.from_numpy(self.hours[rows]), extra


class Net(nn.Module):
    def __init__(self, grid, dem, inputs=14, width=64, onehot=False):
        super().__init__()
        w = width
        self.onehot = onehot
        self.register_buffer("grid", grid, persistent=False)
        self.register_buffer("dem", dem, persistent=False)
        self.enc = nn.Sequential(nn.Conv2d(inputs, w, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(w, w, 3, padding=1), nn.GELU())
        self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        self.dec = nn.Sequential(nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(w, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, 4, 1))

    def forward(self, x, hour, extra=None):
        b, lag, lat, lon, channels = x.shape
        x = x.permute(0, 1, 4, 2, 3).reshape(b, lag * channels, lat, lon)
        if self.onehot:
            h = F.one_hot(hour, num_classes=24).to(x.dtype)
        else:
            angle = 2 * np.pi * hour.float() / 24
            h = torch.stack([angle.sin(), angle.cos()], dim=1)
        h = h[:, :, None, None].expand(-1, -1, lat, lon)
        fields = [x, h] if extra is None else [x, h, extra]
        z = self.enc(torch.cat(fields, dim=1).contiguous(memory_format=torch.channels_last))
        z = F.grid_sample(z, self.grid.expand(b, -1, -1, -1), align_corners=True)
        z = torch.cat([z.contiguous(memory_format=torch.channels_last), self.dem.expand(b, -1, -1, -1),
                       self.static.expand(b, -1, -1, -1)], dim=1)
        return self.dec(z.contiguous(memory_format=torch.channels_last)).permute(0, 2, 3, 1)


def build(grid, stats, options):
    dem = ((grid["dem"] - stats["dem_mean"]) / stats["dem_sd"])[None, None]
    onehot = options.get("onehot", False)
    width = 4 * len(options.get("lags", (2, 1, 0))) + (24 if onehot else 2)
    if options.get("derived"):
        width += 9
    return Net(grid["interpolation"], dem, inputs=width,
               onehot=onehot).to(memory_format=torch.channels_last)


@torch.inference_mode()
def predict(model, inputs, stats):
    model.eval()
    x, hours, extra = inputs
    pieces = []
    for start in range(0, len(x), BATCH):
        selected = slice(start, start + BATCH)
        h = hours[selected]
        z = None if extra is None else extra[selected]
        pieces.append(stats["clim"][h] + model(x[selected], h, z) * stats["anomaly_sd"])
    return torch.cat(pieces)


def measures(prediction, target, hours):
    sq = (prediction - target).square().mean(dim=(1, 2), dtype=torch.float64)
    groups = [("todas", torch.ones_like(hours, dtype=torch.bool)), ("cada3h", hours % 3 == 0)]
    groups += [(f"{h:02d}UTC", hours == h) for h in range(24)]
    records = {}
    for name, selected in groups:
        n = int(selected.sum())
        if n:
            rmse = sq[selected].mean(0).sqrt()
            if not rmse.isfinite().all():
                raise RuntimeError("Métricas no finitas")
            records[name] = {"horas": n, "score": float(1 - (rmse / REF).mean()),
                             **dict(zip(("RH", "T", "U", "V"), rmse.tolist()))}
    return records


def evaluate(folder, checkpoint, data, val, name):
    options = VARIANTS[name]
    stats = checkpoint["normalizacion"]
    model = build(checkpoint["grilla"], stats, options)
    model.load_state_dict(checkpoint["state_dict"])
    nominal = data.inputs(stats, val, options)
    shifted_input = data.inputs(stats, val, options, shifted=True)
    hours = nominal[1]
    normal = predict(model, nominal, stats)
    shifted = predict(model, shifted_input, stats)
    if options.get("clip"):
        for prediction in (normal, shifted):
            prediction[..., 0].clamp_(0, 100)
    save({"normal": normal, "desfase": shifted, "indices": torch.from_numpy(val),
          "horas": hours, "fechas": data.times[val].tolist()}, folder / "predicciones.pt")
    target = torch.as_tensor(data.y[val], dtype=torch.float32)
    return {"normal": measures(normal, target, hours), "desfase": measures(shifted, target, hours)}


def fit(data, folder, train, seed, deadline, name="starter"):
    options = VARIANTS[name]
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    stats = data.stats(train, options)
    x, hours, extra = data.inputs(stats, train, options)
    target = (torch.as_tensor(data.y[train]) - stats["clim"][hours]) / stats["anomaly_sd"]
    x, hours, target = x.to(DEV), hours.to(DEV), target.to(DEV)
    extra = None if extra is None else extra.to(DEV)
    sd_dev = stats["anomaly_sd"].to(DEV)
    model = build(data.grid, stats, options).to(DEV)
    average_model = None
    if name == "starter" or options.get("ema"):
        average_model = deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=2e-3, total_steps=EPOCHS * int(np.ceil(len(train) / BATCH)))
    indices = np.arange(len(train))
    weights = torch.from_numpy(np.where(data.times[train] >= "2024050100", 2.0, 1.0).astype(np.float32)).to(DEV)
    with (folder / "epocas.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["epoca", "perdida_media_lotes", "segundos"])
        for epoch in range(1, EPOCHS + 1):
            start = time.monotonic()
            deadline()
            model.train()
            rng.shuffle(indices)
            losses = []
            for step, start_batch in enumerate(range(0, len(train), BATCH)):
                if step % 8 == 0:
                    deadline()
                b = torch.as_tensor(indices[start_batch:start_batch + BATCH], device=DEV)
                add = None if extra is None else extra[b]
                output = model(x[b], hours[b], add)
                if options.get("rmse"):
                    err = (output - target[b]) * sd_dev
                    loss = ((err.square().mean(dim=(0, 1, 2)) + 1e-6).sqrt() / REF.to(DEV)).mean()
                elif options.get("recent"):
                    per_sample = (output - target[b]).square().mean(dim=(1, 2, 3))
                    loss = (per_sample * weights[b]).sum() / weights[b].sum()
                else:
                    loss = F.mse_loss(output, target[b])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                if average_model is not None:
                    with torch.no_grad():
                        torch._foreach_lerp_(tuple(average_model.parameters()),
                                             tuple(model.parameters()), 0.01)
                losses.append(loss.detach())
            average = torch.stack(losses).mean().item()
            if not np.isfinite(average):
                raise RuntimeError("Pérdida no finita")
            elapsed = time.monotonic() - start
            writer.writerow((epoch, average, elapsed))
            file.flush()
            log(f"{folder.name} época={epoch}/{EPOCHS} pérdida={average:.6f} s={elapsed:.1f}")
    final_model = average_model if options.get("ema") else model
    checkpoint = {"schema": 1, "state_dict": {k: v.cpu() for k, v in final_model.state_dict().items()}, "normalizacion": stats,
            "grilla": data.grid, "referencia": REF, "semilla": seed, "epocas": EPOCHS,
            "modelo": name, "opciones": options,
            "arquitectura": {"ancho": 64, "canales": model.enc[0].in_channels,
                             "rezagos": list(options.get("lags", (2, 1, 0))),
                             "hora": "onehot" if options.get("onehot") else "sincos"},
            "hiperparametros": {"batch": BATCH, "lr_max": 2e-3, "weight_decay": 1e-4,
                                "optimizador": "AdamW", "calendario": "OneCycleLR"}}
    if average_model is not None:
        checkpoint["hiperparametros"]["ema_decay"] = 0.99
        if name == "starter":
            checkpoint["ema_state_dict"] = {k: v.cpu() for k, v in average_model.state_dict().items()}
    return checkpoint


def run(data, cut, seed, deadline, name="starter"):
    root = RESULTS if name == "starter" else RESULTS.parent / "ablaciones" / name
    folder = root / f"{cut}_{seed}"
    folder.mkdir(exist_ok=True, parents=True)
    metrics = folder / "metricas.json"
    if metrics.exists():
        return
    train, val = data.split(cut)
    deadline()
    log(f"{name} {folder.name} Inicio entrenamiento={len(train)} validación={len(val)}")
    path = folder / "modelo.pt"
    if name == "rh_recorte":
        path = RESULTS / folder.name / "modelo.pt"
    elif not path.exists():
        checkpoint = None
        if name == "ema":
            baseline = read(RESULTS / folder.name / "modelo.pt")
            if "ema_state_dict" in baseline:
                baseline["state_dict"] = baseline.pop("ema_state_dict")
                baseline.update({"modelo": name, "opciones": VARIANTS[name]})
                checkpoint = baseline
                log(f"{name} {folder.name} Pesos EMA de la misma trayectoria del starter")
        if checkpoint is None:
            checkpoint = fit(data, folder, train, seed, deadline, name)
        save(checkpoint, path)
    result = evaluate(folder, read(path), data, val, name)
    write_json({"modelo": name, "corte": cut, "semilla": seed, "epocas": EPOCHS,
                **result}, metrics)
    for label in ("todas", "cada3h"):
        log(f"{name} {folder.name} {label} score={result['normal'][label]['score']:.6f} "
            f"desfase={result['desfase'][label]['score']:.6f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hilos", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--horas", type=float, default=6)
    parser.add_argument("--corte", choices=CUTS)
    parser.add_argument("--semilla", choices=SEEDS, type=int)
    args = parser.parse_args()
    if not 0 < args.horas <= 6:
        parser.error("--horas debe estar en (0, 6]")
    end = time.monotonic() + args.horas * 3600

    def deadline():
        if time.monotonic() >= end:
            raise TimeoutError("Se alcanzó el límite de ejecución")

    RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(args.hilos)
    torch.set_num_interop_threads(1)
    log(f"Inicio starter completo. Torch={torch.__version__} hilos={args.hilos}")
    try:
        data = Data(ROOT / "data")
        for seed in ([args.semilla] if args.semilla is not None else SEEDS):
            for cut in ([args.corte] if args.corte else CUTS):
                run(data, cut, seed, deadline)
        log("Finalizó el starter solicitado")
    except TimeoutError as error:
        log(str(error))


if __name__ == "__main__":
    main()



