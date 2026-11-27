"""Ejecuta uno o todos los experimentos reproducibles."""
import argparse
import csv
import json
from pathlib import Path
import time

import torch

import cnn
import ridge
from informe import publish


ROOT = Path(__file__).resolve().parent
DATA = ROOT.parent.parent / "data"  # data/ en la raíz del repo
RESULTS = ROOT / "results"
EXPERIMENTS = {
    "mse": ("mse", False),
    "rmse": ("rmse", False),
    "mse_mayjun": ("mse", True),
    "rmse_mayjun": ("rmse", True),
}
SEEDS = (17, 37, 73)


def log(message):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}"
    print(line, flush=True)
    with (RESULTS / "corrida.log").open("a", encoding="utf-8") as file:
        print(line, file=file)


def save(value, path):
    temp = path.with_suffix(".tmp")
    torch.save(value, temp)
    temp.replace(path)


def read(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def measures(prediction, target, hours):
    sq = (cnn.clip_rh(prediction) - target).square().mean(dim=1, dtype=torch.float64)
    records = {}
    subsets = [("todas", torch.ones_like(hours, dtype=torch.bool)), ("cada3h", hours % 3 == 0)]
    subsets += [(f"{hour:02d}UTC", hours == hour) for hour in range(24)]
    for label, select in subsets:
        n = int(select.sum())
        if n:
            rmse = sq[select].mean(0).sqrt()
            if not rmse.isfinite().all():
                raise RuntimeError("Métricas externas no finitas")
            records[label] = {"horas": n, "score": float(1 - (rmse / ridge.REF).mean()),
                              **dict(zip(("RH", "T", "U", "V"), rmse.tolist()))}
    return records


def finish(path, experiment, cut, seed, data, val, normal, shifted, training=None):
    result = {"experimento": experiment, "corte": cut, "semilla": seed, "entrenamiento": training,
              "normal": measures(normal, data.y[val], data.hours[val]),
              "desfase": measures(shifted, data.y[val], data.hours[val])}
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temp.replace(path)
    publish(RESULTS, EXPERIMENTS, SEEDS)
    for label in ("todas", "cada3h"):
        log(f"{path.stem} {label} score={result['normal'][label]['score']:.6f} "
            f"desfase={result['desfase'][label]['score']:.6f}")


def prepare_cut(data, cut, deadline, oof):
    train, dev, internal, val = ridge.split(data, cut)
    folder = RESULTS / cut
    folder.mkdir(exist_ok=True)
    path = folder / "lineal.pt"
    evaluation = folder / "lineal_validacion.pt"
    if not path.exists():
        deadline()
        log(f"{cut} Ajuste de lineal externa. Entrenamiento={len(train)}, validación={len(val)}")
        model = ridge.fit(data, train)
        save({"schema": 1, "grid": data.grid, "ref": ridge.REF, "linear": model,
              "cut": cut, "boundaries": ridge.CUTS[cut]}, path)
    if not evaluation.exists():
        deadline()
        model = read(path)["linear"]
        shifted = ridge.shifted(data, val)
        save({"normal": ridge.predict(data, model, val),
              "desfase": ridge.predict(data, model, val, shifted),
              "validation": torch.as_tensor(val),
              "hours": data.hours[val]}, evaluation)
    base = read(evaluation)
    metrics = folder / "lineal.json"
    if not metrics.exists():
        finish(metrics, "lineal", cut, None, data, val, base["normal"], base["desfase"])
    cache = folder / "lineal_fuera_de_muestra.pt"
    if oof and not cache.exists():
        deadline()
        log(f"{cut} Desarrollo={len(dev)}, ventana interna={len(internal)}")
        oof_predictions = ridge.blocked_predictions(data, dev, deadline, log)
        deadline()
        inside = ridge.predict(data, ridge.fit(data, dev), internal)
        save({"dev": torch.as_tensor(dev), "internal": torch.as_tensor(internal),
              "dev_pred": oof_predictions, "internal_pred": inside}, cache)


def train_cut(data, cut, name, seeds, deadline):
    folder = RESULTS / cut
    pending = [seed for seed in seeds if not (folder / f"{name}_{seed}.json").exists()]
    if not pending:
        return
    cache = read(folder / "lineal_fuera_de_muestra.pt")
    base = read(folder / "lineal_validacion.pt")
    dev, internal, val = (cache["dev"].numpy(), cache["internal"].numpy(),
                          base["validation"].numpy())
    shifted = ridge.shifted(data, val)
    loss, recent = EXPERIMENTS[name]
    for seed in pending:
        deadline()
        stem = f"{name}_{seed}"
        log(f"{cut} {stem} Inicio")
        with (folder / f"{stem}_epocas.csv").open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("epoca", "perdida_media_lotes", "score_interno", "mejor_epoca", "lr"))

            def record(*fields):
                writer.writerow(fields)
                file.flush()
                epoch, _, score, best, _ = fields
                if epoch == 1 or epoch % 5 == 0:
                    log(f"{cut} {stem} época={epoch} interno={score:.6f} mejor_época={best}")

            model, scales, training = cnn.train(data, dev, internal, cache["dev_pred"],
                                                cache["internal_pred"], loss, recent, seed,
                                                deadline, record)
        checkpoint = read(folder / "lineal.pt")
        checkpoint.update({"state_dict": model.state_dict(), "cnn_input": scales,
                           "architecture": cnn.ARCH, "training": training,
                           "seed": seed, "loss": loss, "recent_weight": 2.0 if recent else 1.0,
                           "epochs_min": cnn.MIN_EPOCHS, "epochs_max": cnn.MAX_EPOCHS,
                           "ema": cnn.EMA, "weight_decay": cnn.WEIGHT_DECAY,
                           "batch": cnn.BATCH, "patience": cnn.PATIENCE,
                           "units": ["%", "K", "m/s", "m/s"], "time_zone": "UTC"})
        path = folder / f"{stem}.pt"
        save(checkpoint, path)
        loaded = read(path)
        restored = cnn.Correction(loaded["architecture"]["width"], loaded["architecture"]["groups"])
        restored.to(memory_format=torch.channels_last)
        restored.load_state_dict(loaded["state_dict"])
        restored.eval()
        h = data.hours[val]
        x = cnn.normalize(cnn.maps(data, val, base["normal"]), loaded["cnn_input"])
        normal = base["normal"] + cnn.correction(restored, x, h)
        x = cnn.normalize(cnn.maps(data, val, base["desfase"], shifted), loaded["cnn_input"])
        shifted_prediction = base["desfase"] + cnn.correction(restored, x, h)
        finish(folder / f"{stem}.json", name, cut, seed, data, val, normal, shifted_prediction,
               training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", choices=["todos", "lineal", *EXPERIMENTS], default="todos")
    parser.add_argument("--corte", choices=ridge.CUTS)
    parser.add_argument("--semilla", choices=SEEDS, type=int)
    parser.add_argument("--hilos", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--horas", type=float, default=6.0)
    args = parser.parse_args()
    if not 0 < args.horas <= 6:
        parser.error("--horas debe estar en el intervalo (0, 6]")
    end = time.monotonic() + args.horas * 3600
    torch.set_num_threads(args.hilos)
    torch.set_num_interop_threads(1)
    RESULTS.mkdir(exist_ok=True)

    def deadline():
        if time.monotonic() >= end:
            raise TimeoutError("Se alcanzó el límite de corrida")

    log(f"Inicio experimento={args.experimento} torch={torch.__version__} hilos={torch.get_num_threads()}")
    cuts = [args.corte] if args.corte else list(ridge.CUTS)
    configs = list(EXPERIMENTS) if args.experimento == "todos" else [args.experimento]
    seeds = [args.semilla] if args.semilla is not None else SEEDS
    try:
        data = ridge.Dataset(DATA)
        for cut in cuts:
            prepare_cut(data, cut, deadline, False)
        if args.experimento != "lineal":
            for cut in cuts:
                prepare_cut(data, cut, deadline, True)
            for name in configs:
                for cut in cuts:
                    train_cut(data, cut, name, seeds, deadline)
        log("Corrida solicitada finalizada")
    except TimeoutError as error:
        log(str(error))
    finally:
        publish(RESULTS, EXPERIMENTS, SEEDS)


if __name__ == "__main__":
    main()
