"""Ejecuta el starter completo, ablaciones independientes y ensambles."""
import argparse
import datetime as dt
import json
import time

import numpy as np
import torch

import experimento as residual
import ridge
import starter as s
from resumen_starter import CNN_WEIGHT, collect, make_rows, save_report


RESULTS = s.ROOT / "results"
ORDER = ("hora_onehot", "perdida_rmse", "ema", "gfs_derivados",
         "t6", "t12", "t24", "peso_mayjun")


def report(status=None):
    try:
        save_report(RESULTS, s.VARIANTS, s.SEEDS, status)
    except PermissionError as error:
        s.log(f"No se pudo actualizar la tabla o el informe. {error}")


def only_clip(data, cut, seed):
    folder = RESULTS / "ablaciones" / "rh_recorte" / f"{cut}_{seed}"
    path = folder / "metricas.json"
    if path.exists():
        return
    folder.mkdir(exist_ok=True, parents=True)
    baseline = s.read(s.RESULTS / folder.name / "predicciones.pt")
    for scenario in ("normal", "desfase"):
        baseline[scenario][..., 0].clamp_(0, 100)
    s.save(baseline, folder / "predicciones.pt")
    target = torch.as_tensor(data.y[baseline["indices"].numpy()], dtype=torch.float32)
    results = {scenario: s.measures(baseline[scenario], target, baseline["horas"])
               for scenario in ("normal", "desfase")}
    s.write_json({"modelo": "rh_recorte", "corte": cut, "semilla": seed, "epocas": s.EPOCHS,
                  **results}, path)


def ensemble(data, cut, seed, name):
    label = f"{cut}_{seed}"
    folder = RESULTS / "ensambles" / name / label
    path = folder / "metricas.json"
    if path.exists():
        return
    source = s.RESULTS if name == "starter" else RESULTS / "ablaciones" / name
    cnn_path = source / label / "predicciones.pt"
    if not cnn_path.exists():
        return
    cnn = s.read(cnn_path)
    linear = s.read(RESULTS / cut / "lineal_validacion.pt")
    if not torch.equal(cnn["indices"], linear["validation"]):
        raise RuntimeError(f"Índices desalineados en {label}")
    y = torch.as_tensor(data.y[cnn["indices"].numpy()], dtype=torch.float32)
    output = {}
    for scenario in ("normal", "desfase"):
        base = linear[scenario].reshape_as(cnn[scenario])
        base[..., 0].clamp_(0, 100)
        mixed = CNN_WEIGHT * cnn[scenario] + (1 - CNN_WEIGHT) * base
        output[scenario] = s.measures(mixed, y, cnn["horas"])
    folder.mkdir(exist_ok=True, parents=True)
    s.write_json({"modelo": f"ensamble_{name}", "corte": cut, "semilla": seed, "epocas": s.EPOCHS,
                  "peso_cnn": CNN_WEIGHT, **output}, path)


def prepare_linear(cuts, deadline):
    if all((RESULTS / cut / "lineal_validacion.pt").exists() and
           (RESULTS / cut / "lineal.json").exists() for cut in cuts):
        return
    data = ridge.Dataset(s.DATA)
    for cut in cuts:
        residual.prepare_cut(data, cut, deadline, False)


def final_fit(data, deadline):
    rows, _ = make_rows(collect(RESULTS), s.SEEDS)
    candidates = [row for row in rows if row["corte"] == "media cortes" and
                  row["horas_UTC"] == "cada3h" and row["n_semillas"] == len(s.SEEDS) and
                  row["modelo"] != "lineal"]
    best = max(candidates, key=lambda row: row["delta_starter"])["modelo"]
    name = best.removeprefix("ensamble_")
    status = RESULTS / "final" / "seleccion.json"
    if status.exists() and json.loads(status.read_text(encoding="utf-8"))["modelo"] == best:
        return
    rows = np.arange(len(data.times))
    folder = status.parent
    folder.mkdir(exist_ok=True)
    deadline()
    s.log(f"Ajuste final con todos los datos. Modelo={best} semilla={s.SEEDS[0]}")
    checkpoint = s.fit(data, folder, rows, s.SEEDS[0], deadline, name)
    if best.startswith("ensamble_"):
        deadline()
        checkpoint["lineal"] = ridge.fit(ridge.Dataset(s.DATA), rows)
        checkpoint["peso_cnn"] = CNN_WEIGHT
    s.save(checkpoint, folder / "modelo.pt")
    s.read(folder / "modelo.pt")
    s.write_json({"modelo": best, "semilla": s.SEEDS[0], "n_horas": len(rows)}, status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experimento", choices=["todos", "lineal", "ensamble", "informe", *s.VARIANTS],
                        default="todos")
    parser.add_argument("--corte", choices=ridge.CUTS)
    parser.add_argument("--semilla", type=int, choices=s.SEEDS)
    parser.add_argument("--hilos", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--horas", type=float, default=6)
    parser.add_argument("--hasta", type=dt.datetime.fromisoformat, help="límite absoluto con zona horaria")
    args = parser.parse_args()
    if not 0 < args.horas <= 6:
        parser.error("--horas debe estar en (0, 6]")
    if args.hasta is not None and args.hasta.utcoffset() is None:
        parser.error("--hasta requiere zona horaria")
    end = time.monotonic() + args.horas * 3600

    def deadline():
        if time.monotonic() >= end or (args.hasta is not None and dt.datetime.now().astimezone() >= args.hasta):
            raise TimeoutError("Se alcanzó el límite de corrida. Se conservaron todos los resultados terminados.")

    s.RESULTS.mkdir(exist_ok=True)
    torch.set_num_threads(args.hilos)
    torch.set_num_interop_threads(1)
    if args.experimento == "informe":
        report()
        return
    cuts = [args.corte] if args.corte else list(ridge.CUTS)
    seeds = [args.semilla] if args.semilla is not None else s.SEEDS
    status = None
    try:
        data = s.Data(s.DATA)
        if args.experimento != "lineal":
            for seed in seeds:
                for cut in cuts:
                    s.run(data, cut, seed, deadline)
                    report()
        if args.experimento == "starter":
            return
        prepare_linear(cuts, deadline)
        if args.experimento == "lineal":
            return
        if args.experimento in ("todos", "ensamble", "rh_recorte"):
            for seed in seeds:
                for cut in cuts:
                    deadline()
                    only_clip(data, cut, seed)
                    ensemble(data, cut, seed, "starter")
                    ensemble(data, cut, seed, "rh_recorte")
                    if args.experimento == "todos":
                        checkpoint = s.read(s.RESULTS / f"{cut}_{seed}" / "modelo.pt")
                        if "ema_state_dict" in checkpoint:
                            s.run(data, cut, seed, deadline, "ema")
                            ensemble(data, cut, seed, "ema")
                    report()
        if args.experimento in ("ensamble", "rh_recorte"):
            if args.experimento == "ensamble":
                for name in ORDER:
                    for seed in seeds:
                        for cut in cuts:
                            deadline()
                            ensemble(data, cut, seed, name)
            return
        wanted = ORDER if args.experimento == "todos" else (args.experimento,)
        # Completa el barrido inicial en los tres cortes antes de las réplicas adicionales.
        for seed in seeds:
            for name in wanted:
                for cut in cuts:
                    deadline()
                    s.run(data, cut, seed, deadline, name)
                    ensemble(data, cut, seed, name)
                    report()
        if args.experimento == "todos" and args.corte is None and args.semilla is None:
            final_fit(data, deadline)
            status = "Matriz de evaluación completada. El ajuste final con todo el conjunto quedó guardado."
        s.log("Finalizó el experimento solicitado")
    except (TimeoutError, KeyboardInterrupt) as error:
        status = str(error) or "Corrida interrumpida. Se conservaron todos los resultados terminados."
        s.log(status)
    finally:
        report(status)


if __name__ == "__main__":
    main()


