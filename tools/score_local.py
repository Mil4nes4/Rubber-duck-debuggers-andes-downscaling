"""Score local uniforme sobre la validación temporal (times >= '2024061600').

Formato del archivo de predicciones (.npz):
    pred  : float, forma (N_val, 70, 35, 4), unidades originales (RH %, T K, U m/s, V m/s),
            orden de variables RH, T, U, V; fila 0 = sur (igual que Y de train.npz).
    times : (opcional pero recomendado) str 'YYYYMMDDHH' de cada fila de pred.
            Si falta, se asume el orden de las horas de validación de train.npz.

Para guardarlo desde un notebook:
    val = ttr >= '2024061600'
    np.savez_compressed('pred_val_expXX.npz', pred=P_val.astype(np.float32), times=ttr[val])

Uso (desde la raíz del repo):
    python tools/score_local.py pred_val_expXX.npz [otro.npz ...]
    python tools/score_local.py --demo-clim pred_val_clim.npz   # genera la climatología como referencia

Se informan dos scores:
- score_local: con el RMSE de la climatología horaria de train (antes del corte) evaluada en validación,
  igual que el starter notebook.
- score_ref_oficial: con los RMSE_clim de referencia oficial (RH 25.360, T 2.0652, U 2.6776, V 2.0876).
  Sirve para comparar con el leaderboard, aunque la validación de junio puede ser optimista.
"""
import argparse
import os
import sys

import numpy as np

VARS = ["RH", "T", "U", "V"]
CORTE_VAL = "2024061600"
RMSE_CLIM_OFICIAL = np.array([25.360, 2.0652, 2.6776, 2.0876])


def buscar_train():
    aqui = os.path.dirname(os.path.abspath(__file__))
    for c in [os.path.join(aqui, "..", "data", "train.npz"), os.path.join("data", "train.npz"),
              os.path.join("..", "data", "train.npz")]:
        if os.path.exists(c):
            return c
    return None


def cargar_validacion(ruta_train):
    tr = np.load(ruta_train)
    Y, t = tr["Y"], tr["times"].astype(str)
    val = t >= CORTE_VAL
    trn = ~val
    hora = np.array([int(k[-2:]) for k in t])
    # float64 para que los promedios no acumulen error de redondeo de float32
    clim = np.stack([Y[trn & (hora == h)].mean(0, dtype=np.float64) for h in range(24)])  # (24,70,35,4)
    Yv, tv, hv = Y[val].astype(np.float64), t[val], hora[val]
    rmse_clim = np.sqrt(((clim[hv] - Yv) ** 2).mean(axis=(0, 1, 2)))
    return Yv, tv, clim[hv], rmse_clim


def puntuar(pred, Yv, rmse_clim):
    rmse = np.sqrt(((pred.astype(np.float64) - Yv) ** 2).mean(axis=(0, 1, 2)))
    return {
        "rmse": rmse,
        "score_local": 1 - (rmse / rmse_clim).mean(),
        "score_ref_oficial": 1 - (rmse / RMSE_CLIM_OFICIAL).mean(),
        "ratio_local": rmse / rmse_clim,
    }


def alinear(arch, Yv, tv):
    pred = arch["pred"]
    if pred.ndim != 4 or pred.shape[1:] != (70, 35, 4):
        raise ValueError(f"pred con forma {pred.shape}; se espera (N, 70, 35, 4)")
    if not np.all(np.isfinite(pred)):
        raise ValueError(f"pred tiene {int((~np.isfinite(pred)).sum())} valores no finitos")
    if "times" in arch.files:
        tp = arch["times"].astype(str)
        if len(tp) != len(pred):
            raise ValueError("times y pred tienen distinto largo")
        pos = {k: i for i, k in enumerate(tp)}
        faltan = [k for k in tv if k not in pos]
        if faltan:
            raise ValueError(f"faltan {len(faltan)} horas de validación en times (ej. {faltan[0]})")
        return pred[[pos[k] for k in tv]]
    if len(pred) != len(tv):
        raise ValueError(f"pred tiene {len(pred)} filas y la validación {len(tv)}; agrega 'times' al npz")
    return pred


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("npz", nargs="+", help="archivos .npz de predicciones de validación")
    ap.add_argument("--train", default=None, help="ruta de train.npz (por defecto, data/)")
    ap.add_argument("--demo-clim", action="store_true",
                    help="en vez de puntuar, guarda la climatología horaria en el primer npz indicado")
    a = ap.parse_args()

    ruta_train = a.train or buscar_train()
    if ruta_train is None:
        print("ERROR: no se encuentra train.npz; usa --train")
        sys.exit(1)
    Yv, tv, clim_v, rmse_clim = cargar_validacion(ruta_train)

    if a.demo_clim:
        np.savez_compressed(a.npz[0], pred=clim_v.astype(np.float32), times=tv)
        print(f"Climatología de validación guardada en {a.npz[0]} ({len(tv)} horas)")
        return

    print(f"Validación: {len(tv)} horas ({tv[0]} a {tv[-1]})")
    print("RMSE climatología local: " + ", ".join(f"{v} {r:.3f}" for v, r in zip(VARS, rmse_clim)))
    print()
    cab = f"{'archivo':40s} {'score_local':>11s} {'score_ofic':>10s} " + " ".join(f"{'RMSE_' + v:>8s}" for v in VARS)
    print(cab)
    print("-" * len(cab))
    hubo_error = False
    for ruta in a.npz:
        try:
            pred = alinear(np.load(ruta), Yv, tv)
        except Exception as e:  # noqa: BLE001
            print(f"{os.path.basename(ruta):40s} ERROR: {e}")
            hubo_error = True
            continue
        r = puntuar(pred, Yv, rmse_clim)
        print(f"{os.path.basename(ruta)[:40]:40s} {r['score_local']:11.4f} {r['score_ref_oficial']:10.4f} "
              + " ".join(f"{x:8.3f}" for x in r["rmse"]))
    sys.exit(1 if hubo_error else 0)


if __name__ == "__main__":
    main()
