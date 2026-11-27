"""Valida un CSV de submission contra data/sample_submission.csv.

Uso (desde la raíz del repo):
    python tools/verificar_submission.py submission.csv
    python tools/verificar_submission.py submission.csv --sample data/sample_submission.csv

Comprueba:
- columnas exactas: id, RH, T, U, V (en ese orden)
- 646.800 filas, ids únicos, mismas ids y mismo orden que sample_submission
- valores numéricos y finitos (sin NaN ni inf)
- rangos físicos plausibles: RH en [0, 100] %, T en Kelvin, |U| y |V| acotados

Devuelve código de salida 0 si todo está bien y 1 si hay algún error.
Las advertencias (valores raros pero posibles) no cambian el código de salida.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

N_FILAS = 646_800
COLUMNAS = ["id", "RH", "T", "U", "V"]

# (mínimo, máximo) para error duro y para advertencia.
# Error: fuera de lo físicamente posible en la zona. Advertencia: raro para la zona (2 jul - 4 ago, 115-5782 m).
RANGOS = {
    "RH": {"error": (0.0, 100.0), "aviso": (1.0, 100.0)},
    "T": {"error": (200.0, 330.0), "aviso": (240.0, 315.0)},   # Kelvin
    "U": {"error": (-75.0, 75.0), "aviso": (-40.0, 40.0)},     # m/s
    "V": {"error": (-75.0, 75.0), "aviso": (-40.0, 40.0)},     # m/s
}


def buscar_sample():
    aqui = os.path.dirname(os.path.abspath(__file__))
    candidatos = [
        os.path.join(aqui, "..", "data", "sample_submission.csv"),
        os.path.join("data", "sample_submission.csv"),
        os.path.join("..", "data", "sample_submission.csv"),
    ]
    for c in candidatos:
        if os.path.exists(c):
            return c
    return None


def verificar(ruta_csv, ruta_sample):
    errores, avisos = [], []

    sub = pd.read_csv(ruta_csv, dtype={"id": str})
    sample = pd.read_csv(ruta_sample, usecols=["id"], dtype={"id": str})

    # Columnas
    if list(sub.columns) != COLUMNAS:
        errores.append(f"columnas {list(sub.columns)}; se esperaba {COLUMNAS}")
        faltan = [c for c in COLUMNAS if c not in sub.columns]
        if faltan:
            return errores, avisos

    # Filas e ids
    if len(sub) != N_FILAS:
        errores.append(f"{len(sub)} filas; se esperaban {N_FILAS}")
    if len(sample) != N_FILAS:
        avisos.append(f"sample_submission tiene {len(sample)} filas (se esperaban {N_FILAS})")
    n_unicos = sub["id"].nunique()
    if n_unicos != len(sub):
        errores.append(f"ids repetidos: {len(sub) - n_unicos}")
    if sub["id"].isna().any():
        errores.append(f"ids vacíos: {int(sub['id'].isna().sum())}")

    ids_sub, ids_ref = sub["id"].values, sample["id"].values
    if len(ids_sub) == len(ids_ref):
        distintos = np.flatnonzero(ids_sub != ids_ref)
        if len(distintos):
            set_sub, set_ref = set(ids_sub), set(ids_ref)
            if set_sub == set_ref:
                errores.append(f"mismas ids pero distinto orden ({len(distintos)} posiciones; primera en fila {distintos[0]})")
            else:
                errores.append(f"ids distintas: {len(set_ref - set_sub)} faltan, {len(set_sub - set_ref)} sobran "
                               f"(ej. fila {distintos[0]}: '{ids_sub[distintos[0]]}' vs '{ids_ref[distintos[0]]}')")
    else:
        set_sub, set_ref = set(ids_sub), set(ids_ref)
        errores.append(f"ids: {len(set_ref - set_sub)} faltan, {len(set_sub - set_ref)} sobran")

    # Valores
    resumen = {}
    for v in ["RH", "T", "U", "V"]:
        col = pd.to_numeric(sub[v], errors="coerce").values.astype(np.float64)
        no_num = int(np.isnan(col).sum() - sub[v].isna().sum())
        if no_num > 0:
            errores.append(f"{v}: {no_num} valores no numéricos")
        no_fin = int((~np.isfinite(col)).sum())
        if no_fin:
            errores.append(f"{v}: {no_fin} valores NaN o infinitos")
        fin = col[np.isfinite(col)]
        if len(fin) == 0:
            continue
        lo, hi = RANGOS[v]["error"]
        fuera = int(((fin < lo) | (fin > hi)).sum())
        if fuera:
            errores.append(f"{v}: {fuera} valores fuera de [{lo}, {hi}] (min {fin.min():.3f}, max {fin.max():.3f})")
        lo_a, hi_a = RANGOS[v]["aviso"]
        raros = int(((fin < lo_a) | (fin > hi_a)).sum())
        if raros and not fuera:
            avisos.append(f"{v}: {raros} valores fuera del rango habitual [{lo_a}, {hi_a}]")
        resumen[v] = (fin.min(), fin.mean(), fin.max(), fin.std())

    # Señales de unidades equivocadas
    if "T" in resumen and resumen["T"][1] < 100:
        errores.append(f"T media {resumen['T'][1]:.2f}: parece estar en °C; debe ir en Kelvin")
    if "RH" in resumen and resumen["RH"][2] <= 1.0:
        errores.append("RH máxima <= 1: parece estar en fracción; debe ir en % (0-100)")
    for v, (mn, me, mx, sd) in resumen.items():
        if sd == 0:
            avisos.append(f"{v}: todos los valores son iguales ({me:.3f})")

    return errores, avisos, resumen


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="ruta del CSV a validar")
    ap.add_argument("--sample", default=None, help="ruta de sample_submission.csv (por defecto, data/)")
    a = ap.parse_args()

    ruta_sample = a.sample or buscar_sample()
    if ruta_sample is None or not os.path.exists(ruta_sample):
        print("ERROR: no se encuentra sample_submission.csv; usa --sample")
        sys.exit(1)

    res = verificar(a.csv, ruta_sample)
    errores, avisos = res[0], res[1]
    resumen = res[2] if len(res) > 2 else {}

    print(f"Archivo: {a.csv}")
    print(f"Referencia: {ruta_sample}")
    for v, (mn, me, mx, sd) in resumen.items():
        print(f"  {v}: min {mn:9.3f} | media {me:9.3f} | max {mx:9.3f} | desv {sd:8.3f}")
    for w in avisos:
        print("AVISO:", w)
    for e in errores:
        print("ERROR:", e)
    if errores:
        print(f"RESULTADO: NO VÁLIDO ({len(errores)} errores)")
        sys.exit(1)
    print("RESULTADO: VÁLIDO")


if __name__ == "__main__":
    main()
