"""Actualiza las tablas y el informe a partir de evaluaciones guardadas."""
import csv
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean

from ridge import CUTS


METRICS = ("score", "RH", "T", "U", "V")
GROUPS = ("todas", "cada3h", *(f"{h:02d}UTC" for h in range(24)))


def write_csv(path, rows, fields):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def aggregate(items):
    return {key: mean(row[key] for row in items) for key in METRICS}


def markdown(rows, records, experiments, seeds):
    baseline = {row["corte"]: row for row in rows
                if row["experimento"] == "lineal" and row["horas_UTC"] == "todas"}
    complete = []
    summarized = defaultdict(dict)
    for row in rows:
        if row["semilla"] == "media" and row["horas_UTC"] in ("todas", "cada3h", "00UTC"):
            summarized[row["experimento"]][row["corte"], row["horas_UTC"]] = row
    lines = [
        "# Corrección convolucional sobre ridge por celda",
        "",
        "## Referencia externa",
        "",
        "Según el registro del equipo, el baseline oficial es el starter completo de 25 épocas. "
        "La fila `submission_baseline_cnn.csv` del leaderboard público registró 0,42782 en Kaggle "
        "sobre el test de julio y agosto. El equipo aún no había enviado una predicción. "
        "La variante C original se evaluó localmente en mayo y junio. La evaluación local del starter "
        "se limitó a una corrida de dos épocas. El presente estudio compara solo la lineal adaptada "
        "y su corrección CNN en los mismos cortes locales. El score de Kaggle se cita como contexto externo.",
        "",
        "## Protocolo",
        "",
        "Se comparó la variante C adaptada del ridge con una CNN residual en la grilla de 3 km.",
        "El ridge usa GFS en t, t-1 y t-2, gradientes en t, PCA de 30 componentes del campo completo, "
        "24 indicadores de hora UTC e interacciones entre cada indicador y el GFS local en t centrado por celda. "
        "Se ajusta al residuo sobre GFS bilineal con regularización de 0,01 por observación e intercepto libre.",
        "",
        "Los cortes externos de 2024 comienzan el 16 de mayo, el 1 de junio y el 16 de junio. "
        "Cada corte termina al inicio del siguiente o al terminar junio. "
        "La lineal de evaluación usa todo el historial disponible anterior al corte. "
        "Para la CNN, el tramo de desarrollo se divide en cinco bloques contiguos. "
        "Cada predicción lineal de desarrollo excluye su bloque y al menos 24 horas a cada lado. "
        "Los ajustes de estos bloques pueden usar fechas posteriores al bloque dentro de desarrollo. "
        "Los siete días anteriores al corte se reservan para seleccionar la época con una separación adicional de un día. "
        "Su lineal se ajusta solo en desarrollo.",
        "",
        "La CNN recibe la lineal, los tres campos GFS interpolados, el DEM, su diferencia con un promedio móvil de 9 × 9 "
        "y la proyección de la hora one-hot. Predice el residuo dividido por la RMSE climatológica oficial de cada variable. "
        "La capa de salida inicia en cero. Se combinan MSE normalizada y RMSE normalizada con peso uniforme "
        "o peso doble para observaciones de mayo y junio. Se usa AdamW, promedio exponencial de pesos y "
        "selección por score interno tras un mínimo de 40 épocas de entrenamiento, con un máximo de 64. "
        "Puede restaurarse un punto anterior a la época 40 si obtuvo el mejor score interno.",
        "",
        "Se acota a [0, 100] la RH final de ambos modelos. En la prueba de desfase, todas las entradas GFS "
        "usadas por la validación se enfrían 1,5 K y se secan 8 puntos de RH, incluidos los rezagos. "
        "Se conservan las observaciones WRF y los parámetros ajustados. La prueba mide sensibilidad a "
        "una perturbación del GFS con WRF fijo.",
        "",
        "## Reproducción",
        "",
        "Desde PowerShell en la raíz del espacio de trabajo:",
        "",
        "```powershell",
        r"$python = '.\Rubber-duck-debuggers-andes-downscaling\.venv\Scripts\python.exe'",
        r"& $python -B .\rdd-codex-cnn\experimento.py --experimento todos",
        r"& $python -B .\rdd-codex-cnn\experimento.py --experimento lineal",
        r"& $python -B .\rdd-codex-cnn\experimento.py --experimento mse",
        r"& $python -B .\rdd-codex-cnn\experimento.py --experimento rmse",
        r"& $python -B .\rdd-codex-cnn\experimento.py --experimento mse_mayjun",
        r"& $python -B .\rdd-codex-cnn\experimento.py --experimento rmse_mayjun",
        "```",
        "",
        "El primer comando de entrenamiento ejecuta la matriz completa. Los siguientes permiten ejecutar un solo "
        "experimento. Se omiten los resultados completos ya guardados y se reutilizan los ajustes lineales. "
        "Cada carpeta de corte guarda los checkpoints, las métricas por réplica y el seguimiento por época.",
        "",
        "## Resultados",
        "",
        "Las medias por corte ponderan por igual las semillas disponibles. La fila «media cortes» pondera "
        "cada corte por igual y se publica solo con todas las semillas configuradas. "
        "Δ CNN compara con la lineal del mismo corte y escenario. Δ desfase compara con el score nominal del mismo modelo.",
        "",
        "| Modelo | Corte | Semillas | Score | Δ CNN | Score cada 3 h | RH | T | U | V | Score desfase | Δ desfase | Δ CNN con desfase |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("lineal", *experiments):
        lookup = summarized[name]
        for cut in (*CUTS, "media cortes"):
            row = lookup.get((cut, "todas"))
            qualified = lookup.get((cut, "cada3h"))
            if row is None:
                continue
            b = baseline.get(cut)
            delta = row["score"] - b["score"] if b else 0.0
            shifted = row["score_desfase"] - b["score_desfase"] if b else 0.0
            deltas = [delta, row["desfase_delta"], shifted]
            rmse = " | ".join(f"{row[key]:.3f}" for key in METRICS[1:])
            step = qualified["score"]
            lines.append(f"| {name} | {cut} | {row['n_semillas']} | {row['score']:.4f} | "
                         f"{deltas[0]:+.4f} | {step:.4f} | {rmse} | "
                         f"{row['score_desfase']:.4f} | {deltas[1]:+.4f} | {deltas[2]:+.4f} |")
        avg = lookup.get(("media cortes", "todas"))
        if name == "lineal" or avg is None or "media cortes" not in baseline:
            continue
        delta = avg["score"] - baseline["media cortes"]["score"]
        qualifies = delta >= 0.01 and all(
            lookup[cut, "todas"]["score_desfase"] >= baseline[cut]["score_desfase"]
            for cut in CUTS)
        complete.append((delta, name, qualifies))

    lines.extend([
        "",
        "### RMSE a las 00 UTC",
        "",
        "| Modelo | Corte | RH | T | U | V |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for name in ("lineal", *experiments):
        for cut in (*CUTS, "media cortes"):
            row = summarized[name].get((cut, "00UTC"))
            if row:
                rmse = " | ".join(f"{row[key]:.3f}" for key in METRICS[1:])
                lines.append(f"| {name} | {cut} | {rmse} |")
    lines += [
        "",
        "El [CSV de resultados](tabla_residual_previa.csv) incluye las 24 horas UTC, el subconjunto de "
        "cada tres horas, los valores por semilla, las medias y el detalle de ambos escenarios.",
        "",
        "## Estado y recomendación",
        "",
    ]
    pending = []
    for name in ("lineal", *experiments):
        expected = [(cut, seed) for cut in CUTS for seed in ([None] if name == "lineal" else seeds)]
        present = {(item["corte"], item["semilla"]) for item in records if item["experimento"] == name}
        missing = [f"{cut}" + (f"/{seed}" if seed is not None else "") for cut, seed in expected
                   if (cut, seed) not in present]
        if missing:
            pending.append(f"- {name}. Pendientes " + ", ".join(missing) + ".")
    if pending:
        lines += pending + [""]
    accepted = [item for item in complete if item[2]]
    if complete:
        gain, name, _ = max(accepted or complete)
        descriptor = "La configuración recomendada" if accepted else "La mejor configuración completa"
        lines.append(f"{descriptor} fue **{name}**, con una ganancia media de **{gain:+.4f}**.")
        lookup = summarized[name]
        avg = lookup["media cortes", "todas"]
        shifted_gain = avg["score_desfase"] - baseline["media cortes"]["score_desfase"]
        nominal3 = lookup["media cortes", "cada3h"]
        baseline3 = summarized["lineal"]["media cortes", "cada3h"]
        lines.append(f"La ganancia media en el escenario perturbado fue **{shifted_gain:+.4f}**. "
                     f"En las horas evaluables, la ganancia fue **{nominal3['score'] - baseline3['score']:+.4f}** "
                     f"en condiciones nominales y **{nominal3['score_desfase'] - baseline3['score_desfase']:+.4f}** "
                     "con desfase.")
    if accepted:
        lines += ["", "Sí. La CNN alcanzó el umbral medio y conservó al menos el score de la lineal bajo "
                  "desfase en cada corte. El siguiente paso es confirmar la variante en un periodo sin usar "
                  "para elegir la configuración y medir el coste de inferencia."]
    elif complete:
        lines += ["", "**No se recomienda la CNN evaluada.** Ninguna configuración completa alcanzó "
                  "simultáneamente la ganancia media de +0,01 y un score bajo desfase igual o superior "
                  "a la lineal en cada corte. El siguiente paso es conservar el ridge y evaluar su "
                  "ponderación estacional con el mismo protocolo."]
    else:
        lines += ["Aún no hay una configuración CNN completa en los tres cortes. La recomendación queda "
                  "pendiente. El siguiente paso es completar las combinaciones indicadas arriba."]
    return "\n".join(lines) + "\n"


def publish(root, experiments, seeds):
    records = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(root.glob("*/*.json"))]
    rows = []
    buckets = defaultdict(list)
    for record in records:
        for group in GROUPS:
            if group not in record["normal"]:
                continue
            normal, shifted = (record[scenario][group] for scenario in ("normal", "desfase"))
            exp, cut, seed = (record[key] for key in ("experimento", "corte", "semilla"))
            training = record["entrenamiento"] or {}
            row = {"experimento": exp, "corte": cut, "semilla": seed if seed is not None else "lineal",
                   "horas_UTC": group, "n_horas": normal["horas"], "n_semillas": int(seed is not None),
                   "epocas": training.get("epochs"), "mejor_epoca": training.get("best_epoch"),
                   **{key: normal[key] for key in METRICS},
                   **{key + "_desfase": shifted[key] for key in METRICS},
                   "desfase_delta": shifted["score"] - normal["score"]}
            rows.append(row)
            buckets[exp, cut, group].append(row)
    by_experiment = defaultdict(list)
    metric_names = [*METRICS, *(key + "_desfase" for key in METRICS), "desfase_delta"]
    for (exp, cut, group), items in buckets.items():
        row = {"experimento": exp, "corte": cut, "semilla": "media", "horas_UTC": group,
               "n_horas": items[0]["n_horas"], "n_semillas": 0 if exp == "lineal" else len(items),
               **{key: mean(item[key] for item in items) for key in metric_names}}
        rows.append(row)
        if exp == "lineal" or len(items) == len(seeds):
            by_experiment[exp, group].append(row)
    for (exp, group), items in by_experiment.items():
        if len(items) == len(CUTS):
            rows.append({"experimento": exp, "corte": "media cortes", "semilla": "media",
                         "horas_UTC": group, "n_horas": sum(item["n_horas"] for item in items),
                         "n_semillas": items[0]["n_semillas"],
                         **{key: mean(item[key] for item in items) for key in metric_names}})
    baseline = {(row["corte"], row["horas_UTC"]): row for row in rows
                if row["experimento"] == "lineal" and row["semilla"] == "media"}
    for row in rows:
        reference = baseline.get((row["corte"], row["horas_UTC"]))
        if reference is not None:
            row["ganancia"] = row["score"] - reference["score"]
            row["ganancia_desfase"] = row["score_desfase"] - reference["score_desfase"]
    columns = ("experimento", "corte", "semilla", "horas_UTC", "n_horas", "n_semillas",
               "epocas", "mejor_epoca", *metric_names, "ganancia", "ganancia_desfase")
    write_csv(root / "tabla_residual_previa.csv", rows, columns)
    temp = root / "Documentacion_CNN.tmp"
    temp.write_text(markdown(rows, records, experiments, seeds), encoding="utf-8")
    temp.replace(root / "protocolo_residual_interrumpido.md")



