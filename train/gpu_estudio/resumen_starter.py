"""Tabula la evaluación local con referencia climatológica oficial."""
import csv
import json
from pathlib import Path
from statistics import mean

from ridge import CUTS


GROUPS = ("todas", "cada3h", *(f"{h:02d}UTC" for h in range(24)))
METRICS = ("score", "RH", "T", "U", "V")
KAGGLE = 0.42782
CNN_WEIGHT = 0.75


def collect(root):
    paths = [*root.glob("starter/*/metricas.json"), *root.glob("ablaciones/*/*/metricas.json"),
             *root.glob("ensambles/*/*/metricas.json")]
    records = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(paths)]
    for cut in CUTS:
        path = root / cut / "lineal.json"
        if path.exists():
            item = json.loads(path.read_text(encoding="utf-8"))
            records.append({"modelo": "lineal", "corte": cut, "semilla": None,
                            "normal": item["normal"], "desfase": item["desfase"]})
    return records


def make_rows(records, seeds):
    rows = []
    pairs = {(item["corte"], item["semilla"]): item for item in records if item["modelo"] == "starter"}
    by_cut = {}
    models = list(dict.fromkeys(item["modelo"] for item in records))
    for item in records:
        name, cut, seed = (item[field] for field in ("modelo", "corte", "semilla"))
        reference = pairs.get((cut, seed))
        for group in GROUPS:
            if group not in item["normal"]:
                continue
            normal, shift = item["normal"][group], item["desfase"][group]
            row = {"modelo": name, "corte": cut, "semilla": seed if seed is not None else "lineal",
                   "horas_UTC": group, "n_horas": normal["horas"], "n_semillas": int(seed is not None),
                   "epocas": item.get("epocas"), **{key: normal[key] for key in METRICS},
                   **{key + "_desfase": shift[key] for key in METRICS},
                   "delta_desfase": shift["score"] - normal["score"]}
            if reference:
                row["delta_starter"] = normal["score"] - reference["normal"][group]["score"]
                row["delta_starter_desfase"] = shift["score"] - reference["desfase"][group]["score"]
            rows.append(row)
            by_cut.setdefault((name, cut, group), []).append(row)

    numeric = (*METRICS, *(m + "_desfase" for m in METRICS), "delta_desfase",
               "delta_starter", "delta_starter_desfase")

    def averages(group_rows):
        return {key: mean(row[key] for row in group_rows) for key in numeric
                if all(key in row for row in group_rows)}

    averages_by_cut = {}
    for (name, cut, group), items in by_cut.items():
        summary = {"modelo": name, "corte": cut, "semilla": "media", "horas_UTC": group,
                   "n_horas": items[0]["n_horas"],
                   "n_semillas": 0 if name == "lineal" else len(items), **averages(items)}
        if name == "lineal":
            starter = by_cut.get(("starter", cut, group), [])
            if starter:
                summary["delta_starter"] = summary["score"] - mean(x["score"] for x in starter)
                summary["delta_starter_desfase"] = summary["score_desfase"] - mean(x["score_desfase"] for x in starter)
        averages_by_cut[name, cut, group] = summary
        rows.append(summary)
    for name in models:
        for group in GROUPS:
            keyed = [by_cut.get((name, cut, group), []) for cut in CUTS]
            if not all(keyed):
                continue
            common = set.intersection(*(set(x["semilla"] for x in items) for items in keyed))
            if not common:
                continue
            matched = [[x for x in items if x["semilla"] in common] for items in keyed]
            items = [{**averages(x), "n_horas": x[0]["n_horas"]} for x in matched]
            if name == "lineal":
                for row, cut in zip(items, CUTS):
                    raw = averages_by_cut[name, cut, group]
                    for key in ("delta_starter", "delta_starter_desfase"):
                        if key in raw:
                            row[key] = raw[key]
            rows.append({"modelo": name, "corte": "media cortes", "semilla": "media", "horas_UTC": group,
                         "n_horas": sum(x["n_horas"] for x in items),
                         "n_semillas": 0 if name == "lineal" else len(common), **averages(items)})
    return rows, (*numeric,)


def fmt(row, key, decimals=4, signed=False):
    value = row.get(key)
    return "s/d" if value is None else format(value, f"{'+' if signed else ''}.{decimals}f")


def save_report(root, variants, seeds, status=None):
    rows, numeric = make_rows(collect(root), seeds)
    columns = ("modelo", "corte", "semilla", "horas_UTC", "n_horas", "n_semillas", "epocas", *numeric)
    csv_path = root / "tabla_resultados.csv"
    temp = csv_path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(csv_path)

    levels = {(row["modelo"], row["corte"], row["horas_UTC"]): row
              for row in rows if row["semilla"] == "media"}
    model_order = ["starter", "lineal", *(name for name in variants if name != "starter"),
                   *(f"ensamble_{name}" for name in variants)]
    lines = [
        "# Evaluación del starter y sus mejoras",
        "",
        "El baseline de la organización es el starter completo de 25 épocas. Según el registro del equipo, "
        "la fila `submission_baseline_cnn.csv` obtuvo 0,42782 en el leaderboard público sobre el test "
        "de julio y agosto. La comparación de este estudio es local y usa la referencia climatológica oficial. "
        "La brecha con Kaggle puede reflejar el periodo, el subconjunto de horas, los datos de ajuste y "
        "la regla de selección de pesos. La lineal C histórica y el starter local de dos épocas "
        "pertenecen a evaluaciones anteriores.",
        "",
        "## Protocolo",
        "",
        "Se ajusta cada modelo con datos previos al corte y se evalúa en el tramo siguiente. Los cortes "
        "comienzan el 16 de mayo, 1 de junio y 16 de junio de 2024. El starter conserva la arquitectura "
        "de 64 canales, ventanas t-2, t-1 y t, climatología horaria y MSE de anomalías normalizadas. "
        "Se conserva AdamW, OneCycleLR y el tamaño de lote. El checkpoint corresponde a la última de "
        "25 épocas, sin inspección de la validación durante el entrenamiento. La semilla 0 replica "
        "la del notebook. Las otras dos semillas se comparan de forma pareada.",
        "",
        "Cada ablación cambia un elemento del starter. Los rezagos t-6, t-12 y t-24 se prueban "
        "por separado. Los rasgos derivados son gradientes espaciales por variable y rapidez del viento "
        "en t. La RMSE se calcula por variable en cada lote usando unidades físicas y la referencia "
        "oficial. El recorte de RH se evalúa sobre las predicciones de la CNN ya entrenada. Para EMA "
        "se usa el promedio acumulado hasta la última actualización. Se incluye una prueba de ponderación "
        "doble para mayo y junio.",
        "",
        "La lineal C adaptada se incorpora al ensamble con pesos fijos de 75 % para la CNN y 25 % para "
        "la lineal. Los pesos se fijaron sin usar scores externos. La lineal usa RH acotada a [0,100]. "
        "La perturbación estacional enfría el GFS 1,5 K y reduce RH en 8 puntos, también en los rezagos. "
        "Se conserva el WRF observado para medir sensibilidad a la entrada.",
        "",
        "## Resultados",
        "",
        "El score es 1 menos el promedio de RMSE por variable dividido entre su referencia oficial. "
        "Cada fila con media de los tres cortes usa las semillas disponibles en común. "
        "Las deltas de las ablaciones y ensambles se calculan contra las réplicas correspondientes del starter. "
        "Se usa el subconjunto cada 3 h para la comparación principal porque coincide con el horario de Kaggle.",
        "",
        "| Modelo | Corte | Semillas | Score | Score cada 3 h | Δ starter cada 3 h | RMSE RH | T | U | V | Score desfase | Δ desfase | Desfase cada 3 h | Δ desfase cada 3 h |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in model_order:
        for cut in (*CUTS, "media cortes"):
            row = levels.get((name, cut, "todas"))
            step = levels.get((name, cut, "cada3h"))
            if row is None or step is None:
                continue
            values = [name, cut, str(row["n_semillas"]), fmt(row, "score"), fmt(step, "score"),
                      fmt(step, "delta_starter", signed=True),
                      *(fmt(row, m, 3) for m in METRICS[1:]), fmt(row, "score_desfase"),
                      fmt(row, "delta_desfase", signed=True), fmt(step, "score_desfase"), fmt(step, "delta_desfase", signed=True)]
            lines.append("| " + " | ".join(values) + " |")
    lines += ["", "### RMSE a las 00 UTC", "", "| Modelo | Corte | RH | T | U | V |", "|---|---|---:|---:|---:|---:|"]
    for name in model_order:
        for cut in (*CUTS, "media cortes"):
            row = levels.get((name, cut, "00UTC"))
            if row:
                lines.append("| " + " | ".join([name, cut, *(fmt(row, m, 3) for m in METRICS[1:])]) + " |")
    lines += ["", "El [CSV de resultados](results/tabla_resultados.csv) detalla todas las horas UTC, "
              "cada semilla y las métricas con y sin perturbación.", "", "## Brecha respecto de Kaggle", ""]
    reference = levels.get(("starter", "media cortes", "cada3h"))
    if reference:
        lines.append(f"El starter completo obtuvo **{reference['score']:.5f}** de score local en las "
                     f"horas evaluables, una brecha local menos Kaggle de **{reference['score'] - KAGGLE:+.5f}**.")
    else:
        lines.append("La brecha se calcula cuando hay una misma semilla del starter completo en los tres cortes.")

    lines += ["", "## Estado", ""]
    if status is None:
        session_log = root / "starter" / "corrida.log"
        if session_log.exists():
            events = session_log.read_text(encoding="utf-8").splitlines()
            if events and "límite de corrida" in events[-1]:
                status = "Se alcanzó el límite de seis horas de reloj. Se conservaron las corridas completas."
    if status:
        lines += [status, ""]
    if not (root / "final" / "seleccion.json").exists():
        lines += ["El ajuste con todo el conjunto queda pendiente del cierre de las comparaciones.", ""]
    recommendation = []
    candidates = [levels[name, "media cortes", "cada3h"] for name in model_order
                  if name not in ("starter", "lineal") and (name, "media cortes", "cada3h") in levels]
    full = [row for row in candidates if row["n_semillas"] == len(seeds)]
    best = max(full, key=lambda x: x["delta_starter"], default=None)
    if best and best["delta_starter"] > 0:
        recommendation.append(f"Sí se recomienda **{best['modelo']}** entre las variantes completas. Su ganancia "
                     f"media fue **{best['delta_starter']:+.5f}** en las horas evaluables. Bajo perturbación, "
                     f"su diferencia frente al starter fue **{best['delta_starter_desfase']:+.5f}**. "
                     "La selección de variante se basa en estos cortes. El siguiente paso es verificarla "
                     "en un periodo independiente.")
    elif best:
        recommendation.append(f"Se recomienda conservar el starter entre las variantes completas. La mejor alternativa "
                     f"evaluada fue **{best['modelo']}**, con una diferencia media de "
                     f"**{best['delta_starter']:+.5f}** en las horas evaluables. El siguiente paso es "
                     "completar las comparaciones pendientes si quedan variantes incompletas.")
    else:
        partial = [row for row in candidates if row["n_semillas"] < len(seeds)]
        first = max(partial, key=lambda x: x["delta_starter"], default=None)
        if first and first["delta_starter"] > 0:
            name = first["modelo"]
            n = first["n_semillas"]
            replicates = "una semilla" if n == 1 else f"{n} semillas"
            recommendation.append(
                f"**Sí, de forma provisional se recomienda {name}.** Con {replicates} en los "
                f"tres cortes alcanzó **{first['score']:.5f}** en las horas evaluables, "
                f"una ganancia de **{first['delta_starter']:+.5f}** frente al starter completo. "
                f"Con el GFS perturbado, la ventaja fue de **{first['delta_starter_desfase']:+.5f}** "
                f"y el score cambió **{first['delta_desfase']:+.5f}** respecto del escenario nominal. "
                "El siguiente paso es confirmar la ganancia con las semillas pendientes y completar "
                "las ablaciones antes de elegir el modelo que se ajustará con todos los datos.")
        else:
            recommendation.append("La recomendación queda pendiente de completar las comparaciones "
                                  "en los tres cortes y las tres semillas.")
    lines += ["", "### Cobertura", ""]
    raw = [row for row in rows if row["horas_UTC"] == "cada3h" and isinstance(row["semilla"], int)]
    for name in variants:
        missing = [f"{cut}/{seed}" for cut in CUTS for seed in seeds
                   if not any(row["modelo"] == name and row["corte"] == cut and row["semilla"] == seed for row in raw)]
        if missing:
            lines.append(f"- {name}. Pendientes " + ", ".join(missing) + ".")
    lines += [
        "",
        "## Reproducción",
        "",
        "Los comandos se ejecutan en PowerShell desde la raíz del espacio de trabajo.",
        "",
        "```powershell",
        r"$python = '.\Rubber-duck-debuggers-andes-downscaling\.venv\Scripts\python.exe'",
        *[rf"& $python -B .\rdd-codex-cnn\estudio.py --experimento {name}"
          for name in ("todos", *variants, "lineal", "ensamble", "informe")],
        "```",
        "",
        "Los checkpoints contienen tensores de pesos, coordenadas y estadísticas de normalización. "
        "Las predicciones guardadas incluyen sus índices, fechas UTC y los dos escenarios. "
        "Cada entrenamiento registra el tiempo y la pérdida de cada época.",
    ]
    lines += ["", "## Recomendación", "", *recommendation]
    report_tmp = root / "informe_starter.tmp"
    report_tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_tmp.replace(root.parent / "Documentacion_CNN.md")


