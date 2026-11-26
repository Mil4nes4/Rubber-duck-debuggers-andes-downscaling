"""
Backend mínimo del prototipo (FastAPI).

Correr desde la raíz del repo:
    python -m uvicorn app.servidor:app --port 8000
Endpoints:
    GET /            página del prototipo (app/static/index.html)
    GET /salud       estado del modelo y del GFS
    GET /pronostico  resumen por hora (°C, RH %, rapidez del viento m/s), horas UTC y Perú
    GET /mapa?hora=  grilla 70x35 con lat/lon; hora = índice (0..N-1) o 'YYYYMMDDHH' UTC
"""
import os
import sys
import threading

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inferencia import Pronostico  # noqa: E402

app = FastAPI(title="Prototipo downscaling Cordillera Blanca", version="0.1-prototipo")
_lock = threading.Lock()
_estado = {"p": None}
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def pronostico():
    with _lock:
        if _estado["p"] is None:
            _estado["p"] = Pronostico()
        return _estado["p"].actualizar()


@app.get("/")
def inicio():
    return FileResponse(os.path.join(STATIC, "index.html"))


@app.get("/salud")
def salud():
    p = pronostico()
    return dict(ok=True, prototipo=True, modelo=p.pred.fuente, avisos=p.pred.avisos, gfs=p.info,
                horas=len(p.times), desde_utc=str(p.times[0]), hasta_utc=str(p.times[-1]))


@app.get("/pronostico")
def get_pronostico():
    p = pronostico()
    return dict(prototipo=True, modelo=p.pred.fuente, avisos=p.pred.avisos, gfs=p.info,
                zona_horaria_peru="UTC-5", unidades=dict(T="°C", RH="%", viento="m/s"),
                horas=p.resumen())


@app.get("/mapa")
def get_mapa(hora: str = Query("0", description="índice de hora o 'YYYYMMDDHH' UTC")):
    p = pronostico()
    times = [str(t) for t in p.times]
    if len(hora) == 10 and hora in times:
        i = times.index(hora)
    else:
        try:
            i = int(hora)
        except ValueError:
            raise HTTPException(400, "hora debe ser un índice o 'YYYYMMDDHH' UTC")
        if not 0 <= i < len(times):
            raise HTTPException(404, f"índice fuera de rango (0..{len(times) - 1})")
    return dict(prototipo=True, modelo=p.pred.fuente, indice=i, **p.mapa(i))
