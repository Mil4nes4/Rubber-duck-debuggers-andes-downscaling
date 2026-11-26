"""
Inferencia del prototipo: GFS actual (Open-Meteo) -> mapas 3 km (70x35) con los pesos del modelo.

Orden de carga del modelo (el primero que exista):
  1. models/modelo_final.pt   checkpoint del contrato modelo-app (dict con 'state_dict' + estadísticas)
  2. models/modelo_app_rapido.pt  checkpoint del contrato generado por este módulo (respaldo)
  3. models/modelo.pt         formato del starter (solo state_dict); las estadísticas se recalculan con data/train.npz
  4. Sin pesos: entrena el modelo del starter 2 épocas (respaldo, se guarda como modelo_app_rapido.pt)
  5. Si todo falla: interpolación bilineal del GFS (respaldo explícito, sin modelo)

Contrato del checkpoint (propuesta del equipo):
  {'state_dict': ..., 'arquitectura': 'starter', 'mu_x': (4,), 'sd_x': (4,),
   'clim': (24,70,35,4), 'anom_sd': (4,)}
  entrada ventanas (N,3,17,13,4) + hora UTC -> salida (N,70,35,4) en unidades originales (RH %, T K, U, V m/s).
  Si otro modelo usa otra arquitectura, basta con registrar su clase en ARQUITECTURAS.
"""
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.set_num_threads(2)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(APP_DIR)
DATA = os.path.join(ROOT, "data")
MODELS = os.path.join(ROOT, "models")
CACHE_DIR = os.path.join(APP_DIR, "cache")
GFS_CACHE = os.path.join(CACHE_DIR, "gfs_cache.npz")
GFS_ULTIMA = os.path.join(CACHE_DIR, "gfs_ultima_ok.npz")
CACHE_MAX_AGE_H = float(os.environ.get("GFS_CACHE_HORAS", "3"))
FORECAST_HOURS = int(os.environ.get("GFS_HORAS", "48"))

sys.path.insert(0, APP_DIR)
from gfs_forecast import get_gfs_forecast, make_windows  # noqa: E402

VARS = ["RH", "T", "U", "V"]
st = np.load(os.path.join(DATA, "static.npz"))
DEM_M = st["dem"].astype(np.float32)
WRF_LATS, WRF_LONS = st["wrf_lats"], st["wrf_lons"]
GFS_LATS, GFS_LONS = st["gfs_lats"], st["gfs_lons"]

# Coordenadas WRF dentro de la grilla GFS para grid_sample (igual que el starter)
_LAT, _LON = np.meshgrid(WRF_LATS, WRF_LONS, indexing="ij")
_gy = (_LAT - GFS_LATS[0]) / (GFS_LATS[-1] - GFS_LATS[0]) * 2 - 1
_gx = (_LON - GFS_LONS[0]) / (GFS_LONS[-1] - GFS_LONS[0]) * 2 - 1
GRID = torch.tensor(np.stack([_gx, _gy], -1)[None], dtype=torch.float32)       # (1,70,35,2)
DEM = torch.tensor((DEM_M - DEM_M.mean()) / DEM_M.std(), dtype=torch.float32)[None, None]


class NetStarter(nn.Module):
    """Misma arquitectura que train/starter_notebook.ipynb (claves del state_dict compatibles)."""

    def __init__(self, w=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv2d(14, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, w, 3, padding=1), nn.GELU())
        self.static = nn.Parameter(torch.zeros(1, 8, 70, 35))
        self.dec = nn.Sequential(nn.Conv2d(w + 9, w, 3, padding=1), nn.GELU(),
                                 nn.Conv2d(w, w, 3, padding=1), nn.GELU(), nn.Conv2d(w, 4, 1))

    def forward(self, x, h):
        B = x.shape[0]
        x = x.permute(0, 1, 4, 2, 3).reshape(B, 12, 17, 13)
        ang = 2 * np.pi * h / 24
        hh = torch.stack([torch.sin(ang), torch.cos(ang)], 1)[..., None, None].expand(B, 2, 17, 13)
        f = self.enc(torch.cat([x, hh], 1))
        f = F.grid_sample(f, GRID.expand(B, -1, -1, -1), align_corners=True)
        f = torch.cat([f, DEM.expand(B, -1, -1, -1), self.static.expand(B, -1, -1, -1)], 1)
        return self.dec(f).permute(0, 2, 3, 1)


ARQUITECTURAS = {"starter": NetStarter}


# ----------------------------------------------------------------------------- estadísticas
def _hour(t):
    return np.array([int(str(k)[-2:]) for k in t])


def _stats_desde_train():
    """Estadísticas del starter (split temporal: train = times < 2024061600)."""
    tr = np.load(os.path.join(DATA, "train.npz"))
    X, Y, t = tr["X"], tr["Y"], tr["times"].astype(str)
    trn = t < "2024061600"
    h = _hour(t[trn])
    clim = np.stack([Y[trn][h == k].mean(0) for k in range(24)]).astype(np.float32)
    return dict(mu_x=X[trn].mean((0, 1, 2)), sd_x=X[trn].std((0, 1, 2)), clim=clim,
                anom_sd=(Y[trn] - clim[h]).std((0, 1, 2))), (X, Y, t, trn)


def _windows_train(X, t):
    import pandas as pd
    ts = pd.to_datetime(t, format="%Y%m%d%H")
    pos = {k: i for i, k in enumerate(ts)}
    out = np.empty((len(X), 3) + X.shape[1:], np.float32)
    for i, k in enumerate(ts):
        for j, lag in enumerate([2, 1, 0]):
            idx = pos.get(k - pd.Timedelta(hours=lag))
            out[i, j] = X[idx] if idx is not None else X[i]
    return out


def _entrenar_rapido(epochs=2):
    """Respaldo: entrena el modelo del starter pocas épocas en CPU y lo guarda con el contrato."""
    torch.manual_seed(0)
    np.random.seed(0)
    stats, (X, Y, t, trn) = _stats_desde_train()
    W = _windows_train(X, t)
    mu, sd = stats["mu_x"], stats["sd_x"]
    CLIM, ASD = torch.tensor(stats["clim"]), torch.tensor(stats["anom_sd"], dtype=torch.float32)
    model = NetStarter()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    itr = np.where(trn)[0]
    for ep in range(epochs):
        model.train()
        np.random.shuffle(itr)
        t0 = time.time()
        for s in range(0, len(itr), 32):
            b = itr[s:s + 32]
            x = torch.tensor((W[b] - mu) / sd, dtype=torch.float32)
            h = torch.tensor(_hour(t[b]))
            y = (torch.tensor(Y[b]) - CLIM[h]) / ASD
            loss = F.mse_loss(model(x, h.float()), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"[inferencia] entrenamiento rápido época {ep + 1}/{epochs} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
    ck = dict(state_dict=model.state_dict(), arquitectura="starter", origen=f"entrenamiento rápido {epochs} épocas",
              **{k: np.asarray(v, dtype=np.float32) for k, v in stats.items()})
    os.makedirs(MODELS, exist_ok=True)
    torch.save(ck, os.path.join(MODELS, "modelo_app_rapido.pt"))
    return ck


# ----------------------------------------------------------------------------- carga del modelo
class Predictor:
    def __init__(self):
        self.model, self.stats, self.fuente, self.avisos = None, None, "interpolacion", []
        self._cargar()

    def _desde_checkpoint(self, ck, origen):
        arq = ck.get("arquitectura", "starter")
        model = ARQUITECTURAS[arq]()
        model.load_state_dict(ck["state_dict"])
        model.eval()
        self.model = model
        self.stats = {k: torch.tensor(np.asarray(ck[k]), dtype=torch.float32) for k in ("mu_x", "sd_x", "clim", "anom_sd")}
        self.fuente = origen

    def _cargar(self):
        for nombre in ("modelo_final.pt", "modelo_app_rapido.pt"):
            ruta = os.path.join(MODELS, nombre)
            if os.path.exists(ruta):
                try:
                    ck = torch.load(ruta, map_location="cpu", weights_only=False)
                    if isinstance(ck, dict) and "state_dict" in ck:
                        self._desde_checkpoint(ck, f"models/{nombre}")
                        if nombre != "modelo_final.pt":
                            self.avisos.append("Pesos de respaldo (entrenamiento rápido), no el modelo final.")
                        return
                except Exception as e:  # checkpoint incompatible: seguir con el siguiente
                    self.avisos.append(f"No se pudo cargar models/{nombre}: {e}")
        ruta = os.path.join(MODELS, "modelo.pt")
        if os.path.exists(ruta):
            try:
                sd = torch.load(ruta, map_location="cpu")
                stats, _ = _stats_desde_train()
                self._desde_checkpoint(dict(state_dict=sd, arquitectura="starter", **stats), "models/modelo.pt (starter)")
                self.avisos.append("Formato del starter: estadísticas recalculadas con data/train.npz.")
                return
            except Exception as e:
                self.avisos.append(f"No se pudo cargar models/modelo.pt: {e}")
        if os.environ.get("NO_ENTRENAR") != "1":
            try:
                print("[inferencia] No hay pesos: entrenando el modelo del starter 2 épocas (respaldo)...")
                self._desde_checkpoint(_entrenar_rapido(2), "entrenamiento rápido (2 épocas)")
                self.avisos.append("No había pesos: se entrenó el starter 2 épocas como respaldo. Calidad baja.")
                return
            except Exception as e:
                self.avisos.append(f"Falló el entrenamiento rápido: {e}")
        self.avisos.append("Sin modelo: se usa interpolación bilineal del GFS (respaldo explícito).")

    @torch.no_grad()
    def predecir(self, W, times, bs=16):
        """W: (N,3,17,13,4) unidades originales; times 'YYYYMMDDHH' UTC -> (N,70,35,4) unidades originales."""
        h = torch.tensor(_hour(times))
        if self.model is None:  # interpolación bilineal de la hora t
            x = torch.tensor(W[:, -1], dtype=torch.float32).permute(0, 3, 1, 2)
            return F.grid_sample(x, GRID.expand(len(W), -1, -1, -1), align_corners=True).permute(0, 2, 3, 1).numpy()
        s = self.stats
        out = []
        for i in range(0, len(W), bs):
            x = (torch.tensor(W[i:i + bs], dtype=torch.float32) - s["mu_x"]) / s["sd_x"]
            hb = h[i:i + bs]
            out.append((s["clim"][hb] + self.model(x, hb.float()) * s["anom_sd"]).numpy())
        return np.concatenate(out)


# ----------------------------------------------------------------------------- GFS con caché y respaldo
def obtener_gfs():
    """Devuelve (X, times, info). Usa caché en disco; si Open-Meteo falla, usa la última descarga buena."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    fresca = os.path.exists(GFS_CACHE) and (time.time() - os.path.getmtime(GFS_CACHE)) / 3600 < CACHE_MAX_AGE_H
    try:
        X, t = get_gfs_forecast(forecast_hours=FORECAST_HOURS, past_hours=2, cache_path=GFS_CACHE,
                                cache_max_age_h=CACHE_MAX_AGE_H)
        if not fresca:
            np.savez_compressed(GFS_ULTIMA, X=X, times=t)
        origen = "cache" if fresca else "open-meteo"
    except Exception as e:
        for ruta in (GFS_ULTIMA, GFS_CACHE):
            if os.path.exists(ruta):
                d = np.load(ruta)
                X, t = d["X"], d["times"]
                origen = f"respaldo última descarga ({os.path.basename(ruta)}); error: {e}"
                break
        else:
            raise RuntimeError(f"Open-Meteo falló y no hay descarga previa: {e}")
    ruta_usada = GFS_CACHE if os.path.exists(GFS_CACHE) else GFS_ULTIMA
    descargado = datetime.fromtimestamp(os.path.getmtime(ruta_usada), timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return X, t.astype(str), dict(origen=origen, descargado=descargado, segundos=round(time.time() - t0, 2))


def utc_a_peru(t):
    d = datetime.strptime(str(t), "%Y%m%d%H") - timedelta(hours=5)
    return d.strftime("%Y-%m-%d %H:00")


class Pronostico:
    """Mantiene el último pronóstico en memoria; se recalcula si cambia la caché del GFS."""

    def __init__(self):
        self.pred = Predictor()
        self._mtime, self.Y, self.times, self.info = None, None, None, None

    def actualizar(self):
        X, t, info = obtener_gfs()
        mtime = os.path.getmtime(GFS_CACHE) if os.path.exists(GFS_CACHE) else None
        if self.Y is None or mtime != self._mtime:
            t0 = time.time()
            W, _ = make_windows(X, t)
            # Se descartan las 2 primeras horas (pasadas, sin ventana completa)
            W, t = W[2:], t[2:]
            self.Y = self.predecir_cache(W, t)
            info["segundos_inferencia"] = round(time.time() - t0, 2)
            self.times, self.info, self._mtime = t, info, mtime
        return self

    def predecir_cache(self, W, t):
        return self.pred.predecir(W, t).astype(np.float32)

    def resumen(self):
        """Resumen por hora (promedio, mínimo y máximo del dominio)."""
        Y = self.Y
        T = Y[..., 1] - 273.15
        RH = np.clip(Y[..., 0], 0, 100)
        WS = np.hypot(Y[..., 2], Y[..., 3])
        horas = []
        for i, t in enumerate(self.times):
            horas.append(dict(
                hora_utc=str(t), hora_peru=utc_a_peru(t), indice=i,
                T_C=dict(media=round(float(T[i].mean()), 2), min=round(float(T[i].min()), 2), max=round(float(T[i].max()), 2)),
                RH=dict(media=round(float(RH[i].mean()), 1), min=round(float(RH[i].min()), 1), max=round(float(RH[i].max()), 1)),
                viento_ms=dict(media=round(float(WS[i].mean()), 2), max=round(float(WS[i].max()), 2)),
            ))
        return horas

    def mapa(self, indice):
        y = self.Y[indice]
        return dict(
            hora_utc=str(self.times[indice]), hora_peru=utc_a_peru(self.times[indice]),
            filas=70, columnas=35, nota="fila 0 = sur, columna 0 = oeste",
            lats=[round(float(a), 4) for a in WRF_LATS], lons=[round(float(b), 4) for b in WRF_LONS],
            T_C=np.round(y[..., 1] - 273.15, 2).tolist(),
            RH=np.round(np.clip(y[..., 0], 0, 100), 1).tolist(),
            U=np.round(y[..., 2], 2).tolist(), V=np.round(y[..., 3], 2).tolist(),
            viento_ms=np.round(np.hypot(y[..., 2], y[..., 3]), 2).tolist(),
        )


if __name__ == "__main__":
    p = Pronostico().actualizar()
    print("Modelo:", p.pred.fuente, "| avisos:", p.pred.avisos)
    print("GFS:", p.info)
    print("Horas:", len(p.times), p.times[0], "->", p.times[-1], "UTC")
    for h in p.resumen()[:3]:
        print(h)
