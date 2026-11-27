# Prototipo técnico de la app

Prototipo de punta a punta: GFS actual (Open-Meteo) -> modelo -> mapas 3 km (70x35) -> API -> página. No define todavía al usuario final; muestra temperatura, humedad relativa y rapidez del viento por hora en hora de Perú (UTC-5).

## Archivos
- `app/inferencia.py`: carga del modelo, descarga del GFS con caché y predicción por hora.
- `app/servidor.py`: backend FastAPI.
- `app/static/index.html`: página simple (canvas, sin librerías externas), marcada como prototipo.
- `app/cache/`: caché del GFS (`gfs_cache.npz`) y copia de la última descarga buena (`gfs_ultima_ok.npz`). No conviene subirla al repo.

## Requisitos
```
pip install numpy requests torch pandas fastapi uvicorn
```
Necesita `data/static.npz` (siempre) y `data/train.npz` (solo si se usan pesos del starter o el entrenamiento de respaldo).

## Correr (desde la raíz del repo)
```
# 1) Probar la inferencia sola (descarga el GFS una vez y lo guarda en app/cache)
python app/inferencia.py

# 2) Levantar el servidor
python -m uvicorn app.servidor:app --port 8000
# abrir http://127.0.0.1:8000
```
En Windows con el entorno del equipo: `D:\hackathon\Summer\Rubber-duck-debuggers-andes-downscaling\.venv\Scripts\python.exe` en lugar de `python`.

## Endpoints
- `GET /salud`: modelo usado, avisos, origen del GFS (caché, Open-Meteo o respaldo) y rango de horas.
- `GET /pronostico`: resumen por hora del dominio (T en °C media/mín/máx, RH % media/mín/máx, viento m/s medio/máx), con `hora_utc` y `hora_peru`.
- `GET /mapa?hora=N` o `GET /mapa?hora=YYYYMMDDHH`: grilla 70x35 (fila 0 = sur, columna 0 = oeste) con `lats`, `lons`, `T_C`, `RH`, `U`, `V`, `viento_ms`.

## Modelo: orden de carga
1. `models/modelo_final.pt` con el contrato modelo-app: dict con `state_dict`, `arquitectura` (por ahora `starter`), `mu_x`, `sd_x`, `clim` (24,70,35,4), `anom_sd`.
2. `models/modelo_app_rapido.pt` (mismo formato, lo genera el respaldo de entrenamiento).
3. `models/modelo.pt` del starter (solo `state_dict`); las estadísticas se recalculan con `data/train.npz`.
4. Sin pesos: entrena el modelo del starter 2 épocas en CPU y lo guarda como `models/modelo_app_rapido.pt` (calidad baja, se avisa en `/salud` y en la página).
5. Si todo falla (o con `NO_ENTRENAR=1` y sin pesos): interpolación bilineal del GFS, avisada como respaldo.

Para otra arquitectura (por ejemplo U-Net), registrar su clase en `ARQUITECTURAS` dentro de `app/inferencia.py` y guardar `arquitectura` con ese nombre en el checkpoint.

## GFS y caché
- Cada descarga completa cuesta unas 221 llamadas del plan gratuito de Open-Meteo. La caché se reutiliza mientras tenga menos de `GFS_CACHE_HORAS` horas (por defecto 3).
- Si Open-Meteo falla, se usa la última descarga buena y `/salud` lo indica en `gfs.origen`.
- Horas pedidas: `GFS_HORAS` (por defecto 48) más 2 horas pasadas para la ventana t-2, t-1, t. Las 2 horas pasadas no se muestran.

## Limitaciones conocidas
- Sin pesos finales, el resultado viene del respaldo (2 épocas o interpolación): sirve para probar el flujo, no para tomar decisiones.
- El GFS de Open-Meteo es distinto del GFS de entrenamiento (fuente y época del año): posible sesgo, sin validar.
- El servidor calcula el pronóstico en la primera petición; con el respaldo de entrenamiento esa primera petición puede tardar minutos.
