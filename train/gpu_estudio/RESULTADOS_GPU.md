# Resultados del estudio con GPU: starter, ablaciones y ensambles

Estudio con el starter de 25 épocas, la regresión lineal por celda (ridge) y su ensamble. Cada modelo se ajusta con los datos anteriores a un corte y se evalúa en el tramo siguiente. Cortes: 16 de mayo, 1 de junio y 16 de junio de 2024. Semillas: 0, 37 y 73. El score usa las referencias oficiales y solo las horas 00, 03, ..., 21 UTC (como Kaggle).

## Tabla (media de los 3 cortes y 3 semillas)

La columna Δ es la diferencia con el starter de la misma semilla y corte. La columna "desfase" evalúa con la entrada enfriada 1,5 K y secada 8 puntos de RH, dejando el WRF fijo: mide sensibilidad a la entrada, no calidad. El ensamble es 75 % CNN + 25 % lineal, con pesos fijados antes de ver los resultados.

| Modelo | Semillas | Score 3 h | Δ starter | Score con desfase |
|---|---:|---:|---:|---:|
| ensamble_peso_mayjun | 3 | 0.4689 | +0.0267 | 0.3434 |
| ensamble_hora_onehot | 3 | 0.4685 | +0.0263 | 0.3450 |
| ensamble_perdida_rmse | 3 | 0.4658 | +0.0236 | 0.3451 |
| ensamble_rh_recorte | 3 | 0.4657 | +0.0235 | 0.3394 |
| ensamble_ema | 3 | 0.4657 | +0.0235 | 0.3397 |
| ensamble_starter | 3 | 0.4657 | +0.0235 | 0.3392 |
| ensamble_t6 | 3 | 0.4631 | +0.0209 | 0.3537 |
| ensamble_gfs_derivados | 3 | 0.4628 | +0.0207 | 0.3520 |
| ensamble_t12 | 3 | 0.4609 | +0.0187 | 0.3637 |
| ensamble_t24 | 3 | 0.4590 | +0.0168 | 0.3546 |
| peso_mayjun | 3 | 0.4463 | +0.0041 | 0.3318 |
| hora_onehot | 3 | 0.4451 | +0.0029 | 0.3310 |
| ema | 3 | 0.4423 | +0.0002 | 0.3256 |
| rh_recorte | 3 | 0.4422 | +0.0001 | 0.3252 |
| starter | 3 | 0.4422 | +0.0000 | 0.3249 |
| perdida_rmse | 3 | 0.4422 | -0.0000 | 0.3321 |
| lineal | 0 | 0.4395 | -0.0027 | 0.2064 |
| t6 | 3 | 0.4368 | -0.0054 | 0.3395 |
| gfs_derivados | 3 | 0.4364 | -0.0057 | 0.3400 |
| t12 | 3 | 0.4323 | -0.0098 | 0.3497 |
| t24 | 3 | 0.4305 | -0.0116 | 0.3360 |

## Lectura

- La ganancia principal viene de ensamblar la CNN con la lineal (≈ +0,024 sobre el starter).
- Sobre el ensamble, el peso doble mayo-junio y la hora one-hot suman ≈ +0,003; esa diferencia no se ha contrastado con la variación entre semillas.
- Rezagos t-6, t-12 y t-24 y rasgos derivados del GFS empeoran la CNN sola. EMA, pérdida RMSE y recorte de RH no cambian el resultado.
- Los resultados son locales: el starter local con la semilla 0 (0,4404) queda +0,013 sobre su score en Kaggle (0,42782), lo que no garantiza la misma brecha para otros modelos.
- El ajuste final con todo el train (`results/final/modelo.pt`) no se versiona. No hay submission enviado.

## Cómo reproducir

Entorno con PyTorch (CUDA opcional; sin GPU corre en CPU, unas 20 veces más lento) y `train.npz`, `test.npz` y `static.npz` en la carpeta `data/` de la raíz del repositorio (ver `data/README.md`). Para instalar torch con CUDA y el resto de dependencias, ver `requirements_propuesto.txt`. Los scripts se ejecutan desde `train/gpu_estudio/`.

```
python run_gpu.py --experimento todos --hilos 4 --horas 5
python bench.py   # mide segundos por época en GPU y CPU
```

Notas: con laptop a batería la GPU baja su límite de potencia y va unas 6 veces más lenta; conviene correr conectada a la corriente. `run_gpu.py` evita la suspensión en Windows. Los resultados terminados quedan guardados y se reanudan al relanzar.
