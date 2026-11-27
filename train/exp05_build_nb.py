"""Genera train/exp05_regularizacion.ipynb a partir de los resultados de exp05_run.py."""
import os, nbformat as nbf
HERE = os.path.dirname(os.path.abspath(__file__))
nb = nbf.v4.new_notebook(); C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s)); code = lambda s: C.append(nbf.v4.new_code_cell(s))

md("""# exp05 · Sobreajuste y regularización (modelo del starter)

Estudio rápido sobre la CNN del `starter_notebook.ipynb` como base fija. Por tiempo de CPU (CPU compartida, ~380 s por época completa con 1 hilo), todas las corridas comparativas usan **3 épocas** con OneCycle (lr máx. 2e-3, batch 32) y **1 de cada 4 muestras de train** (`STRIDE=4`, ~550 muestras); la validación sí es completa. Los scores absolutos son, por tanto, menores que los del starter de 25 épocas; lo que vale aquí es la comparación entre configuraciones dentro del mismo régimen. Al final se muestra además la época 1 de dos corridas con train completo que se detuvieron por tiempo.

- Curva train vs val por época (score en una muestra de train del mismo tamaño que val, y en val).
- Efecto aislado (un cambio a la vez respecto a la base): weight decay 1e-2 (base 1e-4), Dropout2d 0,1, ruido gaussiano 0,05 sobre la entrada normalizada, anchura 32 (base 64), parada temprana (mejor época de la base).
- Estabilidad: semilla 0 vs 1; corte temporal 16 jun (val 16–30 jun) vs 1 jun (entrena solo con datos anteriores al 1 jun, val 1–15 jun).

El entrenamiento está en `train/exp05_run.py` (mismo modelo y pérdida que el starter; solo se añaden las perillas). Este notebook lee sus resultados. Para reproducir:

```
EPOCHS=3 STRIDE=4 THREADS=1 python train/exp05_run.py base wd1e-2 dropout01 ancho32 semilla1 ruido005 corte0601
```

`score_val` usa la climatología del propio periodo de validación (como el starter); `score_val_ref_oficial` usa las RMSE_clim oficiales (RH 25,360; T 2,0652; U 2,6776; V 2,0876). Por la nota del contexto, la validación local de junio puede ser optimista o pesimista respecto a test.""")

code("""import glob, os
import pandas as pd, matplotlib.pyplot as plt
os.chdir(os.path.dirname(os.path.abspath('__file__')) if os.path.isdir('results') else 'train')
parts = sorted(glob.glob('results/exp05_parts/*.csv'))
ep = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
ep[['config','epoca','loss_train','score_train','score_val','score_val_ref_oficial','RMSE_RH','RMSE_T','RMSE_U','RMSE_V','seg']].round(4)""")

md("## 1. Curva de entrenamiento vs validación")
code("""fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
for name, g in ep.groupby('config'):
    l, = ax[0].plot(g.epoca, g.score_val, marker='o', label=name)
    ax[0].plot(g.epoca, g.score_train, ls='--', color=l.get_color(), alpha=0.6)
    ax[1].plot(g.epoca, g.score_train - g.score_val, marker='o', color=l.get_color(), label=name)
ax[0].set_title('score por época (línea = val, discontinua = train)'); ax[0].set_xlabel('época'); ax[0].legend(fontsize=8)
ax[1].set_title('brecha train − val'); ax[1].set_xlabel('época'); ax[1].axhline(0, color='k', lw=0.5)
plt.tight_layout(); plt.show()""")

md("## 2 y 3. Tabla final: última época y mejor época (parada temprana)")
code("""last = ep.sort_values('epoca').groupby('config').tail(1).assign(checkpoint='ultima')
best = ep.loc[ep.groupby('config').score_val.idxmax()].assign(checkpoint='mejor_val')
cols = ['config','checkpoint','epoca','score_train','score_val','score_val_ref_oficial','RMSE_RH','RMSE_T','RMSE_U','RMSE_V','n_train','n_val','w','drop','wd','noise','seed','cut','val_end']
res = pd.concat([last, best])[cols].sort_values(['config','checkpoint']).reset_index(drop=True)
res.to_csv('results/exp05_resultados.csv', index=False)
b = res[(res.config=='base') & (res.checkpoint=='ultima')].score_val.values
if len(b): res['delta_vs_base'] = res.score_val - b[0]
res.round(4)""")
md("""Lectura de la tabla (las comparaciones solo son válidas dentro del mismo corte; `corte0601` tiene otra validación):

- `delta_vs_base` compara contra la última época de la base con el mismo corte.
- La diferencia entre `semilla1` y `base` da el ruido de una corrida: cambios menores que esa diferencia no deben tomarse como mejoras.""")

md("## Anexo: época 1 con train completo (corridas detenidas por tiempo; OneCycle de 5 épocas truncado)")
code("""full = pd.concat([pd.read_csv(p) for p in sorted(glob.glob('results/exp05_full_parcial/*.csv'))], ignore_index=True)
full[['config','epoca','score_train','score_val','score_val_ref_oficial','RMSE_RH','RMSE_T','RMSE_U','RMSE_V','seg']].round(4)""")

nbf.write(nb, os.path.join(HERE, 'exp05_regularizacion.ipynb'))
print('ok')
