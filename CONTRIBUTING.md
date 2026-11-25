# Guía de trabajo del equipo

Este documento describe cómo configurar el entorno y cómo trabajar en el repositorio. Aplica a todas las personas que colaboran, y debe leerse antes de hacer cambios.

## 1. Configuración inicial

1. Aceptar la invitación de colaborador enviada por correo.
2. Clonar el repositorio:
   ```bash
   git clone https://github.com/Mil4nes4/Rubber-duck-debuggers-andes-downscaling
   cd Rubber-duck-debuggers-andes-downscaling
   ```
3. Descargar los datos desde la pestaña *Data* de la competencia (https://www.kaggle.com/competitions/hackathon-ia-pucp-2026/data) y descomprimirlos en `data/`. Los cuatro archivos deben quedar directamente en esa carpeta, sin subcarpetas:
   ```text
   data/
   ├── train.npz
   ├── test.npz
   ├── static.npz
   └── sample_submission.csv
   ```
   Los datos no se suben al repositorio (están excluidos en `.gitignore`) y no se redistribuyen fuera del equipo.
4. Crear el entorno e instalar las dependencias:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # Linux / macOS
   pip install -r requirements.txt
   ```
5. Ejecutar `train/starter_notebook.ipynb` de principio a fin y registrar el resultado en `RESULTADOS.md` (el score de referencia es ≈ 0.40).

## 2. Flujo de ramas

Se usa un flujo simple basado en ramas cortas y Pull Requests.

- `main` contiene siempre una versión que funciona. **No se hacen commits directos en `main`.**
- Cada cambio se desarrolla en una rama propia, creada desde `main` actualizada:
  - `modelo/<nombre>-<tema>`, por ejemplo `modelo/ana-unet`
  - `producto/<nombre>-<tema>`, por ejemplo `producto/luis-alerta-heladas`
  - `docs/<tema>` para documentación
- Los cambios se integran mediante un Pull Request, revisado y aprobado por al menos otra persona del equipo.
- Los commits son pequeños y con mensajes claros, en infinitivo y describiendo el cambio (`Agregar U-Net con relieve`, `Corregir orientación de filas en el DEM`).
- Las ramas se eliminan después de integrarse.

```bash
git checkout main && git pull
git checkout -b modelo/nombre-tema
# ... trabajo y commits ...
git push -u origin modelo/nombre-tema
# abrir el Pull Request en GitHub
```

Como el plazo es corto, las revisiones deben ser rápidas: verificar que el cambio corre y que no incluye archivos pesados ni claves.

## 3. Estructura del repositorio

| Ruta | Contenido |
|---|---|
| `train/` | Notebooks de experimentos y el notebook final `submission_notebook.ipynb` |
| `models/` | Pesos del modelo final (solo pesos, sin estado del optimizador) |
| `app/` | Producto, que carga los mismos pesos del modelo final |
| `data/` | Datos oficiales (local, no versionados) |
| `RESULTADOS.md` | Registro de experimentos |
| `REPORTE_PRODUCTO.md` | Reporte del producto exigido por la organización |

Los experimentos se nombran con un número y un tema: `train/exp01_unet.ipynb`, `train/exp02_residuo_gfs.ipynb`.

## 4. Reglas del proyecto

- **Datos:** para entrenar y generar la submission solo se usa la data oficial de la competencia. El script `app/gfs_forecast.py` es exclusivo del producto.
- **Validación:** siempre con corte temporal (por ejemplo, las últimas dos semanas de junio). Nunca con división aleatoria.
- **Coherencia con el producto:** toda entrada o característica del modelo debe poder calcularse en la app a partir del GFS de Open-Meteo.
- **Reproducibilidad:** semillas fijas, rutas relativas y notebooks guardados con sus salidas.
- **Secretos:** ninguna clave de API en el código ni en el historial; usar variables de entorno.
- **Archivos pesados:** no versionar datos ni pesos de más de 100 MB. Los pesos grandes se alojan en Drive o Hugging Face y se enlazan en el README.
- **Pesos:** `.gitignore` excluye `*.pt`, `*.pth` y `*.ckpt`. Solo el modelo final, guardado como `models/modelo_final.pt`, se versiona.
- **Notebooks de experimentos:** evitar subir salidas pesadas (imágenes o tablas largas incrustadas). El notebook final sí se entrega con sus salidas guardadas.
- **Unidades:** las predicciones se entregan en las unidades del dataset (RH en %, T en K, U y V en m/s). La conversión a °C y hora de Perú (UTC−5) solo se hace al mostrar resultados.

## 5. Registro de experimentos

Cada experimento relevante se anota en `RESULTADOS.md` con su fecha, responsable, rama, cambios, score de validación por variable y score en Kaggle. Antes de repetir una prueba conviene revisar ese registro.

## 6. Envíos a Kaggle

- Se permiten 5 envíos diarios; deben distribuirse entre el equipo.
- El nombre del equipo en Kaggle debe ser exactamente **Rubber Duck Debuggers**.
- No se elige un modelo solo por el leaderboard público: cuenta el privado.
- Antes del cierre (8 dic, 23:59) se seleccionan dos envíos finales.
