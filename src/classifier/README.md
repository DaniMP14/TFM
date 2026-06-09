# Modulo de clasificador ML de calidad fotometrica

Este modulo entrena un clasificador binario de calidad:

- `label = 1`: transito valido (TESS, ExoClock, sinteticos batman, observaciones propias buenas)
- `label = 0`: no valido / ruido / artefactos instrumentales

## Jerarquia inicial

```text
src/classifier/
  code/
    dataset.py
    features.py
    training.py
    train.py
  data/
    training/
      raw/
      processed/
      manifest_example.csv
  artifacts/
  configs/
    random_forest_default.json
```

## Formato del manifest

`manifest.csv` debe incluir estas columnas:

- `sample_id`: identificador unico de curva
- `label`: 0 o 1
- `source_type`: `tess`, `exoclock`, `batman`, `own`, etc.
- `curve_path`: ruta al CSV de la curva (absoluta o relativa al manifest)

Cada CSV de curva debe contener al menos una columna de flujo (por defecto `detrended_flux`).
Si existe `time_jd`, se utiliza como tiempo; si no, se usa el indice de muestra.

## Formatos mixtos en `raw/`

Puedes meter curvas de distintos formatos en `src/classifier/data/training/raw/`.
El flujo recomendado es:

1. Crear `raw_manifest.csv` con etiquetas y metadatos de parseo.
2. Convertir todo a formato estandar (`time_jd`, `detrended_flux`, `flux_err` opcional) en `processed/`.
3. Entrenar usando el manifest generado para entrenamiento.

Ejemplo de raw manifest: `src/classifier/data/training/raw_manifest_example.csv`

Comando de conversion:

```bash
python -m classifier.code.prepare_dataset \
  --raw-manifest src/classifier/data/training/raw_manifest_example.csv \
  --processed-dir src/classifier/data/training/processed \
  --output-manifest src/classifier/data/training/manifest_training.csv
```

Formatos soportados por el conversor:

- `csv`: detecta columnas de tiempo/flujo (o puedes forzarlas en raw_manifest)
- `hops`: TXT con cabecera comentada `#` y columnas estilo HOPS/TRESCA (usa columna 5 como de-trended flux)
- `threecol`: TXT numerico de 2-3 columnas (tiempo, flujo, error opcional)

## Entrenamiento RF inicial

Desde la raiz del proyecto:

```bash
python -m classifier.code.train \
  --manifest src/classifier/data/training/manifest_training.csv \
  --output-dir src/classifier/artifacts \
  --flux-column detrended_flux \
  --time-column time_jd
```

## Salidas

El entrenamiento genera en `artifacts/`:

- `random_forest_quality_classifier.pkl`
- `metrics.json`
- `feature_importance.csv`
- `feature_columns.json`
- `training_config.json`
- `training_features.csv`

Con esto tienes una base reproducible para iterar hiperparametros, balanceo de clases y validacion cruzada.
