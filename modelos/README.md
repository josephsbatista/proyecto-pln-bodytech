# Modelos

Esta carpeta no se versiona (los pesos de BETO ocupan ~440 MB por corrida). Se llena al ejecutar
`notebooks/Proyecto_Final_PLN.ipynb`:

| Archivo / carpeta | Qué es | Sección del notebook |
|---|---|---|
| `baseline_tfidf_lemas.joblib` | Pipeline TF-IDF de lemas + regresión logística | 12 |
| `bilstm.keras` + `bilstm_vocabulario.json` | Mejor red recurrente (semilla 42) y su vocabulario | 12 |
| `beto/beto_L{longitud}_s{semilla}/` | Corridas de BETO: logits de validación y test, metadatos y (semilla 42) el modelo ajustado | 8 |
| `politica.json` | Umbrales elegidos en validación (abstención, regla de negativas, temperatura), versiones y hardware | 13 |
| `cache/` | Resultados de los experimentos recurrentes con varias semillas | 7 |

**Regenerar todo:** `REENTRENAR=1` en `.env` y ejecutar el notebook completo (≈ 3,5 h en un Apple M5; en Colab, usar GPU).

**Modelo base:** `dccuchile/bert-base-spanish-wwm-cased` (BETO). Verificar su licencia en su repositorio oficial antes de un uso comercial.
