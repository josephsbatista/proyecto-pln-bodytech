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

**Modelo base:** `dccuchile/bert-base-spanish-wwm-cased` (BETO, Universidad de Chile).
- **Licencia:** el repositorio oficial (https://github.com/dccuchile/beto) indica **CC BY 4.0**, con este aviso de los autores:
  *"The license CC BY 4.0 best describes our intentions for our work. However we are not sure that all the datasets used
  to train BETO have licenses compatible with CC BY 4.0 (specially for commercial use). Please use at your own discretion
  and verify that the licenses of the original text resources match your needs."*
- **Implicación para Bodytech:** el uso académico es válido; antes de un uso comercial hay que revisar con el área legal
  las licencias de los corpus con que se preentrenó BETO (o elegir un modelo con licencia comercial clara).
- **Cita:** Cañete, J., Chaperon, G., Fuentes, R., Ho, J.-H., Kang, H., & Pérez, J. (2020). *Spanish Pre-Trained BERT
  Model and Evaluation Data*. PML4DC at ICLR 2020.
