# Clasificación del sentimiento de reseñas por sede · Caso Bodytech

Proyecto final de **Procesamiento de Lenguaje Natural** (Universidad Ean).
Equipo interdisciplinario: Ciencia de Datos · Especialista en Machine Learning · Gerencia en IA.

## Qué problema resuelve
Bodytech recibe reseñas de sus usuarios sobre cada sede. Leerlas a mano no escala y una queja sin atender puede terminar
en una cancelación. El sistema clasifica cada reseña en **negativa, neutra o positiva**, envía las posibles quejas al
coordinador de la sede, manda a revisión humana lo que no sabe clasificar y resume la satisfacción por sede y ciudad.

| | |
|---|---|
| **Usuario** | Coordinador de servicio de cada sede y equipo de experiencia del cliente |
| **Decisión que apoya** | Qué reseñas atender primero (las negativas, el mismo día) |
| **Entrada → salida** | Texto de una reseña → clase, probabilidades y acción (automático / revisión humana / coordinador) |
| **Error crítico** | Una reseña negativa que no se detecta |
| **KPI** | Recall de negativas auditado por sede (meta ≥ 0,90), % automatizado, horas de lectura manual |

> **Datos:** no existe un corpus público de reseñas de gimnasios en español. Se usa `Big_AHR.csv` (reseñas **reales y
> públicas** de hoteles de Andalucía) como **proxy**: hotel = sede, ubicación = ciudad. **No hay datos de Bodytech en
> este repositorio**; sus volúmenes y costos son supuestos declarados. Ver `datos/LICENCIA_DATOS.md`.

## Resultados clave
Partición por sede (sedes no vistas en test); el modelo se eligió con validación y el test se miró una sola vez.

| Modelo | F1 macro (test) | Recall negativas (regla) | Precisión negativas (regla) | Latencia p95 (CPU) | Tamaño |
|---|---|---|---|---|---|
| TF-IDF de lemas + regresión logística (baseline) | 0,755 | 0,891 | 0,728 | ≈ 5 ms | 2,8 MB |
| BiLSTM (mejor red recurrente, 5 semillas) | 0,723 | 0,874 | 0,716 | ≈ 5 ms | 16 MB |
| **BETO fine-tuned** (`dccuchile/bert-base-spanish-wwm-cased`, 128 tokens) | **0,780** (media de 3 semillas: 0,784) | 0,888 | **0,806** | 41–56 ms | 440 MB |

- **BETO es el mejor modelo**: supera al baseline en las 3 semillas (+0,029 de F1 macro de media) y mejora sobre todo la clase neutra.
  Ninguna de las 46 corridas de redes recurrentes supera al baseline.
- **Criterio de adopción:** BETO cumple 2 de 4 (mejora de F1 ≥ 0,02 y latencia ≤ 150 ms). No llega al recall de negativas ≥ 0,90
  (el baseline tampoco) ni a calidad ≥ 0,90 en lo que automatiza. Además pierde 0,19 de F1 con texto en mayúsculas.

**Recomendación: pilotear.** El baseline opera con revisión humana; BETO calibrado corre en *shadow mode* y solo pasa a
operar si cumple los 4 criterios con reseñas reales de Bodytech. Detalle, condiciones del piloto, gobierno y criterio de
suspensión en la sección 14 del notebook.

## Estructura
```
├── notebooks/Proyecto_Final_PLN.ipynb   notebook principal: EDA, preparación, baseline, modelos avanzados,
│                                         evaluación, errores, eficiencia, gobierno y conclusiones
├── datos/                               licencia y origen, diccionario de datos, partición por sede
├── src/                                 funciones reutilizables (preprocesamiento y política de decisión)
├── app/                                 demo en vivo
├── modelos/                             artefactos entrenados (no se versionan; ver modelos/README.md)
├── docs/                                registro de experimentos y declaración de uso de IA
├── entregables/                         informe técnico, presentación, tablero, matriz de riesgos y caso de negocio
├── requirements.txt                     dependencias con versiones exactas
└── .env.example                         configuración (sin secretos)
```

## Cómo ejecutarlo
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # ajustar si hace falta (SEED, DATA_DIR, REENTRENAR)
# copiar Big_AHR.csv en datos/ (ver datos/LICENCIA_DATOS.md)
jupyter nbconvert --to notebook --execute --inplace notebooks/Proyecto_Final_PLN.ipynb
```
- **Primera ejecución:** entrena todo (≈ 3,5 h en un Apple M5 con MPS, de las cuales BETO ≈ 2,5 h; en Colab, activar GPU). BETO necesita GPU (CUDA o
  Apple MPS); en CPU cada corrida tarda horas.
- **Siguientes ejecuciones:** reutilizan los experimentos guardados en `modelos/` y terminan en minutos.
  Con `REENTRENAR=1` se entrena todo de nuevo.
- **Colab:** subir `Big_AHR.csv` a `/content`; el notebook instala lo que falta.

## Demo en vivo
```bash
python app/demo.py          # abre http://127.0.0.1:7860 (sin internet, carga los modelos de modelos/)
```
- **Pestaña 1 · Una reseña:** filtros previos (vacío, repetitivo, otro idioma, mayúsculas) → clase y probabilidades →
  acción (automático / revisión humana / coordinador de la sede) con los umbrales de `modelos/politica.json` → evidencia
  (palabras del baseline y oraciones que pesan en BETO). Decide el baseline; BETO se muestra al lado en *shadow mode*.
- **Pestaña 2 · Resumen por sede:** procesa un CSV (por defecto `datos/muestra_controlada.csv`: 400 reseñas de test,
  20 sedes, 5 ciudades) y muestra % de negativas, quejas al coordinador y revisiones por sede, más la auditoría con la
  etiqueta real.
- **Trazabilidad:** cada predicción se guarda en `logs/predicciones.csv` con fecha, decisión y versión de la política,
  pero **sin el texto** (solo una huella SHA-256): minimización de datos.
- `python app/crear_muestra.py` regenera la muestra controlada desde `datos/Big_AHR.csv` y `datos/particion.csv`.

## Reproducibilidad
- Semilla declarada `SEED=42`; las redes se repiten con 5 semillas y BETO con 3, y se reporta la variación.
- **El test se mira una sola vez**, después de elegir el modelo con validación (sección 9 del notebook).
- La partición por sede queda guardada en `datos/particion.csv`; todos los modelos usan exactamente los mismos índices.
- Rutas relativas; configuración por variables de entorno; sin credenciales en el código.
- Hardware, versiones, tiempos de entrenamiento y tamaño de los artefactos quedan registrados en el notebook,
  en `modelos/politica.json` y en `docs/registro_experimentos.csv`.

## Uso responsable de IA generativa
Ver `docs/declaracion_uso_ia.md`.
