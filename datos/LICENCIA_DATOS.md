# Licencia y origen de los datos

| Campo | Valor |
|---|---|
| Nombre del conjunto | **Andalusian Hotels' Reviews** (AHR) · "A corpus of TripAdvisor reviews in spanish" |
| Autora / publicadora | Mariia Chizhikova (`chizhikchi` en Kaggle) |
| Enlace | https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced |
| Versión | 3, del 6 de diciembre de 2021 |
| Archivo usado | `Big_AHR.csv` (18.172 reseñas) |
| Fecha de descarga | noviembre de 2026 |
| **Licencia** | **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** · texto completo: https://creativecommons.org/licenses/by-nc/4.0/legalcode |

## Qué permite la licencia y cómo la cumplimos
- **Compartir y adaptar** el material: permitido. Por eso el CSV puede incluirse en este repositorio.
- **Atribución** (obligatoria): se cita a la autora y el enlace en este archivo y en el README, y se indica que hubo cambios.
- **Cambios realizados:** se eliminaron 4.841 filas duplicadas, se excluyó la columna `rating` del modelo (determina la etiqueta), se normalizó el texto (Unicode NFC, URL y correos enmascarados) y se concatenó el título con la reseña.
- **No comercial (NC):** el uso en este proyecto es **académico**. **Este corpus no puede usarse para entrenar un modelo que Bodytech opere comercialmente.** Para producción hay que reentrenar con reseñas propias de Bodytech, como ya exige la recomendación del proyecto.

## Origen de las reseñas (según la ficha del conjunto)
- **16.356 reseñas** descargadas de **TripAdvisor** por la autora en **diciembre de 2021**.
- El resto (≈ 1.816) proviene del **corpus COAH** (*Corpus of Opinions about Andalusian Hotels*), compilado por el grupo de investigación **SINAI (Universidad de Jaén) en 2014**. COAH no trae hotel ni ubicación: por eso esas filas tienen `hotel` y `location` vacíos. La auditoría del notebook (sección 1.3) las había detectado como "otra fuente" (más negativas y más largas).
- Columnas: `title`, `rating` (1 a 5 estrellas), `review_text`, `location`, `hotel`, `label` (0 negativa, 1 positiva, 3 neutra).

## Cita solicitada por la autora (corpus COAH)
Molina-González, M. D., Martínez-Cámara, E., Martín-Valdivia, M. T., & Ureña-López, L. A. (2014). *Cross-domain sentiment analysis using Spanish opinionated words*. Lecture Notes in Computer Science, vol. 8455, pp. 214-219. https://doi.org/10.1007/978-3-319-07983-7_28

## Qué datos son reales, anonimizados o sintéticos
- `Big_AHR.csv`: reseñas **reales y públicas** de huéspedes de hoteles de Andalucía (España), usadas como **proxy** de reseñas de sedes de Bodytech.
- **No hay datos de Bodytech** en este repositorio. Los volúmenes y costos de Bodytech del notebook y del caso de negocio son **supuestos declarados**.
- La auditoría (sección 1.5) no encontró correos reales y solo 0,01 % de URL; los "@" son lenguaje inclusivo ("chic@s").
- `muestra_controlada.csv`: 400 reseñas reales del conjunto de test, para la demo, **anonimizadas**: los nombres de empleados y
  anfitriones (por ejemplo, "el recepcionista Fernando") se reemplazan por `[NOMBRE]`, y los correos, URL y teléfonos por
  `[CORREO]`, `[URL]` y `[TELEFONO]`. Quedan 77 nombres enmascarados en 55 reseñas. La lista de nombres salió de revisar a mano
  las entidades de persona de spaCy y las palabras con mayúscula poco comunes; se conservan lugares, marcas y nombres de hoteles,
  que no son datos personales. El enmascarado no cambia ninguna predicción de la demo. Lo hace `app/crear_muestra.py`.
- `Big_AHR.csv` se redistribuye **tal como lo publicó su autora** (corpus público, CC BY-NC 4.0), sin anonimizar, para que el
  notebook sea reproducible. Sus textos pueden traer nombres de pila de empleados que los huéspedes mencionan; por eso el
  proyecto no los usa para evaluar personas (ver "Usos no previstos" en la ficha del dataset del informe técnico).
