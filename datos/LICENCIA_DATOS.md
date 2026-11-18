# Licencia y origen de los datos

> **PENDIENTE (Ciencia de Datos):** completar con lo que dice la página de Kaggle de donde se descargó `Big_AHR.csv`.
> Copiar la licencia **tal cual**, sin resumirla.

| Campo | Valor |
|---|---|
| Nombre del conjunto en Kaggle | _por completar_ |
| Autor / publicador | _por completar_ |
| Enlace | _por completar_ |
| Fecha de descarga | noviembre de 2026 |
| Licencia (textual) | _por completar_ |
| ¿Permite redistribuir? | _por completar_ → si **sí**, quitar `datos/Big_AHR.csv` del `.gitignore`; si **no**, dejar solo este archivo y las instrucciones de descarga |

## Qué datos son reales, anonimizados o sintéticos
- `Big_AHR.csv`: reseñas **reales y públicas** de hoteles de Andalucía (España), escritas por huéspedes. Se usan como **proxy** de reseñas de sedes de Bodytech.
- **No hay datos de Bodytech** en este repositorio. Los volúmenes y costos de Bodytech que aparecen en el notebook son **supuestos declarados**.
- La auditoría del notebook (sección 1.5) no encontró correos reales y solo 0,01 % de URL; los "@" son lenguaje inclusivo ("chic@s").

## Cómo obtener el corpus si no está en el repositorio
1. Descargar `Big_AHR.csv` desde el enlace de Kaggle indicado arriba.
2. Copiarlo en `datos/Big_AHR.csv` (o definir otra carpeta con `DATA_DIR` en `.env`).
