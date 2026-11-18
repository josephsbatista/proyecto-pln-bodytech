# Declaración de uso de IA generativa

| Herramienta | Finalidad |
|---|---|
| Claude (Anthropic), como asistente de código | Estructurar el notebook, depurar código, proponer pruebas (varias semillas, validación frente a test, robustez, calibración) y redactar borradores del README, el informe y la presentación |

**Verificación:** todas las cifras del notebook, el informe y la presentación salen de celdas ejecutadas del notebook.
Ninguna cifra se presenta como evidencia experimental sin haberse ejecutado. El equipo revisó el código, los resultados
y las conclusiones, y es responsable de su contenido.

**Datos:** no se enviaron datos confidenciales a servicios externos. No se usaron datos de Bodytech; el corpus es público
y todo el entrenamiento se hizo en local.

**Trazabilidad:** las decisiones de diseño (partición por sede, selección con validación, regla para la clase crítica,
criterio de adopción) quedan registradas en el notebook y en `docs/registro_experimentos.csv`.
