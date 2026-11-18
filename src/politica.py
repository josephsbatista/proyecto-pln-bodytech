"""Política de decisión: filtros previos + regla para la clase crítica + umbral de abstención (secciones 9 y 14).
Los umbrales se leen de modelos/politica.json, que guarda los valores elegidos en VALIDACIÓN."""
import json
from pathlib import Path

from .preprocesamiento import es_espanol, es_repetitivo, proporcion_mayusculas, normalizar_mayusculas, tokenizar

AUTOMATICO = 'Automático'
REVISION = 'A revisión humana'
COORDINADOR = 'Al coordinador de la sede'
MIN_PALABRAS = 3


def cargar_politica(raiz: Path) -> dict:
    return json.load(open(raiz / 'modelos' / 'politica.json', encoding='utf-8'))


def filtrar(texto: str):
    """Devuelve (texto listo para el modelo, motivo de revisión o None, avisos)."""
    avisos = []
    t = (texto or '').strip()
    if len(tokenizar(t)) < MIN_PALABRAS:
        return t, f'Texto vacío o con menos de {MIN_PALABRAS} palabras: no parece una reseña.', avisos
    if es_repetitivo(t):
        return t, 'Texto repetitivo (casi todas las palabras iguales): entrada no válida.', avisos
    if not es_espanol(t):
        return t, 'No parece estar en español: el modelo solo se validó en español.', avisos
    if proporcion_mayusculas(t) > .6:
        t = normalizar_mayusculas(t)
        avisos.append('El texto venía casi todo en mayúsculas y se normalizó (BETO pierde 0,19 de F1 con mayúsculas).')
    return t, None, avisos


def decidir(probs: dict, umbral: float, t_negativa: float) -> tuple[str, str]:
    """Regla de la clase crítica primero (no perder quejas); después, abstención por baja confianza."""
    if probs['negativa'] >= t_negativa:
        return COORDINADOR, f"P(negativa) = {probs['negativa']:.2f} ≥ {t_negativa:.2f}: posible queja"
    clase, conf = max(probs.items(), key=lambda kv: kv[1])
    if conf < umbral:
        return REVISION, f'Confianza {conf:.3f} < umbral {umbral:.2f}'
    return AUTOMATICO, f'Confianza {conf:.3f} ≥ umbral {umbral:.2f}'
