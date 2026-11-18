"""Preprocesamiento y filtros previos, idénticos a los del notebook principal (secciones 2 y 11.2)."""
import re
import unicodedata

import numpy as np

# Stopwords frecuentes del español: una reseña en español tiene al menos un 8 % de ellas (sección 11.2)
SW_ES = set('de la que el en y a los se del las un por con no una su para es al lo como más pero sus le ya o '
            'fue este ha muy'.split())


def normalizar(s: str) -> str:
    """Unicode NFC, enmascara URL y correos (sin tocar el "@" inclusivo de "chic@s") y compacta espacios."""
    s = unicodedata.normalize('NFC', str(s))
    s = re.sub(r'https?://\S+|www\.\S+', ' URL ', s)
    s = re.sub(r'[\w.%-]+@[\w.-]+\.[A-Za-z]{2,}', ' EMAIL ', s)
    return re.sub(r'\s+', ' ', s).strip()


def tokenizar(s: str) -> list[str]:
    return re.sub(r'[^\wáéíóúñü\s]', ' ', s.lower()).split()


def es_espanol(s: str, minimo: float = .08) -> bool:
    t = tokenizar(s)
    return bool(t) and float(np.mean([w in SW_ES for w in t])) >= minimo


def proporcion_mayusculas(s: str) -> float:
    letras = [c for c in s if c.isalpha()]
    return sum(c.isupper() for c in letras) / len(letras) if letras else 0.0


def normalizar_mayusculas(s: str) -> str:
    """Pasa a minúsculas un texto escrito casi todo en mayúsculas y pone mayúscula al inicio de cada oración.
    BETO pierde 0,19 de F1 con texto en mayúsculas (sección 11.1). Control propuesto para el piloto."""
    s = s.lower()
    return re.sub(r'(^|[.!?¡¿]\s+)(\w)', lambda m: m.group(1) + m.group(2).upper(), s)


def es_repetitivo(s: str) -> bool:
    """Entradas absurdas como "bien bien bien…": muchas palabras y casi ninguna distinta."""
    t = tokenizar(s)
    return len(t) >= 20 and len(set(t)) / len(t) < .2


def lematizar(textos, nlp) -> list[str]:
    """Entrada del baseline: lemas en minúsculas con spaCy (es_core_news_sm), igual que en la sección 2."""
    return [' '.join(t.lemma_.lower() for t in d if not t.is_space) for d in nlp.pipe(textos)]
