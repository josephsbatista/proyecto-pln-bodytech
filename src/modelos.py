"""Carga los artefactos guardados por el notebook (no reentrena) y predice con el baseline y con BETO."""
import re
from pathlib import Path

import joblib
import numpy as np
import spacy
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .preprocesamiento import lematizar, normalizar


def softmax(z):
    e = np.exp(z - z.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class Clasificadores:
    """Baseline TF-IDF de lemas (decide en operación) y BETO calibrado (shadow mode). Inferencia en CPU."""

    def __init__(self, raiz: Path, politica: dict):
        self.clases = politica['clases']
        pb, pt = politica['modelos']['Baseline'], politica['modelos']['BETO']
        self.umbral = {'Baseline': pb['umbral_abstencion'], 'BETO': pt['umbral_abstencion']}
        self.t_neg = {'Baseline': pb['t_negativa'], 'BETO': pt['t_negativa']}
        self.temperatura, self.max_len = pt['temperatura'], pt['max_length']
        self.nlp = spacy.load('es_core_news_sm', disable=['parser', 'ner'])
        self.baseline = joblib.load(raiz / 'modelos' / 'baseline_tfidf_lemas.joblib')
        ruta_beto = raiz / 'modelos' / 'beto' / f'beto_L{self.max_len}_s{politica["semilla"]}' / 'modelo'
        self.tok = AutoTokenizer.from_pretrained(ruta_beto)
        self.beto = AutoModelForSequenceClassification.from_pretrained(ruta_beto).eval()
        assert list(self.baseline.classes_) == self.clases

    # ---------- predicción
    def proba_baseline(self, textos):
        return self.baseline.predict_proba(lematizar([normalizar(t) for t in textos], self.nlp))

    @torch.inference_mode()   # sin gradientes; necesario en cada hilo (Gradio atiende en hilos distintos)
    def proba_beto(self, textos, lote=32):
        textos = [normalizar(t) for t in textos]; salida = []
        for i in range(0, len(textos), lote):
            b = self.tok(textos[i:i + lote], truncation=True, max_length=self.max_len, padding=True, return_tensors='pt')
            salida.append(self.beto(**b).logits.numpy())
        return softmax(np.concatenate(salida) / self.temperatura)   # probabilidades calibradas (sección 9.3)

    def como_dict(self, p):
        return {c: float(v) for c, v in zip(self.clases, p)}

    # ---------- evidencia
    def evidencia_baseline(self, texto):
        """Aporte de cada palabra a la clase predicha: valor TF-IDF de su lema × coeficiente de la regresión."""
        texto = normalizar(texto)
        doc = self.nlp(texto)
        vec, clf = self.baseline.named_steps['tfidf'], self.baseline.named_steps['clf']
        x = vec.transform([' '.join(t.lemma_.lower() for t in doc if not t.is_space)])
        k = int(clf.predict_proba(x)[0].argmax())
        vocab = vec.vocabulary_
        aportes = []
        for t in doc:
            j = vocab.get(t.lemma_.lower())
            aportes.append((t.text_with_ws, float(x[0, j] * clf.coef_[k, j]) if j is not None else 0.0))
        m = max((abs(a) for _, a in aportes), default=0) or 1.0
        return [(w, round(a / m, 3) if abs(a) / m > .25 else None) for w, a in aportes], self.clases[k]

    def evidencia_beto(self, texto):
        """Oclusión por oración: cuánto baja la probabilidad de la clase predicha si se quita cada oración."""
        texto = normalizar(texto)
        oraciones = [o for o in re.split(r'(?<=[.!?…])\s+', texto) if o.strip()]
        base = self.proba_beto([texto])[0]; k = int(base.argmax())
        if len(oraciones) < 2:
            return [(texto, 1.0)], self.clases[k]
        sin = self.proba_beto([' '.join(oraciones[:i] + oraciones[i + 1:]) for i in range(len(oraciones))])
        caidas = base[k] - sin[:, k]
        m = max(abs(caidas).max(), 1e-6)
        return [(o + ' ', round(float(c / m), 3)) for o, c in zip(oraciones, caidas)], self.clases[k]
