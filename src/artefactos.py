"""Ubica los modelos que usa la demo: primero en modelos/ (los genera el notebook) y, si faltan, los descarga
del Hub de Hugging Face cuando la variable de entorno MODELOS_REPO indica el repositorio."""
import json
import os
from pathlib import Path

BASELINE = 'baseline_tfidf_lemas.joblib'
POLITICA = 'politica.json'


def ruta_beto(politica: dict) -> str:
    return f'beto/beto_L{politica["modelos"]["BETO"]["max_length"]}_s{politica["semilla"]}/modelo'


def faltantes(raiz: Path) -> list:
    m = raiz / 'modelos'
    if not (m / POLITICA).exists():
        return [POLITICA]
    beto = m / ruta_beto(json.load(open(m / POLITICA, encoding='utf-8')))
    return [str(p.relative_to(m)) for p in (m / BASELINE, beto / 'model.safetensors', beto / 'tokenizer.json')
            if not p.exists()]


def asegurar_modelos(raiz: Path) -> None:
    """Deja en modelos/ la política, el baseline y el BETO elegido. Falla con un mensaje claro si no puede."""
    if not faltantes(raiz):
        return
    repo = os.getenv('MODELOS_REPO', '').strip()
    if not repo:
        raise SystemExit(
            f'Faltan los modelos en {raiz / "modelos"}: {", ".join(faltantes(raiz))}.\n'
            '  Opción 1: ejecutar notebooks/Proyecto_Final_PLN.ipynb (≈ 3,5 h con GPU), que los genera.\n'
            '  Opción 2: descargarlos del Hub con MODELOS_REPO=<usuario>/<repositorio> (ver README, sección Demo).')
    from huggingface_hub import snapshot_download
    print(f'Descargando los modelos de {repo}…')
    snapshot_download(repo_id=repo, local_dir=raiz / 'modelos', token=os.getenv('HF_TOKEN') or None,
                      allow_patterns=[POLITICA, BASELINE, 'beto/*/modelo/*'])
    if faltantes(raiz):
        raise SystemExit(f'El repositorio {repo} no tiene: {", ".join(faltantes(raiz))}.')
