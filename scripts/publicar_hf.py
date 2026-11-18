"""Publica la demo en Hugging Face: un repositorio de modelo con los artefactos y un Space de Gradio que los descarga.

Requisitos: haber entrenado (modelos/ completo) y haber iniciado sesión con `hf auth login` (token de escritura).

    python scripts/publicar_hf.py --usuario <usuario_hf>            # público
    python scripts/publicar_hf.py --usuario <usuario_hf> --privado  # solo visible para ti y a quien invites
    python scripts/publicar_hf.py --usuario <usuario_hf> --solo-preparar   # arma dist/ sin subir nada

El token nunca se escribe en archivos: si el Space es privado, se guarda como *secret* del Space.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.artefactos import BASELINE, POLITICA, faltantes, ruta_beto  # noqa: E402

NOMBRE = 'Clasificador de reseñas por sede'
REPO_MODELO = 'clasificador-resenas-sede-modelos'
REPO_SPACE = 'clasificador-resenas-por-sede'
GRADIO = '6.32.0'

REQUISITOS_SPACE = """--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.15.0
transformers==5.23.0
scikit-learn==1.9.1
scipy==1.18.1
numpy==2.5.4
pandas==3.1.0
joblib==1.6.0
spacy==3.8.18
es_core_news_sm @ https://github.com/explosion/spacy-models/releases/download/es_core_news_sm-3.8.0/es_core_news_sm-3.8.0-py3-none-any.whl
huggingface_hub==2.4.0
"""

LICENCIAS = """## Licencias y atribución
- **Datos de entrenamiento:** Andalusian Hotels' Reviews (Big_AHR), de Mariia Chizhikova,
  https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced · **CC BY-NC 4.0**
  (incluye el corpus COAH, SINAI, Universidad de Jaén, 2014). Por eso los modelos y la demo son de **uso no comercial**.
- **Modelo base:** BETO, `dccuchile/bert-base-spanish-wwm-cased` (Universidad de Chile) · CC BY 4.0.
- **Código:** MIT.
"""


def tarjeta_modelo(usuario, pol):
    b, t = pol['modelos']['Baseline'], pol['modelos']['BETO']
    return f"""---
license: cc-by-nc-4.0
language: es
library_name: transformers
pipeline_tag: text-classification
base_model: dccuchile/bert-base-spanish-wwm-cased
tags: [sentiment-analysis, spanish, reviews]
---
# {NOMBRE} · modelos

Artefactos que usa la demo [{usuario}/{REPO_SPACE}](https://huggingface.co/spaces/{usuario}/{REPO_SPACE}).
Clasifican el sentimiento de una reseña en español en **{', '.join(pol['clases'])}**. Entrenados con reseñas
públicas de hoteles (proxy: hotel = sede), con partición por hotel para medir la generalización a sedes nuevas.

| Archivo | Qué es |
|---|---|
| `{BASELINE}` | Baseline: TF-IDF de lemas (spaCy `es_core_news_sm`) + regresión logística (scikit-learn) |
| `{ruta_beto(pol)}/` | BETO ajustado, longitud máxima {t['max_length']}, semilla {pol['semilla']} |
| `{POLITICA}` | Umbrales elegidos en **validación**: abstención (baseline {b['umbral_abstencion']}, BETO {t['umbral_abstencion']}), regla P(negativa) ≥ t (baseline {b['t_negativa']}, BETO {t['t_negativa']}) y temperatura de calibración de BETO ({t['temperatura']:.2f}) |

**Uso previsto:** demostración académica de priorización de reseñas negativas con revisión humana.
**Fuera de alcance:** decisiones automáticas sobre personas, uso comercial y textos que no sean reseñas en español.
Para producción hay que reentrenar con datos propios de la organización.

{LICENCIAS}"""


def readme_space(usuario):
    return f"""---
title: {NOMBRE}
emoji: 💬
colorFrom: green
colorTo: gray
sdk: gradio
sdk_version: {GRADIO}
python_version: "3.12"
app_file: app/demo.py
pinned: false
license: cc-by-nc-4.0
short_description: Sentimiento de reseñas por sede, con abstención
models:
  - {usuario}/{REPO_MODELO}
  - dccuchile/bert-base-spanish-wwm-cased
---
# {NOMBRE}

Demo de un proyecto final de Procesamiento de Lenguaje Natural.
- **Pestaña 1 · Una reseña:** filtros previos (vacío, repetitivo, otro idioma, mayúsculas) → clase y probabilidades →
  acción (automático, revisión humana o coordinador de la sede) → evidencia de qué pesó en la decisión.
  Decide el baseline; BETO se muestra al lado en *shadow mode*.
- **Pestaña 2 · Resumen por sede:** procesa un CSV (por defecto, 400 reseñas reales del conjunto de test) y muestra,
  por sede, el porcentaje de negativas y las quejas que llegan al coordinador.

Corre en CPU. Los modelos se descargan de [{usuario}/{REPO_MODELO}](https://huggingface.co/{usuario}/{REPO_MODELO})
al arrancar. Las predicciones se registran sin guardar el texto (solo una huella SHA-256) y el registro se borra
cuando el Space se reinicia.

{LICENCIAS}"""


def preparar_space(destino: Path, usuario: str):
    if destino.exists():
        shutil.rmtree(destino)
    for rel in ['app/__init__.py', 'app/demo.py', 'app/textos_publico.py', 'datos/muestra_controlada.csv', 'LICENSE',
                *[f'src/{p.name}' for p in sorted((RAIZ / 'src').glob('*.py'))]]:
        (destino / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(RAIZ / rel, destino / rel)
    (destino / 'requirements.txt').write_text(REQUISITOS_SPACE, encoding='utf-8')
    (destino / 'README.md').write_text(readme_space(usuario), encoding='utf-8')
    revisar = [p for p in destino.rglob('*') if p.is_file() and p.suffix in {'.py', '.md', '.txt'}
               and 'bodytech' in p.read_text(encoding='utf-8').lower()]
    if revisar:
        raise SystemExit(f'Estos archivos del Space mencionan Bodytech: {revisar}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--usuario', required=True, help='usuario (u organización) de Hugging Face')
    ap.add_argument('--privado', action='store_true', help='crear el repositorio de modelo y el Space como privados')
    ap.add_argument('--solo-preparar', action='store_true', help='armar dist/ y no subir nada')
    a = ap.parse_args()

    if faltantes(RAIZ):
        raise SystemExit(f'Faltan modelos en modelos/: {faltantes(RAIZ)}. Ejecute primero el notebook.')
    pol = json.load(open(RAIZ / 'modelos' / POLITICA, encoding='utf-8'))
    dist = RAIZ / 'dist'
    preparar_space(dist / 'space', a.usuario)
    (dist / 'MODELO_README.md').write_text(tarjeta_modelo(a.usuario, pol), encoding='utf-8')
    print(f'Preparado: {dist / "space"} y {dist / "MODELO_README.md"}')
    if a.solo_preparar:
        return

    from huggingface_hub import HfApi, get_token
    token = get_token()
    if not token:
        raise SystemExit('No hay sesión de Hugging Face. Ejecute `hf auth login` con un token de escritura.')
    api = HfApi(token=token)
    modelo, space = f'{a.usuario}/{REPO_MODELO}', f'{a.usuario}/{REPO_SPACE}'

    print(f'1/2 · Subiendo los modelos a {modelo} (≈ 440 MB)…')
    api.create_repo(modelo, private=a.privado, exist_ok=True)
    api.upload_folder(repo_id=modelo, folder_path=RAIZ / 'modelos', commit_message='Artefactos de la demo',
                      allow_patterns=[POLITICA, BASELINE, f'{ruta_beto(pol)}/*.json', f'{ruta_beto(pol)}/*.safetensors'])
    api.upload_file(repo_id=modelo, path_or_fileobj=dist / 'MODELO_README.md', path_in_repo='README.md',
                    commit_message='Tarjeta del modelo')

    print(f'2/2 · Creando el Space {space}…')
    api.create_repo(space, repo_type='space', space_sdk='gradio', private=a.privado, exist_ok=True)
    api.add_space_variable(space, 'MODELOS_REPO', modelo)
    api.add_space_variable(space, 'DEMO_PUBLICA', '1')
    if a.privado:   # el Space necesita permiso para leer el repositorio de modelo privado
        api.add_space_secret(space, 'HF_TOKEN', token)
    api.upload_folder(repo_id=space, repo_type='space', folder_path=dist / 'space', commit_message='Demo')
    print(f'Listo: https://huggingface.co/spaces/{space}  (la primera construcción tarda unos minutos)')


if __name__ == '__main__':
    main()
