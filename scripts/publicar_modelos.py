"""Publica los modelos de la demo en el Hub de Hugging Face, para levantarla sin entrenar (MODELOS_REPO).

Requisitos: haber entrenado (modelos/ completo) y haber iniciado sesión con `hf auth login` (token de escritura).

    python scripts/publicar_modelos.py --usuario <usuario_hf>            # público
    python scripts/publicar_modelos.py --usuario <usuario_hf> --privado

Sube el baseline, el BETO elegido y politica.json con su tarjeta (licencia CC BY-NC 4.0). Si los pesos no cambiaron,
Hugging Face no los vuelve a subir. El token nunca se escribe en archivos.
"""
import argparse
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
from src.artefactos import BASELINE, POLITICA, faltantes, ruta_beto  # noqa: E402

NOMBRE = 'Clasificador de reseñas por sede'
REPO_MODELO = 'clasificador-resenas-sede-modelos'
GITHUB = 'https://github.com/josephsbatista/proyecto-pln-bodytech'


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

Artefactos que usa la demo en Gradio del proyecto. Código, notebook e informe:
{GITHUB}

Este repositorio **solo guarda los modelos**: la interfaz gráfica es la demo, que se levanta en local y los descarga de aquí:
```bash
git clone {GITHUB} && cd {GITHUB.rsplit('/', 1)[-1]}
pip install -r requirements.txt
MODELOS_REPO={usuario}/{REPO_MODELO} python app/demo.py      # abre http://127.0.0.1:7860
```

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

## Licencias y atribución
- **Datos de entrenamiento:** Andalusian Hotels' Reviews (Big_AHR), de Mariia Chizhikova,
  https://www.kaggle.com/datasets/chizhikchi/andalusian-hotels-reviews-unbalanced · **CC BY-NC 4.0**
  (incluye el corpus COAH, SINAI, Universidad de Jaén, 2014). Por eso los modelos son de **uso no comercial**.
- **Modelo base:** BETO, `dccuchile/bert-base-spanish-wwm-cased` (Universidad de Chile) · CC BY 4.0.
- **Código:** MIT.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--usuario', required=True, help='usuario (u organización) de Hugging Face')
    ap.add_argument('--privado', action='store_true', help='crear el repositorio como privado')
    a = ap.parse_args()

    if faltantes(RAIZ):
        raise SystemExit(f'Faltan modelos en modelos/: {faltantes(RAIZ)}. Ejecute primero el notebook.')
    pol = json.load(open(RAIZ / 'modelos' / POLITICA, encoding='utf-8'))

    from huggingface_hub import HfApi, get_token
    token = get_token()
    if not token:
        raise SystemExit('No hay sesión de Hugging Face. Ejecute `hf auth login` con un token de escritura.')
    api = HfApi(token=token)
    repo = f'{a.usuario}/{REPO_MODELO}'
    print(f'Subiendo los modelos a {repo} (≈ 440 MB la primera vez)…')
    api.create_repo(repo, private=a.privado, exist_ok=True)
    api.upload_folder(repo_id=repo, folder_path=RAIZ / 'modelos', commit_message='Artefactos de la demo',
                      allow_patterns=[POLITICA, BASELINE, f'{ruta_beto(pol)}/*.json', f'{ruta_beto(pol)}/*.safetensors'])
    api.upload_file(repo_id=repo, path_or_fileobj=tarjeta_modelo(a.usuario, pol).encode('utf-8'),
                    path_in_repo='README.md', commit_message='Tarjeta del modelo')
    print(f'Listo: https://huggingface.co/{repo}')


if __name__ == '__main__':
    main()
