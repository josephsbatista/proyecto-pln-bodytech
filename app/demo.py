"""Demo en vivo · Clasificación del sentimiento de reseñas por sede.

Uso:  python app/demo.py   → abrir http://127.0.0.1:7860
Carga los modelos guardados por el notebook (no reentrena) y funciona sin internet. Si faltan y MODELOS_REPO está
definida, los descarga del Hub de Hugging Face. DEMO_COMPARTIR=1 crea además un enlace público temporal.
"""
import hashlib
import html
import os
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import gradio as gr
import pandas as pd

from src.artefactos import asegurar_modelos
from src.modelos import Clasificadores
from src.politica import AUTOMATICO, COORDINADOR, REVISION, cargar_politica, decidir, filtrar

asegurar_modelos(RAIZ)
POL = cargar_politica(RAIZ)
print('Cargando modelos…')
M = Clasificadores(RAIZ, POL)
LOGS = RAIZ / 'logs'; LOGS.mkdir(exist_ok=True)
REGISTRO = LOGS / 'predicciones.csv'
COLUMNAS = ['fecha', 'sede', 'huella_texto', 'palabras', 'accion', 'motivo', 'clase_baseline', 'conf_baseline',
            'accion_beto_shadow', 'clase_beto_shadow', 'conf_beto_shadow', 'version_politica']
COLOR = {AUTOMATICO: '#0F7B6C', REVISION: '#D08A1E', COORDINADOR: '#C24E3A'}
ICONO = {AUTOMATICO: '✓', REVISION: '?', COORDINADOR: '!'}

EJEMPLOS = [
    ['Excelente. El personal fue muy amable, la habitación impecable y el desayuno variado. Volveremos sin duda.', 'Sede Centro'],
    ['Pésima experiencia. Nos cobraron dos veces, la habitación estaba sucia y nadie en recepción quiso ayudarnos.', 'Sede Norte'],
    ['Bien, pero mejorable. La ubicación es buena, pero las instalaciones están viejas y el aire acondicionado no funcionaba.', 'Sede Centro'],
    ['La habitación no estaba limpia y el personal no fue amable.', 'Sede Sur'],
    ['Genial, solo tardaron dos horas en darnos la habitación. Maravilloso.', 'Sede Sur'],
    ['HORRIBLE. NUNCA MÁS VOLVEMOS. EL PEOR SERVICIO QUE HEMOS RECIBIDO.', 'Sede Norte'],
    ['The room was dirty and the staff was rude.', 'Sede Centro'],
    ['La fotosíntesis depende de la luz.', 'Sede Centro'],
    ['', 'Sede Centro'],
]
ETIQUETAS = ['Positiva', 'Queja', 'Mixta', 'Negación', 'Ironía', 'Mayúsculas', 'Otro idioma', 'Fuera de dominio', 'Vacío']

# ---------- diseño
TEMA = gr.themes.Soft(
    primary_hue=gr.themes.Color(c50='#E8F4F1', c100='#CDE8E1', c200='#9FD1C4', c300='#6CB8A6', c400='#3E9E89',
                                c500='#0F7B6C', c600='#0C6A5D', c700='#0A584D', c800='#08463E', c900='#06352F',
                                c950='#042621'),
    neutral_hue='slate', radius_size='lg', font=[gr.themes.GoogleFont('Inter'), 'system-ui', 'sans-serif'],
).set(button_primary_background_fill='*primary_500', button_primary_background_fill_hover='*primary_600',
      block_title_text_weight='600')

CSS = """
.gradio-container {max-width: 1240px !important; margin: 0 auto !important;}
.banner {padding: 22px 26px; border-radius: 16px; background: linear-gradient(135deg, #0A4A42 0%, #0F7B6C 60%, #2E9C8A 100%);
         color: #fff; margin-bottom: 6px;}
.banner h1 {color: #fff !important; margin: 0 0 6px; font-size: 1.65rem; font-weight: 700; letter-spacing: -.01em;}
.banner p {color: #DDF0EB !important; margin: 0; font-size: .95rem; line-height: 1.45;}
.chips {display: flex; gap: 8px; flex-wrap: wrap; margin-top: 14px;}
.chip {background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28); padding: 4px 11px; border-radius: 999px;
       font-size: .8rem; color: #fff;}
.banner b, .chip b {color: #fff !important; background: none !important; font-weight: 700;}
.tabla, .tabla * {font-family: var(--font) !important;}
.tabla th, .tabla th * {word-break: normal !important; overflow-wrap: normal !important; hyphens: none;}
.card {border: 1px solid var(--border-color-primary); border-left: 6px solid var(--acc); border-radius: 14px;
       padding: 16px 18px; background: var(--block-background-fill); margin-bottom: 12px;}
.card.vacia {border: 1.5px dashed var(--neutral-300); color: var(--body-text-color-subdued); text-align: center;
             padding: 34px 18px;}
.eyebrow {font-size: .72rem; text-transform: uppercase; letter-spacing: .08em; color: var(--body-text-color-subdued);
          font-weight: 600;}
.accion {display: flex; align-items: center; gap: 10px; font-size: 1.45rem; font-weight: 700; color: var(--acc); margin: 6px 0 10px;}
.accion.chica {font-size: 1.1rem; margin: 4px 0 8px;}
.icono {display: inline-flex; width: 30px; height: 30px; border-radius: 50%; background: var(--acc); color: #fff;
        align-items: center; justify-content: center; font-size: 1rem; flex: none;}
.chica .icono {width: 24px; height: 24px; font-size: .85rem;}
.meta {display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 10px;}
.meta span {background: var(--background-fill-secondary); border-radius: 6px; padding: 3px 9px; font-size: .84rem;}
.barra {height: 8px; background: var(--background-fill-secondary); border-radius: 999px; overflow: hidden; margin: 2px 0 10px;}
.barra > div {height: 100%; background: var(--acc); border-radius: 999px;}
.motivo {font-size: .9rem; line-height: 1.45;}
.aviso {margin-top: 10px; padding: 8px 11px; border-radius: 8px; background: rgba(208,138,30,.13); font-size: .84rem;}
.acuerdo {font-size: .84rem; color: var(--body-text-color-subdued);}
.kpis {display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin: 4px 0 8px;}
.kpi {border: 1px solid var(--border-color-primary); border-radius: 14px; padding: 14px 16px; background: var(--block-background-fill);
      border-top: 4px solid var(--acc, #64748B);}
.kpi .v {font-size: 1.9rem; font-weight: 700; line-height: 1.1; color: var(--acc, var(--body-text-color));}
.kpi .l {font-size: .8rem; color: var(--body-text-color-subdued); margin-top: 4px;}
.nota {font-size: .86rem; color: var(--body-text-color-subdued); margin-top: 4px;}
.seccion {font-size: 1.02rem; font-weight: 650; margin: 14px 0 2px;}
@media (max-width: 820px) {.kpis {grid-template-columns: repeat(2, minmax(0, 1fr));}}
footer {display: none !important;}
"""

BANNER = """
<div class="banner">
  <h1>Clasificación del sentimiento de reseñas por sede</h1>
  <p>Prototipo del proyecto final de PLN · caso Bodytech. Entrenado con reseñas <b>reales y públicas de hoteles</b>
     (proxy: hotel = sede); <b>no son datos de Bodytech</b>.</p>
  <div class="chips">
    <span class="chip">Decide: <b>baseline</b> TF-IDF de lemas</span>
    <span class="chip">BETO calibrado en <b>shadow mode</b></span>
    <span class="chip">Umbrales elegidos en <b>validación</b></span>
    <span class="chip">Se registra sin guardar el texto</span>
  </div>
</div>"""

VACIA_DECISION = '<div class="card vacia">Escriba o elija una reseña y pulse <b>Clasificar</b>.</div>'
VACIA_RESUMEN = '<div class="card vacia">Pulse <b>Procesar</b> para resumir la muestra de 400 reseñas o el CSV que suba.</div>'


def tarjeta(accion, clase, conf, motivo, avisos, titulo, chica=False, extra=''):
    meta = (f'<div class="meta"><span>clase <b>{clase}</b></span><span>confianza <b>{conf:.2f}</b></span></div>'
            f'<div class="barra"><div style="width:{100 * conf:.0f}%"></div></div>') if clase else ''
    avisos_html = ''.join(f'<div class="aviso">⚠️ {html.escape(a)}</div>' for a in avisos)
    return (f'<div class="card" style="--acc:{COLOR[accion]}"><div class="eyebrow">{titulo}</div>'
            f'<div class="accion{" chica" if chica else ""}"><span class="icono">{ICONO[accion]}</span>{accion}</div>'
            f'{meta}<div class="motivo">{motivo}</div>{extra}{avisos_html}</div>')


def kpi(valor, etiqueta, color=None):
    estilo = f' style="--acc:{color}"' if color else ''
    return f'<div class="kpi"{estilo}><div class="v">{valor}</div><div class="l">{etiqueta}</div></div>'


def registrar(texto, sede, fila):
    """Trazabilidad sin guardar el texto (minimización): huella SHA-256, longitud y decisión."""
    nuevo = not REGISTRO.exists()
    linea = pd.DataFrame([{'fecha': datetime.now().isoformat(timespec='seconds'), 'sede': sede or '',
                           'huella_texto': hashlib.sha256((texto or '').encode()).hexdigest()[:16],
                           'palabras': len((texto or '').split()), **fila,
                           'version_politica': POL['fecha']}]).reindex(columns=COLUMNAS)
    linea.to_csv(REGISTRO, mode='a', header=nuevo, index=False)


def analizar(texto, sede):
    t0 = time.perf_counter()
    t, motivo, avisos = filtrar(texto)
    if motivo:
        registrar(texto, sede, {'accion': REVISION, 'motivo': motivo})
        card = tarjeta(REVISION, None, 0, f'<b>Filtro previo:</b> {html.escape(motivo)} No se envía al modelo.', avisos,
                       'Decisión')
        sombra = ('<div class="card vacia" style="padding:18px">BETO tampoco se ejecuta: '
                  'el filtro previo detiene el texto antes de cualquier modelo.</div>')
        return card, sombra, None, None, [], [], f'Registrado en `logs/predicciones.csv` · {1000 * (time.perf_counter() - t0):.0f} ms'
    pb = M.como_dict(M.proba_baseline([t])[0]); pt = M.como_dict(M.proba_beto([t])[0])
    acc_b, por_b = decidir(pb, M.umbral['Baseline'], M.t_neg['Baseline'])
    acc_t, por_t = decidir(pt, M.umbral['BETO'], M.t_neg['BETO'])
    cb, ct = max(pb, key=pb.get), max(pt, key=pt.get)
    ev_b, _ = M.evidencia_baseline(t); ev_t, _ = M.evidencia_beto(t)
    ms = 1000 * (time.perf_counter() - t0)
    registrar(texto, sede, {'accion': acc_b, 'motivo': por_b, 'clase_baseline': cb, 'conf_baseline': round(pb[cb], 3),
                            'accion_beto_shadow': acc_t, 'clase_beto_shadow': ct, 'conf_beto_shadow': round(pt[ct], 3)})
    card = tarjeta(acc_b, cb, pb[cb], f'<b>Modelo en operación:</b> baseline TF-IDF de lemas. {html.escape(por_b)}.',
                   avisos, f'Decisión{" · " + html.escape(sede) if sede else ""}')
    coincide = (acc_t, ct) == (acc_b, cb)
    acuerdo = f'<div class="acuerdo">{"✓ Coincide" if coincide else "≠ No coincide"} con el baseline.</div>'
    sombra = tarjeta(acc_t, ct, pt[ct], f'Confianza calibrada (temperatura {M.temperatura:.2f}). {html.escape(por_t)}.',
                     [], 'BETO · shadow mode (no decide)', chica=True, extra=acuerdo)
    traza = (f"Umbrales elegidos en validación (`modelos/politica.json`, {POL['fecha']}): baseline abstención "
             f"{M.umbral['Baseline']:.2f}, P(negativa) ≥ {M.t_neg['Baseline']:.2f} · BETO abstención {M.umbral['BETO']:.2f}, "
             f"P(negativa) ≥ {M.t_neg['BETO']:.2f}, temperatura {M.temperatura:.2f} · {ms:.0f} ms en CPU · "
             f"registrado en `logs/predicciones.csv`")
    return card, sombra, pb, pt, ev_b, ev_t, traza


def resumen(archivo, quien_decide):
    ruta = archivo if archivo else RAIZ / 'datos' / 'muestra_controlada.csv'
    df = pd.read_csv(ruta)
    if 'texto' not in df: raise gr.Error('El CSV debe tener una columna "texto" (y opcionalmente "sede" y "ciudad").')
    for c in ('sede', 'ciudad'):
        if c not in df: df[c] = '(sin dato)'
    filtrados = [filtrar(x) for x in df.texto.fillna('')]
    df['texto_modelo'] = [f[0] for f in filtrados]; df['filtro'] = [f[1] for f in filtrados]
    ok = df.filtro.isna()
    modelo = 'BETO' if quien_decide.startswith('BETO') else 'Baseline'
    probs = (M.proba_beto if modelo == 'BETO' else M.proba_baseline)(df.loc[ok, 'texto_modelo'].tolist())
    df['clase'] = None; df['accion'] = REVISION
    for i, p in zip(df.index[ok], probs):
        d = M.como_dict(p); df.at[i, 'clase'] = max(d, key=d.get)
        df.at[i, 'accion'] = decidir(d, M.umbral[modelo], M.t_neg[modelo])[0]
    g = df.groupby(['ciudad', 'sede'])
    tabla = pd.DataFrame({'reseñas': g.size(),
                          '% negativas (predicho)': g.clase.apply(lambda s: (s == 'negativa').mean() * 100),
                          'al coordinador': g.accion.apply(lambda s: (s == COORDINADOR).sum()),
                          'a revisión humana': g.accion.apply(lambda s: (s == REVISION).sum()),
                          'automáticas': g.accion.apply(lambda s: (s == AUTOMATICO).sum())})
    n_coord, n_rev, n_auto = (df.accion == COORDINADOR).sum(), (df.accion == REVISION).sum(), (df.accion == AUTOMATICO).sum()
    tarjetas = [kpi(len(df), f'reseñas · decide {modelo}'), kpi(n_coord, 'al coordinador de la sede', COLOR[COORDINADOR]),
                kpi(n_rev, f'a revisión humana (filtro previo: {(~ok).sum()})', COLOR[REVISION]),
                kpi(n_auto, 'automáticas', COLOR[AUTOMATICO])]
    nota = ''
    if 'etiqueta_real' in df:   # auditoría: solo disponible en la muestra etiquetada
        neg = df.etiqueta_real == 'negativa'
        tabla['negativas reales'] = g.etiqueta_real.apply(lambda s: (s == 'negativa').sum())
        tabla['quejas detectadas'] = df[neg].groupby(['ciudad', 'sede']).accion.apply(lambda s: (s == COORDINADOR).sum())
        tabla['quejas detectadas'] = tabla['quejas detectadas'].fillna(0).astype(int)
        detectadas = (df.accion[neg] == COORDINADOR).sum()
        tarjetas.append(kpi(f'{detectadas / max(neg.sum(), 1):.2f}', f'recall de negativas ({detectadas} de {neg.sum()} quejas reales)'))
        nota = ('<div class="nota"><b>Auditoría:</b> la muestra trae la etiqueta real, así que se puede comprobar '
                'cuántas quejas reales llegaron al coordinador.</div>')
    else:
        tarjetas.append(kpi(f'{100 * n_coord / max(len(df), 1):.0f} %', 'del total va al coordinador'))
    resumen_html = f'<div class="kpis">{"".join(tarjetas)}</div>{nota}'
    ciudades = (df.groupby('ciudad').accion.apply(lambda s: (s == COORDINADOR).mean() * 100)
                  .rename('% al coordinador').reset_index().sort_values('% al coordinador', ascending=False))
    salida = LOGS / f"resumen_{modelo}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.drop(columns='texto_modelo').to_csv(salida, index=False)
    return resumen_html, tabla.round(1).reset_index(), ciudades, str(salida)


with gr.Blocks(title='Reseñas por sede · Bodytech') as demo:
    gr.HTML(BANNER)
    with gr.Tab('1 · Una reseña'):
        with gr.Row(equal_height=False):
            with gr.Column(scale=6):
                txt = gr.Textbox(label='Reseña (título y texto)', lines=6, placeholder='Escriba o pegue una reseña…')
                with gr.Row():
                    sede = gr.Textbox(label='Sede (opcional)', value='Sede Centro', scale=2)
                    btn = gr.Button('Clasificar', variant='primary', scale=1, size='lg')
                gr.Examples(EJEMPLOS, inputs=[txt, sede], example_labels=ETIQUETAS,
                            label='Casos de prueba: cada uno muestra un comportamiento distinto')
            with gr.Column(scale=5):
                decision = gr.HTML(VACIA_DECISION)
                sombra = gr.HTML()
        gr.HTML('<div class="seccion">Probabilidades</div>')
        with gr.Row():
            prob_b = gr.Label(label='Baseline (decide)', num_top_classes=3)
            prob_t = gr.Label(label='BETO calibrado (shadow)', num_top_classes=3)
        gr.HTML('<div class="seccion">Evidencia: ¿qué pesó en la decisión?</div>')
        with gr.Row():
            ev_b = gr.HighlightedText(label='Baseline · palabras que empujan hacia (+) o en contra (−) de la clase predicha',
                                      combine_adjacent=False, show_legend=True)
            ev_t = gr.HighlightedText(label='BETO · cuánto baja la clase predicha si se quita cada oración',
                                      combine_adjacent=False, show_legend=True)
        with gr.Accordion('Detalles técnicos y trazabilidad', open=False):
            traza = gr.Markdown()
        salidas = [decision, sombra, prob_b, prob_t, ev_b, ev_t, traza]
        btn.click(analizar, [txt, sede], salidas)
        txt.submit(analizar, [txt, sede], salidas)

    with gr.Tab('2 · Resumen por sede'):
        gr.Markdown('Suba un CSV con columnas `texto`, `sede` y `ciudad`. Sin archivo, se usa `datos/muestra_controlada.csv`: '
                    '400 reseñas de **test** (sedes que ningún modelo vio), con su etiqueta real para auditar.')
        with gr.Row(equal_height=True):
            archivo = gr.File(label='CSV de reseñas (opcional)', file_types=['.csv'], type='filepath', height=150, scale=3)
            with gr.Column(scale=2):
                quien = gr.Radio(['Baseline (operación)', 'BETO (shadow)'], value='Baseline (operación)', label='Modelo que decide')
                btn2 = gr.Button('Procesar', variant='primary', size='lg')
        res_html = gr.HTML(VACIA_RESUMEN)
        grafico = gr.BarPlot(x='ciudad', y='% al coordinador', title='% de reseñas enviadas al coordinador, por ciudad',
                             height=280, sort='-y')
        tabla = gr.Dataframe(label='Resumen por sede', wrap=True, max_height=420, elem_classes='tabla')
        descarga = gr.File(label='Predicciones (CSV)')
        btn2.click(resumen, [archivo, quien], [res_html, tabla, grafico, descarga])

if __name__ == '__main__':
    # http://127.0.0.1:7860
    # DEMO_COMPARTIR=1 crea además un enlace público temporal (*.gradio.live, 1 semana) mientras el equipo esté encendido.
    demo.launch(show_error=True, share=os.getenv('DEMO_COMPARTIR', '0') == '1', theme=TEMA, css=CSS)
