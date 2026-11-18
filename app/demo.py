"""Demo en vivo · Clasificación del sentimiento de reseñas por sede.

Uso:  python app/demo.py   → abrir http://127.0.0.1:7860
Carga los modelos guardados por el notebook (no reentrena) y funciona sin internet. Si faltan y MODELOS_REPO está
definida, los descarga del Hub de Hugging Face (así corre en el Space). DEMO_PUBLICA=1 usa un nombre neutro.
"""
import hashlib
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

PUBLICA = os.getenv('DEMO_PUBLICA', '0') == '1'
try:          # el Space solo lleva textos_publico.py (nombre neutro)
    if PUBLICA:
        raise ImportError
    from app.textos import ENCABEZADO, TITULO
except ImportError:
    from app.textos_publico import ENCABEZADO, TITULO
asegurar_modelos(RAIZ)
POL = cargar_politica(RAIZ)
print('Cargando modelos…')
M = Clasificadores(RAIZ, POL)
LOGS = RAIZ / 'logs'; LOGS.mkdir(exist_ok=True)
REGISTRO = LOGS / 'predicciones.csv'
COLUMNAS = ['fecha', 'sede', 'huella_texto', 'palabras', 'accion', 'motivo', 'clase_baseline', 'conf_baseline',
            'accion_beto_shadow', 'clase_beto_shadow', 'conf_beto_shadow', 'version_politica']
COLOR = {AUTOMATICO: '#0F7B6C', REVISION: '#E39B2D', COORDINADOR: '#C24E3A'}

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


def tarjeta(accion, clase, conf, motivo, avisos, titulo):
    avisos_md = ''.join(f'\n> ⚠️ {a}' for a in avisos)
    detalle = f' · clase **{clase}** · confianza **{conf:.2f}**' if clase else ''
    return (f"### {titulo}\n<span style='color:{COLOR[accion]};font-size:1.4em;font-weight:700'>{accion}</span>"
            f"{detalle}\n\n{motivo}{avisos_md}")


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
        md = tarjeta(REVISION, None, 0, f'**Filtro previo:** {motivo} No se envía al modelo.', avisos, 'Decisión')
        return md, None, None, [], [], '', f'Registrado en `logs/predicciones.csv` · {1000 * (time.perf_counter() - t0):.0f} ms'
    pb = M.como_dict(M.proba_baseline([t])[0]); pt = M.como_dict(M.proba_beto([t])[0])
    acc_b, por_b = decidir(pb, M.umbral['Baseline'], M.t_neg['Baseline'])
    acc_t, por_t = decidir(pt, M.umbral['BETO'], M.t_neg['BETO'])
    cb, ct = max(pb, key=pb.get), max(pt, key=pt.get)
    ev_b, _ = M.evidencia_baseline(t); ev_t, _ = M.evidencia_beto(t)
    ms = 1000 * (time.perf_counter() - t0)
    registrar(texto, sede, {'accion': acc_b, 'motivo': por_b, 'clase_baseline': cb, 'conf_baseline': round(pb[cb], 3),
                            'accion_beto_shadow': acc_t, 'clase_beto_shadow': ct, 'conf_beto_shadow': round(pt[ct], 3)})
    md = tarjeta(acc_b, cb, pb[cb], f'**Modelo en operación:** baseline TF-IDF de lemas. {por_b}.', avisos, 'Decisión')
    acuerdo = 'coincide con el baseline' if (acc_t, ct) == (acc_b, cb) else '**no coincide** con el baseline'
    shadow = (f"### BETO · *shadow mode* (no decide)\nHaría: <span style='color:{COLOR[acc_t]};font-weight:700'>{acc_t}</span>"
              f" · clase **{ct}** · confianza calibrada **{pt[ct]:.2f}** — {acuerdo}.")
    traza = (f"Umbrales elegidos en validación (`modelos/politica.json`, {POL['fecha']}): baseline abstención "
             f"{M.umbral['Baseline']:.2f}, P(negativa) ≥ {M.t_neg['Baseline']:.2f} · BETO abstención {M.umbral['BETO']:.2f}, "
             f"P(negativa) ≥ {M.t_neg['BETO']:.2f}, temperatura {M.temperatura:.2f} · {ms:.0f} ms en CPU · "
             f"registrado en `logs/predicciones.csv`")
    return md, pb, pt, ev_b, ev_t, shadow, traza


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
    resumen_md = (f"**{len(df)} reseñas** · decide **{modelo}** · al coordinador **{(df.accion == COORDINADOR).sum()}** · "
                  f"a revisión humana **{(df.accion == REVISION).sum()}** (filtro previo: {(~ok).sum()}) · "
                  f"automáticas **{(df.accion == AUTOMATICO).sum()}**")
    if 'etiqueta_real' in df:   # auditoría: solo disponible en la muestra etiquetada
        neg = df.etiqueta_real == 'negativa'
        tabla['negativas reales'] = g.etiqueta_real.apply(lambda s: (s == 'negativa').sum())
        tabla['quejas detectadas'] = df[neg].groupby(['ciudad', 'sede']).accion.apply(lambda s: (s == COORDINADOR).sum())
        tabla['quejas detectadas'] = tabla['quejas detectadas'].fillna(0).astype(int)
        recall = (df.accion[neg] == COORDINADOR).mean()
        resumen_md += (f"\n\n**Auditoría** (la muestra trae la etiqueta real): {neg.sum()} quejas reales, "
                       f"**{(df.accion[neg] == COORDINADOR).sum()} llegaron al coordinador** (recall de negativas {recall:.2f}).")
    ciudades = (df.groupby('ciudad').accion.apply(lambda s: (s == COORDINADOR).mean() * 100)
                  .rename('% al coordinador').reset_index())
    salida = LOGS / f"resumen_{modelo}_{datetime.now():%Y%m%d_%H%M%S}.csv"
    df.drop(columns='texto_modelo').to_csv(salida, index=False)
    return resumen_md, tabla.round(1).reset_index(), ciudades, str(salida)


with gr.Blocks(title=TITULO) as demo:
    gr.Markdown(ENCABEZADO)
    with gr.Tab('1 · Una reseña'):
        with gr.Row():
            with gr.Column(scale=3):
                txt = gr.Textbox(label='Reseña (título y texto)', lines=5, placeholder='Escriba o pegue una reseña…')
            with gr.Column(scale=1):
                sede = gr.Textbox(label='Sede (opcional)', value='Sede Centro')
                btn = gr.Button('Clasificar', variant='primary')
        gr.Examples(EJEMPLOS, inputs=[txt, sede], label='Ejemplos: positiva, queja, mixta, negación, ironía, mayúsculas, otro idioma, fuera de dominio y vacío')
        with gr.Row():
            decision = gr.Markdown()
            sombra = gr.Markdown()
        with gr.Row():
            prob_b = gr.Label(label='Probabilidades · baseline (decide)', num_top_classes=3)
            prob_t = gr.Label(label='Probabilidades · BETO calibrado (shadow)', num_top_classes=3)
        gr.Markdown('#### Evidencia: ¿qué pesó en la decisión?')
        with gr.Row():
            ev_b = gr.HighlightedText(label='Baseline · palabras que empujan hacia (+) o en contra (−) de la clase predicha',
                                      combine_adjacent=False, show_legend=True)
            ev_t = gr.HighlightedText(label='BETO · cuánto baja la clase predicha si se quita cada oración',
                                      combine_adjacent=False, show_legend=True)
        traza = gr.Markdown()
        btn.click(analizar, [txt, sede], [decision, prob_b, prob_t, ev_b, ev_t, sombra, traza])
        txt.submit(analizar, [txt, sede], [decision, prob_b, prob_t, ev_b, ev_t, sombra, traza])

    with gr.Tab('2 · Resumen por sede'):
        gr.Markdown('Suba un CSV con columnas `texto`, `sede` y `ciudad`. Sin archivo, se usa `datos/muestra_controlada.csv`: '
                    '400 reseñas de **test** (sedes que ningún modelo vio), con su etiqueta real para auditar.')
        with gr.Row():
            archivo = gr.File(label='CSV de reseñas (opcional)', file_types=['.csv'], type='filepath')
            quien = gr.Radio(['Baseline (operación)', 'BETO (shadow)'], value='Baseline (operación)', label='Modelo que decide')
        btn2 = gr.Button('Procesar', variant='primary')
        res_md = gr.Markdown()
        grafico = gr.BarPlot(x='ciudad', y='% al coordinador', title='% de reseñas enviadas al coordinador, por ciudad', height=280)
        tabla = gr.Dataframe(label='Resumen por sede', wrap=True)
        descarga = gr.File(label='Predicciones (CSV)')
        btn2.click(resumen, [archivo, quien], [res_md, tabla, grafico, descarga])

if __name__ == '__main__':
    demo.launch(show_error=True)   # local: 127.0.0.1:7860; en el Space, Gradio toma host y puerto del entorno
