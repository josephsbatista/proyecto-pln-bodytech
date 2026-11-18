"""Genera datos/muestra_controlada.csv: 400 reseñas de TEST (sedes que ningún modelo vio) para la demo.
Uso: python app/crear_muestra.py   (requiere datos/Big_AHR.csv y datos/particion.csv)"""
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ETQ = {0: 'negativa', 3: 'neutra', 1: 'positiva'}

raw = pd.read_csv(RAIZ / 'datos' / 'Big_AHR.csv').drop(columns='Unnamed: 0')
df = raw.drop_duplicates(['title', 'review_text']).reset_index(drop=True)   # misma deduplicación que el notebook
part = pd.read_csv(RAIZ / 'datos' / 'particion.csv', index_col='id')
assert len(part) == len(df) and (part.etiqueta.values == df.label.map(ETQ).values).all(), 'la partición no coincide'

te = df[(part.particion == 'test').values & df.hotel.notna().values].copy()
te['sede'] = te.hotel.str.replace('_', ' ')
te['ciudad'] = te.location.str.extract(r'^([A-Za-z]+)')[0]
# 20 sedes de test con 20 reseñas cada una: 4 sedes en cada una de las 5 ciudades con más reseñas de test
top_ciudades = te.ciudad.value_counts().index[:5]
sedes = (te[te.ciudad.isin(top_ciudades)].groupby(['ciudad', 'sede']).size().reset_index(name='n')
           .query('n >= 20').groupby('ciudad').head(4))
muestra = te[te.sede.isin(sedes.sede)].groupby('sede').head(20)
muestra = muestra.assign(texto=muestra.title + '. ' + muestra.review_text, etiqueta_real=muestra.label.map(ETQ))
muestra = muestra[['sede', 'ciudad', 'texto', 'etiqueta_real']].sample(frac=1, random_state=42).reset_index(drop=True)
muestra.index.name = 'id'
muestra.to_csv(RAIZ / 'datos' / 'muestra_controlada.csv')
print(f'{len(muestra)} reseñas de test · {muestra.sede.nunique()} sedes · {muestra.ciudad.nunique()} ciudades')
print(muestra.groupby('ciudad').sede.nunique().to_dict()); print(muestra.etiqueta_real.value_counts().to_dict())
