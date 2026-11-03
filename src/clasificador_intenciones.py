"""Clasificador académico: TF-IDF + Regresión Logística (no diagnóstico clínico)."""
from pathlib import Path
import csv
import re
import unicodedata
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATOS = Path(__file__).resolve().parent.parent / 'data' / 'mensajes_sinteticos.csv'
ETIQUETAS = {
    'saludo_conversacion': 'Saludo y conversación',
    'alegria_bienestar': 'Alegría y bienestar',
    'tristeza_soledad': 'Tristeza o soledad',
    'solicitud_orientacion': 'Solicitud de orientación',
    'despedida': 'Despedida',
}


def normalizar(texto):
    """Normalización sencilla; no borra negaciones como 'no'."""
    texto = unicodedata.normalize('NFKC', texto).lower().strip()
    return re.sub(r'\s+', ' ', texto)


@lru_cache(maxsize=1)
def cargar_modelo():
    with DATOS.open(encoding='utf-8', newline='') as f:
        registros = list(csv.DictReader(f))
    x = [normalizar(r['mensaje']) for r in registros]
    y = [r['categoria'] for r in registros]
    modelo = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), strip_accents='unicode')),
        ('logreg', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
    ])
    modelo.fit(x, y)
    return modelo


def clasificar(texto):
    modelo = cargar_modelo()
    etiquetas = modelo.named_steps['logreg'].classes_
    probabilidades = modelo.predict_proba([normalizar(texto)])[0]
    indice = int(probabilidades.argmax())
    codigo = etiquetas[indice]
    return {'codigo': codigo, 'categoria': ETIQUETAS[codigo], 'confianza_experimental': float(probabilidades[indice])}
