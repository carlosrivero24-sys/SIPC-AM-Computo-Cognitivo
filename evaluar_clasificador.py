"""Evaluación exploratoria por validación cruzada, no validación clínica."""
import csv
from pathlib import Path
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ruta = Path(__file__).parent / 'data' / 'mensajes_sinteticos.csv'
with ruta.open(encoding='utf-8', newline='') as archivo:
    filas = list(csv.DictReader(archivo))
X = [r['mensaje'] for r in filas]
y = [r['categoria'] for r in filas]
modelo = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), strip_accents='unicode')),
    ('regresion_logistica', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)),
])
pred = cross_val_predict(modelo, X, y, cv=LeaveOneOut())
print('EVALUACIÓN EXPLORATORIA SOBRE SOLO 25 FRASES SINTÉTICAS')
print('Exactitud:', round(accuracy_score(y, pred), 3))
print(classification_report(y, pred, zero_division=0))
print('Matriz de confusión (filas: real, columnas: predicho):')
print(confusion_matrix(y, pred, labels=sorted(set(y))))
print('Orden de etiquetas:', sorted(set(y)))
print('AVISO: no demuestra desempeño en personas reales ni eficacia clínica.')
