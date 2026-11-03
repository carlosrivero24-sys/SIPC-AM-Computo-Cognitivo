# SIPC-AM — chatbot + clasificador experimental

Esta versión conserva la interfaz Streamlit y las respuestas generadas por Gemini del proyecto original. Agrega **un módulo propio del equipo** para clasificar la intención o necesidad expresada en mensajes **escritos** mediante TF-IDF y Regresión Logística.

## Qué cambió

- `src/clasificador_intenciones.py`: carga 25 mensajes sintéticos, normaliza texto, entrena `TfidfVectorizer` + `LogisticRegression` y devuelve una categoría.
- `data/mensajes_sinteticos.csv`: cinco ejemplos por categoría, preparados como **datos sintéticos**.
- `src/app.py`: muestra la categoría experimental del último mensaje escrito y la pasa como orientación a Gemini. **No reemplaza** su respuesta generativa ni el historial de Streamlit.
- `src/chatbot_logic.py`: recibe una categoría opcional y la contextualiza como etiqueta experimental, no como diagnóstico.
- `evaluar_clasificador.py`: validación cruzada dejando una muestra fuera, con exactitud, F1 y matriz de confusión. Evalúa *solo* los 25 ejemplos; resultados no generalizables.

## Instalación y ejecución

```bash
pip install -r requirements.txt
streamlit run src/app.py
```

Configure `GEMINI_API_KEY` en `.streamlit/secrets.toml` o en los secretos del despliegue (nunca suba la clave al repositorio):

```toml
GEMINI_API_KEY = "SU_CLAVE_AQUI"
```

Para evaluar el clasificador (no requiere Gemini):

```bash
python evaluar_clasificador.py
```

## Pipeline propuesto

Texto → limpieza/normalización → TF-IDF → Regresión Logística → categoría experimental → lógica de respuesta (categoría + historial) → API de Gemini → respuesta en Streamlit.

Audio → `st.audio_input` → archivo temporal → Gemini (interpreta audio y genera respuesta) → Streamlit. **No hay transcripción independiente ni clasificación del audio** en esta versión. Si se quiere clasificar por voz, habrá que añadir una transcripción explícita primero.

## Categorías

- `saludo_conversacion`: Saludo y conversación
- `alegria_bienestar`: Alegría y bienestar
- `tristeza_soledad`: Tristeza o soledad
- `solicitud_orientacion`: Solicitud de orientación
- `despedida`: Despedida

Estas categorías son intenciones/necesidades conversacionales; **no son diagnósticos** ni niveles de riesgo clínico.

## Responsabilidades

| Equipo (código propio) | API Gemini (externa) |
|---|---|
| Etiquetado de 25 ejemplos sintéticos | Generación de respuestas naturales |
| Normalización, TF-IDF, Regresión Logística | Interpretación directa de audios |
| Predicción y visualización de categoría | Procesamiento del contexto enviado |
| Evaluación exploratoria | Dependencia de una clave, cuota y modelo disponible |

## Limitaciones importantes

- 25 ejemplos son insuficientes para un clasificador robusto; puede fallar ante frases nuevas y ambigüedades. Amplíen el conjunto con más ejemplos, sin datos personales sensibles, y evalúen sobre mensajes no usados para entrenar.
- La etiqueta solo orienta a Gemini; debe ser capaz de ignorarla cuando contradiga al usuario.
- El identificador del modelo Gemini es el del código original (`gemini-3.8-flash`). Si la API devuelve error de modelo no disponible, el equipo debe verificar los IDs disponibles para su propia clave y modificarlo. No se ha probado acceso real a la API.
- El proyecto original **no implementa** análisis acústico MFCC/Jitter/Shimmer, diagnóstico clínico ni consentimiento explícito para grabaciones. Se requiere revisión de privacidad y seguridad antes de pruebas con personas reales.
- Esta es una copia del código, no actualiza automáticamente la app publicada ni el repositorio de GitHub.
