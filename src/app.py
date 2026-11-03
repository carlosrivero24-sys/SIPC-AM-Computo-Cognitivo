import streamlit as st
from chatbot_logic import generar_respuesta_chatbot, limpiar_texto_natural
from clasificador_intenciones import clasificar

st.set_page_config(
    page_title="Asistente Personal de Apoyo",
    page_icon="💙",
    layout="centered"
)

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .main-title { color: #2563EB; font-weight: bold; }
    .chat-text { font-size: 1.15rem !important; line-height: 1.6 !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>💙 Mi Asistente de Compañía</h1>", unsafe_allow_html=True)
st.write("Un espacio tranquilo para conversar mediante voz o texto.")


st.components.v1.html("""
    <div style="margin-bottom: 10px;">
        <button onclick="parent.window.speechSynthesis.cancel(); window.speechSynthesis.cancel();" style="background: #dc2626; color: white; border: none; padding: 8px 14px; border-radius: 4px; cursor: pointer; font-weight: bold; font-size: 14px;">
            Silenciar voz 🔇
        </button>
    </div>
""", height=45)

if "historial_chat" not in st.session_state:
    st.session_state.historial_chat = [
        {"role": "assistant", "content": "Hola. ¿En qué puedo ayudarte hoy?"}
    ]


if "ultima_categoria" not in st.session_state:
    st.session_state.ultima_categoria = None

if "ultimo_audio_procesado" not in st.session_state:
    st.session_state.ultimo_audio_procesado = None

def reproducir_voz(texto):
    texto_seguro = limpiar_texto_natural(texto).replace('"', "'").replace('\n', ' ')
    script_js = f"""
    <script>
        window.speechSynthesis.cancel();
        const utterance = new SpeechSynthesisUtterance("{texto_seguro}");
        utterance.lang = "es-MX";
        utterance.rate = 0.95; 
        window.speechSynthesis.speak(utterance);
    </script>
    """
    st.components.v1.html(script_js, height=0)


for msg in st.session_state.historial_chat:
    with st.chat_message(msg["role"]):
        texto_mostrable = limpiar_texto_natural(msg["content"])
        st.markdown(f'<p class="chat-text">{texto_mostrable}</p>', unsafe_allow_html=True)


if st.session_state.historial_chat[-1]["role"] == "assistant":
    reproducir_voz(st.session_state.historial_chat[-1]["content"])


st.markdown("### 🎙️ Dictado por Voz")
audio_grabado = st.audio_input("Presiona el botón para grabar tu mensaje de voz:")

if audio_grabado:
    audio_bytes = audio_grabado.getvalue()
    
    if audio_bytes != st.session_state.ultimo_audio_procesado:
        st.session_state.ultimo_audio_procesado = audio_bytes
        st.session_state.ultima_categoria = None  # Sin transcripción, no se clasifica el audio.
        
        st.session_state.historial_chat.append({"role": "user", "content": "[Mensaje de voz enviado]"})
        with st.chat_message("user"):
            st.markdown('<p class="chat-text">🎙️ [Mensaje de voz enviado]</p>', unsafe_allow_html=True)

        with st.spinner("Escuchando y procesando audio..."):
            respuesta_bot = generar_respuesta_chatbot(st.session_state.historial_chat, audio_bytes=audio_bytes)
        
        st.session_state.historial_chat.append({"role": "assistant", "content": respuesta_bot})
        st.rerun()


if st.session_state.ultima_categoria:
    st.info(f"Categoría experimental del último mensaje escrito: {st.session_state.ultima_categoria}")
    st.caption("Clasificador académico con 25 mensajes sintéticos. No determina estados de salud ni constituye un diagnóstico.")

if entrada_usuario := st.chat_input("O escribe tu mensaje aquí..."):
    st.session_state.historial_chat.append({"role": "user", "content": entrada_usuario})
    with st.chat_message("user"):
        st.markdown(f'<p class="chat-text">{entrada_usuario}</p>', unsafe_allow_html=True)

    # La clasificación es un módulo propio y separado de la generación con Gemini.
    try:
        prediccion = clasificar(entrada_usuario)
        st.session_state.ultima_categoria = prediccion["categoria"]
    except Exception as exc:
        st.session_state.ultima_categoria = "No disponible"
        st.warning(f"No se pudo ejecutar el clasificador: {exc}")

    with st.spinner("Procesando respuesta..."):
        respuesta_bot = generar_respuesta_chatbot(
            st.session_state.historial_chat,
            categoria=st.session_state.ultima_categoria if st.session_state.ultima_categoria != "No disponible" else None
        )
    
    st.session_state.historial_chat.append({"role": "assistant", "content": respuesta_bot})
    st.rerun()