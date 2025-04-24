from utils import db_connect
engine = db_connect()


import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter

#Titulo
st.markdown(
    """
    <style>
    .titulo-app {
        background: linear-gradient(to right, #D6ECFA, #FFFFFF); /* Degradado azul claro a blanco */
        padding: 25px;
        border-radius: 12px;
        text-align: center;
        color: #003A63; /* Azul profundo */
        font-size: 32px;
        font-weight: 600;
        font-family: 'Trebuchet MS', sans-serif;
        margin-bottom: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra sutil */
    }
    .subtitulo-app {
        text-align: center;
        color: #4C738B; /* Azul suave */
        font-size: 20px;
        font-family: 'Georgia', serif;
        margin-bottom: 25px;
    }
    </style>
    <div class="titulo-app">InsightPulse</div>
    <div class="subtitulo-app">Explora las emociones y tendencias escondidas en redes sociales.</div>
    """,
    unsafe_allow_html=True
)


#Para que sirve la app
st.markdown(
    """
    <style>
    .intro-message {
        text-align: center;
        background-color: #E6F7FF; /* Azul claro pastel */
        color: #005B96; /* Azul profundo */
        padding: 20px;
        border-radius: 10px;
        font-size: 18px;
        font-family: 'Verdana', sans-serif;
        margin-bottom: 20px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1); /* Sombra estética */
    }
    </style>
    <div class="intro-message">
        💬 <b>InsightPulse</b> te ayuda a analizar los sentimientos y descubrir las tendencias en publicaciones de redes sociales.<br>
        Ingresa un link, explora las emociones y visualiza gráficos interactivos.
    </div>
    """,
    unsafe_allow_html=True
)





#
st.markdown(
    """
    <style>
    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-10px); }
    }
    .emoji-bounce {
        display: inline-block;
        animation: bounce 1.5s infinite;
    }
    </style>
    <p style="text-align:center;">
        Ingresa el link de la publicación de Facebook <span class="emoji-bounce">🔗</span>
    </p>
    """
    ,
    unsafe_allow_html=True
)

link_publicacion = st.text_input("Link de la publicación:")
if st.button("Analizar"):
    if link_publicacion:
        st.write(f"Procesando el link: {link_publicacion}")
    else:
        st.warning("Por favor ingresa un link válido.")











# Simulación de extracción de palabras clave
comentarios = ["Me encantó la película", "Es muy aburrida", "Buena trama", "Muy mala producción", "Espectacular actuación"]
texto_completo = " ".join(comentarios)
palabras = texto_completo.split()
contador_palabras = Counter(palabras).most_common(10)

# Gráfica
palabras, frecuencias = zip(*contador_palabras)
fig, ax = plt.subplots()
ax.bar(palabras, frecuencias, color='blue')
ax.set_title("Palabras Más Repetidas")
ax.set_ylabel("Frecuencia")
ax.set_xlabel("Palabras")
st.pyplot(fig)








#Velocimetro de satisfacción
import plotly.graph_objects as go

# Simulación de datos de satisfacción
indice_satisfaccion = 75  # Cambia este valor dinámicamente según los resultados del análisis

# Crear el velocímetro con la paleta personalizada
fig = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=indice_satisfaccion,
        title={"text": "Índice de Satisfacción"},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 2, "tickcolor": "#014040"},  # Color de los ticks
            "bar": {"color": "#03A678"},  # Color del puntero
            "steps": [
                {"range": [0, 25], "color": "#F27405"},   # Negativo
                {"range": [25, 75], "color": "#02735E"},  # Neutral
                {"range": [75, 100], "color": "#014040"}, # Positivo
            ],
        },
    )
)

# Mostrar el velocímetro en Streamlit
st.subheader("Medidor de Satisfacción Global")
st.plotly_chart(fig)

# Mensaje destacado
if indice_satisfaccion > 75:
    st.success("🎉 ¡Los comentarios reflejan una tendencia positiva!")
elif indice_satisfaccion < 25:
    st.error("⚠️ Los comentarios reflejan una tendencia negativa.")
else:
    st.warning("🤔 La tendencia parece ser neutral.")


