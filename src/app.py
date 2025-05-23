import os
import facebook
import requests
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import regex as re
import spacy
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import tqdm
from tqdm import tqdm as tqdm_streamlit
import pickle
from pickle import load
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk
import seaborn as sns
import numpy as np
import plotly.express as px  # Importar plotly express


import streamlit as st

st.markdown("<h1 style='text-align: center; color: #00CCBF;'>🔍InsightPulse </h1>", unsafe_allow_html=True)

# Introducción con fondo translúcido y texto en color #04BF9D
st.markdown("""
<div style='text-align: center; background-color: rgba(4, 191, 157, 0.1); padding: 20px 30px; border-radius: 15px; font-size: 18px; color: #04BF9D; font-weight: 600; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);'>
    Descubre cómo percibe tu audiencia tus publicaciones. <br>
</div>
""", unsafe_allow_html=True)

# Espacio entre la intro y el campo de ID
st.markdown("<br><br>", unsafe_allow_html=True)





# Cargar las variables de entorno AL INICIO del script
load_dotenv()

IdentificadorApp = os.getenv('IdentificadorApp')
ClaveSecretaApp = os.getenv('ClaveSecretaApp')
TokenAcceso = os.getenv('TokenAcceso')

# Descargar stopwords (solo se ejecuta una vez)
try:
    stopwords.words('spanish')
except LookupError:
    nltk.download('stopwords')


# Convertir stopwords a LISTA
stop_words_es = list(stopwords.words('spanish'))


# Inicializar el TfidfVectorizer (AHORA INTENTAMOS CARGAR EL GUARDADO)
try:
    with open('models/tfidf_vectorizer.pkl', 'rb') as file:
        tfidf = pickle.load(file)
except FileNotFoundError:
    st.error("Error: No se encontró el archivo del vectorizador TF-IDF ('models/tfidf_vectorizer.pkl'). Asegúrate de haberlo guardado durante el entrenamiento del modelo y que la ruta sea correcta.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el vectorizador TF-IDF: {e}")
    st.stop()


# Cargar el modelo de spaCy en español
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    st.error("Error: No se pudo cargar el modelo de spaCy en español. Asegúrate de haberlo descargado con: python -m spacy download es_core_news_sm")
    st.stop()

# Cargar el modelo de análisis de sentimiento
try:
    modelo = load(open("models/svm_classifier_linear_probabilityTrue_42.sav", "rb"))
except FileNotFoundError:
    st.error("Error: No se encontró el archivo del modelo de sentimiento en 'models/svm_classifier_linear_probabilityTrue_42.sav'. Asegúrate de que la ruta sea correcta.")
    st.stop()
except Exception as e:
    st.error(f"Error al cargar el modelo de sentimiento: {e}")
    st.stop()

# Widget de Streamlit para obtener el ID de la publicación
post_id = st.text_input("Por favor ingresa el ID (122098322012852515) de la publicación de Facebook:")

# Diccionario de reemplazos (caracter mal codificado → caracter correcto)
reemplazos = {
    r'Ã±': 'ñ',     # ñ
    r'Ã¡': 'a',     # á
    r'Ã©': 'e',     # é
    r'Ã3': 'o',     # ó
    r'Ã­': 'i',     # í
    r'Ãº': 'u',     # ú
    r'Ã¼': 'ü',     # ü (por si acaso)
    r'á': 'a',     # á
    r'é': 'e',     # é
    r'ó': 'o',     # ó
    r'í­': 'i',     # í
    r'ú': 'u',     # ú
}

def preprocess_text(text):

    for caracter_mal, caracter_bien in reemplazos.items():
        text = text.replace(caracter_mal, caracter_bien)


    # Permitir letras (incluyendo acentuadas y ñ), espacios y algunos símbolos básicos
    text = re.sub(r'[^\wáéíóúüñÁÉÍÓÚÜÑ ]', " ", text, flags=re.UNICODE)

    # Eliminar palabras de una sola letra rodeadas por espacios
    text = re.sub(r'\s+[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]\s+', " ", text)
    text = re.sub(r'^\s*[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]\s+', " ", text)
    text = re.sub(r'\s+[a-zA-ZáéíóúüñÁÉÍÓÚÜÑ]\s*$', " ", text)

    # Reducir espacios múltiples y convertir a minúsculas
    text = re.sub(r'\s+', " ", text.lower()).strip()

    # Eliminar tags (opcional, si aún es necesario)
    text = re.sub(r'&lt;/?.*?&gt;', " ", text)

    return text.split()

def lematizar_texto_es(words):
    if isinstance(words, list):  # Asegurar que 'words' sea una lista
        doc = nlp(" ".join(words))
        tokens = [token.lemma_ for token in doc if not token.is_stop and len(token.text) > 3]
        return tokens
    return []  # Si no es una lista, devolver lista vacía




# Función para obtener los comentarios de Facebook
def obtener_comentarios(post_id, access_token):
    if not post_id or not access_token:
        st.warning("Por favor, ingresa el ID de la publicación y asegúrate de que el Token de Acceso esté configurado.")
        return []

    api_version = "v22.0"
    url = f"https://graph.facebook.com/{api_version}/{post_id}/comments?access_token={access_token}"
    all_comments = []
    status_text = st.empty()
    status_text.text("Obteniendo comentarios...")

    try:
        response = requests.get(url)
        response.raise_for_status()
        comments_data = response.json()

        if 'data' in comments_data:
            for i, comment in enumerate(comments_data['data']):
                message = comment.get('message')
                if message:
                    all_comments.append(message)
                status_text.text(f"Obteniendo comentarios... {len(all_comments)} comentarios obtenidos.")

            next_url = comments_data.get('paging', {}).get('next')
            while next_url:
                next_response = requests.get(next_url)
                next_response.raise_for_status()
                next_comments_data = next_response.json()
                if 'data' in next_comments_data:
                    for comment in next_comments_data['data']:
                        message = comment.get('message')
                        if message:
                            all_comments.append(message)
                    status_text.text(f"Obteniendo comentarios... {len(all_comments)} comentarios obtenidos.")
                    next_url = next_comments_data.get('paging', {}).get('next')
                else:
                    next_url = None

        status_text.empty()
        return all_comments

    except requests.exceptions.RequestException as e:
        st.error(f"Ocurrió un error al consultar la API de Facebook: {e}")
        st.empty()
        return []





# Definir la paleta de colores para los gráficos
colores_paleta = ['#014040', '#02735E', '#03A678', '#F27405', '#731702']


#Lógica principal para obtener y mostrar los comentarios y el análisis
if post_id and TokenAcceso:
    all_comments = obtener_comentarios(post_id, TokenAcceso)

    if all_comments:
        df_comentarios = pd.DataFrame({'Comentario': all_comments})

        # Aplicar reemplazo de caracteres especiales
        for caracter_mal, caracter_bien in reemplazos.items():
            df_comentarios['Comentario'] = df_comentarios['Comentario'].str.replace(caracter_mal, caracter_bien, regex=False)

        tab1, tab2, tab3 = st.tabs(["Comentarios", "Análisis de Sentimiento", "Visualización de Métricas"])

        # Pestaña 1: Comentarios
        with tab1:
            st.subheader(f"Comentarios de la publicación con ID: {post_id}")
            st.dataframe(df_comentarios)
            st.write(f"Se han capturado **{len(all_comments)}** comentarios de la publicación.")

        # Pestaña 2: Análisis de Sentimiento
        with tab2:
            st.subheader("Análisis de Sentimiento y Nube de Palabras")

            # Inicializar variables de sesión
            if 'mostrar_filtros' not in st.session_state:
                st.session_state.mostrar_filtros = False
            if 'analizado' not in st.session_state:
                st.session_state.analizado = False

            col1, col2 = st.columns([2, 1])

            with col1:
                if st.button("✨ Realizar análisis de sentimiento y ver nube de palabras ✨"):
                    st.session_state.analizado = True
            with col2:
                if st.button("🔄 Recargar comentarios"):
                    with st.spinner("Recargando comentarios..."):
                        all_comments = obtener_comentarios(post_id, TokenAcceso)
                        if all_comments:
                            df_comentarios['Comentario'] = all_comments
                            st.session_state.analizado = False
                            st.experimental_rerun()

            if st.session_state.analizado:
                with st.spinner("Procesando texto para análisis..."):
                    # Preprocesamiento
                    df_comentarios['Comentario'] = df_comentarios['Comentario'].str.lower()
                    df_comentarios['Comentario_preprocesado'] = df_comentarios['Comentario'].apply(preprocess_text)

                    all_lemas_corpus = []
                    lemas_por_comentario = []
                    for comentario in df_comentarios['Comentario_preprocesado']:
                        lemas = lematizar_texto_es(comentario)
                        all_lemas_corpus.extend(lemas)
                        lemas_por_comentario.append(lemas)

                    df_comentarios['Comentario_lematizado'] = lemas_por_comentario

                    # Análisis de sentimiento
                    sentimientos = []
                    probabilidades = []
                    for lemas in df_comentarios['Comentario_lematizado']:
                        if lemas:
                            texto = " ".join(lemas)
                            vector_tfidf = tfidf.transform([texto])
                            proba = modelo.predict_proba(vector_tfidf)[0]
                            prob_negativa, prob_positiva = proba

                            if prob_positiva > prob_negativa:
                                sentimientos.append("positivo")
                                probabilidades.append(f"{prob_positiva:.2%}")
                            else:
                                sentimientos.append("negativo")
                                probabilidades.append(f"{prob_negativa:.2%}")
                        else:
                            sentimientos.append("neutral")
                            probabilidades.append("N/A")

                    df_comentarios['Sentimiento'] = sentimientos
                    df_comentarios['Probabilidad'] = probabilidades

                    # Mostrar resultados
                    st.subheader("Resultados del Análisis de Sentimiento")
                    st.dataframe(df_comentarios[['Comentario', 'Sentimiento', 'Probabilidad']])

                    # Nube de palabras
                    st.subheader("Nube de Palabras de los Comentarios")
                    if all_lemas_corpus:
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(Counter(all_lemas_corpus))
                        fig, ax = plt.subplots()
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.warning("No hay suficientes palabras para generar la nube de palabras.")

                    # Gráfico de barra horizontal de sentimientos
                    st.subheader("📊 Distribución de Sentimientos")
                    conteo_sentimientos = df_comentarios['Sentimiento'].value_counts()
                    fig2, ax2 = plt.subplots(figsize=(8, 2))

                    sentimientos_barra = ['positivo', 'neutral', 'negativo']
                    cantidad = [conteo_sentimientos.get(s, 0) for s in sentimientos_barra]

                    ax2.barh(['Comentarios'], [cantidad[0]], color=colores_paleta[1], label='Positivo')
                    ax2.barh(['Comentarios'], [cantidad[1]], color=colores_paleta[2], left=cantidad[0], label='Neutral')
                    ax2.barh(['Comentarios'], [cantidad[2]], color=colores_paleta[4], left=cantidad[0] + cantidad[1], label='Negativo')

                    ax2.set_xlim(0, sum(cantidad))
                    ax2.set_xlabel('Cantidad de Comentarios')
                    ax2.set_title('Distribución de Sentimientos')
                    ax2.legend()
                    st.pyplot(fig2)

                    # Opciones de Filtro
                    st.subheader("🎯 Aplicar Filtros a Comentarios Analizados")

                    colf1, colf2 = st.columns(2)
                    with colf1:
                        if st.button("🎛️ Mostrar/Ocultar Filtros"):
                            st.session_state.mostrar_filtros = not st.session_state.mostrar_filtros
                    with colf2:
                        if st.button("🧹 Borrar Filtros"):
                            st.session_state.mostrar_filtros = False
                            st.session_state.aplicar_sentimiento = False
                            st.session_state.aplicar_palabra = False
                            st.session_state.aplicar_probabilidad = False
                            st.session_state.sentimiento_seleccionado = None
                            st.session_state.palabra_clave = ""
                            st.session_state.nivel_probabilidad = 50
                            st.success("Filtros borrados correctamente.")
                            st.experimental_rerun()

                    if st.session_state.mostrar_filtros:
                        st.markdown("### Opciones de Filtro")

                        aplicar_sentimiento = st.checkbox("Filtrar por tipo de sentimiento", key="aplicar_sentimiento")
                        aplicar_palabra = st.checkbox("Filtrar por palabra clave", key="aplicar_palabra")
                        aplicar_probabilidad = st.checkbox("Filtrar por nivel de probabilidad (%)", key="aplicar_probabilidad")

                        if aplicar_sentimiento:
                            st.session_state.sentimiento_seleccionado = st.selectbox("Selecciona el sentimiento:", ("positivo", "negativo", "neutral"))

                        if aplicar_palabra:
                            st.session_state.palabra_clave = st.text_input("Escribe la palabra clave:")

                        if aplicar_probabilidad:
                            st.session_state.nivel_probabilidad = st.slider("Selecciona el porcentaje mínimo de certeza:", min_value=0, max_value=100, value=50, step=5)

                        # Aplicar filtros
                        df_filtrado = df_comentarios.copy()

                        if aplicar_sentimiento and st.session_state.sentimiento_seleccionado:
                            df_filtrado = df_filtrado[df_filtrado['Sentimiento'] == st.session_state.sentimiento_seleccionado]

                        if aplicar_palabra and st.session_state.palabra_clave:
                            df_filtrado = df_filtrado[df_filtrado['Comentario'].str.contains(st.session_state.palabra_clave, case=False, na=False)]

                        if aplicar_probabilidad:
                            df_filtrado = df_filtrado[
                                df_filtrado['Probabilidad'].apply(
                                    lambda x: (x != "N/A") and float(x.replace('%', '')) >= st.session_state.nivel_probabilidad
                               )
                           ]

                        st.success(f"🔎 Se encontraron {len(df_filtrado)} comentarios filtrados.")
                        st.dataframe(df_filtrado[['Comentario', 'Sentimiento', 'Probabilidad']])

        # Pestaña 3: Visualización de Métricas
        with tab3:
            st.subheader("📊 Visualización de Métricas")

            

            # 📝 Gráfico de Palabras Clave por Sentimiento  
            with st.expander("📝 Palabras Más Comunes por Sentimiento"): 
                st.write("📝 Este análisis muestra las palabras más comunes en los comentarios agrupados por tipo de sentimiento (positivo, neutral o negativo).")
                from collections import Counter
                for sentimiento in ['positivo', 'neutral', 'negativo']:
                    comentarios_sentimiento = df_comentarios[df_comentarios['Sentimiento'] == sentimiento]['Comentario']
                    palabras = " ".join(comentarios_sentimiento).lower().split()
                    palabras_filtradas = [p for p in palabras if len(p) > 3]
                    conteo = Counter(palabras_filtradas).most_common(10)

                    if conteo:
                        palabras, frecuencias = zip(*conteo)
                        fig4, ax4 = plt.subplots()
                        ax4.barh(palabras, frecuencias, color=colores_paleta[1] if sentimiento == 'positivo' else (colores_paleta[2] if sentimiento == 'neutral' else colores_paleta[4]))
                        ax4.set_title(f"Top Palabras en comentarios {sentimiento}")
                        ax4.invert_yaxis()
                        st.pyplot(fig4)
                    else:
                        st.info(f"No hay suficientes palabras para mostrar en '{sentimiento}'.")

            # 🚀 Velocímetro de Comentarios Positivos    
            with st.expander("🚀 Velocímetro de Sentimiento Positivo"):
                st.write("🚀 El velocímetro indica el porcentaje de comentarios que fueron clasificados como positivos respecto al total de comentarios analizados.")
                import plotly.graph_objects as go
                total = len(df_comentarios)
                positivos = len(df_comentarios[df_comentarios['Sentimiento'] == 'positivo'])
                porcentaje_positivo = (positivos / total) * 100 if total > 0 else 0

                fig5 = go.Figure(go.Indicator(
                    mode="gauge+number+delta",
                    value=porcentaje_positivo,
                    title={'text': "Comentarios Positivos (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': colores_paleta[1]},
                        'steps': [
                            {'range': [0, 50], 'color': colores_paleta[3]},
                            {'range': [50, 100], 'color': colores_paleta[2]}
                        ],
                    }
                ))

                fig5.update_layout(
                    annotations=[{
                        'x': 0.5,
                        'y': -0.25,
                        'text': f'{round(porcentaje_positivo, 2)}%',
                        'showarrow': False,
                        'font': {'size': 20}
                    }]
                )

                st.plotly_chart(fig5)
            