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
import plotly.express as px  # Importar plotly express


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



import streamlit as st
import pandas as pd
import numpy as np
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Suponiendo que ya tienes estas funciones
# obtener_comentarios(post_id, TokenAcceso)
# y tu diccionario reemplazos

from textblob import TextBlob

def analizar_comentarios(comentarios):
    resultados = []
    for comentario in comentarios:
        blob = TextBlob(comentario)
        polaridad = blob.sentiment.polarity

        if polaridad > 0.1:
            sentimiento = 'positivo'
        elif polaridad < -0.1:
            sentimiento = 'negativo'
        else:
            sentimiento = 'neutral'

        probabilidad = f"{int(abs(polaridad) * 100)}%"

        resultados.append({
            'sentimiento': sentimiento,
            'probabilidad': probabilidad
        })
    
    return resultados


# Lógica principal para obtener y mostrar los comentarios y el análisis
if post_id and TokenAcceso:
    all_comments = obtener_comentarios(post_id, TokenAcceso)

    if all_comments:
        df_comentarios = pd.DataFrame({'Comentario': all_comments})

        # Aplicar reemplazo de caracteres especiales
        for caracter_mal, caracter_bien in reemplazos.items():
            df_comentarios['Comentario'] = df_comentarios['Comentario'].str.replace(caracter_mal, caracter_bien, regex=False)

        tab1, tab2, tab3 = st.tabs(["Comentarios", "Análisis de Sentimiento", "Visualización de Métricas"])

        with tab1:
            st.subheader(f"Comentarios de la publicación con ID: {post_id}")
            st.dataframe(df_comentarios)
            st.write(f"Se han capturado **{len(all_comments)}** comentarios de la publicación.")

        with tab2:
            st.subheader("Análisis de Sentimiento y Nube de Palabras")
            col1, col2 = st.columns([2, 1])

            with col1:
                analizar_sentimiento = st.button("✨ Realizar análisis de sentimiento y ver nube de palabras ✨")
            with col2:
                recargar_comentarios = st.button("🔄 Recargar comentarios")

            if recargar_comentarios:
                with st.spinner("Recargando comentarios..."):
                    all_comments = obtener_comentarios(post_id, TokenAcceso)
                    if all_comments:
                        df_comentarios['Comentario'] = all_comments
                        if 'analizado' in st.session_state:
                            del st.session_state['analizado']  # 🔥 BORRAR análisis viejo
                        st.rerun()

            if analizar_sentimiento or st.session_state.get('analizado', False):
                st.session_state['analizado'] = True  # Guardar que ya analizamos

                with st.spinner("Procesando texto para análisis..."):
                    # --- Aquí realiza el análisis de sentimientos ---
                    resultados = analizar_comentarios(df_comentarios['Comentario'].tolist())
                    df_comentarios['Sentimiento'] = [r['sentimiento'] for r in resultados]
                    df_comentarios['Probabilidad'] = [r['probabilidad'] for r in resultados]

                    # --- Nube de Palabras ---
                    st.subheader("☁️ Nube de Palabras de Comentarios")
                    texto_completo = " ".join(df_comentarios['Comentario'])
                    nube = WordCloud(width=800, height=400, background_color='white').generate(texto_completo)

                    fig1, ax1 = plt.subplots()
                    ax1.imshow(nube, interpolation='bilinear')
                    ax1.axis('off')
                    st.pyplot(fig1)

                    # --- Barra de Sentimientos apilada ---
                    st.subheader("📊 Distribución de Sentimientos")
                    conteo_sentimientos = df_comentarios['Sentimiento'].value_counts()

                    fig2, ax2 = plt.subplots(figsize=(8, 2))
                    sentimientos = ['positivo', 'neutral', 'negativo']
                    cantidad = [conteo_sentimientos.get(sent, 0) for sent in sentimientos]

                    ax2.barh(['Comentarios'], [cantidad[0]], color='green', label='Positivo')
                    ax2.barh(['Comentarios'], [cantidad[1]], color='gray', left=cantidad[0], label='Neutral')
                    ax2.barh(['Comentarios'], [cantidad[2]], color='red', left=cantidad[0]+cantidad[1], label='Negativo')

                    ax2.set_xlim(0, sum(cantidad))
                    ax2.set_xlabel('Cantidad de Comentarios')
                    ax2.set_title('Distribución de Sentimientos')
                    ax2.legend()
                    st.pyplot(fig2)

                    # --- Botón para mostrar filtros ---
                    st.subheader("🎯 Aplicar Filtros a Comentarios Analizados")

                    if 'mostrar_filtros' not in st.session_state:
                        st.session_state.mostrar_filtros = False

                    if st.button("🧹 Borrar Filtros"):
                        # Resetear todos los filtros
                        st.session_state.mostrar_filtros = False
                        st.session_state.aplicar_sentimiento = False
                        st.session_state.aplicar_palabra = False
                        st.session_state.aplicar_probabilidad = False
                        st.session_state.sentimiento_seleccionado = None
                        st.session_state.palabra_clave = ""
                        st.session_state.nivel_probabilidad = 50
                        st.success("Filtros borrados.")
                        st.rerun()

                    if st.button("🧹 Filtros"):
                        st.session_state.mostrar_filtros = not st.session_state.mostrar_filtros

                    if st.session_state.mostrar_filtros:
                        st.markdown("### Opciones de Filtro")

                        aplicar_sentimiento = st.checkbox("Filtrar por tipo de sentimiento")
                        if aplicar_sentimiento:
                            sentimiento_seleccionado = st.selectbox(
                                "Selecciona el sentimiento:",
                                ("positivo", "negativo", "neutral")
                            )

                        aplicar_palabra = st.checkbox("Filtrar por palabra clave")
                        if aplicar_palabra:
                            palabra_clave = st.text_input("Escribe la palabra clave:")

                        aplicar_probabilidad = st.checkbox("Filtrar por nivel de probabilidad (%)")
                        if aplicar_probabilidad:
                            nivel_probabilidad = st.slider(
                                "Selecciona el porcentaje mínimo de certeza:",
                                min_value=0,
                                max_value=100,
                                value=50,
                                step=5
                            )

                        # Aplicar filtros
                        if aplicar_sentimiento or aplicar_palabra or aplicar_probabilidad:
                            df_filtrado = df_comentarios.copy()

                            if aplicar_sentimiento:
                                df_filtrado = df_filtrado[df_filtrado['Sentimiento'] == sentimiento_seleccionado]

                            if aplicar_palabra and palabra_clave:
                                df_filtrado = df_filtrado[df_filtrado['Comentario'].str.contains(palabra_clave, case=False, na=False)]

                            if aplicar_probabilidad:
                                # Ajuste para el filtro de probabilidad en un rango de 10%
                                df_filtrado = df_filtrado[ 
                                    df_filtrado['Probabilidad'].apply(
                                        lambda x: float(x.replace('%', '')) >= nivel_probabilidad and 
                                                  float(x.replace('%', '')) < (nivel_probabilidad + 10)
                                    )
                                ]

                            st.success(f"🔎 Se encontraron {len(df_filtrado)} comentarios filtrados.")
                            st.dataframe(df_filtrado[['Comentario', 'Sentimiento', 'Probabilidad']])
                        else:
                            st.info("Selecciona al menos un filtro para aplicar.")







             

                                 
     
                        













        with tab3:
            st.subheader("Visualización de Métricas")
            if 'Sentimiento' in df_comentarios:
                # Gráfico de pastel de sentimientos (sin cambios)
                sentiment_counts = df_comentarios['Sentimiento'].value_counts().reset_index()
                sentiment_counts.columns = ['Sentimiento', 'Cantidad']

                fig_pie = px.pie(sentiment_counts,
                                 values='Cantidad',
                                 names='Sentimiento',
                                 title='Distribución de Sentimientos',
                                 color='Sentimiento',
                                 color_discrete_map={'positivo': 'green',
                                                     'negativo': 'red',
                                                     'neutral': 'blue'},
                                 hover_data=['Cantidad'])

                fig_pie.update_traces(textinfo='percent+label', texttemplate='%{percent:.1%} (%{value})')
                st.plotly_chart(fig_pie)

                # Gráfico de barras de palabras más frecuentes (MODIFICADO)
                st.subheader("Palabras Más Frecuentes")
                if all_lemas_corpus:
                    word_counts = Counter(all_lemas_corpus)
                    most_common_words = pd.DataFrame(word_counts.most_common(20), columns=['Palabra', 'Frecuencia'])

                    fig_bar = px.bar(most_common_words,
                                     x='Frecuencia',
                                     y='Palabra',
                                     orientation='h',
                                     title='Top 20 Palabras Más Frecuentes',
                                     labels={'Frecuencia': 'Frecuencia', 'Palabra': 'Palabra'},
                                     height=600)  # Aumentar la altura del gráfico

                    fig_bar.update_traces(texttemplate='%{x}', textposition='outside') # Mostrar la frecuencia al final de la barra

                    st.plotly_chart(fig_bar)
                else:
                    st.info("No hay suficientes palabras lematizadas para mostrar las más frecuentes.")

            else:
                st.info("El análisis de sentimiento aún no se ha realizado.")
                
    else:
        st.warning(f"No se encontraron comentarios para la publicación con ID: {post_id}")

elif not TokenAcceso:
    st.error("Error: TokenAcceso no está configurada en tu archivo .env")