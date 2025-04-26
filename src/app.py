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

# Lógica principal para obtener y mostrar los comentarios y el análisis
if post_id and TokenAcceso:
    all_comments = obtener_comentarios(post_id, TokenAcceso)

    if all_comments:
        df_comentarios = pd.DataFrame({'Comentario': all_comments})

        # Aplicar reemplazo de caracteres especiales en la columna 'Comentario'
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
                        df_comentarios.Comentario = all_comments
                        st.rerun()
                    else:
                        st.warning("No se pudieron recargar los comentarios.")

            if analizar_sentimiento:
                with st.spinner("Procesando texto para análisis..."):
                    df_comentarios['Comentario'] = df_comentarios['Comentario'].str.lower()
                    df_comentarios['Comentario_preprocesado'] = df_comentarios['Comentario'].apply(preprocess_text)

                    num_comentarios = len(df_comentarios)
                    all_lemas_corpus = []
                    lemas_por_comentario = []
                    for i, row in df_comentarios.iterrows():
                        processed_words = row['Comentario_preprocesado']
                        lemas = lematizar_texto_es(processed_words)
                        all_lemas_corpus.extend(lemas)
                        lemas_por_comentario.append(lemas)

                    df_comentarios['Comentario_lematizado'] = lemas_por_comentario

                    st.subheader("Resultados del Análisis de Sentimiento")
                    sentimientos = []
                    probabilidades = []
                    for lemas in df_comentarios['Comentario_lematizado']:
                        if lemas:
                            texto_para_predecir = " ".join(lemas)
                            vector_tfidf = tfidf.transform([texto_para_predecir])
                            proba = modelo.predict_proba(vector_tfidf)[0]
                            prob_negativa = proba[0]
                            prob_positiva = proba[1]

                            if prob_positiva > prob_negativa:
                                sentimiento = "positivo"
                                probabilidad = f"{prob_positiva:.2%}"
                            elif prob_positiva < prob_negativa:
                                sentimiento = "negativo"
                                probabilidad = f"{prob_negativa:.2%}"
                            else:
                                sentimiento = "neutral"
                                probabilidad = "50.00%"
                        else:
                            sentimiento = "neutral"
                            probabilidad = "N/A"

                        sentimientos.append(sentimiento)
                        probabilidades.append(probabilidad)

                    df_comentarios['Sentimiento'] = sentimientos
                    df_comentarios['Probabilidad'] = probabilidades

                    st.dataframe(df_comentarios[['Comentario', 'Sentimiento', 'Probabilidad']])

                    # Gráfico de porcentaje de sentimientos

                    st.subheader("Distribución de Sentimientos")

                    conteo_sentimientos = Counter(df_comentarios['Sentimiento'])
                    total = sum(conteo_sentimientos.values())
                    porcentajes = {k: (v / total) * 100 for k, v in conteo_sentimientos.items()}

                    colores_sent = {
                            "positivo": "#03A678",
                            "negativo": "#F27405",
                            "neutral": "#A0A0A0"

                    }


                    labels = list(porcentajes.keys())
                    sizes = [porcentajes[k] for k in labels]
                    colores = [colores_sent.get(k, "#888") for k in labels]


                    fig, ax = plt.subplots(figsize=(8, 1.5))
                    fig.patch.set_facecolor('#014040')
                    left = 0
                    for i in range(len(sizes)):
                            ax.barh(0, sizes[i], left=left, color=colores[i])
                            ax.text(left + sizes[i]/2, 0, f"{labels[i]}: {sizes[i]:.1f}%", va='center', ha='center', color='white', fontsize=10)
                            left += sizes[i]

                    ax.set_xlim(0, 100)  
                    ax.axis('off')    
                    st.pyplot(fig)  



















                    # Nube de palabras
                    st.subheader("Nube de Palabras de los Comentarios")

                    from wordcloud import WordCloud
                    import random

                    custom_palette = ['#014040', '#02735E', '#03A678', '#F27405']

                    def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
                        return random.choice(custom_palette)

                    if all_lemas_corpus:
                        word_counts = Counter(all_lemas_corpus)
                        wordcloud = WordCloud(
                            width=800,
                            height=400,
                            background_color='white',
                            color_func=custom_color_func
                        ).generate_from_frequencies(word_counts)

                        fig, ax = plt.subplots()
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.pyplot(fig)
                    else:
                        st.warning("No hay suficientes palabras para generar la nube de palabras.")



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