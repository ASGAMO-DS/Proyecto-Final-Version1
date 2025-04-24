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


# Inicializar el TfidfVectorizer con la MISMA configuración que usaste al entrenar tu modelo
tfidf = TfidfVectorizer(
    max_features=15_000,
    ngram_range=(1, 3),
    stop_words=stop_words_es,
    sublinear_tf=True
)


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
    r'Ã¡': 'á',     # á
    r'Ã©': 'é',     # é
    r'Ã3': 'ó',     # ó
    r'Ã­': 'í',     # í
    r'Ãº': 'ú',     # ú
    r'Ã¼': 'ü',     # ü (por si acaso)
}

def preprocess_text(text):
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

# Lógica para obtener y mostrar los comentarios
if post_id and TokenAcceso:
    api_version = "v22.0"
    url = f"https://graph.facebook.com/{api_version}/{post_id}/comments?access_token={TokenAcceso}"
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

            if all_comments:
                df_comentarios = pd.DataFrame({'Comentario': all_comments})

                # Aplicar reemplazo de caracteres especiales en la columna 'Comentario'
                for columna in df_comentarios.select_dtypes(include=['object']).columns:
                    df_comentarios[columna] = df_comentarios[columna].str.replace('|'.join(reemplazos.keys()),
                                                                                 lambda x: reemplazos[x.group()],
                                                                                 regex=True)

                st.subheader(f"Comentarios de la publicación con ID: {post_id}")
                st.dataframe(df_comentarios)

                # Insertando el recuento de comentarios
                st.write(f"Se han capturado **{len(all_comments)}** comentarios de la publicación.")

                status_text.empty()

                # Botón para realizar análisis de sentimiento y ver nube de palabras
                if st.button("✨ Realizar análisis de sentimiento y ver nube de palabras ✨"):
                    st.info("Procesando texto para análisis y generando nube de palabras...")

                    # Convertir la columna de comentarios a minúsculas
                    df_comentarios['Comentario'] = df_comentarios['Comentario'].str.lower()

                    # Aplicar la función de preprocesamiento a cada comentario
                    df_comentarios['Comentario_preprocesado'] = df_comentarios['Comentario'].apply(preprocess_text)

                    # Lematización con barra de progreso de Streamlit
                    num_comentarios = len(df_comentarios)
                    progress_bar = st.progress(0)
                    all_lemas_corpus = [] # Lista para almacenar todos los lemas para ajustar el vectorizador
                    lemas_por_comentario = [] # Lista para almacenar los lemas de cada comentario
                    for i, row in df_comentarios.iterrows():
                        processed_words = row['Comentario_preprocesado']
                        lemas = lematizar_texto_es(processed_words)
                        all_lemas_corpus.extend(lemas) # Extender la lista para el corpus
                        lemas_por_comentario.append(lemas) # Añadir los lemas del comentario actual
                        progress = (i + 1) / num_comentarios
                        progress_bar.progress(progress)

                    df_comentarios['Comentario_lematizado'] = lemas_por_comentario # Crear la columna

                    st.success("¡Texto preprocesado y lematizado!")

                    # Ajustar el vectorizador TF-IDF con el corpus de lemas
                    corpus_para_tfidf = [" ".join(lemas) for lemas in df_comentarios['Comentario_lematizado']]
                    tfidf.fit(corpus_para_tfidf)

                    # Generar nube de palabras
                    if all_lemas_corpus:
                        word_counts = Counter(all_lemas_corpus)
                        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_counts)
                        fig, ax = plt.subplots()
                        ax.imshow(wordcloud, interpolation='bilinear')
                        ax.axis("off")
                        st.subheader("Nube de Palabras de los Comentarios")
                        st.pyplot(fig)
                    else:
                        st.warning("No hay suficientes palabras para generar la nube de palabras.")
                    
                    # Predicción de sentimiento
                    st.subheader("Predicciones de Sentimiento de los Comentarios")
                    sentimientos = []
                    probabilidades = []
                    for lemas in df_comentarios['Comentario_lematizado']:
                        if lemas:
                            # Unir los lemas en una cadena
                            texto_para_predecir = " ".join(lemas)
                            # Vectorizar el texto usando el vectorizador TF-IDF YA AJUSTADO
                            vector_tfidf = tfidf.transform([texto_para_predecir])
                            # Realizar la predicción de probabilidad
                            proba = modelo.predict_proba(vector_tfidf)[0]
                            # Asumimos que la primera probabilidad es para la clase negativa y la segunda para positiva
                            prob_negativa = proba[0]
                            prob_positiva = proba[1]

                            # Determinar el sentimiento basado en la probabilidad
                            if prob_positiva > prob_negativa:
                                sentimiento = "positivo"
                                probabilidad = f"{prob_positiva:.2%}"
                            else:
                                sentimiento = "negativo"
                                probabilidad = f"{prob_negativa:.2%}"

                            sentimientos.append(sentimiento)
                            probabilidades.append(probabilidad)
                        else:
                            sentimientos.append("neutral")  # O podrías indicar "sin texto"
                            probabilidades.append("N/A")

                    df_comentarios['Sentimiento'] = sentimientos
                    df_comentarios['Probabilidad'] = probabilidades
                    st.subheader("Comentarios con Predicción de Sentimiento")
                    st.dataframe(df_comentarios)

            else:
                st.warning(f"No se encontraron comentarios para la publicación con ID: {post_id}")
                status_text.empty()

        else:
            st.warning(f"No se encontraron comentarios para la publicación con ID: {post_id}")
            st.empty()

    except requests.exceptions.RequestException as e:
        st.error(f"Ocurrió un error al consultar la API de Facebook: {e}")
        st.empty()
elif not TokenAcceso:
    st.error("Error: TokenAcceso no está configurada en tu archivo .env")