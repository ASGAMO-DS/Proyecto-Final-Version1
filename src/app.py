import os
import facebook
import requests
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
import regex as re
import spacy
import tqdm
from tqdm import tqdm as tqdm_streamlit


# Cargar las variables de entorno AL INICIO del script
load_dotenv()

IdentificadorApp = os.getenv('IdentificadorApp')
ClaveSecretaApp = os.getenv('ClaveSecretaApp')
TokenAcceso = os.getenv('TokenAcceso')

# Cargar el modelo de spaCy en español
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    st.error("Error: No se pudo cargar el modelo de spaCy en español. Asegúrate de haberlo descargado con: python -m spacy download es_core_news_sm")
    st.stop()

# Widget de Streamlit para obtener el ID de la publicación
post_id = st.text_input("Por favor ingresa el ID de la publicación de Facebook:")

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

                # Botón para realizar análisis de sentimiento y procesamiento del texto
                if st.button("✨ Realizar análisis de sentimiento ✨"):
                    st.info("Procesando texto para análisis de sentimiento...")

                    # Convertir la columna de comentarios a minúsculas
                    df_comentarios['Comentario'] = df_comentarios['Comentario'].str.lower()

                    # Aplicar la función de preprocesamiento a cada comentario
                    df_comentarios['Comentario_preprocesado'] = df_comentarios['Comentario'].apply(preprocess_text)

                    # Lematización con barra de progreso de Streamlit
                    num_comentarios = len(df_comentarios)
                    progress_bar = st.progress(0)
                    lematized_comments = []
                    for i, row in df_comentarios.iterrows():
                        processed_words = row['Comentario_preprocesado']
                        lemas = lematizar_texto_es(processed_words)
                        lematized_comments.append(lemas)
                        progress = (i + 1) / num_comentarios
                        progress_bar.progress(progress)

                    df_comentarios['Comentario_lematizado'] = lematized_comments
                    st.success("¡Texto de los comentarios preprocesado y lematizado!")
                    # Por ahora, no mostramos el resultado de la lematización
            else:
                st.warning(f"No se encontraron comentarios para la publicación con ID: {post_id}")
                status_text.empty()

        else:
            st.warning(f"No se encontraron comentarios para la publicación con ID: {post_id}")
            status_text.empty()

    except requests.exceptions.RequestException as e:
        st.error(f"Ocurrió un error al consultar la API de Facebook: {e}")
        status_text.empty()
elif not TokenAcceso:
    st.error("Error: TokenAcceso no está configurada en tu archivo .env")