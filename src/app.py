from utils import db_connect
engine = db_connect()



import os
import facebook
from dotenv import load_dotenv

# load the .env file variables
load_dotenv()

IdentificadorApp = os.getenv('IdentificadorApp')
ClaveSecretaApp = os.getenv('ClaveSecretaApp')
TokenAcceso = os.getenv('TokenAcceso')






import os
import requests
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

access_token = os.getenv("TokenAcceso")
post_id = '122098322012852515'
api_version = "v22.0"  # Prueba con una versión reciente de la API

if not access_token:
    print("Error: TokenAcceso no está configurada en tu archivo .env")
    exit()

url = f"https://graph.facebook.com/{api_version}/{post_id}/comments?access_token={access_token}"
all_comments = []

try:
    response = requests.get(url)
    response.raise_for_status()  # Lanza una excepción para códigos de error HTTP
    comments_data = response.json()

    if 'data' in comments_data:
        for comment in comments_data['data']:
            message = comment.get('message')
            if message:
                all_comments.append(message)

        # Manejo de la paginación (si existe 'next' en la respuesta)
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
                next_url = next_comments_data.get('paging', {}).get('next')
            else:
                next_url = None

    else:
        print(f"No se encontraron comentarios para la publicación con ID: {post_id}")

except requests.exceptions.RequestException as e:
    print(f"Ocurrió un error al consultar la API de Facebook: {e}")

# Crear el DataFrame de pandas con solo la columna de comentarios
df_comentarios = pd.DataFrame({'Comentario': all_comments})

# Imprimir el DataFrame (opcional)
print(df_comentarios)









