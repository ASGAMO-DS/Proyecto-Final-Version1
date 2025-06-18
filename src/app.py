from flask import Flask, redirect, request, session, url_for
import requests
import os

app = Flask(__name__)
app.secret_key = '18f481ca7f13ffc5437a1697dc8a9eb3'

# Configuración OAuth
FB_CLIENT_ID = '1366085591316447'
FB_CLIENT_SECRET = '18f481ca7f13ffc5437a1697dc8a9eb3'
FB_REDIRECT_URI = 'http://localhost:5001/callback'  # Asegúrate que esté en FB Developers

FB_AUTH_BASE = 'https://www.facebook.com/v18.0/dialog/oauth'
FB_TOKEN_BASE = 'https://graph.facebook.com/v18.0/oauth/access_token'

@app.route('/')
def index():
    return '''
        <h1>Bienvenido a Insight Pulse</h1>
        <p>Aquí conectaremos con Facebook OAuth2.0</p>
        <a href="/login" style="font-size:20px;color:blue;">👉 Login con Facebook</a>
    '''


@app.route('/login')
def login():
    fb_auth_url = (
        f"{FB_AUTH_BASE}?client_id={FB_CLIENT_ID}"
        f"&redirect_uri={FB_REDIRECT_URI}"
        f"&scope=pages_read_engagement,pages_read_user_content"
        f"&response_type=code"
    )
    return redirect(fb_auth_url)

@app.route('/callback')
def callback():
    code = request.args.get('code')
    if not code:
        return 'Error: no se recibió el código de autorización'
    
    # Intercambiar código por token
    token_res = requests.get(FB_TOKEN_BASE, params={
        'client_id': FB_CLIENT_ID,
        'redirect_uri': FB_REDIRECT_URI,
        'client_secret': FB_CLIENT_SECRET,
        'code': code
    }).json()

    if 'access_token' in token_res:
        session['fb_token'] = token_res['access_token']
        return redirect(url_for('metrics'))
    else:
        return f"Error al obtener token: {token_res}"

@app.route('/metrics')
def metrics():
    token = session.get('fb_token')
    if not token:
        return redirect(url_for('login'))

    # Aquí haremos las métricas
    return '¡Aquí irán tus métricas!'

if __name__ == '__main__':
    app.run(debug=True, port=5001)




