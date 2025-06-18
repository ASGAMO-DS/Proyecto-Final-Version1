from flask import Flask, request, redirect, url_for, session, render_template_string 

app = Flask(__name__)
app.secret_key = '4e13f3c21c4e2fbdde69a3a7ab5cc472'  # Cambia esto por algo seguro 

@app.route("/")
def index():
    return "<h1>Bienvenido a Insight Pulse</h1><p>Aquí conectaremos con Facebook OAuth2.0</p>"

@app.route("/login")
def login():
    return "<p>Ruta de login: aquí vamos a hacer el redirect a Facebook</p>"

@app.route("/callback")
def callback():
    return "<p>Callback recibido, procesando token...</p>"

@app.route("/metrics")
def metrics():
    return "<p>Visualización de métricas</p>"

if __name__ == "__main__":
    app.run(debug=True)

