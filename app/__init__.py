from flask import Flask
from flask_caching import Cache

from app.nn import load_model

app = Flask(__name__, template_folder="../static/", static_folder="../static")
cache = Cache(app, config={"CACHE_TYPE": "simple"})
model = load_model("models/mnist.pt")

from app import routes
