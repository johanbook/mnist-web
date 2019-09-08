import io
import numpy as np

from flask import Response, render_template, request, jsonify
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from app import app, cache
from app import plot


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cache.cached(timeout=1)
def predict():
    preds = plot.predict(request.form.get("image"))
    result = {
        'accuracy': f' ({100*float(np.max(preds)):.2f}%)',
        'data': preds.tolist(),
        'candidate': int(np.argmax(preds)),
        'labels': list(range(10))
    }
    return jsonify(result)


@app.route("/imshow", methods=["POST"])
@cache.cached(timeout=1)
def imshow():
    data = request.form.get("image")
    output = io.BytesIO()
    FigureCanvas(plot.plot_image(data)).print_png(output)
    return Response(output.getvalue(), mimetype="image/png")
