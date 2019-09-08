"""
This module contains tools for prediction from images and plotting the results.
"""
import base64
import io

import numpy as np
import torch
from matplotlib.figure import Figure
from PIL import Image

from app import model


def preprocess(data):
    """
    Convert base64 string to shrinked monochrome image.

    Args:
        data (str): Image encoded as a base64 string.

    Returns:
        np.array with shape (28, 28)
    """
    # Remove prefix generated in html canvases
    length = len("data:image/png;base64")
    data = data[length:]

    image = Image.open(io.BytesIO(base64.standard_b64decode(data)))
    image.thumbnail((28, 28), Image.NEAREST)
    return np.array(image)[:, :, 3] / 255


def plot_image(data):
    """
    Plot the given image after preprocessing.

    Args:
        data (str): Image encoded as a base64 string.

    Returns:
        matplotlib figure
    """
    image = preprocess(data)
    figure = Figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.imshow(image, cmap="Greys")
    ax.set_xticks([])
    ax.set_yticks([])
    figure.tight_layout()
    return figure


def predict(data, samples=1):
    """
    Predict probability for different digits in incoming data and
    make a bar plot out of it.

    Args:
        data (str): Image encoded as a base64 string.
        samples (int): Number of times the network should be sampled.
            An average is then used. This is only useful if network
            is stochastic (default 1).

    Returns:
        matplotlib figure
    """
    image = preprocess(data)
    x = torch.Tensor(image).view(1, 1, 28, 28)

    figure = Figure()
    ax = figure.add_subplot(1, 1, 1)

    preds = []
    for _ in range(samples):
        pred = model(x).detach().numpy()
        pred = np.squeeze(pred)
        preds.append(pred)
    pred = np.mean(preds, axis=0)
    pred = np.exp(pred)

    return pred
