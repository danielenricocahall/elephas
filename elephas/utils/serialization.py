from typing import Dict, Any, Optional

import io
import numpy as np

from tensorflow.keras.models import model_from_json, Model


def model_to_dict(model: Model) -> Dict[str, Any]:
    """Turns a Keras model into a Python dictionary

    :param model: Keras model instance
    :return: dictionary with model information
    """
    return dict(model=model.to_json(), weights=model.get_weights())


def dict_to_model(
    _dict: Dict[str, Any], custom_objects: Optional[Dict[str, Any]] = None
):
    """Turns a Python dictionary with model architecture and weights
    back into a Keras model

    :param _dict: dictionary with `model` and `weights` keys.
    :param custom_objects: custom objects i.e; layers/activations, required for model
    :return: Keras model instantiated from dictionary
    """
    model = model_from_json(_dict["model"], custom_objects)
    model.set_weights(_dict["weights"])
    return model


def weights_to_npz_bytes(weights):
    """weights: list[np.ndarray]"""
    buf = io.BytesIO()
    # name each array deterministically for order
    np.savez_compressed(buf, **{f"arr{i}": w for i, w in enumerate(weights)})
    return buf.getvalue()


def npz_bytes_to_weights(b):
    buf = io.BytesIO(b)
    with np.load(buf) as z:
        return [z[f"arr{i}"] for i in range(len(z.files))]
