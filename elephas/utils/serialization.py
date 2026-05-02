from typing import Dict, Any, List, Optional

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


def weights_to_bytes(weights: List[np.ndarray]) -> bytes:
    """Serialize a list of numpy arrays to a self-describing byte string.

    Format (all little-endian, no compression): uint32 count, then for each array
    uint8 ndim, int64[ndim] shape, uint16 dtype_str_len, ascii dtype_str,
    uint64 nbytes, raw payload.
    """
    out = io.BytesIO()
    out.write(np.uint32(len(weights)).tobytes())
    for w in weights:
        w = np.ascontiguousarray(w)
        dt = str(w.dtype).encode("ascii")
        out.write(np.uint8(w.ndim).tobytes())
        out.write(np.asarray(w.shape, dtype=np.int64).tobytes())
        out.write(np.uint16(len(dt)).tobytes())
        out.write(dt)
        out.write(np.uint64(w.nbytes).tobytes())
        out.write(w.tobytes())
    return out.getvalue()


def bytes_to_weights(b: bytes) -> List[np.ndarray]:
    """Inverse of :func:`weights_to_bytes`."""
    mv = memoryview(b)
    off = 0
    n = int(np.frombuffer(mv[off : off + 4], dtype=np.uint32)[0])
    off += 4
    out: List[np.ndarray] = []
    for _ in range(n):
        ndim = int(np.frombuffer(mv[off : off + 1], dtype=np.uint8)[0])
        off += 1
        shape = tuple(np.frombuffer(mv[off : off + 8 * ndim], dtype=np.int64).tolist())
        off += 8 * ndim
        dt_len = int(np.frombuffer(mv[off : off + 2], dtype=np.uint16)[0])
        off += 2
        dtype = np.dtype(bytes(mv[off : off + dt_len]).decode("ascii"))
        off += dt_len
        nbytes = int(np.frombuffer(mv[off : off + 8], dtype=np.uint64)[0])
        off += 8
        # frombuffer returns a read-only view into `b`; copy so the array is
        # independently owned and may be mutated by callers.
        arr = np.frombuffer(mv[off : off + nbytes], dtype=dtype).reshape(shape).copy()
        off += nbytes
        out.append(arr)
    return out
