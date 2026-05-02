import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from elephas.utils import serialization


def test_model_to_dict():
    model = Sequential()
    model.add(Dense(1, "linear"))
    dict_model = serialization.model_to_dict(model)
    assert list(dict_model.keys()) == ["model", "weights"]


def test_dict_to_model():
    model = Sequential()
    model.add(Dense(1, "linear"))
    dict_model = serialization.model_to_dict(model)

    recovered = serialization.dict_to_model(dict_model)
    assert recovered.to_json() == model.to_json()


def test_weights_bytes_roundtrip_preserves_shape_dtype_and_values():
    rng = np.random.default_rng(42)
    weights = [
        rng.standard_normal((4, 3), dtype=np.float32),
        rng.standard_normal((3,), dtype=np.float32),
        rng.integers(-5, 5, size=(2, 2), dtype=np.int64),
        np.array([], dtype=np.float64),
    ]

    blob = serialization.weights_to_bytes(weights)
    recovered = serialization.bytes_to_weights(blob)

    assert len(recovered) == len(weights)
    for original, restored in zip(weights, recovered):
        assert original.shape == restored.shape
        assert original.dtype == restored.dtype
        assert np.array_equal(original, restored)


def test_bytes_to_weights_returns_writable_arrays():
    # frombuffer alone returns read-only views; callers mutate weights in place,
    # so the decoder must return independently-owned arrays.
    blob = serialization.weights_to_bytes([np.zeros(3, dtype=np.float32)])
    arr = serialization.bytes_to_weights(blob)[0]
    arr[0] = 1.0
    assert arr[0] == 1.0
