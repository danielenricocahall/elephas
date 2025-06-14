import pytest
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from elephas.utils import serialization


def test_model_to_dict():
    model = Sequential()
    model.add(Dense(1, "linear"))
    dict_model = serialization.model_to_dict(model)
    assert list(dict_model.keys()) == ['model', 'weights']


def test_dict_to_model():
    model = Sequential()
    model.add(Dense(1, "linear"))
    dict_model = serialization.model_to_dict(model)

    recovered = serialization.dict_to_model(dict_model)
    assert recovered.to_json() == model.to_json()
