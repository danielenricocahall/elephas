import json
from enum import Enum


class ModelType(Enum):
    CLASSIFICATION = 1
    REGRESSION = 2


class _Singleton(type):
    """ A metaclass that creates a Singleton base class when called. """
    _instances = {}

    def __call__(cls, *args):
        if cls not in cls._instances:
            cls._instances[cls] = super(_Singleton, cls).__call__(*args)
        return cls._instances[cls]


class Singleton(_Singleton('SingletonMeta', (object,), {})):
    pass


class LossModelTypeMapper(Singleton):
    """
    Mapper for losses -> model type
    """

    def __init__(self):
        loss_to_model_type = {}
        loss_to_model_type.update(
            {'mean_squared_error': ModelType.REGRESSION,
             'mean_absolute_error': ModelType.REGRESSION,
             'mse': ModelType.REGRESSION,
             'mae': ModelType.REGRESSION,
             'cosine_proximity': ModelType.REGRESSION,
             'mean_absolute_percentage_error': ModelType.REGRESSION,
             'mean_squared_logarithmic_error': ModelType.REGRESSION,
             'logcosh': ModelType.REGRESSION,
             'binary_crossentropy': ModelType.CLASSIFICATION,
             'categorical_crossentropy': ModelType.CLASSIFICATION,
             'sparse_categorical_crossentropy': ModelType.CLASSIFICATION})
        self.__mapping = loss_to_model_type

    def get_model_type(self, loss):
        return self.__mapping.get(loss)

    def register_loss(self, loss, model_type):
        if callable(loss):
            loss = loss.__name__
        self.__mapping.update({loss: model_type})


class ModelTypeEncoder(json.JSONEncoder):
    def default(self, obj):
        if obj in [e for e in ModelType]:
            return {"__enum__": str(obj)}
        return json.JSONEncoder.default(self, obj)


def as_enum(d):
    if "__enum__" in d:
        name, member = d["__enum__"].split(".")
        return getattr(ModelType, member)
    else:
        return d


def is_multiple_input_model(model) -> bool:
    """Check if a model has multiple inputs
    """
    return isinstance(model.input_shape, list)


def is_multiple_output_model(model) -> bool:
    """Check if a model has multiple outputs
    """
    return isinstance(model.output_shape, list)
