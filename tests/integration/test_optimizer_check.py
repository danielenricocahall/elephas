import numpy as np
import pytest
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from elephas.enums.modes import Mode
from elephas.spark_model import SparkModel
from elephas.utils import to_simple_rdd
from elephas.utils.versioning_utils import get_minor_version


@pytest.mark.skipif(get_minor_version(tf.__version__) < 13, reason="This test only applies to Tensorflow 2.13+")
def test_exception_raised_if_not_legacy_optimizer_in_tf_13(spark_context, boston_housing_dataset, regression_model):
    # import SGD not from legacy - this cannot be serialized correctly so we try to catch that in advance
    from tensorflow.keras.optimizers import SGD

    model = Sequential()
    model.add(Dense(1, input_dim=1))
    model.add(Dense(1, activation='sigmoid'))

    sgd = SGD(learning_rate=0.1)
    model.compile(sgd, 'binary_crossentropy', ['acc'])

    x_train = np.random.rand(100)
    y_train = np.zeros(100)
    y_train[:50] = 1
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    spark_model = SparkModel(model, frequency='epoch', mode=Mode.SYNCHRONOUS)
    with pytest.raises(ValueError):
        # Train Spark model
        spark_model.fit(rdd, epochs=1, batch_size=32,
                        verbose=0, validation_split=0.1)
