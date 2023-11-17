from itertools import count
from math import isclose

from tensorflow.keras.optimizers.legacy import SGD

from elephas.enums.modes import Mode
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

import pytest
import numpy as np


def _generate_port_number(port=3000, _count=count(1)):
    return port + next(_count)

# enumerate possible combinations for training mode and parameter server for a classification model while also
# validatiing multiple workers for repartitioning
@pytest.mark.parametrize('mode,parameter_server_mode,num_workers',
                         [(Mode.SYNCHRONOUS, None, None),
                          (Mode.SYNCHRONOUS, None, 2),
                          (Mode.ASYNCHRONOUS, 'http', None),
                          (Mode.ASYNCHRONOUS, 'http', 2),
                          (Mode.ASYNCHRONOUS, 'socket', None),
                          (Mode.ASYNCHRONOUS, 'socket', 2),
                          (Mode.HOGWILD, 'http', None),
                          (Mode.HOGWILD, 'http', 2),
                          (Mode.HOGWILD, 'socket', None),
                          (Mode.HOGWILD, 'socket', 2)])
def test_training_classification(spark_context, mode, parameter_server_mode, num_workers, mnist_data, classification_model):
    # Define basic parameters
    batch_size = 64
    epochs = 10

    # Load data
    x_train, y_train, x_test, y_test = mnist_data
    x_train = x_train[:1000]
    y_train = y_train[:1000]

    sgd = SGD(lr=0.1)
    classification_model.compile(sgd, 'categorical_crossentropy', ['acc'])

    # Build RDD from numpy features and labels
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Initialize SparkModel from keras model and Spark context
    spark_model = SparkModel(classification_model, frequency='epoch', num_workers=num_workers,
                             mode=mode, parameter_server_mode=parameter_server_mode, port=_generate_port_number())

    # Train Spark model
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)

    # run inference on trained spark model
    predictions = spark_model.predict(x_test)
    # run evaluation on trained spark model
    evals = spark_model.evaluate(x_test, y_test)

    # assert we can supply rdd and get same prediction results when supplying numpy array
    test_rdd = spark_context.parallelize(x_test)
    assert [np.argmax(x) for x in predictions] == [np.argmax(x) for x in spark_model.predict(test_rdd)]

    # assert we get the same prediction result with calling predict on keras model directly
    assert [np.argmax(x) for x in predictions] == [np.argmax(x) for x in spark_model.master_network.predict(x_test)]

    # assert we get the same evaluation results when calling evaluate on keras model directly
    assert isclose(evals[0], spark_model.master_network.evaluate(x_test, y_test)[0], abs_tol=0.01)
    assert isclose(evals[1], spark_model.master_network.evaluate(x_test, y_test)[1], abs_tol=0.01)


# enumerate possible combinations for training mode and parameter server for a regression model while also validating
# multiple workers for repartitioning
@pytest.mark.parametrize('mode,parameter_server_mode,num_workers',
                         [(Mode.SYNCHRONOUS, None, None),
                          (Mode.SYNCHRONOUS, None, 2),
                          (Mode.ASYNCHRONOUS, 'http', None),
                          (Mode.ASYNCHRONOUS, 'http', 2),
                          (Mode.ASYNCHRONOUS, 'socket', None),
                          (Mode.ASYNCHRONOUS, 'socket', 2),
                          (Mode.HOGWILD, 'http', None),
                          (Mode.HOGWILD, 'http', 2),
                          (Mode.HOGWILD, 'socket', None),
                          (Mode.HOGWILD, 'socket', 2)])
def test_training_regression(spark_context, mode, parameter_server_mode, num_workers, boston_housing_dataset,
                             regression_model):
    x_train, y_train, x_test, y_test = boston_housing_dataset
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Define basic parameters
    batch_size = 64
    epochs = 10
    sgd = SGD(lr=0.0000001)
    regression_model.compile(sgd, 'mse', ['mae', 'mean_absolute_percentage_error'])
    spark_model = SparkModel(regression_model, frequency='epoch', mode=mode, num_workers=num_workers,
                             parameter_server_mode=parameter_server_mode, port=_generate_port_number())

    # Train Spark model
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size,
                    verbose=0, validation_split=0.1)

    # run inference on trained spark model
    predictions = spark_model.predict(x_test)
    # run evaluation on trained spark model
    evals = spark_model.evaluate(x_test, y_test)

    # assert we can supply rdd and get same prediction results when supplying numpy array
    test_rdd = spark_context.parallelize(x_test)
    assert all(np.isclose(x, y, 0.01) for x, y in zip(predictions, spark_model.predict(test_rdd)))

    # assert we get the same prediction result with calling predict on keras model directly
    assert all(np.isclose(x, y, 0.01) for x, y in zip(predictions, spark_model.master_network.predict(x_test)))

    # assert we get the same evaluation results when calling evaluate on keras model directly
    assert isclose(evals[0], spark_model.master_network.evaluate(x_test, y_test)[0], abs_tol=0.01)
    assert isclose(evals[1], spark_model.master_network.evaluate(x_test, y_test)[1], abs_tol=0.01)
    assert isclose(evals[2], spark_model.master_network.evaluate(x_test, y_test)[2], abs_tol=0.01)


def test_training_regression_no_metrics(spark_context, boston_housing_dataset, regression_model):
    x_train, y_train, x_test, y_test = boston_housing_dataset
    rdd = to_simple_rdd(spark_context, x_train, y_train)

    # Define basic parameters
    batch_size = 64
    epochs = 1
    sgd = SGD(lr=0.0000001)
    regression_model.compile(sgd, 'mse')
    spark_model = SparkModel(regression_model, frequency='epoch', mode=Mode.SYNCHRONOUS, port=_generate_port_number())

    # Train Spark model
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size, verbose=0, validation_split=0.1)

    # run inference on trained spark model
    predictions = spark_model.predict(x_test)

    # assert we can supply rdd and get same prediction results when supplying numpy array
    test_rdd = spark_context.parallelize(x_test)
    assert all(np.isclose(x, y, 0.01) for x, y in zip(predictions, spark_model.predict(test_rdd)))

    # assert we get the same prediction result with calling predict on keras model directly
    assert all(np.isclose(x, y, 0.01) for x, y in zip(predictions, spark_model.master_network.predict(x_test)))

    # assert we get the same evaluation results when calling evaluate on keras model directly
    assert isclose(spark_model.evaluate(x_test, y_test),
                   spark_model.master_network.evaluate(x_test, y_test), abs_tol=0.01)



