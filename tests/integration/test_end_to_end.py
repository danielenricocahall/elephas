import os
from itertools import count
from math import isclose
from keras import Model
from pyspark.ml.feature import StringIndexer, VectorAssembler
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Flatten, Dot

from elephas.enums.modes import Mode
from elephas.enums.frequency import Frequency
from elephas.spark_model import SparkModel
from elephas.utils.rdd_utils import to_simple_rdd

import pytest
import numpy as np
import tensorflow as tf

from elephas.utils.versioning_utils import get_minor_version


def _generate_port_number(port=3000, _count=count(1)):
    return port + next(_count)


COMBINATIONS = [(Mode.SYNCHRONOUS, None, None),
                (Mode.SYNCHRONOUS, None, 2),
                (Mode.ASYNCHRONOUS, 'http', None),
                (Mode.ASYNCHRONOUS, 'http', 2),
                (Mode.ASYNCHRONOUS, 'socket', None),
                (Mode.ASYNCHRONOUS, 'socket', 2),
                (Mode.HOGWILD, 'http', None),
                (Mode.HOGWILD, 'http', 2),
                (Mode.HOGWILD, 'socket', None),
                (Mode.HOGWILD, 'socket', 2)]


# enumerate possible combinations for training mode and parameter server for a classification model while also
# validatiing multiple workers for repartitioning
@pytest.mark.parametrize('mode,parameter_server_mode,num_workers', COMBINATIONS)
def test_training_classification(spark_context, mode, parameter_server_mode, num_workers, mnist_data,
                                 classification_model):
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
@pytest.mark.parametrize('mode,parameter_server_mode,num_workers', COMBINATIONS)
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


@pytest.mark.parametrize('frequency', [Frequency.EPOCH, Frequency.BATCH])
def test_multiple_input_model(spark_session, frequency):
    def row_to_tuple(row):
        return [row.user_id_encoded, row.track_id_encoded], row.frequency

    # Read and preprocess data
    current_dir = os.path.dirname(__file__)
    parent_dir = os.path.dirname(current_dir)
    df = spark_session.read.csv(f'{parent_dir}/data/sample_data.csv', header=True, inferSchema=True)

    indexer_user = StringIndexer(inputCol="user_id", outputCol="user_id_encoded")
    df = indexer_user.fit(df).transform(df)
    indexer_track = StringIndexer(inputCol="track_id", outputCol="track_id_encoded")
    df = indexer_track.fit(df).transform(df)

    assembler = VectorAssembler(inputCols=["user_id_encoded", "track_id_encoded"], outputCol="features")
    df_transformed = assembler.transform(df)

    # Keras model inputs
    n_users = df.select('user_id').distinct().count()
    n_items = df.select('track_id').distinct().count()
    n_latent_factors = 50  # Example

    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    user_embedding = Embedding(output_dim=n_latent_factors, input_dim=n_users)(user_input)
    item_embedding = Embedding(output_dim=n_latent_factors, input_dim=n_items)(item_input)
    user_vec = Flatten()(user_embedding)
    item_vec = Flatten()(item_embedding)
    dot = Dot(axes=1)([user_vec, item_vec])
    model = Model([user_input, item_input], dot)
    model.compile(optimizer=SGD(), loss='mean_squared_error')

    rdd_final = df_transformed.rdd.map(row_to_tuple)

    spark_model = SparkModel(model, frequency=frequency, mode=Mode.ASYNCHRONOUS, port=_generate_port_number())
    spark_model.fit(rdd_final, epochs=5, batch_size=32, verbose=0, validation_split=0.1)
    rdd_test_data = rdd_final.map(lambda x: x[0])
    rdd_test_targets = rdd_final.map(lambda x: x[1])
    assert spark_model.predict(rdd_test_data)
    assert spark_model.evaluate(np.array(rdd_test_data.collect()), np.array(rdd_test_targets.collect()))
