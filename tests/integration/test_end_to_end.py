import os
from itertools import count
from math import isclose

from datasets import load_dataset
from keras import Model
from pyspark.ml.feature import StringIndexer, VectorAssembler
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.optimizers.legacy import SGD, Adam
from tensorflow.keras import Input
from tensorflow.keras.layers import Embedding, Flatten, Dot
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification, TFAutoModelForCausalLM, \
    TFAutoModelForTokenClassification

from elephas.enums.modes import Mode
from elephas.enums.frequency import Frequency
from elephas.spark_model import SparkModel, SparkHFModel
from elephas.utils.huggingface_utils import pad_labels
from elephas.utils.rdd_utils import to_simple_rdd

import pytest
import numpy as np

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


def test_training_huggingface_classification(spark_context):
    batch_size = 5
    epochs = 1
    num_workers = 2

    newsgroups = fetch_20newsgroups(subset='train')
    x = newsgroups.data[:50]  # Limit the data size for the test
    y = newsgroups.target[:50]

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.5)

    model_name = 'albert-base-v2'  # use the smallest classification model for testing

    rdd = to_simple_rdd(spark_context, x_train, y_train)

    model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(np.unique(y_encoded)))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer_kwargs = {'padding': True, 'truncation': True}

    model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    spark_model = SparkHFModel(model, num_workers=num_workers, mode=Mode.SYNCHRONOUS, tokenizer=tokenizer,
                               tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForSequenceClassification)

    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)

    # Run inference on trained Spark model
    predictions = spark_model.predict(spark_context.parallelize(x_test))
    samples = tokenizer(x_test, padding=True, truncation=True, return_tensors="tf")
    # Evaluate results
    assert all(np.isclose(x, y, 0.01).all() for x, y in zip(predictions, spark_model.master_network(**samples)[0]))


def test_training_huggingface_generation(spark_context):
    batch_size = 5
    epochs = 1
    num_workers = 2

    newsgroups = fetch_20newsgroups(subset='train')
    x = newsgroups.data[:60]

    x_train, x_test = train_test_split(x, test_size=0.2)

    model_name = 'sshleifer/tiny-gpt2'  # use the smaller generative model for testing

    model = TFAutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer_kwargs = {'max_length': 15, 'padding': True, 'truncation': True}

    model.compile(optimizer=SGD(), metrics=['accuracy'], loss='sparse_categorical_crossentropy')

    spark_model = SparkHFModel(model, num_workers=num_workers, mode=Mode.SYNCHRONOUS, tokenizer=tokenizer,
                               tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForCausalLM)
    rdd = spark_context.parallelize(x_train)
    rdd_test = spark_context.parallelize(x_test)
    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)
    generations = spark_model.generate(rdd_test, max_length=20, num_return_sequences=1)
    generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generations]
    assert generated_texts == [tokenizer.decode(output, skip_special_tokens=True) for output in
                               spark_model.master_network.generate(
                                   **tokenizer(x_test, max_length=15, padding=True, truncation=True,
                                               return_tensors="tf"), num_return_sequences=1)]


def test_training_huggingface_token_classification(spark_context):
    batch_size = 5
    epochs = 2
    num_workers = 2
    model_name = 'hf-internal-testing/tiny-bert-for-token-classification'  # use the smallest classification model for testing

    model = TFAutoModelForTokenClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples[f"ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)  # Map tokens to their respective word.
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:  # Set the special tokens to -100.
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    dataset = load_dataset("conll2003", split='train[:5%]', trust_remote_code=True)
    dataset = dataset.map(tokenize_and_align_labels, batched=True)

    x = dataset['tokens']
    y = dataset['labels']

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    rdd = to_simple_rdd(spark_context, x_train, y_train)

    tokenizer_kwargs = {'padding': True, 'truncation': True, 'is_split_into_words': True}

    model.compile(optimizer=Adam(learning_rate=5e-5), metrics=['accuracy'])
    spark_model = SparkHFModel(model, num_workers=num_workers, mode=Mode.SYNCHRONOUS, tokenizer=tokenizer,
                               tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForTokenClassification)

    spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)

    # Run inference on trained Spark model
    samples = tokenizer(x_test, **tokenizer_kwargs, return_tensors="tf")
    distributed_predictions = spark_model(**samples)
    regular_predictions = spark_model.master_network(**samples)
    # Evaluate results
    assert all(np.isclose(x, y, 0.01).all() for x, y in zip(distributed_predictions[0], regular_predictions[0]))