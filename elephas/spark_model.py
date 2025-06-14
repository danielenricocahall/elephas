import json
import shutil
import subprocess
from contextlib import contextmanager
from uuid import uuid4
from pathlib import Path
from copy import deepcopy
from functools import partial
from itertools import tee
from typing import Union, List, Dict, Any, Optional, Callable

import h5py
import numpy as np
import pyspark
import tensorflow as tf
from pyspark import RDD
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import (
    serialize as serialize_optimizer,
    deserialize as deserialize_optimizer,
)

from .enums.modes import Mode
from .enums.frequency import Frequency
from .mllib import to_matrix, from_matrix, to_vector, from_vector
from .parameter.factory import ClientServerFactory
from .utils import lp_to_simple_rdd, to_simple_rdd
from .utils import model_to_dict
from .utils import subtract_params, divide_by
from .utils.model_utils import is_multiple_input_model, is_multiple_output_model
from .worker import AsynchronousSparkWorker, SparkWorker


class SparkModel:
    def __init__(
        self,
        model,
        num_workers=None,
        custom_objects=None,
        batch_size=32,
        *args,
        **kwargs,
    ):
        """SparkModel

        Base class for distributed training on RDDs. Spark model takes a Keras
        model as master network, an optimization scheme, a parallelisation mode
        and an averaging frequency.

        :param model: Compiled Keras model
        :param num_workers: int, number of workers used for training (defaults to None)
        :param custom_objects: Keras custom objects
        :param batch_size: batch size used for training and inference
        """
        self._training_histories = []
        self._master_network = model
        if not hasattr(model, "loss"):
            raise Exception(
                "Compile your Keras model before initializing an Elephas model with it"
            )
        metrics = model._compile_metrics._user_metrics if model._compile_metrics else []
        loss = model.loss

        if custom_objects is None:
            custom_objects = {}
        self.num_workers = num_workers
        self.weights = self._master_network.get_weights()
        self.pickled_weights = None
        self.master_optimizer = model.optimizer
        self.master_loss = loss
        self.master_metrics = metrics
        self.custom_objects = custom_objects
        self.batch_size = batch_size
        self.kwargs = kwargs
        self.serialized_model = model_to_dict(model)

    def get_config(self):
        base_config = {"num_workers": self.num_workers, "batch_size": self.batch_size}
        config = base_config.copy()
        config.update(self.kwargs)
        return config

    def save(self, file_name: str, overwrite: bool = False, to_hadoop: bool = False):
        """
        Save an elephas model as a h5 or keras file. The default functionality is to
        save the model file locally without overwriting. This can be changed to toggle
        overwriting and saving directly into a network-accessible Hadoop cluster.

        :param file_name: String, name or full path of the model file to be saved
        :param overwrite: Boolean, toggles between overwriting or raising error if
        file already excists, default is False
        :param to_hadoop: Boolean, toggles between saving locally or on a Hadoop
        cluster, default is False
        """
        assert file_name[-3:] == ".h5" or file_name[-6:] == ".keras", (
            "File name must end with either '.h5' or '.keras'"
        )

        if overwrite and not to_hadoop and Path(file_name).exists():
            Path(file_name).unlink()

        if to_hadoop:
            cluster_file_path = deepcopy(file_name)
            file_name = str(uuid4()) + "-temp-model-file." + file_name.split(".")[-1]

        model = self._master_network
        model.save(file_name)
        f = h5py.File(file_name, mode="a")

        f.attrs["distributed_config"] = json.dumps(
            {"class_name": self.__class__.__name__, "config": self.get_config()}
        ).encode("utf8")

        f.flush()
        f.close()

        if to_hadoop:
            # TODO: Consider implementing a try-except clause to use "hdfs dfs" instead
            cli = ["hadoop", "fs", "-moveFromLocal"]
            if overwrite:
                cli.append("-f")
            cli.append(file_name)
            cli.append(cluster_file_path)
            subprocess.run(cli)

    @property
    def training_histories(self):
        return self._training_histories

    @property
    def master_network(self):
        return self._master_network

    @master_network.setter
    def master_network(self, network):
        self._master_network = network

    def predict(self, data: Union[RDD, np.array]) -> List[np.ndarray]:
        """Perform distributed inference with the model"""
        if isinstance(data, (np.ndarray,)):
            from pyspark.sql import SparkSession

            sc = SparkSession.builder.getOrCreate().sparkContext
            data = sc.parallelize(data)
        return self._predict(data)

    def evaluate(
        self, x_test: np.array, y_test: np.array, **kwargs
    ) -> Union[List[float], float]:
        """Perform distributed evaluation with the model"""
        from pyspark.sql import SparkSession

        sc = SparkSession.builder.getOrCreate().sparkContext
        test_rdd = to_simple_rdd(sc, x_test, y_test)
        return self._evaluate(test_rdd, **kwargs)

    def fit(self, rdd: RDD, **kwargs):
        """
        Train an elephas model on an RDD. The Keras model configuration as specified
        in the elephas model is sent to Spark workers, abd each worker will be trained
        on their data partition.

        :param rdd: RDD with features and labels
        :param epochs: number of epochs used for training
        :param batch_size: batch size used for training
        :param verbose: logging verbosity level (0, 1 or 2)
        :param validation_split: percentage of data set aside for validation
        """
        print(">>> Fit model")
        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)

        self._fit(rdd, **kwargs)

    def _fit(self, rdd: RDD, **kwargs):
        """Protected train method to make wrapping of modes easier"""
        self._master_network.compile(
            optimizer=self.master_optimizer,
            loss=self.master_loss,
            metrics=self.master_metrics,
        )
        train_config = kwargs
        loss = self.master_loss
        metrics = self.master_metrics
        custom = self.custom_objects
        serialized_optimizer = serialize_optimizer(self.master_optimizer)

        model_json = self._master_network.to_json()
        init = self._master_network.get_weights()
        parameters = rdd.context.broadcast(init)
        worker = SparkWorker(
            model_json,
            parameters,
            train_config,
            serialized_optimizer,
            loss,
            metrics,
            custom,
        )
        training_outcomes = rdd.mapPartitions(worker.train).collect()
        new_parameters = self._master_network.get_weights()
        number_of_sub_models = len(training_outcomes)
        for training_outcome in training_outcomes:
            grad, history = training_outcome
            self.training_histories.append(history)
            weighted_grad = divide_by(grad, number_of_sub_models)
            new_parameters = subtract_params(new_parameters, weighted_grad)
        print(">>> Synchronous training complete.")
        self._master_network.set_weights(new_parameters)

    def _predict(self, rdd: RDD) -> List[np.ndarray]:
        """
        Private distributed predict method called by public predict method, after data has been verified to be an RDD
        """
        json_model = self.master_network.to_json()
        weights = self.master_network.get_weights()
        weights = rdd.context.broadcast(weights)
        custom_objs = self.custom_objects

        def _predict(
            model_as_json: str, custom_objects: Dict[str, Any], data
        ) -> np.array:
            model = model_from_json(model_as_json, custom_objects)
            model.set_weights(weights.value)
            data = np.array([x for x in data])
            if is_multiple_input_model(model):
                data = np.hsplit(data, len(model.input_shape))
            return model.predict(data)

        def _predict_with_indices(
            model_as_json: str, custom_objects: Dict[str, Any], data
        ):
            model = model_from_json(model_as_json, custom_objects)
            model.set_weights(weights.value)
            data, indices = zip(*data)
            data = np.array(data)
            if is_multiple_input_model(model):
                data = np.hsplit(data, len(model.input_shape))
            return zip(model.predict(data), indices)

        return self._call_and_collect(
            rdd,
            partial(_predict, json_model, custom_objs),
            partial(_predict_with_indices, json_model, custom_objs),
        )

    def _evaluate(self, rdd: RDD, **kwargs) -> Union[List[float], float]:
        """Private distributed evaluate method called by public evaluate method, after data has been verified to be an RDD"""
        json_model = self.master_network.to_json()
        serialized_optimizer = serialize_optimizer(self.master_optimizer)
        loss = self.master_loss
        weights = self.master_network.get_weights()
        weights = rdd.context.broadcast(weights)
        custom_objects = self.custom_objects
        metrics = self.master_metrics

        def _evaluate(
            model,
            serialized_optimizer,
            loss: Callable[[tf.Tensor, tf.Tensor], tf.Tensor],
            custom_objects: Dict[str, Any],
            metrics: List[str],
            kwargs: Dict[str, Any],
            data_iterator,
        ) -> List[Union[float, int]]:
            model = model_from_json(model, custom_objects)
            model.compile(
                deserialize_optimizer(serialized_optimizer), loss, metrics=metrics
            )
            model.set_weights(weights.value)
            feature_iterator, label_iterator = tee(data_iterator, 2)
            x_test = np.asarray([x for x, y in feature_iterator])
            y_test = np.asarray([y for x, y in label_iterator])
            if is_multiple_input_model(model):
                x_test = np.hsplit(x_test, len(model.input_shape))
            if is_multiple_output_model(model):
                y_test = np.hsplit(y_test, len(model.output_shape))
            evaluation_results = model.evaluate(x_test, y_test, **kwargs)
            evaluation_results = (
                [evaluation_results]
                if not isinstance(evaluation_results, list)
                else evaluation_results
            )
            # return the evaluation results and the size of the sample
            return [evaluation_results + [len(x_test)]]

        if self.num_workers:
            rdd = rdd.repartition(self.num_workers)
        results = rdd.mapPartitions(
            partial(
                _evaluate,
                json_model,
                serialized_optimizer,
                loss,
                custom_objects,
                metrics,
                kwargs,
            )
        )
        mapping_function = lambda x: tuple(x[-1] * x[i] for i in range(len(x) - 1)) + (  # noqa: E731
            x[-1],
        )
        reducing_function = lambda x, y: tuple(x[i] + y[i] for i in range(len(x)))  # noqa: E731
        agg_loss, *agg_metrics, number_of_samples = results.map(
            mapping_function
        ).reduce(reducing_function)
        avg_loss = agg_loss / number_of_samples
        avg_metrics = [agg_metric / number_of_samples for agg_metric in agg_metrics]
        # return loss and list of metrics if there are metrics, otherwise just return the scalar loss
        return [avg_loss, *avg_metrics] if avg_metrics else avg_loss

    def _call_and_collect(
        self, rdd: RDD, predict_func: Callable, predict_with_indices_func: Callable
    ) -> List[np.ndarray]:
        if self.num_workers and self.num_workers > 1:
            rdd = rdd.zipWithIndex()
            rdd = rdd.repartition(self.num_workers)
            predictions_and_indices = rdd.mapPartitions(
                partial(predict_with_indices_func)
            )
            predictions_sorted_by_index = predictions_and_indices.sortBy(lambda x: x[1])
            predictions = predictions_sorted_by_index.map(lambda x: x[0]).collect()
        else:
            predictions = rdd.mapPartitions(partial(predict_func)).collect()
        return predictions


class AsynchronousSparkModel(SparkModel):
    def __init__(
        self,
        model,
        num_workers=None,
        custom_objects=None,
        batch_size=32,
        mode=Mode.ASYNCHRONOUS,
        frequency=Frequency.EPOCH,
        parameter_server_mode="http",
        port=4000,
        *args,
        **kwargs,
    ):
        super().__init__(
            model,
            batch_size=batch_size,
            num_workers=num_workers,
            custom_objects=custom_objects,
            *args,
            **kwargs,
        )
        self.mode = mode
        self.parameter_server_mode = parameter_server_mode
        self.frequency = frequency
        factory = ClientServerFactory.get_factory(parameter_server_mode)
        self.parameter_server = factory.create_server(
            self.serialized_model, port, self.mode, custom_objects=self.custom_objects
        )
        self.client = factory.create_client(port)

    def get_config(self):
        config = super().get_config()
        return {
            **config,
            "parameter_server_mode": self.parameter_server_mode,
            "mode": self.mode,
            "frequency": self.frequency,
        }

    def _fit(self, rdd: RDD, **kwargs):
        """Protected train method to make wrapping of modes easier"""
        self._master_network.compile(
            optimizer=self.master_optimizer,
            loss=self.master_loss,
            metrics=self.master_metrics,
        )
        with self.run_parameter_server():
            train_config = kwargs
            freq = self.frequency
            serialized_optimizer = serialize_optimizer(self.master_optimizer)
            loss = self.master_loss
            metrics = self.master_metrics
            custom = self.custom_objects

            model_json = self._master_network.to_json()
            init = self._master_network.get_weights()
            parameters = rdd.context.broadcast(init)

            print(">>> Initialize workers")
            worker = AsynchronousSparkWorker(
                model_json,
                parameters,
                self.client,
                train_config,
                freq,
                serialized_optimizer,
                loss,
                metrics,
                custom,
            )
            print(">>> Distribute load")
            rdd.mapPartitions(worker.train).collect()
            print(">>> Async training complete.")
            new_parameters = self.client.get_parameters()
            self._master_network.set_weights(new_parameters)

    @contextmanager
    def run_parameter_server(self):
        try:
            self.parameter_server.start()
            yield
        finally:
            self.parameter_server.stop()


class SparkMLlibModel(SparkModel):
    def fit(
        self,
        labeled_points: RDD,
        epochs: int = 10,
        batch_size: int = 32,
        verbose: int = 0,
        validation_split: float = 0.1,
        categorical: bool = False,
        nb_classes: Optional[int] = None,
    ):
        """Train an elephas model on an RDD of LabeledPoints"""
        rdd = lp_to_simple_rdd(labeled_points, categorical, nb_classes)
        super().fit(
            rdd=rdd,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
            validation_split=validation_split,
        )

    def predict(self, mllib_data):
        """Predict probabilities for an RDD of features"""
        if isinstance(mllib_data, pyspark.mllib.linalg.Matrix):
            return to_matrix(self.predict(from_matrix(mllib_data)))
        elif isinstance(mllib_data, pyspark.mllib.linalg.Vector):
            return to_vector(self.predict(from_vector(mllib_data)))
        else:
            raise ValueError(
                "Provide either an MLLib matrix or vector, got {}".format(
                    mllib_data.__name__
                )
            )


def load_spark_model(
    file_name: str, from_hadoop: bool = False
) -> Union[SparkModel, SparkMLlibModel]:
    """
    Load an elephas model from a h5 or keras file. Assumes file is located locally by
    default, but can be toggled to load from a network-connected Hadoop cluster as well.

    :param file_name: String, name or full path of the model file to be loaded
    :param from_hadoop: Boolean, toggles between local or Hadoop cluster file loading,
    default is False
    :return: SparkModel or SparkMLlibModel, loaded elephas model
    """
    assert file_name[-3:] == ".h5" or file_name[-6:] == ".keras", (
        "File name must end with either '.h5' or '.keras'"
    )

    if from_hadoop:
        temp_file = str(uuid4()) + "-temp-model-file." + file_name.split(".")[-1]
        subprocess.run(["hadoop", "fs", "-copyToLocal", file_name, temp_file])
        file_name = temp_file

    model = load_model(file_name)
    f = h5py.File(file_name, mode="r")

    elephas_conf = json.loads(f.attrs.get("distributed_config"))
    class_name = elephas_conf.get("class_name")
    config = elephas_conf.get("config")

    if from_hadoop:
        Path(file_name).unlink()

    if class_name == SparkModel.__name__:
        return SparkModel(model=model, **config)
    elif class_name == SparkMLlibModel.__name__:
        return SparkMLlibModel(model=model, **config)


def save_and_zip_model(model, temp_dir):
    # Save model to the temporary directory
    model.save_pretrained(temp_dir)
    # Create a zip file of the model directory
    zip_path = shutil.make_archive(temp_dir, "zip", temp_dir)
    return zip_path
