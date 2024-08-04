import numpy as np
import tensorflow as tf
from itertools import tee

from pyspark import SparkFiles
from tensorflow.keras.models import model_from_json
from tensorflow.keras.optimizers import get as get_optimizer
from tensorflow.python.keras.utils.generic_utils import slice_arrays

from .enums.frequency import Frequency
from .utils import subtract_params
from .parameter import BaseParameterClient

from .utils.versioning_utils import get_minor_version
from .utils.model_utils import is_multiple_input_model, is_multiple_output_model


class SparkWorker(object):
    """Synchronous Spark worker. This code will be executed on workers.
    """

    def __init__(self, json, parameters, train_config, master_optimizer,
                 master_loss, master_metrics, custom_objects):
        self.json = json
        self.parameters = parameters
        self.train_config = train_config
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics
        self.custom_objects = custom_objects
        self.model = None

    @property
    def master_optimizer(self):
        return self._master_optimizer

    @master_optimizer.setter
    def master_optimizer(self, optimizer):
        from tensorflow.keras.optimizers.legacy import Optimizer as LegacyOptimizer
        if get_minor_version(tf.__version__) >= 13:
            if not isinstance(optimizer, LegacyOptimizer):
                raise ValueError(f"In Tensorflow 2.13+, optimizer {optimizer} needs to be a legacy optimizer - "
                                 f"subclass of "
                                 f"tensorflow.keras.optimizers.legacy.Optimizer. Currently, non-legacy optimizers ("
                                 f"subclasses of tensorflow.keras.optimizers) are not supported.")
        self._master_optimizer = optimizer

    def train(self, data_iterator):
        """Train a keras model on a worker
        """
        history = None
        optimizer = get_optimizer(self.master_optimizer)
        self.model = model_from_json(self.json, self.custom_objects)
        self.model.compile(optimizer=optimizer,
                           loss=self.master_loss, metrics=self.master_metrics)
        self.model.set_weights(self.parameters.value)

        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])
        if is_multiple_input_model(self.model):
            x_train = np.hsplit(x_train, len(self.model.input_shape))
        if is_multiple_output_model(self.model):
            y_train = np.hsplit(y_train, len(self.model.output_shape))
        weights_before_training = self.model.get_weights()
        if x_train.shape[0] > self.train_config.get('batch_size'):
            history = self.model.fit(x_train, y_train, **self.train_config)
        weights_after_training = self.model.get_weights()
        deltas = subtract_params(
            weights_before_training, weights_after_training)
        if history:
            yield [deltas, history.history]
        else:
            yield [deltas, None]


class AsynchronousSparkWorker(object):
    """Asynchronous Spark worker. This code will be executed on workers.
    """

    def __init__(self, json, parameters, client, train_config, frequency,
                 master_optimizer, master_loss, master_metrics, custom_objects):

        if isinstance(client, BaseParameterClient):
            # either supply a client object directly
            self.client = client
        else:
            # or a string to create a client
            self.client = BaseParameterClient.get_client(client)

        self.train_config = train_config
        self.frequency = frequency
        self.master_optimizer = master_optimizer
        self.master_loss = master_loss
        self.master_metrics = master_metrics
        self.json = json
        self.parameters = parameters
        self.custom_objects = custom_objects
        self.model = None

    def train(self, data_iterator):
        """Train a keras model on a worker and send asynchronous updates
        to parameter server
        """
        feature_iterator, label_iterator = tee(data_iterator, 2)
        x_train = np.asarray([x for x, y in feature_iterator])
        y_train = np.asarray([y for x, y in label_iterator])

        if x_train.size == 0:
            return
        nb_train_sample = x_train.shape[0]

        self.model = model_from_json(self.json, self.custom_objects)
        if is_multiple_input_model(self.model):
            x_train = np.hsplit(x_train, len(self.model.input_shape))
        if is_multiple_output_model(self.model):
            y_train = np.hsplit(y_train, len(self.model.output_shape))
        self.model.compile(optimizer=get_optimizer(self.master_optimizer),
                           loss=self.master_loss, metrics=self.master_metrics)
        self.model.set_weights(self.parameters.value)

        epochs = self.train_config['epochs']
        batch_size = self.train_config.get('batch_size')
        nb_batch = int(np.ceil(nb_train_sample / float(batch_size)))
        index_array = np.arange(nb_train_sample)
        batches = [
            (i * batch_size, min(nb_train_sample, (i + 1) * batch_size))
            for i in range(0, nb_batch)
        ]

        if self.frequency == Frequency.EPOCH:
            for epoch in range(epochs):
                weights_before_training = self.client.get_parameters()
                self.model.set_weights(weights_before_training)
                self.train_config['epochs'] = 1
                if nb_train_sample > batch_size:
                    self.model.fit(x_train, y_train, **self.train_config)
                self.train_config['epochs'] = epochs
                weights_after_training = self.model.get_weights()
                deltas = subtract_params(
                    weights_before_training, weights_after_training)
                self.client.update_parameters(deltas)
        elif self.frequency == Frequency.BATCH:
            for epoch in range(epochs):
                if nb_train_sample > batch_size:
                    for (batch_start, batch_end) in batches:
                        weights_before_training = self.client.get_parameters()
                        self.model.set_weights(weights_before_training)
                        batch_ids = index_array[batch_start:batch_end]
                        x = slice_arrays(x_train, batch_ids)
                        y = slice_arrays(y_train, batch_ids)
                        self.model.train_on_batch(x, y)
                        weights_after_training = self.model.get_weights()
                        deltas = subtract_params(
                            weights_before_training, weights_after_training)
                        self.client.update_parameters(deltas)
        else:
            raise ValueError(
                'frequency parameter can be `epoch` or `batch, got {}'.format(self.frequency))
        yield []


class SparkHFWorker(SparkWorker):
    """Synchronous worker for Huggingface models"""

    def __init__(self, json, parameters, train_config, master_optimizer,
                 master_loss, master_metrics, custom_objects, temp_dir, tokenizer, loader):
        super().__init__(json, parameters, train_config, master_optimizer,
                 master_loss, master_metrics, custom_objects)
        self.tokenizer = tokenizer
        self.temp_dir = temp_dir
        self.loader = loader

    def train(self, data_iterator):
        """Train a Huggingface model on a worker
        """
<<<<<<< Updated upstream
=======
        from transformers import TFAutoModelForSequenceClassification, TFAutoModelForCausalLM, TFAutoModelForTokenClassification

>>>>>>> Stashed changes
        temp_dir = self.temp_dir.value
        config = SparkFiles.get(temp_dir)
        x_train, y_train = zip(*data_iterator)

        self.model = self.loader.from_pretrained(config, local_files_only=True)
        self.model.compile(optimizer=get_optimizer(self.master_optimizer),
                           loss=self.master_loss, metrics=self.master_metrics)
<<<<<<< Updated upstream
=======
        weights_before_training = self.model.get_weights()
        if self.loader.__name__ in (TFAutoModelForSequenceClassification.__name__, TFAutoModelForTokenClassification.__name__):
            # TODO: would be nice to have a better way to check if the model is a sequence classification model
            x_train, y_train = zip(*data_iterator)
            x_train = self.tokenizer(list(x_train), **self.tokenizer_kwargs, return_tensors="tf")
            max_length = max(len(seq) for seq in x_train['input_ids'])
            y_train_padded = process_labels(y_train, max_length, -100)

            y_train = np.array(y_train_padded)
            history = self.model.fit(dict(x_train), y_train, **self.train_config)
        elif self.loader.__name__ == TFAutoModelForCausalLM.__name__:
            x_train = self.tokenizer(list(data_iterator), **self.tokenizer_kwargs, return_tensors="tf")
            x_train, y_train = x_train['input_ids'][:, :-1], x_train['input_ids'][:, 1:]
            history = self.model.fit(x_train, y_train, **self.train_config)
        else:
            raise ValueError(f"Unsupported loader type: {self.loader.__name__}")
>>>>>>> Stashed changes

        x_train = self.tokenizer(list(x_train), padding=True, truncation=True, return_tensors="tf")
        y_train = np.array(y_train)

        weights_before_training = self.model.get_weights()
        history = self.model.fit(dict(x_train), y_train, **self.train_config)
        weights_after_training = self.model.get_weights()
        deltas = subtract_params(
            weights_before_training, weights_after_training)
        if history:
            yield [deltas, history.history]
        else:
            yield [deltas, None]


<<<<<<< Updated upstream

=======
def process_labels(labels, max_length, pad_token_label_id):
    processed_labels = []
    for label_seq in labels:
        padded_seq = label_seq + [pad_token_label_id] * (max_length - len(label_seq))
        processed_labels.append(padded_seq)
    return np.array(processed_labels)
>>>>>>> Stashed changes
