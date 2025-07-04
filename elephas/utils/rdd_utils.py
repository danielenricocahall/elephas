from typing import Optional, Tuple

from pyspark import RDD, SparkContext
from pyspark.mllib.regression import LabeledPoint
import numpy as np

from ..mllib.adapter import to_vector, from_vector


def to_simple_rdd(sc: SparkContext, features: np.array, labels: np.array) -> RDD:
    """Convert numpy arrays of features and labels into
    an RDD of pairs.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :return: Spark RDD with feature-label pairs
    """
    pairs = [(x, y) for x, y in zip(features, labels)]
    return sc.parallelize(pairs)


def to_labeled_point(sc: SparkContext, features: np.array, labels: np.array, categorical: bool = False) -> RDD[
    LabeledPoint]:
    """Convert numpy arrays of features and labels into
    a LabeledPoint RDD for MLlib and ML integration.

    :param sc: Spark context
    :param features: numpy array with features
    :param labels: numpy array with labels
    :param categorical: boolean, whether labels are already one-hot encoded or not
    :return: LabeledPoint RDD with features and labels
    """
    labeled_points = [LabeledPoint(np.argmax(y) if categorical else y, to_vector(x)) for x, y in zip(features, labels)]
    return sc.parallelize(labeled_points)


def from_labeled_point(rdd: RDD[LabeledPoint], categorical: bool = False, nb_classes: Optional[int] = None) -> Tuple[
    np.array, np.array]:
    """Convert a LabeledPoint RDD back to a pair of numpy arrays

    :param rdd: LabeledPoint RDD
    :param categorical: boolean, if labels should be one-hot encode when returned
    :param nb_classes: optional int, indicating the number of class labels
    :return: pair of numpy arrays, features and labels
    """
    features_and_labels = rdd.map(lambda lp: (from_vector(lp.features), int(lp.label)))
    features, labels = zip(*features_and_labels.collect())
    features = np.array(features)
    labels = np.array(labels)
    if categorical:
        if not nb_classes:
            nb_classes = np.max(labels) + 1
        labels = np.stack(list(map(lambda x: encode_label(x, nb_classes), labels)))
    return features, labels


def encode_label(label: np.array, nb_classes: int) -> np.array:
    """One-hot encoding of a single label

    :param label: class label (int or double without floating point digits)
    :param nb_classes: int, number of total classes
    :return: one-hot encoded vector
    """
    encoded = np.zeros(nb_classes)
    encoded[int(label)] = 1.
    return encoded


def lp_to_simple_rdd(lp_rdd: RDD[LabeledPoint], categorical: bool = False, nb_classes: int = None) -> RDD:
    """Convert a LabeledPoint RDD into an RDD of feature-label pairs

    :param lp_rdd: LabeledPoint RDD of features and labels
    :param categorical: boolean, if labels should be one-hot encode when returned
    :param nb_classes: int, number of total classes
    :return: Spark RDD with feature-label pairs
    """
    if categorical:
        if not nb_classes:
            nb_classes = lp_rdd.map(lambda lp: lp.label).map(int).max() + 1
        rdd = lp_rdd.map(lambda lp: (from_vector(lp.features),
                                     encode_label(lp.label, nb_classes)))
    else:
        rdd = lp_rdd.map(lambda lp: (from_vector(lp.features), lp.label))
    return rdd
