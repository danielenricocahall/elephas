from typing import Optional

import numpy as np
import pyspark.sql
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from ..utils.rdd_utils import from_labeled_point, to_labeled_point, lp_to_simple_rdd
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors


def to_data_frame(sc: SparkContext, features: np.array, labels: np.array, categorical: bool = False):
    """Convert numpy arrays of features and labels into Spark DataFrame
    """
    from pyspark.sql import SparkSession
    lp_rdd = to_labeled_point(sc, features, labels, categorical)
    df = SparkSession.builder.getOrCreate().createDataFrame(lp_rdd)
    return df


def from_data_frame(df: pyspark.sql.DataFrame, categorical: bool = False, nb_classes: Optional[int] = None):
    """Convert DataFrame back to pair of numpy arrays
    """
    lp_rdd = df.rdd.map(lambda row: LabeledPoint(row.label, row.features))
    features, labels = from_labeled_point(lp_rdd, categorical, nb_classes)
    return features, labels


def df_to_simple_rdd(df: pyspark.sql.DataFrame,
                     categorical: bool = False,
                     nb_classes: Optional[int] = None,
                     features_col: str = 'features',
                     label_col: str = 'label'):
    """Convert DataFrame into RDD of pairs
    """
    from pyspark.sql import SparkSession
    spark_session = SparkSession.builder.getOrCreate()
    df.createOrReplaceTempView("temp_table")
    selected_df = spark_session.sql(
        f"SELECT {features_col} AS features, {label_col} as label from temp_table")
    if isinstance(selected_df.first().features, MLLibVector):
        lp_rdd = selected_df.rdd.map(
            lambda row: LabeledPoint(row.label, row.features))
    else:
        lp_rdd = selected_df.rdd.map(lambda row: LabeledPoint(
            row.label, MLLibVectors.fromML(row.features)))
    rdd = lp_to_simple_rdd(lp_rdd, categorical, nb_classes)
    return rdd
