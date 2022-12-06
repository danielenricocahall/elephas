from pyspark.mllib.regression import LabeledPoint
from ..utils.rdd_utils import from_labeled_point, to_labeled_point, lp_to_simple_rdd
from pyspark.mllib.linalg import Vector as MLLibVector, Vectors as MLLibVectors


def to_data_frame(sc, features, labels, categorical=False):
    """Convert numpy arrays of features and labels into Spark DataFrame
    """
    from pyspark.sql import SparkSession
    lp_rdd = to_labeled_point(sc, features, labels, categorical)
    df = SparkSession.builder.getOrCreate().createDataFrame(lp_rdd)
    return df


def from_data_frame(df, categorical=False, nb_classes=None):
    """Convert DataFrame back to pair of numpy arrays
    """
    lp_rdd = df.rdd.map(lambda row: LabeledPoint(row.label, row.features))
    features, labels = from_labeled_point(lp_rdd, categorical, nb_classes)
    return features, labels


def df_to_simple_rdd(df, categorical=False, nb_classes=None, features_col='features', label_col='label'):
    """Convert DataFrame into RDD of pairs
    """
    from pyspark.sql import SparkSession
    spark_session = SparkSession.builder.getOrCreate()
    df.createOrReplaceTempView("temp_table")
    selected_df = spark_session.sql(
        "SELECT {0} AS features, {1} as label from temp_table".format(features_col, label_col))
    if isinstance(selected_df.first().features, MLLibVector):
        lp_rdd = selected_df.rdd.map(
            lambda row: LabeledPoint(row.label, row.features))
    else:
        lp_rdd = selected_df.rdd.map(lambda row: LabeledPoint(
            row.label, MLLibVectors.fromML(row.features)))
    rdd = lp_to_simple_rdd(lp_rdd, categorical, nb_classes)
    return rdd
