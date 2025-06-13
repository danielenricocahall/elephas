import numpy as np
from pyspark.sql import SparkSession
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import SGD

from transformers import TFAutoModelForSequenceClassification, AutoTokenizer

from elephas.spark_model import SparkHFModel
from elephas.utils import to_simple_rdd

spark_session = SparkSession.builder.appName('HF Text Classification').getOrCreate()
sc = spark_session.sparkContext

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

rdd = to_simple_rdd(sc, x_train, y_train)

model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(np.unique(y_encoded)))
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer_kwargs = {'padding': True, 'truncation': True}

model.compile(optimizer=SGD(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
spark_model = SparkHFModel(model, num_workers=num_workers, mode="synchronous", tokenizer=tokenizer,
                           tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForSequenceClassification)

spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)

# Run inference on trained Spark model
predictions = spark_model.predict(sc.parallelize(x_test))
samples = tokenizer(x_test, padding=True, truncation=True, return_tensors="tf")
# Evaluate results
assert all(np.isclose(x, y, 0.01).all() for x, y in zip(predictions, spark_model.master_network(**samples)[0]))