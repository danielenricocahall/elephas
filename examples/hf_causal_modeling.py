from pyspark.sql import SparkSession
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers.legacy import SGD

from transformers import AutoTokenizer, TFAutoModelForCausalLM

from elephas.spark_model import SparkHFModel

spark_session = SparkSession.builder.appName('HF Causal LM').getOrCreate()
sc = spark_session.sparkContext

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

spark_model = SparkHFModel(model, num_workers=num_workers, mode="synchronous", tokenizer=tokenizer,
                           tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForCausalLM)
rdd = sc.parallelize(x_train)
rdd_test = sc.parallelize(x_test)
spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)
generations = spark_model.generate(rdd_test, max_length=20, num_return_sequences=1)
generated_texts = [tokenizer.decode(output, skip_special_tokens=True) for output in generations]