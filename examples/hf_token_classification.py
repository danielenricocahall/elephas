from datasets import load_dataset
from pyspark.sql import SparkSession
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

from transformers import TFAutoModelForTokenClassification, AutoTokenizer

from elephas.spark_model import SparkHFModel
from elephas.utils import to_simple_rdd

spark_session = SparkSession.builder.appName('HF Token Classification').getOrCreate()
sc = spark_session.sparkContext

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

rdd = to_simple_rdd(sc, x_train, y_train)

tokenizer_kwargs = {'padding': True, 'truncation': True, 'is_split_into_words': True}

model.compile(optimizer=Adam(learning_rate=5e-5), metrics=['accuracy'])
spark_model = SparkHFModel(model, num_workers=num_workers, mode="synchronous", tokenizer=tokenizer,
                           tokenizer_kwargs=tokenizer_kwargs, loader=TFAutoModelForTokenClassification)

spark_model.fit(rdd, epochs=epochs, batch_size=batch_size)

# Run inference on trained Spark model
samples = tokenizer(x_test, **tokenizer_kwargs, return_tensors="tf")
distributed_predictions = spark_model(**samples)