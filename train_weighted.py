import tensorflow as tf
import os
from tensorflow.keras.metrics import Precision, Recall
from datasets import load_dataset
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from transformers import (
    AutoTokenizer,
    PushToHubCallback,
    TFAutoModelForSequenceClassification,
)
import numpy as np
from collections import Counter

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

if tf.test.gpu_device_name():
    print("GPU found:", tf.test.gpu_device_name())
else:
    print("No GPU found")

#base_model = "distilbert-base-uncased"
base_model = "output-toxicity_weighted/final_model"
output_dir = "output-toxicity_weighted/final_model"
checkpoint_path = "output-toxicity_weighted/checkpoint.ckpt"
batch_size = 8

labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
label2id = {label: id for id, label in enumerate(labels)}
id2label = {id: label for id, label in enumerate(labels)}

dataset = load_dataset("csv", data_files="train.csv")
tokenizer = AutoTokenizer.from_pretrained(base_model)

def process_data(row):
    text = row["comment_text"]
    labels_batch = {k: row[k] for k in row.keys() if k in labels}

    encoding = tokenizer(text, padding="max_length", truncation=True)

    label_arr = [0] * len(labels)

    for id, label in enumerate(labels_batch):
        label_arr[id] = labels_batch[label]

    encoding["labels"] = label_arr

    return encoding

model = TFAutoModelForSequenceClassification.from_pretrained(
    base_model,
    problem_type="multi_label_classification",
    num_labels=len(labels),
    label2id=label2id,
    id2label=id2label,
)

encoded = dataset.map(
    process_data,
    remove_columns=["id", "comment_text"],
)

tf_dataset = model.prepare_tf_dataset(
    encoded["train"], batch_size, shuffle=True, tokenizer=tokenizer
)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, save_weights_only=True, verbose=1
)

push_to_hub_callback = PushToHubCallback(
    output_dir=output_dir,
    tokenizer=tokenizer,
    hub_model_id="RobCaamano/toxicity_weighted",
)
push_to_hub_callback.push_frequency = 'epoch'

reduce_lr_callback = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2, verbose=1, min_lr=1e-6)

es_callback = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)

epochs = 5

label_columns = dataset["train"].remove_columns(["id", "comment_text"]).to_pandas()
label_counts = label_columns.sum(axis=0)
total_labels = label_columns.shape[0]

class_weights = {}
for i, label in enumerate(labels):
    class_weight = (1 / label_counts[label]) * total_labels / len(labels)
    class_weights[i] = class_weight

print("Class Weights:")
for i, label in enumerate(labels):
    print(f"{label}: {class_weights[i]:.4f}")

model.compile(
    optimizer=Adam(3e-5),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=[
        Precision(name='precision'),
        Recall(name='recall'),
    ],
)

model.fit(
    tf_dataset,
    epochs=epochs,
    callbacks=[cp_callback, push_to_hub_callback, reduce_lr_callback, es_callback],
    class_weight=class_weights,
)