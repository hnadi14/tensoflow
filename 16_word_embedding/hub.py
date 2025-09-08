import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # غیرفعال کردن هشدار oneDNN

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# بارگذاری دادهها
train_data, validation_data, test_data = tfds.load(
    name="imdb_reviews",
    split=('train[:60%]', 'train[60%:]', 'test'),
    as_supervised=True
)

# لایه Embedding از TF Hub
embedding = "https://tfhub.dev/google/nnlm-en-dim50/2"
hub_layer = hub.KerasLayer(embedding, input_shape=[], dtype=tf.string, trainable=True)

# ساخت مدل
model = Sequential([
    hub_layer,
    Dense(16, activation='relu'),
    Dense(1)
])

# کامپایل مدل
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# آموزش مدل
history = model.fit(
    train_data.shuffle(10000).batch(512),  # buffer_size اضافه شد
    epochs=10,
    validation_data=validation_data.batch(512),
    verbose=1
)

# پیشبینی
examples = tf.constant([
    "This film was absolutely fantastic! The acting was superb.",
    "Terrible movie with poor plot and bad acting.",
    "It had its moments but overall disappointing."
])
result = model.predict(examples)
print(result)