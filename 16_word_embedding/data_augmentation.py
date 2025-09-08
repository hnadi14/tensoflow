import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers

# بارگذاری دادهها
(train_ds, val_ds, test_ds), metadata = tfds.load(
    'tf_flowers',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True
)

IMG_SIZE = 180
num_classes = metadata.features['label'].num_classes

# پیشپردازش تصاویر
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(IMG_SIZE, IMG_SIZE),
    layers.Rescaling(1./255)
])

# افزایش داده
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

# ساخت مدل
model = tf.keras.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),  # اضافه کردن لایه ورودی
    data_augmentation,
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes)
])

# کامپایل مدل
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# آمادهسازی دادهها
BATCH_SIZE = 32
train_ds = train_ds.map(lambda x, y: (resize_and_rescale(x), y)).batch(BATCH_SIZE)
val_ds = val_ds.map(lambda x, y: (resize_and_rescale(x), y)).batch(BATCH_SIZE)

# آموزش مدل
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5
)