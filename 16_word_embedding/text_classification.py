import tensorflow as tf
import tensorflow_datasets as tfds
import re
import string
import numpy as np
import os
import json
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from datetime import datetime


# ******************* پارامترهای پیشرفته *******************
class Config:
    BATCH_SIZE = 32
    SEED = 42
    VOCAB_SIZE = 10000
    EMBEDDING_DIM = 128
    MAX_SEQUENCE_LENGTH = 250
    DROPOUT_RATE = 0.3
    LSTM_UNITS = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 10
    LOG_DIR = "./logs"
    MODEL_DIR = "./saved_model"
    CACHE_DIR = "./tf_cache"


# ******************* تنظیمات محیطی *******************
tf.random.set_seed(Config.SEED)
np.random.seed(Config.SEED)
os.makedirs(Config.LOG_DIR, exist_ok=True)
os.makedirs(Config.MODEL_DIR, exist_ok=True)
os.makedirs(Config.CACHE_DIR, exist_ok=True)


# ******************* بارگذاری و پیش‌پردازش داده‌ها *******************
def load_datasets():
    """بارگذاری و پیش‌پردازش مجموعه داده IMDB"""

    def configure_dataset(dataset):
        return dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    raw_train = tf.keras.utils.text_dataset_from_directory(
        './imdb/train',
        batch_size=Config.BATCH_SIZE,
        validation_split=0.2,
        subset='training',
        seed=Config.SEED
    ).map(lambda x, y: (x, tf.expand_dims(y, -1)))

    raw_val = tf.keras.utils.text_dataset_from_directory(
        './imdb/train',
        batch_size=Config.BATCH_SIZE,
        validation_split=0.2,
        subset='validation',
        seed=Config.SEED
    ).map(lambda x, y: (x, tf.expand_dims(y, -1)))

    raw_test = tf.keras.utils.text_dataset_from_directory(
        './imdb/test',
        batch_size=Config.BATCH_SIZE
    ).map(lambda x, y: (x, tf.expand_dims(y, -1)))

    return configure_dataset(raw_train), configure_dataset(raw_val), configure_dataset(raw_test)


# ******************* پردازش متن پیشرفته *******************
class TextPreprocessor:
    def __init__(self, max_features, sequence_length):
        self.vectorize_layer = tf.keras.layers.TextVectorization(
            standardize=self.custom_standardization,
            max_tokens=max_features,
            output_mode='int',
            output_sequence_length=sequence_length
        )

    def custom_standardization(self, input_data):
        """پیش‌پردازش پیشرفته متن شامل:
        - تبدیل به حروف کوچک
        - حذف HTML
        - حذف علامات نگارشی
        - حذف اعداد
        - نرمال‌سازی Unicode"""
        lowercase = tf.strings.lower(input_data)
        stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
        no_punct = tf.strings.regex_replace(stripped_html, f'[{re.escape(string.punctuation)}]', '')
        no_digits = tf.strings.regex_replace(no_punct, '\d+', '')
        return tf.strings.unicode_transcode(no_digits, 'UTF-8', 'UTF-8')

    def adapt(self, dataset):
        text_ds = dataset.map(lambda x, y: x)
        self.vectorize_layer.adapt(text_ds)

    def vectorize(self, text, label):
        text = tf.expand_dims(text, -1)
        return self.vectorize_layer(text), label


# ******************* معماری مدل پیشرفته *******************
def build_advanced_model():
    """ساخت مدل پیشرفته با استفاده از LSTM دوطرفه و لایه‌های بهبودیافته"""
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(Config.VOCAB_SIZE, Config.EMBEDDING_DIM,
                                  name="embedding"),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            Config.LSTM_UNITS,
            return_sequences=True,
            recurrent_dropout=0.2
        ), name="bidirectional_lstm_1"),

        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
            Config.LSTM_UNITS // 2,
            recurrent_dropout=0.2
        ), name="bidirectional_lstm_2"),

        tf.keras.layers.LayerNormalization(name="layer_norm"),

        tf.keras.layers.Dense(64, activation='relu', name="dense_1"),
        tf.keras.layers.Dropout(Config.DROPOUT_RATE, name="dropout_1"),

        tf.keras.layers.Dense(1, activation='sigmoid', name="output")
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc')
        ]
    )

    return model


# ******************* آموزش پیشرفته *******************
def train_model(model, train_ds, val_ds):
    """آموزش مدل با استفاده از کالبک‌های پیشرفته"""
    log_dir = os.path.join(Config.LOG_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
        TensorBoard(log_dir=log_dir, histogram_freq=1),
        ModelCheckpoint(
            filepath=os.path.join(Config.MODEL_DIR, 'model_{epoch:02d}-{val_loss:.2f}.keras'),
            save_best_only=True,
            monitor='val_auc',
            mode='max'
        )
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=Config.EPOCHS,
        callbacks=callbacks
    )

    return history


# ******************* ارزیابی پیشرفته *******************
def evaluate_model(model, test_ds):
    """ارزیابی جامع مدل با معیارهای مختلف"""
    results = model.evaluate(test_ds, return_dict=True)
    print("\n*** Evaluation Results ***")
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    return results


# ******************* صادرات مدل *******************
def export_full_model(preprocessor, model):
    """ایجاد مدل قابل استفاده برای استنتاج با پیش‌پردازش داخلی"""
    inference_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(), dtype=tf.string),
        preprocessor.vectorize_layer,
        model
    ])

    inference_model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(),
        metrics=['accuracy']
    )

    inference_model.save(os.path.join(Config.MODEL_DIR, 'inference_model.keras'))
    return inference_model


# ******************* پیش‌بینی پیشرفته *******************
def predict_samples(model, samples):
    """اجرای پیش‌بینی و فرمت‌بندی خروجی"""
    samples = tf.constant(samples)
    predictions = model.predict(samples)

    results = []
    for text, prob in zip(samples.numpy(), predictions):
        sentiment = "Positive" if prob > 0.5 else "Negative"
        results.append({
            'text': text.decode('utf-8'),
            'probability': float(prob),
            'sentiment': sentiment
        })

    return results


# ******************* اجرای کلی *******************
if __name__ == "__main__":
    # بارگذاری داده‌ها
    train_ds, val_ds, test_ds = load_datasets()

    # پیش‌پردازش
    preprocessor = TextPreprocessor(Config.VOCAB_SIZE, Config.MAX_SEQUENCE_LENGTH)
    preprocessor.adapt(train_ds)

    train_ds = train_ds.map(lambda x, y: preprocessor.vectorize(x, y))
    val_ds = val_ds.map(lambda x, y: preprocessor.vectorize(x, y))
    test_ds = test_ds.map(lambda x, y: preprocessor.vectorize(x, y))

    # ساخت و آموزش مدل
    model = build_advanced_model()
    history = train_model(model, train_ds, val_ds)

    # ارزیابی
    evaluate_model(model, test_ds)

    # صادرات مدل
    inference_model = export_full_model(preprocessor, model)

    # نمونه پیش‌بینی
    examples = [
        "This film was absolutely fantastic! The acting was superb.",
        "Terrible movie with poor plot and bad acting.",
        "It had its moments but overall disappointing."
    ]

    results = predict_samples(inference_model, examples)
    print(json.dumps(results, indent=2, ensure_ascii=False))