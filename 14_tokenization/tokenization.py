"""
طبقه‌بندی متن با AG News Dataset (با ۵۰۰ نمونه آموزش)
این کد شامل مراحل زیر است:
1. محدود کردن داده آموزش به ۵۰۰ نمونه
2. پیش‌پردازش متن با TextVectorization
3. ساخت مدل ساده برای طبقه‌بندی
"""

import tensorflow as tf
import tensorflow_datasets as tfds

# ***************************************************
# مرحله ۱: بارگذاری و محدود کردن داده
# ***************************************************

# بارگذاری مجموعه داده
dataset = tfds.load('ag_news_subset')

# محدود کردن داده آموزش به ۵۰۰ نمونه
ds_train = dataset['train'].take(500)
ds_test = dataset['test']

# ***************************************************
# مرحله ۲: پیش‌پردازش داده
# ***************************************************

# ترکیب عنوان و توضیحات
def combine_text(features):
    return features['title'] + ' ' + features['description'], features['label']

# پیش‌پردازش داده‌ها
AUTOTUNE = tf.data.experimental.AUTOTUNE

ds_train = ds_train.map(combine_text, num_parallel_calls=AUTOTUNE)
ds_test = ds_test.map(combine_text, num_parallel_calls=AUTOTUNE)

# ***************************************************
# مرحله ۳: Vectorization متن
# ***************************************************

# تنظیمات TextVectorization
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=10000,
    output_mode='int',
    output_sequence_length=200
)

# آماده‌سازی Vectorizer روی داده‌های آموزش (۵۰۰ نمونه)
text_ds = ds_train.map(lambda x, y: x)
vectorizer.adapt(text_ds)

# ***************************************************
# مرحله ۴: ساخت مدل ساده
# ***************************************************

model = tf.keras.Sequential([
    vectorizer,
    tf.keras.layers.Embedding(len(vectorizer.get_vocabulary()), 64),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(4, activation='softmax')  # 4 کلاس در AG News
])

# ***************************************************
# مرحله ۵: کامپایل مدل
# ***************************************************

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ***************************************************
# مرحله ۶: آموزش مدل
# ***************************************************

# تنظیمات آموزش
BATCH_SIZE = 32
ds_train = ds_train.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).cache().prefetch(AUTOTUNE)

history = model.fit(
    ds_train,
    validation_data=ds_test,
    epochs=15,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    ]
)

# ***************************************************
# مرحله ۷: ارزیابی نهایی
# ***************************************************

test_loss, test_acc = model.evaluate(ds_test)
print(f"\nدقت نهایی روی داده آزمون: {test_acc:.2f}")
