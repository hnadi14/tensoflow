"""
شبکه عصبی پیشرفته برای طبقه‌بندی تصاویر CIFAR-10
این کد شامل بهبودهای زیر است:
- استفاده از Batch Normalization برای پایداری آموزش
- اضافه کردن لایه Dropout برای جلوگیری از Overfitting
- استفاده از Data Augmentation برای افزایش تنوع داده
- بهینه‌سازی معماری با لایه‌های عمیق‌تر
- تنظیم نرخ یادگیری و Early Stopping
"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ***************************************************
# مرحله ۱ و ۲: تعریف مسئله، جمع‌آوری داده و پیش‌پردازش
# ***************************************************

# بارگذاری داده‌ها
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# نرمال‌سازی پیکسل‌ها به بازه [0, 1]
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# ***************************************************
# مرحله ۲ (پیشرفته): افزودن Data Augmentation
# ***************************************************

# تنظیمات Augmentation برای افزایش تنوع داده
datagen = ImageDataGenerator(
    rotation_range=15,       # چرخش تصادفی تا ۱۵ درجه
    width_shift_range=0.1,   # جابجایی افقی تا ۱۰٪
    height_shift_range=0.1,  # جابجایی عمودی تا ۱۰٪
    horizontal_flip=True     # قرینه افقی تصاویر
)

# ***************************************************
# مرحله ۳: طراحی معماری بهینه‌شده CNN
# ***************************************************

def create_advanced_cnn_model():
    """
    معماری پیشرفته با ویژگی‌های زیر:
    - سه بلوک Convolutional با BatchNorm و MaxPooling
    - استفاده از Dropout برای کاهش Overfitting
    - لایه‌های Dense کوچکتر برای کاهش پیچیدگی
    - فعال‌ساز ReLU و Softmax در خروجی
    """
    model = tf.keras.models.Sequential([
        # بلوک Convolutional 1
        tf.keras.layers.Conv2D(32, (3,3), padding='same', input_shape=(32,32,3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        # بلوک Convolutional 2
        tf.keras.layers.Conv2D(64, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        # بلوک Convolutional 3
        tf.keras.layers.Conv2D(128, (3,3), padding='same'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.MaxPooling2D((2,2)),

        # تبدیل به بردار یک بعدی
        tf.keras.layers.Flatten(),

        # لایه پنهان با Dropout
        tf.keras.layers.Dense(128),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),

        # لایه خروجی
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

# ***************************************************
# مرحله ۴ و ۵: تنظیمات بهینه‌ساز و تابع هزینه
# ***************************************************

# ایجاد مدل
model = create_advanced_cnn_model()

# تنظیم نرخ یادگیری پویا
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)

# کامپایل مدل با بهینه‌ساز به‌روزرسانی شده
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# ***************************************************
# مرحله ۶: آموزش با Early Stopping و Augmentation
# ***************************************************

# تنظیمات Early Stopping برای جلوگیری از Overfitting
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# آموزش مدل با Augmentation و Batch Size بزرگتر
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=3,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping]
)

# ***************************************************
# مرحله ۹: ارزیابی پیشرفته
# ***************************************************

def plot_training_history(history):
    """
    نمایش نمودارهای دقت و Loss برای آموزش و اعتبارسنجی
    همراه با بررسی Overfitting/Underfitting
    """
    plt.figure(figsize=(14, 5))

    # نمودار دقت
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # نمودار Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# اجرای تابع رسم نمودار
plot_training_history(history)

# ***************************************************
# مرحله ۱۰: ارزیابی نهایی روی داده آزمون
# ***************************************************

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
print(f"\nTest Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# ***************************************************
# مرحله ۱۱: ذخیره مدل بهینه‌شده
# ***************************************************
model.save('advanced_cifar10_cnn.h5')