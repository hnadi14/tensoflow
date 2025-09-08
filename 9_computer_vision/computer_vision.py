import zipfile
import idx2numpy
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# 1. باز کردن فایل ZIP
zip_path = 'mnist.zip'  # مسیر فایل ZIP خود را وارد کنید
# تغییرات در بخش خواندن فایلها:
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # برای فایلهای آموزشی:
    with zip_ref.open('train-images-idx3-ubyte/train-images-idx3-ubyte') as img_file:
        X_train = idx2numpy.convert_from_file(img_file)

    with zip_ref.open('train-labels-idx1-ubyte/train-labels-idx1-ubyte') as lbl_file:
        y_train = idx2numpy.convert_from_file(lbl_file)

    # برای فایلهای تست:
    with zip_ref.open('t10k-images-idx3-ubyte/t10k-images-idx3-ubyte') as img_file:
        X_test = idx2numpy.convert_from_file(img_file)

    with zip_ref.open('t10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte') as lbl_file:
        y_test = idx2numpy.convert_from_file(lbl_file)

# 5. تایید ابعاد داده‌ها
print("ابعاد داده‌های آموزشی:", X_train.shape)  # باید (60000, 28, 28) باشد
print("ابعاد برچسب‌های آموزشی:", y_train.shape)  # باید (60000,) باشد

# 6. نرمالسازی داده‌ها (اختیاری اما توصیه می‌شود)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# 7. نمایش نمونهای از داده‌ها
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# حذف شده: from Tools.i18n.makelocalealias import optimize (کاربردی نبود)

# خواندن دادهها (قسمت قبلی که تصحیح شده صحیح است)
# ... (بخش خواندن دادهها بدون تغییر)

# نرمالسازی (تکراری بود، یکبار حذف شد)
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

# ***** تغییرات در معماری مدل *****
model = tf.keras.models.Sequential([  # تصحیح: models با s
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),  # افزودن لایه پنهان
    tf.keras.layers.Dropout(0.2),  # جلوگیری از Overfitting
    tf.keras.layers.Dense(64, activation='relu'),  # افزودن لایه پنهان
    tf.keras.layers.Dropout(0.2),  # جلوگیری از Overfitting
    tf.keras.layers.Dense(32, activation='relu'),  # افزودن لایه پنهان
    tf.keras.layers.Dropout(0.2),  # جلوگیری از Overfitting
    tf.keras.layers.Dense(16, activation='relu'),  # افزودن لایه پنهان
    tf.keras.layers.Dropout(0.2),  # جلوگیری از Overfitting
    tf.keras.layers.Dense(10, activation='softmax')
])

print(model.summary())

# ***** تغییرات در کامپایل مدل *****
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # تغییر optimizer
    loss='categorical_crossentropy',
    metrics=['accuracy']  # اضافه کردن metric
)

# تبدیل برچسبها به one-hot (همانند قبل)
y_train_onehot = tf.keras.utils.to_categorical(y_train)
y_test_onehot = tf.keras.utils.to_categorical(y_test)

# ***** تغییرات در آموزش مدل *****
history = model.fit(
    X_train, y_train_onehot,
    epochs=10,
    validation_data=(X_test, y_test_onehot),
    verbose=1  # نمایش پیشرفت آموزش
)

# ***** نمودارهای بهبود یافته *****
plt.figure(figsize=(12, 5))

# نمودار loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Evolution')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# نمودار دقت
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Evolution')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# ***** پیشبینی و نمایش نتیجه *****
index = 100
test_image = X_test[index]
prediction = model.predict(np.expand_dims(test_image, 0))  # افزودن بعد batch

plt.imshow(test_image, cmap='gray')
plt.title(f"True: {y_test[index]}\nPredicted: {np.argmax(prediction)}")
plt.axis('off')
plt.show()
