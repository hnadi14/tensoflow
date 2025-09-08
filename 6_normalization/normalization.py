import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import r2_score

# ------------------- تنظیمات اولیه -------------------
plt.rcParams['figure.figsize'] = (12, 6)
sns.set_style("whitegrid")

# ------------------- خواندن و بررسی داده -------------------
df = pd.read_csv('auto_mpg.csv').dropna()

# نمایش اطلاعات کلی داده
print("مشخصات داده اولیه:")
print(df.describe().transpose())

# نمودار همبستگی
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('ماتریس همبستگی ویژگیها')
plt.show()

# ------------------- آمادهسازی داده -------------------
train_dataset = df.sample(frac=0.8, random_state=42)
test_dataset = df.drop(train_dataset.index)

sns.pairplot(train_dataset)
plt.show()

x_train = train_dataset.copy()
x_test = test_dataset.copy()

# بررسی ستون هدف
if 'mpg' not in x_train.columns:
    raise ValueError("ستون هدف 'mpg' در داده وجود ندارد.")

y_train = x_train.pop('mpg')
y_test = x_test.pop('mpg')

# نرمالسازی
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(np.array(x_train))


# ------------------- ساخت مدل پیشرفته -------------------
def build_model():
    model = tf.keras.Sequential([
        normalizer,
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mse',
                  metrics=[tf.keras.metrics.MeanAbsoluteError(name='mae'),
                           tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model


model = build_model()

# ------------------- آموزش مدل با نظارت -------------------
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=20,
    restore_best_weights=True
)

history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=500,
    batch_size=32,
    verbose=0,
    callbacks=[early_stop]
)


# ------------------- تحلیل نتایج آموزش -------------------
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # نمودار Loss
    ax1.plot(history.history['loss'], label='Loss آموزش')
    ax1.plot(history.history['val_loss'], label='Loss اعتبارسنجی', linestyle='--')
    ax1.set_title('تغییرات Loss در طول آموزش')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    # نمودار MAE
    ax2.plot(history.history['mae'], label='MAE آموزش')
    ax2.plot(history.history['val_mae'], label='MAE اعتبارسنجی', linestyle='--')
    ax2.set_title('تغییرات MAE در طول آموزش')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.show()


plot_training_history(history)

# ------------------- ارزیابی مدل -------------------
test_loss, test_mae, test_rmse = model.evaluate(x_test, y_test, verbose=0)
y_pred = model.predict(x_test).flatten()

# محاسبه R-squared
r2 = r2_score(y_test, y_pred)

print(f"\n{'=' * 40}")
print(f"عملکرد مدل روی داده تست:")
print(f"MAE: {test_mae:.2f}")
print(f"RMSE: {test_rmse:.2f}")
print(f"R-squared: {r2:.2f}")
print(f"{'=' * 40}")


# ------------------- نمودارهای تحلیل پیشبینی -------------------
# تابع رسم نمودار Residuals
def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    plt.figure(figsize=(12, 6))

    # نمودار Scatter Residuals
    plt.subplot(1, 2, 1)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.7)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Predictions')

    # نمودار توزیع Residuals
    plt.subplot(1, 2, 2)
    sns.histplot(residuals, kde=True, bins=30)
    plt.xlabel('Residuals')
    plt.title('Distribution of Residuals')
    plt.axvline(0, color='r', linestyle='--')

    plt.tight_layout()
    plt.show()


plot_residuals(y_test, y_pred)

# ------------------- ذخیره و بارگذاری مدل -------------------
model.save('advanced_dnn_model.keras')
reloaded_model = tf.keras.models.load_model('advanced_dnn_model.keras')

# اعتبارسنجی مدل بارگذاریشده
reloaded_mae = reloaded_model.evaluate(x_test, y_test, verbose=0)[1]
print(f"\nMAE مدل بارگذاریشده: {reloaded_mae:.2f}")