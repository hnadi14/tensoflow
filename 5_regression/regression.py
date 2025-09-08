import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # قبل از import tensorflow

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# --- ۱. خواندن دادهها ---
df = pd.read_csv('house_prices.csv')

# --- ۲. آمادهسازی دادهها ---
x = df[['SqFt']].values  # تبدیل به آرایه NumPy
y = df[['Price']].values

# --- ۳. ساخت مدل ---
model = keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(units=1)
])

# --- ۴. کامپایل مدل ---
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    loss='mean_absolute_error'
)

# --- ۵. آموزش مدل ---
history = model.fit(x, y, epochs=100, verbose=0)

# --- ۶. پیشبینی ---
new_sqft = np.array([[100]])  # توجه به دو براکت [[...]]
pred = model.predict(new_sqft)
print(f"قیمت پیشبینیشده برای ۱۰۰ SqFt: {pred[0][0]:,.2f}")
