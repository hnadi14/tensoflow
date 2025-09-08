import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

# بارگذاری داده
df = pd.read_csv('movies.csv')

# نمایش دادههای نمونه
print("دادههای نمونه:")
print(df.head())

# تعداد ژانرهای منحصر به فرد
all_genres = set('|'.join(df['genres']).split('|'))
print(f"\nتعداد ژانرهای منحصر به فرد: {len(all_genres)}")

# تبدیل ژانرها به لیست
df['genres'] = df['genres'].apply(lambda x: x.split('|'))

# One-Hot Encoding با MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(df['genres'])

# تبدیل به DataFrame
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)

# فراوانی هر ژانر
genre_counts = genres_df.sum().sort_values(ascending=False)
plt.figure(figsize=(12, 6))
sns.barplot(x=genre_counts.values, y=genre_counts.index)
plt.title('فراوانی ژانرهای فیلمها')
plt.show()

# ماتریس همرویداد (Co-occurrence Matrix)
co_occurrence = np.dot(genres_encoded.T, genres_encoded)
plt.figure(figsize=(12, 8))
sns.heatmap(co_occurrence, annot=True, fmt='d', cmap='viridis',
            xticklabels=mlb.classes_, yticklabels=mlb.classes_)
plt.title('ماتریس همرویداد ژانرها')
plt.show()

# ساخت مدل Autoencoder
input_dim = genres_encoded.shape[1]
encoding_dim = 5  # بعد فضای نهان (latent space)

input_layer = tf.keras.layers.Input(shape=(input_dim,))
encoded = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoded = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoded)

autoencoder = tf.keras.models.Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

autoencoder.summary()

history = autoencoder.fit(genres_encoded, genres_encoded,
                          epochs=100,
                          batch_size=32,
                          validation_split=0.2,
                          verbose=0)

# رسم نمودار Loss
plt.plot(history.history['loss'], label='Loss آموزش')
plt.plot(history.history['val_loss'], label='Loss اعتبارسنجی', linestyle='--')
plt.title('کاهش Loss در طول آموزش')
plt.legend()
plt.show()

# ------------------- مرحله ۷: استخراج قوانین انجمنی (اصلاحشده) -------------------
# استخراج لایه Encoder
encoder = tf.keras.models.Model(input_layer, encoded)
genre_embeddings = encoder.predict(genres_encoded)

# محاسبه Embedding هر ژانر
genre_vectors = {}
for genre in mlb.classes_:
    # اندیس ستون مربوط به ژانر
    genre_idx = np.where(mlb.classes_ == genre)[0][0]
    # فیلمهایی که شامل این ژانر هستند
    indices = np.where(genres_encoded[:, genre_idx] == 1)[0]
    # میانگین Embedding این فیلمها
    genre_vectors[genre] = genre_embeddings[indices].mean(axis=0)

# تبدیل به ماتریس
genre_vectors_matrix = np.array([genre_vectors[g] for g in mlb.classes_])

# محاسبه Cosine Similarity
from sklearn.metrics.pairwise import cosine_similarity
cosine_sim = cosine_similarity(genre_vectors_matrix)

# ساخت DataFrame
similarity_df = pd.DataFrame(cosine_sim, index=mlb.classes_, columns=mlb.classes_)

# نمودار حرارتی شباهت ژانرها
plt.figure(figsize=(12, 8))
sns.heatmap(similarity_df, annot=True, cmap='coolwarm')
plt.title('شباهت کسینوسی بین ژانرها')
plt.show()

# نمایش قوانین بر اساس حد آستانه
threshold = 0.6
for (genre1, genre2) in combinations(mlb.classes_, 2):
    if similarity_df.loc[genre1, genre2] > threshold:
        print(f"قانون: {genre1} ⟷ {genre2} (میزان شباهت: {similarity_df.loc[genre1, genre2]:.2f})")


# ------------------- مرحله ۸: کشف قوانین چندتایی -------------------

def calculate_support(itemset):
    """محاسبه تعداد فیلمهای حاوی تمام ژانرهای مجموعه"""
    mask = np.ones(genres_encoded.shape[0], dtype=bool)
    for genre in itemset:
        genre_idx = np.where(mlb.classes_ == genre)[0][0]
        mask &= (genres_encoded[:, genre_idx] == 1)
    return mask.sum()

min_support = 5  # حداقل تعداد فیلمهای حاوی مجموعه
min_confidence = 0.6  # حداقل اطمینان

print("\n=== قوانین چندتایی ===")
for r in [2, 3]:  # بررسی ترکیبهای 2 و 3تایی
    for itemset in combinations(mlb.classes_, r):
        support = calculate_support(itemset)
        if support >= min_support:
            # تولید تمام قوانین ممکن از این ترکیب
            for i in range(r):
                antecedent = list(itemset[:i] + itemset[i+1:])
                consequent = itemset[i]
                antecedent_support = calculate_support(antecedent)
                confidence = support / antecedent_support if antecedent_support > 0 else 0
                if confidence >= min_confidence and antecedent_support > 0:
                    print(f"قانون: {{{', '.join(antecedent)}}} → {consequent} | "
                          f"Support: {support}, Confidence: {confidence:.2f}")