import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_curve, auc

# خواندن داده ها و پیش پردازش
df = pd.read_csv('breast_cancer.csv')
df.drop('id', axis=1, inplace=True)
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# تحلیل اکتشافی داده ها
plt.figure(figsize=(18, 6))
sns.countplot(x='diagnosis', data=df)
plt.title('توزیع دسته ها')
plt.show()

# نمودار همبستگی
plt.figure(figsize=(20, 16))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('ماتریس همبستگی')
plt.show()

# انتخاب ویژگی ها و تقسیم داده ها
X = df.drop('diagnosis', axis=1)
y = df['diagnosis']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# مقیاس بندی داده ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ساخت مدل
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),

    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# آموزش مدل با validation
history = model.fit(
    X_train_scaled, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    verbose=0
)

# ارزیابی مدل روی داده های تست
test_metrics = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f"\nTest Accuracy: {test_metrics[1]:.4f}")
print(f"Test Precision: {test_metrics[2]:.4f}")
print(f"Test Recall: {test_metrics[3]:.4f}")
print(f"Test AUC: {test_metrics[4]:.4f}")

# ترسیم نمودارهای آموزش
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
metrics = ['accuracy', 'precision', 'recall', 'auc']

for i, metric in enumerate(metrics):
    ax = axes[i // 2, i % 2]
    ax.plot(history.history[metric], label='Training')
    ax.plot(history.history[f'val_{metric}'], label='Validation')
    ax.set_title(f'{metric.capitalize()} Curve')
    ax.set_xlabel('Epochs')
    ax.set_ylabel(metric.capitalize())
    ax.legend()

plt.tight_layout()
plt.show()

# نمودار AUC-ROC
y_pred_proba = model.predict(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# ماتریس سردرگمی
y_pred = (model.predict(X_test_scaled) > 0.5).astype("int32")
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# نمودارهای توزیع ویژگی ها
selected_features = ['radius_mean', 'texture_mean', 'perimeter_mean',
                     'area_mean', 'smoothness_mean']

g = sns.PairGrid(df, vars=selected_features, hue='diagnosis', palette='husl')
g.map_diag(sns.histplot, multiple='stack')
g.map_offdiag(sns.scatterplot)
g.add_legend(title='Diagnosis', labels=['Benign', 'Malignant'])
plt.suptitle('Pairwise Relationships of Selected Features')
plt.show()