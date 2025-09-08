import tensorflow as tf

# --- ۱. بررسی حالت اجرایی TensorFlow ---
print("حالت اجرایی Eager فعال است:", tf.executing_eagerly())

# --- ۲. تعریف ماتریس و انجام عملیات ماتریسی ---
# ماتریس ۲x۲ تعریف شده
x = [[1, 0],
     [0, 4]]
x_tensor = tf.constant(x, dtype=tf.float32)  # تبدیل به تانسور TensorFlow

# عملیات ضرب ماتریسی (Matrix Multiplication)
matmul_result = tf.matmul(x_tensor, x_tensor)
print("\n--- نتیجه ضرب ماتریسی ---")
print(matmul_result.numpy())

# --- ۳. عملیات تنسوری پیشرفته ---
# ضرب عنصری (Element-wise Multiplication)
elementwise_result = tf.multiply(x_tensor, x_tensor)
print("\n--- ضرب عنصری ---")
print(elementwise_result.numpy())

# ترانهاده ماتریس (Transpose)
transpose_result = tf.transpose(x_tensor)
print("\n--- ترانهاده ماتریس ---")
print(transpose_result.numpy())

# --- ۴. محاسبات آماری ---
# محاسبه دترمینان (Determinant)
det = tf.linalg.det(x_tensor)
print("\n--- دترمینان ماتریس ---")
print(det.numpy())

# محاسبه معکوس ماتریس (Inverse)
inv = tf.linalg.inv(x_tensor)
print("\n--- معکوس ماتریس ---")
print(inv.numpy())

# --- ۵. عملیات بر روی تنسورهای چندبعدی ---
# تعریف یک تنسور ۳ بعدی (۲x۲x۳)
tensor_3d = tf.constant([
    [[1, 2, 3], [4, 5, 6]],
    [[7, 8, 9], [10, 11, 12]]
])
print("\n--- تنسور ۳ بعدی ---")
print(tensor_3d.numpy())

# جمعکردن مقادیر در امتداد محور دوم (axis=2)
sum_axis_2 = tf.reduce_sum(tensor_3d, axis=2)
print("\n--- جمع مقادیر در امتداد axis=2 ---")
print(sum_axis_2.numpy())

# --- ۶. تجزیه مقادیر ویژه (Eigen Decomposition) ---
# فقط برای ماتریسهای مربعی
eigen_values, eigen_vectors = tf.linalg.eigh(x_tensor)
print("\n--- مقادیر ویژه ---")
print(eigen_values.numpy())
print("\n--- بردارهای ویژه ---")
print(eigen_vectors.numpy())