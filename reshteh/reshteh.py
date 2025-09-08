import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import os

# 1. بارگیری داده‌ها
def load_data(file_paths):
    all_data = []
    for file_path in file_paths:
        data = pd.read_excel(file_path, engine='openpyxl')
        all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

# 2. پیش‌پردازش داده‌ها
def preprocess_data(data):
    # حذف ستون‌های توصیفی
    description_columns = [col for col in data.columns if "توصیف" in col]
    data = data.drop(columns=description_columns)

    # جایگزینی عبارت "غایب" با صفر
    data = data.replace("غایب", 0)

    # تبدیل تمام مقادیر به عدد
    for col in data.columns[2:]:
        data[col] = pd.to_numeric(data[col], errors='coerce')

    # تبدیل NaN‌ها به صفر
    data = data.fillna(0)

    return data

# 3. محاسبه تغییرات نمرات
def calculate_changes(grouped_data):
    subjects = [
        'قرآن', 'حیات طیبه', 'مطالعات اجتماعی', 'ادبیات فارسی', 'آمادگی', 'الگوی من',
        'زبان خارجه', 'ریاضی', 'نویسندگی', 'فرهنگ و هنر', 'عربی', 'ورزش', 'کارگاه رایانه', 'علوم'
    ]

    def calculate_differences(row):
        scores = row[subjects]
        differences = []
        for score in scores:
            # اگر فقط یک نمره وجود داشته باشد، تغییرات صفر است
            differences.append([0, 0, 0, 0])
        return differences

    diff_df = grouped_data.apply(calculate_differences, axis=1, result_type='expand')
    diff_df.columns = subjects
    grouped_data = pd.concat([grouped_data[['نام', 'نام خانوادگی']], diff_df], axis=1)
    return grouped_data

# 4. محاسبه ویژگی‌های اضافی
def calculate_additional_features(grouped_data):
    subjects = [
        'قرآن', 'حیات طیبه', 'مطالعات اجتماعی', 'ادبیات فارسی', 'آمادگی', 'الگوی من',
        'زبان خارجه', 'ریاضی', 'نویسندگی', 'فرهنگ و هنر', 'عربی', 'ورزش', 'کارگاه رایانه', 'علوم'
    ]

    grouped_data['میانگین نمرات'] = grouped_data[subjects].mean(axis=1)
    grouped_data['پراکندگی نمرات'] = grouped_data[subjects].var(axis=1)
    grouped_data['حداکثر نمرات'] = grouped_data[subjects].max(axis=1)
    grouped_data['حداقل نمرات'] = grouped_data[subjects].min(axis=1)
    return grouped_data

# 5. تهیه ویژگی‌ها
def prepare_features(grouped_data):
    features = []
    subjects = [
        'قرآن', 'حیات طیبه', 'مطالعات اجتماعی', 'ادبیات فارسی', 'آمادگی', 'الگوی من',
        'زبان خارجه', 'ریاضی', 'نویسندگی', 'فرهنگ و هنر', 'عربی', 'ورزش', 'کارگاه رایانه', 'علوم'
    ]
    for _, row in grouped_data.iterrows():
        feature_row = [
            row['میانگین نمرات'],
            row['پراکندگی نمرات'],
            row['حداکثر نمرات'],
            row['حداقل نمرات']
        ]
        features.append(feature_row)
    return np.array(features)

# 6. خوشه‌بندی
def cluster_students(features):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    total_variance = np.var(features_scaled, axis=0).sum()
    if total_variance == 0:
        print("هشدار: واریانس داده‌ها صفر است. خوشه‌بندی امکان‌پذیر نیست.")
        return np.zeros(len(features), dtype=int)

    agg_clustering = AgglomerativeClustering(n_clusters=4, linkage='ward')
    labels = agg_clustering.fit_predict(features_scaled)

    print(f"تعداد خوشه‌ها: {len(set(labels))}")
    return labels

# 7. تشخیص رشته‌های مناسب
def recommend_fields(labels, grouped_data):
    subjects = [
        'قرآن', 'حیات طیبه', 'مطالعات اجتماعی', 'ادبیات فارسی', 'آمادگی', 'الگوی من',
        'زبان خارجه', 'ریاضی', 'نویسندگی', 'فرهنگ و هنر', 'عربی', 'ورزش', 'کارگاه رایانه', 'علوم'
    ]
    fields = {
        "علمی": ["ریاضی", "علوم", "زبان خارجه"],
        "انسانی": ["ادبیات فارسی", "مطالعات اجتماعی", "فرهنگ و هنر"],
        "فنی": ["کارگاه رایانه", "الگوی من", "آمادگی"],
        "ورزشی": ["ورزش"]
    }

    recommendations = []
    for idx, label in enumerate(labels):
        scores = {}
        for field, subjects_in_field in fields.items():
            score = sum(
                grouped_data.loc[idx, subj] if subj in grouped_data.columns else 0
                for subj in subjects_in_field
            )
            scores[field] = score
        best_field = max(scores, key=scores.get)
        recommendations.append(best_field)
    return recommendations

# 8. ذخیره نتایج
def save_results(recommendations, grouped_data, output_file):
    result_df = grouped_data[["نام", "نام خانوادگی"]].copy()
    result_df.loc[:, "رشته‌های پیشنهادی"] = recommendations
    result_df.to_excel(output_file, index=False)

# اجرای کامل
if __name__ == "__main__":
    file_paths = [
        "Book1.xlsx",
        "Book2.xlsx",
        "Book3.xlsx",
        "Book4.xlsx",
        "Book5.xlsx",
    ]

    # 1. بارگیری داده‌ها
    data = load_data(file_paths)

    # 2. پیش‌پردازش داده‌ها
    data = preprocess_data(data)

    # 3. محاسبه تغییرات نمرات
    grouped_data = calculate_changes(data)

    # 4. محاسبه ویژگی‌های اضافی
    grouped_data = calculate_additional_features(grouped_data)

    # 5. تهیه ویژگی‌ها
    features = prepare_features(grouped_data)

    # 6. خوشه‌بندی
    labels = cluster_students(features)

    # 7. تشخیص رشته‌های مناسب
    recommendations = recommend_fields(labels, grouped_data)

    # 8. ذخیره نتایج
    save_results(recommendations, grouped_data, "recommendations.xlsx")