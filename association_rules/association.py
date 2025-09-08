import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

# --- ۱. تنظیمات پارامترها ---
MIN_SUPPORT = 0.03  # کاهش حداقل پشتیبان برای قوانین چندتایی
MIN_CONFIDENCE = 0.2
MAX_LEN = 3  # حداکثر طول مجموعه آیتمها

# --- ۲. خواندن دادهها ---
df = pd.read_csv("movies.csv")
transactions = df["genres"].str.split("|").tolist()

# --- ۳. تبدیل دادهها ---
te = TransactionEncoder()
df_encoded = pd.DataFrame(te.fit_transform(transactions), columns=te.columns_)

# --- ۴. استخراج مجموعهآیتمهای مکرر ---
frequent_itemsets = fpgrowth(df_encoded, min_support=MIN_SUPPORT, max_len=MAX_LEN, use_colnames=True)

# --- ۵. استخراج قوانین چندتایی ---
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=MIN_CONFIDENCE)
rules["antecedent_len"] = rules["antecedents"].apply(lambda x: len(x))
rules["consequent_len"] = rules["consequents"].apply(lambda x: len(x))

# --- ۶. فیلتر قوانین چندتایی ---
multi_rules = rules[(rules["antecedent_len"] > 1) | (rules["consequent_len"] > 1)]

# --- ۷. نمایش نتایج ---
print("\n--- قوانین چندتایی ---")
print(multi_rules[["antecedents", "consequents", "support", "confidence", "lift"]])

# --- ۸. رسم گراف با NetworkX ---
# --- ۸. رسم گراف با NetworkX (اصلاحشده) ---
plt.figure(figsize=(15, 10))
G = nx.DiGraph()

# اضافه کردن گرهها و یالها
for _, row in multi_rules.iterrows():
    antecedents = ", ".join(list(row["antecedents"]))
    consequents = ", ".join(list(row["consequents"]))
    G.add_edge(antecedents, consequents, weight=row["confidence"], lift=row["lift"])

# تنظیمات گراف
pos = nx.spring_layout(G, k=0.5)
edges = G.edges(data=True)

# استخراج دادههای لیفت برای نرمالیزه کردن رنگها
lift_values = [edge[2]["lift"] for edge in edges]
vmin, vmax = min(lift_values), max(lift_values)

# ایجاد colormap و norm
cmap = plt.cm.Reds
norm = plt.Normalize(vmin=vmin, vmax=vmax)

# رسم گراف
fig, ax = plt.subplots(figsize=(15, 10))  # ایجاد محور جدید
nx.draw_networkx_nodes(G, pos, node_size=3000, node_color="lightblue", ax=ax)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight="bold", ax=ax)
edges_draw = nx.draw_networkx_edges(
    G,
    pos,
    width=[edge[2]["weight"] * 5 for edge in edges],
    edge_color=lift_values,
    edge_cmap=cmap,
    edge_vmin=vmin,
    edge_vmax=vmax,
    ax=ax,
)

# اضافه کردن colorbar
plt.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    label="Lift",
    ax=ax,  # مشخص کردن محور مربوطه
)

plt.title("گراف قوانین چندتایی (رنگ یالها: Lift، ضخامت یالها: Confidence)", fontsize=14)
plt.axis("off")
plt.show()


# --- ۹. تابع پیشبینی بر اساس قوانین ---
def predict_genre(input_genres, rules):
    """
    پیشبینی ژانرهای مرتبط بر اساس قوانین انجمنی
    Args:
        input_genres (list): لیست ژانرهای ورودی (مثلاً ["Comedy", "Drama"])
        rules (pd.DataFrame): جدول قوانین استخراجشده
    Returns:
        pd.DataFrame: قوانین مرتبط و پیشبینیها
    """
    input_set = set(input_genres)
    predictions = []

    for _, rule in rules.iterrows():
        antecedent = set(rule["antecedents"])
        consequent = set(rule["consequents"])

        # بررسی تطابق مقدم قانون با ورودی
        if antecedent.issubset(input_set):
            predictions.append({
                "مقدم": antecedent,
                "تالی": consequent,
                "پشتیبان": rule["support"],
                "اعتماد": rule["confidence"],
                "لیفت": rule["lift"]
            })

    if not predictions:
        return "هیچ قانونی برای این ورودی یافت نشد."

    return pd.DataFrame(predictions).sort_values("لیفت", ascending=False)


# --- ۱۰. مثال پیشبینی ---
sample_input = ["Romance"]
print("\n--- پیشبینی برای ورودی:", sample_input, "---")
predictions_df = predict_genre(sample_input, multi_rules)
print(predictions_df)