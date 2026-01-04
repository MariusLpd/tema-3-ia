import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, association_rules

warnings.filterwarnings("ignore")
pd.set_option("display.max_colwidth", 80)

cons = pd.read_csv("consumption_user.csv", encoding="latin1", low_memory=False)
subjects = pd.read_csv("subject_user.csv", encoding="latin1", low_memory=False)

cons.columns = cons.columns.str.replace("ï»¿", "", regex=False)

cons = cons[cons["INGREDIENT_ENG"].notna()].copy()
cons["INGREDIENT_ENG"] = cons["INGREDIENT_ENG"].astype(str).str.strip()
cons = cons[cons["INGREDIENT_ENG"] != ""]

nr_persoane = cons["SUBJECT"].nunique()
nr_ingrediente = cons["INGREDIENT_ENG"].nunique()

summary_table = pd.DataFrame({
    "Indicator": ["Număr persoane", "Număr ingrediente"],
    "Valoare": [nr_persoane, nr_ingrediente]
})

top_ingrediente = (
    cons.groupby("INGREDIENT_ENG")
        .size()
        .sort_values(ascending=False)
        .head(10)
        .reset_index(name="Frequency")
)

ingrediente_per_persoana = (
    cons.groupby("SUBJECT")["INGREDIENT_ENG"]
        .nunique()
        .describe()
)

transactions = (
    cons.groupby("SUBJECT")["INGREDIENT_ENG"]
        .apply(list)
)

min_user_support = 0.02

users_per_item = cons.groupby("INGREDIENT_ENG")["SUBJECT"].nunique()
threshold = int(nr_persoane * min_user_support)
common_items = users_per_item[users_per_item >= threshold].index

cons_small = cons[cons["INGREDIENT_ENG"].isin(common_items)].copy()

transactions_small = (
    cons_small.groupby("SUBJECT")["INGREDIENT_ENG"]
        .apply(lambda x: list(set(x)))
)

te = TransactionEncoder()
df_bin = pd.DataFrame(
    te.fit(transactions_small).transform(transactions_small),
    columns=te.columns_
)

min_support_apriori = 0.08
min_conf = 0.6

df_bin_small = df_bin.sample(
    n=min(2000, len(df_bin)),
    random_state=42
)

start = time.time()

freq_apriori = apriori(
    df_bin_small,
    min_support=min_support_apriori,
    use_colnames=True,
    max_len=2
)

rules_apriori = association_rules(
    freq_apriori,
    metric="confidence",
    min_threshold=min_conf
)

apriori_time = time.time() - start

subjects_small = cons["SUBJECT"].drop_duplicates().head(200)
cons_fp = cons[cons["SUBJECT"].isin(subjects_small)].copy()

top_items = (
    cons_fp.groupby("INGREDIENT_ENG")["SUBJECT"]
        .nunique()
        .sort_values(ascending=False)
        .head(25)
        .index
)

cons_fp = cons_fp[cons_fp["INGREDIENT_ENG"].isin(top_items)]

transactions_fp = (
    cons_fp.groupby("SUBJECT")["INGREDIENT_ENG"]
        .apply(lambda x: list(set(x)))
)

df_bin_tiny = pd.DataFrame(
    te.fit(transactions_fp).transform(transactions_fp),
    columns=te.columns_
)

start = time.time()

freq_fp = fpgrowth(
    df_bin_tiny,
    min_support=0.2,
    use_colnames=True,
    max_len=2
)

rules_fp = association_rules(
    freq_fp,
    metric="confidence",
    min_threshold=0.6
)

fp_time = time.time() - start

comparison = pd.DataFrame({
    "Algorithm": ["Apriori", "FP-Growth"],
    "Frequent itemsets": [len(freq_apriori), len(freq_fp)],
    "Association rules": [len(rules_apriori), len(rules_fp)],
    "Execution time (sec)": [apriori_time, fp_time]
})

rules_fp_best = rules_fp.sort_values(
    by=["lift", "confidence"],
    ascending=False
)

sex_map = {1: "Male", 2: "Female", "1": "Male", "2": "Female"}
sex_labels = subjects["SEX"].map(sex_map).fillna(subjects["SEX"].astype(str))
sex_dist = sex_labels.value_counts()

plt.figure()
plt.bar(sex_dist.index, sex_dist.values)
plt.xlabel("Sex")
plt.ylabel("Number of respondents")
plt.title("Distribution of respondents by sex")
plt.show()

top20 = (
    cons.groupby("INGREDIENT_ENG")
        .size()
        .sort_values(ascending=False)
        .head(20)
)

plt.figure(figsize=(6, 6))
top20.sort_values().plot(kind="barh")
plt.xlabel("Frequency")
plt.ylabel("Ingredient")
plt.title("Top 20 most frequent food ingredients")
plt.show()

age_labels = {
    0: "0–4",
    1: "5–9",
    2: "10–19",
    3: "20–39",
    4: "40–59",
    5: "60+"
}

age_cat = subjects["AGE_YEAR"].map(age_labels)
age_dist = age_cat.value_counts().sort_index()

plt.figure()
plt.bar(age_dist.index, age_dist.values)
plt.xlabel("Age group (years)")
plt.ylabel("Frequency")
plt.title("Age group distribution of respondents")
plt.show()
