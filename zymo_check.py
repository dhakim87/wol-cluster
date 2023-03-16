from simple_dup_finder import pairwise_pearson
from table_info import BiomTable, CSVTable
import pandas as pd
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

WOLTKA_METADATA_PATH = "./woltka_metadata.tsv"
woltka_meta_table = CSVTable(WOLTKA_METADATA_PATH, delimiter="\t")
woltka_meta_df = woltka_meta_table.load_dataframe()

BIOM_TABLE = [
    "./dataset/biom/zymo_mock_12201/108101_none.biom",
    "./dataset/biom/zymo_mock_12201/140346_none.biom",
    "./dataset/biom/zymo_mock_12201/140574_none.biom",
    "./dataset/biom/zymo_mock_12201/140585_none.biom"
]

if not isinstance(BIOM_TABLE, list):
    BIOM_TABLE = [BIOM_TABLE]

all_dfs = []
for biom_table in BIOM_TABLE:
    bt = BiomTable(biom_table)
    df = bt.load_dataframe()
    print(df.shape)
    print(df.sum(axis=1))
    print(df.sum(axis=1).median())
    all_dfs.append(df)

df = pd.concat(all_dfs).fillna(0)

zymo_meta = CSVTable("./dataset/biom/12201_20220510-095201.txt", sep='\t', index_col="sample_name").load_dataframe()
zymo_samples = zymo_meta.index[zymo_meta["sample_type_2"] == "ZymoBIOMICS Microbial Community Standard II"]
print(zymo_meta["sample_type_2"].value_counts())
# Restrict to zymo samples
df["sample_type_2"] = zymo_meta["sample_type_2"]
df = df[df["sample_type_2"] == "ZymoBIOMICS Microbial Community Standard II"]
df = df.drop(["sample_type_2"], axis=1)

print(df)

print(df.sum().max() / df.sum().sum())
print(df.columns[df.sum().argmax()])
plt.scatter(df["G000196035"], df["G000195995"])

plt.scatter(df["G000567965"], df["G000195995"])
plt.scatter(df["G000619785"], df["G000195995"])
plt.scatter(df["G000620885"], df["G000195995"])
plt.scatter(df["G000620905"], df["G000195995"])

plt.scatter(df["G000972995"], df["G000195995"])
plt.scatter(df["G000973005"], df["G000195995"])

plt.scatter(df["G001463995"], df["G000195995"])
plt.scatter(df["G001564695"], df["G000195995"])
plt.scatter(df["G001564995"], df["G000195995"])
plt.xlabel("Listeria monocytogenes")
plt.ylabel("Salmonella enterica")
plt.show()


df_sum = df.sum()
df_sum_sum = df.sum().sum()
expected_genera_sum = 0
for genus in [
    "Listeria", "Pseudomonas", "Bacillus", "Saccharomyces", "Escherichia",
    "Salmonella", "Lactobacillus", "Enterococcus", "Cryptococcus", "Staphylococcus"
]:
    # Listeria monocytogenes - 89.1%,
    # Pseudomonas aeruginosa - 8.9%,
    # Bacillus subtilis - 0.89%,
    # Saccharomyces cerevisiae - 0.89%,
    # Escherichia coli - 0.089%,
    # Salmonella enterica - 0.089%,
    # Lactobacillus fermentum - 0.0089%,
    # Enterococcus faecalis - 0.00089%,
    # Cryptococcus neoformans - 0.00089%,
    # Staphylococcus aureus - 0.000089%

    glist = list(woltka_meta_df[woltka_meta_df["genus"] == genus]["#genome"].values)
    gsum = df_sum[glist].sum() / df_sum_sum
    print(genus, gsum)
    expected_genera_sum += gsum

print(expected_genera_sum)

# Abundance Filter
total_sum = df.sum().sum()
columns_that_pass = df.sum() > total_sum * 1/10000
print("Pre Abundance Filter", df.shape)
df = df[columns_that_pass[columns_that_pass].index]
print("post abundance filter", df.shape)


# Pairwise Pearson
col_sums = df.sum()
col_sums.name = 'total'
col_sums = col_sums.sort_values(ascending=False)
df = df[col_sums.index]
pre_pp = df
df, sets = pairwise_pearson(df, .99)
print("post pairwise pearson", df.shape)
print(df)

print("HELLO")
merged_into_listeria = []
for m in sets["G000619785"]:
    species = woltka_meta_df[woltka_meta_df["#genome"] == m]["species"].iloc[0]
    merged_into_listeria.append((m, species))
print(merged_into_listeria)
plt.scatter(pre_pp["G000619785"], pre_pp["G900089455"])
plt.show()

final_counts = df.sum() / df.sum().sum()
print(final_counts)


# Target (if same as Microbial Standard II, which it might not be...):
# Listeria monocytogenes - 89.1%,
# Pseudomonas aeruginosa - 8.9%,
# Bacillus subtilis - 0.89%,
# Saccharomyces cerevisiae - 0.89%,
# Escherichia coli - 0.089%,
# Salmonella enterica - 0.089%,
# Lactobacillus fermentum - 0.0089%,
# Enterococcus faecalis - 0.00089%,
# Cryptococcus neoformans - 0.00089%,
# Staphylococcus aureus - 0.000089%.

for idx, val in final_counts.items():
    species = woltka_meta_df[woltka_meta_df["#genome"] == idx]["species"].iloc[0]
    print(idx, species, val)

# for idx1, val in final_counts.items():
#     species1 = woltka_meta_df[woltka_meta_df["#genome"] == idx1]["species"].iloc[0]
#     for idx2, val in final_counts.items():
#         species2 = woltka_meta_df[woltka_meta_df["#genome"] == idx2]["species"].iloc[0]
#         plt.scatter(df[idx1], df[idx2])
#         plt.xlabel(species1)
#         plt.ylabel(species2)
#         plt.show()


# plt.scatter(df["G000195995"], df["G000740655"])
# plt.xlabel("Salmonella")
# plt.ylabel("Bacillus")
# plt.show()
# plt.scatter(df["G000195995"], df["G000159215"])
# plt.xlabel("Salmonella")
# plt.ylabel("Lactobacillus")
# plt.show()
# plt.scatter(df["G000740655"], df["G000159215"])
# plt.xlabel("Bacillus")
# plt.ylabel("Lactobacillus")
# plt.show()


# Borelia seems to just be trash in a single sample, maybe prevalence filter would be appropriate
# Kluyvera and Klebsiella would be merged by the technique except for a single outlier sample.
# plt.scatter(df["G001022135"], df["G001022195"])
# plt.xlabel("Kluyvera")
# plt.ylabel("Klebsiella")
# plt.scatter(df["G001022135"], df["G000568755"])
# plt.xlabel("Kluyvera")
# plt.ylabel("Borrelia")
# plt.scatter(df["G001022195"], df["G000568755"])
# plt.xlabel("Klebsiella")
# plt.ylabel("Borrelia")
# plt.show()
#
# plt.scatter(df["G000568755"], df["G000705255"])
# plt.xlabel("Staphylococcus")
# plt.ylabel("Campylobacter")
# plt.show()
