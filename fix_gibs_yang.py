from qiime2 import Artifact

from table_info import CSVTable, BiomTable
from simple_dup_finder import find_duplicates, pairwise_pearson
from woltka_metadata import filter_and_sort_df
from matplotlib import pyplot as plt

# df = CSVTable("./dataset/biom/celeste_ecoli_kraken.tsv", sep="\t").load_dataframe()
# print(df)

zebra = CSVTable("./dataset/biom/gibsyang/14365_metagenomic_may.tsv", sep="\t").load_dataframe()
# zebra.hist(log=True)
# plt.show()
# zebra2 = zebra[zebra["coverage_ratio"] > 0.10]
# zebra2.hist(log=True)
# plt.show()

filtered_gotu = zebra[zebra["coverage_ratio"] > 0.10]["gotu"]
#
# print("Original Table")
# print(zebra.shape)
# print("After Zebra")
# print(filtered_gotu.shape)

df = BiomTable("./dataset/biom/gibsyang/table.filt.biom").load_dataframe()
print("initial")
print(df.shape)
df = df[filtered_gotu.values]
print("Zebra filtered")
print(df.shape)

# df_table = Artifact.import_data("FeatureTable[Frequency]", df)
# df_table.save("./dataset/biom/gibsyang/table.filt.zebra10.biom")
# This saves as a damn qza file and you have to manually unzip. sorry.  Bah.

df2 = BiomTable("./dataset/biom/gibsyang/table.filt.zebra10.biom").load_dataframe()
print(df2)
print((df - df2).sum().sum())

WOLTKA_METADATA_PATH = "./woltka_metadata.tsv"
woltka_meta_table = CSVTable(WOLTKA_METADATA_PATH, delimiter="\t")
woltka_meta_df = woltka_meta_table.load_dataframe()

print(woltka_meta_df.columns)

df = filter_and_sort_df(df, woltka_meta_df, "all", min_genus_count=0)
prefilter_df = df

#Option 1, pairwise pearson
pp_df, pp_sets = pairwise_pearson(df, thresh=.95)
print("PairwisePearson shape")
print(pp_df.shape)

#Option2 nameless hacky thing
final_cols = find_duplicates(df, thresh=.25, woltka_meta_df= woltka_meta_df)
df = df[final_cols]
print("Weird Hacky Thing.shape")
print(df.shape)

set1 = set(pp_df.columns)
set2 = set(df.columns)
print(len(set1.intersection(set2)))
print("pp only: ", set1.difference(set2))
print("hacky only", set2.difference(set1))

print("hacky thing merges 436135 into 438035")
plt.scatter(prefilter_df["G000436135"], prefilter_df["G000438035"])
plt.title("merged by approach 1 but not approach 2")
plt.show()

for rep in pp_sets:
    if "G000009825" in pp_sets[rep]:
        print("pairwise pearson merges G000009825 into", rep)
        plt.scatter(prefilter_df["G000009825"], prefilter_df[rep])
        plt.title("merged by approach 2 but not approach 1")
        plt.show()


# df_table = Artifact.import_data("FeatureTable[Frequency]", df)
# df_table.save("./dataset/biom/gibsyang/table.filt.zebra10.overlap.biom")
# This saves as a damn qza file and you have to manually unzip. sorry.  Bah.

# pp_df_table = Artifact.import_data("FeatureTable[Frequency]", pp_df)
# pp_df_table.save("./dataset/biom/gibsyang/table.filt.zebra10.pairwise_pearson95.biom")
# This saves as a damn qza file and you have to manually unzip. sorry.  Bah.

# df3 = BiomTable("./dataset/biom/gibsyang/table.filt.zebra10.pairwise_pearson95.biom").load_dataframe()
# print((pp_df - df3).sum().sum())
