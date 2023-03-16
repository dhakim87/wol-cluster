from qiime2 import Artifact

from table_info import CSVTable, BiomTable
from simple_dup_finder import find_duplicates, pairwise_pearson
from woltka_metadata import filter_and_sort_df
from matplotlib import pyplot as plt

import seaborn as sns


def preprocess(df):
    row_sums = df.sum(axis=1)
    df = (df.T / row_sums).T
    return df

WOLTKA_METADATA_PATH = "./woltka_metadata.tsv"
woltka_meta_table = CSVTable(WOLTKA_METADATA_PATH, delimiter="\t")
woltka_meta_df = woltka_meta_table.load_dataframe()

print(woltka_meta_df.columns)
methanobrevibacter = woltka_meta_df[woltka_meta_df["genus"] == "Akkermansia"][["#genome", "species"]]
print(methanobrevibacter)

# Skip gotus that aren't in all datasets.
methanobrevibacter = methanobrevibacter[~methanobrevibacter["#genome"].isin(["G000621965"])]
df_imsms = preprocess(BiomTable("./dataset/biom/imsms-combined-none.biom").load_dataframe())
df_finrisk = preprocess(BiomTable("./dataset/biom/finrisk-combined-none.biom").load_dataframe())
df_sol = preprocess(BiomTable("./dataset/biom/sol_public_99006-none.biom").load_dataframe())


for i, r1 in methanobrevibacter.iterrows():
    for j, r2 in methanobrevibacter.iterrows():
        break
        g1 = r1["#genome"]
        g2 = r2["#genome"]
        if g1 == g2:
            continue
        ax = plt.gca()
        plt.scatter(df_imsms[g1], df_imsms[g2], c='r', label="iMSMS")
        plt.scatter(df_sol[g1], df_sol[g2], c='b', label="SoL")
        plt.scatter(df_finrisk[g1], df_finrisk[g2], c='g', label="FINRISK")
        # ax.set_yscale('log')
        # ax.set_xscale('log')
        plt.xlabel(r1["species"] + "("+g1+")")
        plt.ylabel(r2["species"] + "("+g2+")")
        plt.axvline(0, linestyle=":")
        plt.axhline(0, linestyle=":")
        plt.axis("equal")
        plt.legend()
        plt.show()


def get_genus(woltka_metadata, gotu):
    genus = woltka_metadata[woltka_metadata["#genome"] == gotu]["genus"].iloc[0]
    if not isinstance(genus, str):
        # What...
        genus = "Unknown"
    return genus

def get_species(woltka_metadata, gotu):
    return woltka_metadata[woltka_metadata["#genome"] == gotu]["species"].iloc[0]


fig = plt.figure(figsize=(12, 10), dpi=300) # Gotta tweak fontsize if you touch dpi
df_imsms = df_imsms[methanobrevibacter["#genome"]]
sns.heatmap(df_imsms.corr())
ticks = [x + 0.5 for x in range(len(df_imsms.columns))]
ylabels = [get_species(woltka_meta_df, gotu) for gotu in df_imsms.columns]
xlabels = [get_genus(woltka_meta_df, gotu) for gotu in df_imsms.columns]
plt.xticks(ticks=ticks, labels=xlabels, fontsize=6) # Gotta tweak fontsize if you touch dpi
plt.yticks(ticks=ticks, labels=ylabels, fontsize=6) # Gotta tweak fontsize if you touch dpi
plt.title("Merged GOTU Pearson Correlation")
plt.subplots_adjust(left=0.2, bottom=.15)
plt.savefig("figs/Akkermansia.png")
plt.show()

