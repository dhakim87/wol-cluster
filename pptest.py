from qiime2 import Artifact

from table_info import CSVTable, BiomTable
from simple_dup_finder import pairwise_pearson
from woltka_metadata import filter_and_sort_df
from matplotlib import pyplot as plt

df = BiomTable("./dataset/biom/gibsyang/table.filt.biom").load_dataframe()
print("initial")
print(df)

WOLTKA_METADATA_PATH="./woltka_metadata.tsv"
woltka_meta_table = CSVTable(WOLTKA_METADATA_PATH, delimiter="\t")
woltka_meta_df = woltka_meta_table.load_dataframe()
print(woltka_meta_df.columns)
df = filter_and_sort_df(df, woltka_meta_df, "all", min_genus_count=0)

pp_df, pp_sets = pairwise_pearson(df, thresh=0.85)

print(pp_df)

name1 = ""
name2 = ""
for rep in pp_sets:
    i = 0
    fig = plt.figure(figsize=(12, 12))
    name1 = str(woltka_meta_df[woltka_meta_df["#genome"] == rep]["genus"].iloc[0]) + "\n" + rep
    for reflection in pp_sets[rep]:
        if reflection == rep:
            continue
        name2 = str(woltka_meta_df[woltka_meta_df["#genome"] == reflection]["genus"].iloc[0]) + "\n" + reflection
        i += 1
        ax = fig.add_subplot(4, 3, i)
        ax.scatter(x=list(df[rep].values), y=list(df[reflection].values), s=2)
        # ax.set_yticks([])
        ax.set_ylabel(name2)
        if i < 10:
            ax.set_xticks([])
        if i == 12:
            i = 0
            plt.suptitle(name1)
            fig.tight_layout()
            plt.show()
            fig = plt.figure(figsize=(12,12))
    if i != 0:
        plt.suptitle(name1)
        fig.tight_layout()
        plt.show()