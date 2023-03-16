import pandas as pd
from table_info import BiomTable
from woltka_metadata import filter_and_sort_df
from matplotlib import pyplot as plt

# iMSMS
ZEBRA_DATA_LOCATION = "/Users/djhakim/imsms-ml/zebra.csv"
ZEBRA_SEP = ","
DF_LOCATION = "./dataset/biom/imsms-combined-none.biom"
WOLTKA_METADATA_LOCATION = "./woltka_metadata.tsv"

def get_species(woltka_metadata, gotu):
    return woltka_metadata[woltka_metadata["#genome"] == gotu]["species"].iloc[0]


# Data Loading
zebra_genome_cover = pd.read_csv(ZEBRA_DATA_LOCATION, sep=ZEBRA_SEP)
woltka_metadata = pd.read_csv(WOLTKA_METADATA_LOCATION, sep="\t")

df = BiomTable(DF_LOCATION).load_dataframe()


def _parse_sample_id(index: str):
    # print(index)
    # Input of form Q.71401.0009.2016.02.23
    # Output of form 71401-0009
    ss = index.split('.')
    if len(ss) < 3:
        return "skip"

    sample_id = ss[1] + "-" + ss[2]
    # print("GOOD: ", index, "->", sample_id)
    return sample_id


df.index = df.index.map(_parse_sample_id)
df = df.drop(["skip"])
df = filter_and_sort_df(df, woltka_meta_df=woltka_metadata, genus="Akkermansia")
print(df)

clade_muciniphila = [
    "G000020225",
    "G000436395",
    "G000723745",
    # "G000980515", # Tracked samples don't seem to contain much of these, removing from clade to focus graphs
    # "G001917295",
    # "G001940945"
]

clade_344 = [
    "G000020225", # Not in the 344 clade, just need something to display against
    "G000437075"
]

clade_KLE = [
    "G001578645",
    "G001580195",
    "G001647615"
]

clade_glycaniphila = [
    "G001683795",
    "G900097105"
]

tracked_samples = [
    # "71801-0015",
    # "71801-0038",
    # "71801-0041",
    "71801-0051"
]

df_tracked = df.loc[df.index.map(lambda s: s.startswith('718')), :]
# df_tracked = df.loc[tracked_samples, :]
df_untracked = df.drop(df_tracked.index)

for clade, name in [(clade_muciniphila, "Muciniphila clade"), (clade_344, "Akkermansia sp. CAG 344"), (clade_KLE, "KLE clade")]:
    fig, axes = plt.subplots(len(clade)-1, len(clade)-1, sharex=True, sharey = True)

    if len(clade) - 1 > 1:
        for i in range(len(clade)-1):
            for j in range(1, i+1):
                print(j-1, i)
                ax = axes[j-1, i]
                fig.delaxes(ax)

    for i in range(len(clade)):
        for j in range(i+1, len(clade)):
            if len(clade) - 1 == 1:
                ax = axes
            else:
                ax = axes[j-1, i]
            x = clade[i]
            y = clade[j]

            ax.scatter(df_untracked[x], df_untracked[y])
            ax.scatter(df_tracked[x], df_tracked[y], c="orange")
            for label in df_tracked.index:
                sample_set = df_tracked.loc[label]
                for k in range(sample_set.shape[0]):
                    if isinstance(sample_set, pd.DataFrame):
                        label2 = label + "_" + str(k)
                        ax.annotate(label2, (sample_set[x][k], sample_set[y][k]))
                    else:
                        ax.annotate(label, (sample_set[x], sample_set[y]))

            if j == len(clade) - 1:
                ax.set_xlabel(get_species(woltka_metadata, x) + "\n" + x)
            if i == 0:
                ax.set_ylabel(get_species(woltka_metadata, y) + "\n" + y)

    plt.suptitle(name)
    plt.tight_layout()
    plt.savefig("/Users/djhakim/jorge_plots/718-" + name + ".png")
    plt.show()

