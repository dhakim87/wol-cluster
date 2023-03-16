import pandas as pd
from biom import load_table
from matplotlib.patches import Rectangle
from qiime2 import Artifact

import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import seaborn as sns


# Helper Functions
def load_biom(f):
    biom_table = load_table(f)
    table = Artifact.import_data("FeatureTable[Frequency]", biom_table)
    return table.view(pd.DataFrame)

def get_genus(woltka_metadata, gotu):
    genus = woltka_metadata[woltka_metadata["#genome"] == gotu]["genus"].iloc[0]
    if not isinstance(genus, str):
        # What...
        genus = "Unknown"
    return genus

def get_species(woltka_metadata, gotu):
    return woltka_metadata[woltka_metadata["#genome"] == gotu]["species"].iloc[0]

def sort_by_genus(woltka_metadata, df):
    mapper = {}
    for col in df.columns:
        g = get_genus(woltka_metadata, col)
        mapper[col] = g
    cols_sorted = sorted(df.columns, key=lambda x: mapper[x])
    return df[cols_sorted]

# Data Locations
ZEBRA_DATA_LOCATION = "./dataset/biom/gibsyang/14365_metagenomic_may.tsv"
ABUNDANCE_FILTERED_DF_LOCATION = "./dataset/biom/gibsyang/table.filt.biom"
ZEBRA_FILTERED_DF_LOCATION = "./dataset/biom/gibsyang/table.filt.zebra10.biom"
PAIRWISE_PEARSON_FILTERED_DF_LOCATION = "./dataset/biom/gibsyang/table.filt.zebra10.pairwise_pearson95.biom"
WOLTKA_METADATA_LOCATION = "./woltka_metadata.tsv"

zebra_genome_cover = pd.read_csv(ZEBRA_DATA_LOCATION, sep="\t")
woltka_metadata = pd.read_csv(WOLTKA_METADATA_LOCATION, sep="\t")

abundance_df = load_biom(ABUNDANCE_FILTERED_DF_LOCATION)
df_frac = abundance_df.sum() / abundance_df.sum().sum()
zebra_df = load_biom(ZEBRA_FILTERED_DF_LOCATION)
pp_df = load_biom(PAIRWISE_PEARSON_FILTERED_DF_LOCATION)

# FIGURE S1: ZEBRA
# Zebra histograms - These are kind of terrible, but we can show if we want...
# zebra_genome_cover.hist(log=True)
# plt.show()
# zebra_genome_cover[zebra_genome_cover["coverage_ratio"] > 0.10].hist(log=True)
# plt.show()
# Zebra Num Features At Various Thresholds Scatter
xs = []
ys = []
for thresh in range(1, 100):  # Note that we skip a threshold of 0 (which gives 6000+ features) to avoid blowing out the graph
    x = thresh * 1/100
    features_passing = (zebra_genome_cover["coverage_ratio"] > thresh * 1/100).sum()
    xs.append(x)
    ys.append(features_passing)

fig = plt.figure(figsize=(6, 4), dpi=300)
plt.plot(xs,ys)
plt.axvline(0.1, linestyle=":", c="gray")
plt.axvline(0.25, linestyle=":", c="gray", label="Zebra Recommended Genome Cover Range")
plt.scatter([0.10], [len(zebra_df.columns)], color="r", label="Chosen Threshold")
plt.xlabel("Zebra Genome Cover Threshold")
plt.ylabel("#GOTUs Passing Zebra Filter")
plt.legend()
plt.savefig("figs/gallo_skin_zebra_panel1.png")
plt.show()


# FIGURE S2: PAIRWISE PEARSON
# Correlation Heatmaps
# After Zebra, Before Pairwise Pearson
fig = plt.figure(figsize=(12, 10), dpi=300) # Gotta tweak fontsize if you touch dpi
zebra_df_sorted = sort_by_genus(woltka_metadata, zebra_df)
ax = sns.heatmap(zebra_df_sorted.corr())
ticks = [x + 0.5 for x in range(len(zebra_df_sorted.columns))]
labels = [get_genus(woltka_metadata, gotu) for gotu in zebra_df_sorted.columns]
plt.xticks(ticks=ticks, labels=labels, fontsize=6) # Gotta tweak fontsize if you touch dpi
plt.yticks(ticks=ticks, labels=labels, fontsize=6) # Gotta tweak fontsize if you touch dpi
plt.title("GOTU Pearson Correlation")
first_ecoli = labels.index("Escherichia")
last_ecoli = len(labels) - labels[::-1].index("Escherichia")
first_shigella = labels.index("Shigella")
last_shigella = len(labels) - labels[::-1].index("Shigella")

# Mark lower left
ax.add_patch(Rectangle((first_ecoli, first_shigella), last_ecoli-first_ecoli, last_shigella-first_shigella, edgecolor='blue', fill=False, lw=3))
# Mark upper right
ax.add_patch(Rectangle((first_shigella, first_ecoli), last_shigella-first_shigella, last_ecoli-first_ecoli, edgecolor='blue', fill=False, lw=3))
plt.savefig("figs/gallo_skin_pearson_panel1.png")
plt.show()


# After Pairwise Pearson
fig = plt.figure(figsize=(12, 10), dpi=300) # Gotta tweak fontsize if you touch dpi
pp_df_sorted = sort_by_genus(woltka_metadata, pp_df)
sns.heatmap(pp_df_sorted.corr())
ticks = [x + 0.5 for x in range(len(pp_df_sorted.columns))]
ylabels = [get_species(woltka_metadata, gotu) for gotu in pp_df_sorted.columns]
xlabels = [get_genus(woltka_metadata, gotu) for gotu in pp_df_sorted.columns]
plt.xticks(ticks=ticks, labels=xlabels, fontsize=6) # Gotta tweak fontsize if you touch dpi
plt.yticks(ticks=ticks, labels=ylabels, fontsize=6) # Gotta tweak fontsize if you touch dpi
plt.title("Merged GOTU Pearson Correlation")
plt.subplots_adjust(left=0.2, bottom=.15)
plt.savefig("figs/gallo_skin_pearson_panel2.png")
plt.show()
