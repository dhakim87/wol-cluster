import pandas as pd
from biom import load_table
from matplotlib.patches import Rectangle
from qiime2 import Artifact
import numpy as np

import matplotlib

from simple_dup_finder import pairwise_pearson
from table_info import BiomTable

matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

import seaborn as sns
from matplotlib_venn import venn2, venn3, venn3_circles

import sklearn.decomposition
from skbio.stats.composition import clr
from scipy.spatial.distance import pdist
# from skbio.diversity.beta import pw_distances
import skbio.stats.ordination

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
    return woltka_metadata[woltka_metadata["#genome"] == gotu]["species"].iloc[0] + "\n(" + str(gotu) + ")"

def sort_by_genus(woltka_metadata, df):
    mapper = {}
    for col in df.columns:
        g = get_genus(woltka_metadata, col)
        mapper[col] = g
    cols_sorted = sorted(df.columns, key=lambda x: mapper[x])
    return df[cols_sorted]

def build_venn2(sets, a, b, title):
    A = len(sets[a])
    B = len(sets[b])
    AB = len(sets[a].intersection(sets[b]))
    A -= AB
    B -= AB

    plt.figure(figsize=(6,4))
    # A B AB, C, AC, BC, ABC
    v = venn2(subsets=(1,1,1,1,1,1,1), set_labels = (a,b))
    v.get_label_by_id('10').set_text(str(A))
    v.get_label_by_id('01').set_text(str(B))
    v.get_label_by_id('11').set_text(str(AB))
    plt.title(title)
    plt.show()

def build_venn3(sets, a, b, c, title):
    A = len(sets[a])
    B = len(sets[b])
    C = len(sets[c])
    AB = len(sets[a].intersection(sets[b]))
    AC = len(sets[a].intersection(sets[c]))
    BC = len(sets[b].intersection(sets[c]))
    ABC = len(sets[a].intersection(sets[b]).intersection(sets[c]))
    AB -= ABC
    AC -= ABC
    BC -= ABC
    A -= AB + AC + ABC
    B -= AB + BC + ABC
    C -= AC + BC + ABC
    # print("What the hell gotu is that?")
    # print(sets[a].intersection(sets[c]).difference(sets[b]))

    plt.figure(figsize=(4,4))
    # A B AB, C, AC, BC, ABC
    v = venn3(subsets=(1,1,1,1,1,1,1), set_labels = (a,b,c))
    v.get_label_by_id('100').set_text(str(A))
    v.get_label_by_id('010').set_text(str(B))
    v.get_label_by_id('001').set_text(str(C))
    v.get_label_by_id('110').set_text(str(AB))
    v.get_label_by_id('101').set_text(str(AC))
    v.get_label_by_id('011').set_text(str(BC))
    v.get_label_by_id('111').set_text(str(ABC))
    # v.get_patch_by_id('100').set_alpha(1.0)
    # v.get_patch_by_id('100').set_color('white')
    # v.get_label_by_id('100').set_text('Unknown')
    # v.get_label_by_id('A').set_text('Set "A"')
    # c = venn3_circles(subsets=(1, 1, 1, 1, 1, 1, 1), linestyle='dashed')
    # c[0].set_lw(1.0)
    # c[0].set_ls('dotted')
    plt.title(title)
    # plt.annotate('Unknown set', xy=v.get_label_by_id('100').get_position() - np.array([0, 0.05]), xytext=(-70,-70),
    #              ha='center', textcoords='offset points', bbox=dict(boxstyle='round,pad=0.5', fc='gray', alpha=0.1),
    #              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.5',color='gray'))
    plt.show()

def id_to_color_imsms(idx):
    ss = idx.split(".")
    if ss[1].endswith("1"):
        return "r"
    elif ss[1].endswith("2"):
        return "g"
    return "gray"

def id_to_color_celeste(idx):
    v = idx.split(".")[1]
    if v.startswith("4"):
        return "r"
    elif v.startswith("2"):
        return "g"
    elif v.startswith("3"):
        return "orange"
    elif v.startswith("BLANK"):
        return "gray"


# Data Locations
# TCGA
# Argh, this has two zebra coverage files, merge them.
# zebra_genome_cover_rna = pd.read_csv("/Users/djhakim/tcga_analysis/cov_output_rna_seq.tsv", sep="\t", index_col="gotu")
# zebra_genome_cover_wgs = pd.read_csv("/Users/djhakim/tcga_analysis/cov_output_wgs.tsv", sep="\t", index_col="gotu")
# print(zebra_genome_cover_rna)
# merged = zebra_genome_cover_rna.merge(zebra_genome_cover_wgs, how="outer", suffixes=["_rna", "_wgs"], left_index=True, right_index=True)
# merged["coverage_ratio"] = merged[["coverage_ratio_rna", "coverage_ratio_wgs"]].max(axis=1)
# merged.to_csv("./dataset/tcga_zebra_merged.tsv", sep="\t")
# print(merged)

# iMSMS
# df = BiomTable([
#     "./dataset/biom/imsms-combined-none.biom"
# ]).load_dataframe()
# id_to_color = id_to_color_imsms

# Celeste E. coli samples
df = BiomTable([
    "dataset/biom/Celeste_Prep_1_1428_samples.biom",
    "dataset/biom/Celeste_Prep_2_672_samples.biom",
    "dataset/biom/Celeste_Prep_3_936_samples.biom",
    "dataset/biom/Celeste_Prep_4_792_samples.biom"
]).load_dataframe()
id_to_color = id_to_color_celeste


def plot_pca(df, pca=None, pca_x=0, pca_y=1):
    target = df.index.map(id_to_color).values
    if pca is None:
        pca = sklearn.decomposition.PCA(5)
        df_pca = pca.fit_transform(df)
    else:
        df_pca = pca.transform(df)
    plt.scatter(df_pca[:,pca_x], df_pca[:,pca_y], c=target, alpha=0.05)
    plt.show()
    return pca

# print("Good lord, Jaccard is terrible for distinguishing types of E. coli")
# distmat = pdist(df, 'jaccard')
# print(distmat)
# pcoa = skbio.stats.ordination.pcoa(distmat, number_of_dimensions=2)
# print(pcoa.samples)
# plt.scatter(pcoa.samples["PC1"], pcoa.samples["PC2"], c=df.index.map(id_to_color).values, alpha=0.05)
# plt.show()

def filter_gotus(df, woltka_metadata, do_abundance=True, do_pairwise_pearson=True):
    if do_abundance:
        # Filter GOTUs by abundance
        df_frac = df.sum() / df.sum().sum()
        passing_gotus = list(df_frac[df_frac > 1/10000].index)
        df = df[passing_gotus]

    if do_pairwise_pearson:
        # Collapse GOTUs with pairwise pearson
        col_sums = df.sum()
        col_sums = col_sums.sort_values(ascending=False)
        df = df[col_sums.index]
        df, sets = pairwise_pearson(df, thresh=0.95)

        fig = plt.figure(figsize=(12, 10), dpi=300) # Gotta tweak fontsize if you touch dpi
        pp_df_sorted = sort_by_genus(woltka_metadata, df)
        sns.heatmap(pp_df_sorted.corr())
        ticks = [x + 0.5 for x in range(len(pp_df_sorted.columns))]
        ylabels = [get_species(woltka_metadata, gotu) for gotu in pp_df_sorted.columns]
        xlabels = [get_genus(woltka_metadata, gotu) for gotu in pp_df_sorted.columns]
        plt.xticks(ticks=ticks, labels=xlabels, fontsize=6) # Gotta tweak fontsize if you touch dpi
        plt.yticks(ticks=ticks, labels=ylabels, fontsize=6) # Gotta tweak fontsize if you touch dpi
        plt.title("Merged GOTU Pearson Correlation")
        plt.subplots_adjust(left=0.2, bottom=.15)
        plt.show()
    return df


woltka_metadata = pd.read_csv("./woltka_metadata.tsv", sep="\t")
df_ab = filter_gotus(df, woltka_metadata, True, False)
df_ab_pp = filter_gotus(df, woltka_metadata, True, True)

# for c in df_ab_pp.columns:
#     plt.scatter(df["G000183345"], df[c], c=df.index.map(id_to_color).values)
#     plt.xlabel(get_species(woltka_metadata, "G000183345"))
#     plt.ylabel(get_species(woltka_metadata, c))
#     plt.show()

df_ab_pp_decontam = df_ab_pp[[
    "G000025565",
    "G000183345",
    "G000238795",
    "G000784965",  #Unclear how to tell this is a contaminant from the correlation plot.
    "G001562055",
    "G001941055",
    "G000195995"
]]

do_sample_abundance = True
for df in [df_ab_pp_decontam]:
# for df in [df, df_ab, df_ab_pp]:
    if do_sample_abundance:
        df = df[df.sum(axis=1) > 50000]
    df_pca = plot_pca(df)

    df_clr = pd.DataFrame(data=clr(df + 1), index=df.index)
    plot_pca(df_clr)

    # row_sum[row_sum == 0] = 1 # Don't divide if sample has no counts
    df_normalized = df.div(df.sum(axis=1), axis=0)
    print(df_normalized.shape)
    print(df_normalized.sum(axis=1).sum())
    plot_pca(df_normalized)


