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
from collections import defaultdict
import scipy


# Helper Functions
def load_biom(f):
    biom_table = load_table(f)
    table = Artifact.import_data("FeatureTable[Frequency]", biom_table)
    return table.view(pd.DataFrame)

def get_all(woltka_metadata, gotu):
    try:
        return woltka_metadata[woltka_metadata["#genome"] == gotu][["phylum", "class", "order", "family", "genus", "species"]].iloc[0]
    except KeyError as e:
        return None  # Protozoan parasites like Toxoplasma gondii aren't given phylum, class, etc in woltka metadata...?

def get_diff(gotu_row1, gotu_row2):
    for l in ["phylum", "class", "order", "family", "genus", "species"]:
        if (gotu_row1[l] is not None) and (gotu_row1[l] != gotu_row2[l]):
            return l
    return "strain"

def get_genus(woltka_metadata, gotu):
    genus = woltka_metadata[woltka_metadata["#genome"] == gotu]["genus"].iloc[0]
    if not isinstance(genus, str):
        # What...
        genus = "Unknown"
    return genus

def get_domain(woltka_metadata, gotu):
    return woltka_metadata[woltka_metadata["#genome"] == gotu]["domain"].iloc[0]

def get_species(woltka_metadata, gotu):
    return woltka_metadata[woltka_metadata["#genome"] == gotu]["species"].iloc[0]

def sort_by_genus(woltka_metadata, df):
    mapper = {}
    for col in df.columns:
        g = get_genus(woltka_metadata, col)
        mapper[col] = g
    cols_sorted = sorted(df.columns, key=lambda x: mapper[x])
    return df[cols_sorted]

def sort_by_species(woltka_metadata, df):
    mapper = {}
    for col in df.columns:
        g = get_species(woltka_metadata, col)
        mapper[col] = g
    cols_sorted = sorted(df.columns, key=lambda x: mapper[x])
    return df[cols_sorted]

def build_df(config):
    woltka_metadata = pd.read_csv(WOLTKA_METADATA_LOCATION, sep="\t")
    if "genus" not in woltka_metadata.columns:
        # parse taxon
        if "taxon" in woltka_metadata.columns:
            # rep210
            names = woltka_metadata["taxon"].values
        elif "unique_name" in woltka_metadata.columns:
            # rep200
            names = woltka_metadata["unique_name"].values
        mapper = {
            "d": "domain",
            "p": "phylum",
            "c": "class",
            "o": "order",
            "f": "family",
            "g": "genus",
            "s": "species"
        }
        new_cols_dict = defaultdict(list)
        for s in names:
            ss = s.split("; ")
            ss_map = {}
            for part in ss:
                pp = part.split("__")
                ss_map[pp[0]] = pp[1]
            for key in mapper:
                if key in ss_map:
                    new_cols_dict[mapper[key]].append(ss_map[key])
                else:
                    new_cols_dict[mapper[key]].append(None)
        for key in mapper:
            woltka_metadata[mapper[key]] = new_cols_dict[mapper[key]]

    df = BiomTable(DF_LOCATION).load_dataframe()

    if config.APPLY_RELATIVE_ABUNDANCE_FILTER:
        df_frac = df.sum() / df.sum().sum()

        passing_gotus = list(df_frac[df_frac > 1/10000].index)
        df = df[passing_gotus]

    if config.APPLY_ZEBRA_FILTER:
        zebra_genome_cover = pd.read_csv(ZEBRA_DATA_LOCATION, sep=ZEBRA_SEP)
        # filtered_gotu = zebra_genome_cover[zebra_genome_cover["coverage_ratio"] > 0.25]["genome_id"]
        filtered_gotu = zebra_genome_cover[zebra_genome_cover["coverage_ratio"] > 0.25]["gotu"] # zebra broke compatibility

        all_zebra = set()
        df_cols = set(df.columns)
        for g in filtered_gotu:
            if g in df_cols:
                all_zebra.add(g)
        all_zebra = list(all_zebra)

        df = df[all_zebra]

    if config.APPLY_PAIRWISE_PEARSON:
        # Sort by gotu counts
        col_sums = df.sum()
        col_sums = col_sums.sort_values(ascending=False)
        df = df[col_sums.index]

        # test_df = df
        # for thresh in [0.9, 0.95, 0.99, 0.999]:
        #     pp_df, _ = pairwise_pearson(test_df, thresh=thresh)

        pre_pp = df
        df, sets = pairwise_pearson(df, thresh=0.95)
        diff_count = defaultdict(int)
        for rep in sets:
            rep_all = get_all(woltka_metadata, rep)
            for mem in sets[rep]:
                if rep == mem:
                    continue
                mem_all = get_all(woltka_metadata, mem)

                if rep_all is None or mem_all is None:
                    print("UH OH", rep, mem, rep_all, mem_all)
                    print(woltka_metadata[woltka_metadata["#genome"] == rep])
                    print(woltka_metadata[woltka_metadata["#genome"] == mem])
                    diff_count["indeterminate"] += 1
                    continue
                diff_level = get_diff(rep_all, mem_all)
                if diff_level is not None:
                    diff_count[diff_level] += 1

        for level in ["strain", "species", "genus", "family", "order", "class", "phylum"]:
            print(level, diff_count[level], "{:.2f}".format(diff_count[level] / pre_pp.shape[1] * 100))

    else:
        pre_pp = df

    # FIGURE S2: PAIRWISE PEARSON
    # Correlation Heatmaps
    # After Zebra, Before Pairwise Pearson
    if config.MAKE_HEATMAPS:
        fig = plt.figure(figsize=(12, 10), dpi=300)  # Gotta tweak fontsize if you touch dpi
        interest_pp = pre_pp
        if COLS_OF_INTEREST is not None:
            interest_pp = interest_pp[COLS_OF_INTEREST]

        if SORT_BY == "genus":
            zebra_df_sorted = sort_by_genus(woltka_metadata, interest_pp)
        elif SORT_BY == "species":
            zebra_df_sorted = sort_by_species(woltka_metadata, interest_pp)
        ax = sns.heatmap(zebra_df_sorted.corr(), square=True)
        ticks = [x + 0.5 for x in range(len(zebra_df_sorted.columns))]
        labels_species = [get_species(woltka_metadata, gotu) for gotu in zebra_df_sorted.columns]
        labels_gotus = [gotu for gotu in zebra_df_sorted.columns]
        plt.xticks(ticks=ticks, labels=labels_gotus, fontsize=6)  # Gotta tweak fontsize if you touch dpi
        plt.yticks(ticks=ticks, labels=labels_species, fontsize=6)  # Gotta tweak fontsize if you touch dpi
        plt.title(TITLE + "\n(Before)")
        plt.subplots_adjust(left=0.3, bottom=.15)
        plt.savefig("figs/" + "_".join(TITLE.split()) + "_panel1.png")
        plt.show()

        if config.APPLY_PAIRWISE_PEARSON:
            # After Pairwise Pearson
            fig = plt.figure(figsize=(12, 10), dpi=300)  # Gotta tweak fontsize if you touch dpi
            interest_pp = df
            if COLS_OF_INTEREST is not None:
                cols_intersect = [c for c in COLS_OF_INTEREST if c in interest_pp.columns]
                interest_pp = interest_pp[cols_intersect]
            if SORT_BY == "genus":
                pp_df_sorted = sort_by_genus(woltka_metadata, interest_pp)
            elif SORT_BY == "species":
                pp_df_sorted = sort_by_species(woltka_metadata, interest_pp)

            sns.heatmap(pp_df_sorted.corr(), square=True)
            ticks = [x + 0.5 for x in range(len(pp_df_sorted.columns))]
            labels_species = [get_species(woltka_metadata, gotu) for gotu in pp_df_sorted.columns]
            labels_gotus = [gotu for gotu in pp_df_sorted.columns]
            plt.xticks(ticks=ticks, labels=labels_gotus, fontsize=6) # Gotta tweak fontsize if you touch dpi
            plt.yticks(ticks=ticks, labels=labels_species, fontsize=6) # Gotta tweak fontsize if you touch dpi
            plt.title(TITLE + "\n(After)")
            plt.subplots_adjust(left=0.3, bottom=.15)
            plt.savefig("figs/" + "_".join(TITLE.split()) + "_panel2.png")
            plt.show()

    # interesting_reps = set()
    # for rep in sets:
    #     for mem in sets[rep]:
    #         if mem in ["G002402265", "G000872045", "G000859985"]:
    #             interesting_reps.add(rep)
    # print("bad sample:")
    # print(pre_pp["G003892345"])
    # print(pre_pp["G003892345"].idxmax())
    # print(pre_pp["G003892345"].max())
    if config.MAKE_2D_PLOTS and config.APPLY_PAIRWISE_PEARSON:
        for rep in sets:
        # for rep in sorted(interesting_reps):
            if len(sets[rep]) == 1:
                continue
            i = 1
            fig = plt.figure(figsize=(8, 10))
            rep_name = get_species(woltka_metadata, rep)
            for mem in sets[rep]:
                if rep == mem:
                    continue
                mem_name = get_species(woltka_metadata, mem)
                ax = fig.add_subplot(4, 4, i)
                ax.scatter(pre_pp[mem], pre_pp[rep])
                ax.set_xlabel(mem_name + "\n" + mem)
                if i % 4 != 1:
                    ax.set_yticks([])
                else:
                    ax.set_ylabel(rep_name + "\n" + rep)
                i += 1

                if i == 17:
                    plt.suptitle(rep_name)
                    fig.tight_layout()
                    plt.show()
                    fig = plt.figure(figsize=(8, 10))
                    i = 1
            if i > 1:
                plt.suptitle(rep)
                fig.tight_layout()
                plt.show()
    return df

def _build_venn2( a_label, b_label, A, B, AB, ax=None):
    # A B AB, C, AC, BC, ABC
    v = venn2(subsets=(1,1,1,1,1,1,1), set_labels = (a_label, b_label), ax=ax)
    v.get_label_by_id('10').set_text(str(A))
    v.get_label_by_id('01').set_text(str(B))
    v.get_label_by_id('11').set_text(str(AB))

def build_venn2(sets, a, b, title):
    A = len(sets[a])
    B = len(sets[b])
    AB = len(sets[a].intersection(sets[b]))
    A -= AB
    B -= AB

    plt.figure(figsize=(6,4))
    _build_venn2(a, b, A, B, AB)
    plt.title(title)
    plt.show()

def build_venn2_counts(counts, a, b, title, intersect_label="both", ax=None):
    A = counts[a]
    B = counts[b]
    AB = counts[intersect_label]

    if ax is None:
        plt.figure(figsize=(6, 4))
    _build_venn2(a, b, A, B, AB, ax=ax)

    if ax is None:
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


COLS_OF_INTEREST = None
SORT_BY = "genus"

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

# TCGA
# ZEBRA_DATA_LOCATION = "./dataset/tcga_zebra_merged.tsv"
# ZEBRA_SEP = "\t"
# DF_LOCATION = "/Users/djhakim/tcga_analysis/count_data_all.biom"
# WOLTKA_METADATA_LOCATION = "/Users/djhakim/tcga_analysis/rep200_metadata.tsv"
# TITLE = "TCGA Correlation Matrix"
# SORT_BY = "species"

# ISS
# ZEBRA_DATA_LOCATION = "./dataset/biom/ISS_3DMM/WoLr2_coverages.tsv"
# ZEBRA_SEP = "\t"
# DF_LOCATION = "./dataset/biom/ISS_3DMM/163363_WoLr2.biom"
# WOLTKA_METADATA_LOCATION = "./dataset/biom/ISS_3DMM/WoLr2_metadata.tsv"

# ZEBRA_DATA_LOCATION = "./dataset/biom/ISS_3DMM/rep210_coverages.tsv"
# ZEBRA_SEP = "\t"
# DF_LOCATION = "./dataset/biom/ISS_3DMM/165281_rep210.biom"
# WOLTKA_METADATA_LOCATION = "./dataset/biom/ISS_3DMM/rep210_metadata.tsv"
# COLS_OF_INTEREST = ["G002402265", "G000872045", "G000859985", "G003892345", "G000258445", "G002302495", "G900455895", "G000429265", "G900475445"]

# iMSMS
# ZEBRA_DATA_LOCATION = "/Users/djhakim/imsms-ml/zebra.csv"
# ZEBRA_SEP = ","
# DF_LOCATION = "./dataset/biom/imsms-combined-none.biom"
# WOLTKA_METADATA_LOCATION = "./woltka_metadata.tsv"
# TITLE = "IMSMS Correlation Matrix"
# COLS_OF_INTEREST = [
#     "G000020225", "G000723745", "G000436395", "G001917295", "G001940945",
#     "G000980515", "G000153905", "G001404455", "G000438275", "G001915695",
#     "G000435355", "G000269565", "G001717135", "G000162015", "G000434955",
#     "G001406335", "G000160095", "G000235505", "G900113595", "G000311825",
#     "G000375645", "G000432115", "G001481375", "G000758845", "G000508625",
#     "G000153925"
# ]
#TITLE="IMSMS Correlation Matrix\nDifferentially Abundant Species"

# Gallo skin
# ZEBRA_DATA_LOCATION = "./dataset/biom/gibsyang/14365_metagenomic_may.tsv"
# ZEBRA_SEP = "\t"
# DF_LOCATION = "./dataset/biom/gibsyang/table.filt.biom"
# WOLTKA_METADATA_LOCATION = "./woltka_metadata.tsv"

# Celeste Ecoli
DF_LOCATION = [
    "dataset/biom/Celeste_Prep_1_1428_samples.biom",
    "dataset/biom/Celeste_Prep_2_672_samples.biom",
    "dataset/biom/Celeste_Prep_3_936_samples.biom",
    "dataset/biom/Celeste_Prep_4_792_samples.biom"
]
WOLTKA_METADATA_LOCATION = "./woltka_metadata.tsv"
ZEBRA_DATA_LOCATION = None
ZEBRA_SEP = "\t"
TITLE = "E. coli Isolates Correlation Matrix"

# Finrisk
# DF_LOCATION = "dataset/biom/finrisk-combined-none.biom"
# WOLTKA_METADATA_LOCATION = "./woltka_metadata.tsv"
# ZEBRA_DATA_LOCATION = None
# ZEBRA_SEP = "\t"
# TITLE = "Finrisk Correlation Matrix"


# Configuration
class Config:
    def __init__(self, rel_ab, zebra, pp, heatmaps, scatterplots):
        self.APPLY_RELATIVE_ABUNDANCE_FILTER = rel_ab
        self.APPLY_ZEBRA_FILTER = zebra
        self.APPLY_PAIRWISE_PEARSON = pp
        self.MAKE_HEATMAPS = heatmaps
        self.MAKE_2D_PLOTS = scatterplots

    def __str__(self):
        ss = []
        if self.APPLY_RELATIVE_ABUNDANCE_FILTER:
            ss.append("rel ab")
        if self.APPLY_ZEBRA_FILTER:
            ss.append("zebra")
        if self.APPLY_PAIRWISE_PEARSON:
            ss.append("pearson")
        if len(ss) == 0:
            return "Initial"
        else:
            return "->".join(ss)


gotu_dict = {}
for config in [
    # Config(False, False, False, False, False),
    # Config(True, False, False, False, False),
    # Config(False, True, False, False, False),
    # Config(False, False, True, False, False),
    # Config(True, True, False, False, False),
    # Config(True, False, True, True, True),
    # Config(False, True, True, True, True),
    # Config(True, True, True, True, True),
]:
    df = build_df(config)
    print(config, df.shape)
    # if "G000364165" in df.columns:
    #     print("G000364165 is in ", config)
    gotu_dict[str(config)] = set(df.columns)

# build_venn2(gotu_dict, "rel ab", "rel ab->pearson", "TCGA GOTUs")
# build_venn3(gotu_dict, "rel ab->zebra", "rel ab->pearson", "rel ab->zebra->pearson", "Gallo Skin GOTUs")

thresh_counts = {
    0: {'G000438275': 32912, 'G001406335': 157809, 'G001406335_G000438275': 52491},
    1: {'G001406335_G000438275': 139629, 'G001406335': 340961, 'G000438275': 51013},
    2: {'G001406335_G000438275': 218740, 'G001406335': 478784, 'G000438275': 52330},
    3: {'G001406335_G000438275': 279192, 'G001406335': 592279, 'G000438275': 50214},
    4: {'G001406335': 688683, 'G000438275': 48093, 'G001406335_G000438275': 324765},
    5: {'G001406335': 770815, 'G001406335_G000438275': 359963, 'G000438275': 46456},
    6: {'G001406335': 840937, 'G001406335_G000438275': 387084, 'G000438275': 45152},
    7: {'G001406335': 902845, 'G001406335_G000438275': 408275, 'G000438275': 44038}
}

fig, axes = plt.subplots(3, 3)
fig.delaxes(axes[2][2])
x = 0
y = 0
for edit_distance in thresh_counts:
    fusi_reads = thresh_counts[edit_distance]["G001406335"]
    blautia_reads = thresh_counts[edit_distance]["G000438275"]
    both_reads = thresh_counts[edit_distance]["G001406335_G000438275"]
    total_reads = fusi_reads + blautia_reads + both_reads
    fusi_pct = fusi_reads / total_reads * 100
    blautia_pct = blautia_reads / total_reads * 100
    both_pct = both_reads / total_reads * 100

    ax = axes[y, x]
    build_venn2_counts(
        {
            "Fusicatenibacter\nsaccharivorans": "{:.2f}%".format(fusi_pct),
            "Blautia\nsp. CAG:37": "{:.2f}%".format(blautia_pct),
            "both": "{:.2f}%".format(both_pct)
        },
        "Fusicatenibacter\nsaccharivorans",
        "Blautia\nsp. CAG:37",
        "Read Ambiguity",
        ax=ax
    )
    ax.set_title("ED: " + str(edit_distance) + " #Reads:" + str(total_reads))
    x += 1
    if x == 3:
        x = 0
        y += 1

plt.show()

# Combined Scatter Plots (End of Figure 2)
# gotu1 = "G001406335"
# gotu2 = "G000438275"
#
# df_tcga = BiomTable("/Users/djhakim/tcga_analysis/count_data_all.biom").load_dataframe()
# df_imsms = BiomTable("./dataset/biom/imsms-combined-none.biom").load_dataframe()
# df_finrisk = BiomTable("dataset/biom/finrisk-combined-none.biom").load_dataframe()
# df_ecoli_isolates = BiomTable([
#     "dataset/biom/Celeste_Prep_1_1428_samples.biom",
#     "dataset/biom/Celeste_Prep_2_672_samples.biom",
#     "dataset/biom/Celeste_Prep_3_936_samples.biom",
#     "dataset/biom/Celeste_Prep_4_792_samples.biom"
# ]).load_dataframe()
# woltka_metadata = pd.read_csv(WOLTKA_METADATA_LOCATION, sep="\t")
#
# fig = plt.figure()
# ax = plt.gca()
# for key, df in [
#     ("TCGA", df_tcga),
#     ("iMSMS", df_imsms),
#     ("FINRISK", df_finrisk),
#     ("E. coli Isolates", df_ecoli_isolates)
# ]:
#     if gotu1 in df.columns and gotu2 in df.columns:
#         r, p = scipy.stats.pearsonr(df[gotu1], df[gotu2])
#         # r = res.statistic
#         # p = res.pvalue
#         ax.scatter(df[gotu1], df[gotu2], label=key + ": r={:.2f}".format(r) + ", p={:.2E}".format(p))
#         ax.set_xlabel(get_species(woltka_metadata, gotu1) + "\n" + gotu1)
#         ax.set_ylabel(get_species(woltka_metadata, gotu2) + "\n" + gotu2)
#         ax.set_yscale('log')
#         ax.set_xscale('log')
# plt.legend()
# plt.show()
