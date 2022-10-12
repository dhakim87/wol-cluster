from detnmf.alpha_calculator import AlphaCalculator
from detnmf.beta_calculator import BetaCalculator
from detnmf.determinant_zero import DeterminantZeroCollapse
from table_info import CSVTable, BiomTable
from woltka_metadata import list_woltka_refs
from plotting import plot_taxavec3
from detnmf.detnmf import run_detnmf, L1_normalize, L2_residual, determinant
import math
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

MIN_SAMPLES = 10
MIN_GENUS_COUNT = 500


def get_woltka_label(woltka_meta_df, col):
    series = woltka_meta_df[woltka_meta_df["#genome"] == col]["species"]

    if len(series) > 0:
        label = "\n"+series.iloc[0] + "\n" + col
    else:
        label = str(col)
    return label


def constant_sum_scale(df, normval):
    df_sum = df.sum(axis=1)
    cst = df.divide(df_sum, axis='rows') * normval
    return cst


def filter_df(df, woltka_meta_df, genus):
    if genus == 'all':
        refs_df = list_woltka_refs(df, woltka_meta_df)
    else:
        refs_df = list_woltka_refs(df, woltka_meta_df, genus)
    genomes = refs_df['#genome'].tolist()
    if len(genomes) == 0 and genus =='all':
        print("Probably 16S, falling back to raw table")
        filtered_df = df.copy()
    else:
        filtered_df = df[genomes]
        filtered_df_sum = filtered_df.sum(axis=1)
        filtered_df = filtered_df[filtered_df_sum >= MIN_GENUS_COUNT]
    return filtered_df


def _plot_tv3(filtered_df, woltka_meta_df, title, c1,c2,c3, W, H, subplot_ax):
    plot_taxavec3(filtered_df, title, c1, c2, c3, get_woltka_label(woltka_meta_df, c1), get_woltka_label(woltka_meta_df, c2), get_woltka_label(woltka_meta_df, c3), W, H, subplot_ax)


def write_best_taxavec_genus(filtered_df, title, num_components, num_attempts, alpha_calculator=AlphaCalculator(0.05, 5000)):
    X = filtered_df.to_numpy()

    best_resid = math.inf
    best_W = None
    best_H = None
    for attempt in range(num_attempts):
        try:
            W, H = run_detnmf(X, num_components, alpha_calculator=alpha_calculator, iterations=50000, beta_calculator=BetaCalculator(0.0001, 2500), det0_handler=DeterminantZeroCollapse())

            # TODO FIXME HACK: use the full score, not the L2 resid (argh needs the internal scaling)
            L2_resid = L2_residual(X, W, H)
            if L2_resid < best_resid:
                best_resid = L2_resid
                best_W = W
                best_H = H
        except np.linalg.LinAlgError as e:
            print(e)
            continue

    best_W, best_H = L1_normalize(best_W, best_H)
    extended_title = "./taxavec_best_results/" + title + "_" + str(num_components)
    np.save(extended_title + "_W.npy", best_W)
    np.save(extended_title + "_H.npy", best_H)
    return best_W, best_H


def load_best_taxavec_genus(title, num_components):
    extended_title = "./taxavec_best_results/" + title + "_" + str(num_components)
    W = np.load(extended_title + "_W.npy")
    H = np.load(extended_title + "_H.npy")
    return W, H


# Figure 1, Load iMSMS show Akkermansia with 4 components
def imsms_plots():
    woltka_meta_df = CSVTable("./woltka_metadata.tsv", delimiter="\t").load_dataframe()
    df = BiomTable("./dataset/biom/imsms-combined-none.biom").load_dataframe()

    filtered_df = filter_df(df, woltka_meta_df, "Akkermansia")
    # write_best_taxavec_genus(filtered_df, "Akkermansia_imsms", 4, 100)
    W, H = load_best_taxavec_genus("Akkermansia_imsms", 4)

    W, H = L1_normalize(W, H)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "KLE Clade", "G001580195", "G001647615", "G001578645", W, H, ax)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Muciniphila Clade", "G000020225", "G000723745", "G000436395", W, H, ax)
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Muciniphila/Glycaniphila Clade Overlap", "G000980515", "G900097105", "G001683795", W, H, ax)
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Multiple Clades", "G000437075", "G001917295", "G001580195", W, H, ax)
    plt.show()


    # Figure 1/2?, Load iMSMS show Methanobrevibacter with 2 or 3 components
    filtered_df = filter_df(df, woltka_meta_df, "Methanobrevibacter")
    # write_best_taxavec_genus(filtered_df, "Methanobrevibacter_imsms", 2, 20)
    # write_best_taxavec_genus(filtered_df, "Methanobrevibacter_imsms", 3, 5)
    W, H = load_best_taxavec_genus("Methanobrevibacter_imsms", 3)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Resolving Methanobrevibacter Smithii", "G000437055", "G000016525", "G000824705", W, H, ax)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Resolving Methanobrevibacter Smithii", "G000437055", "G000190095", "G000824705", W, H, ax)
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Resolving Methanobrevibacter Smithii", "G000437055", "G000016525", "G000190095", W, H, ax)
    # ax = fig.add_subplot(2, 2, 4, projection='3d')
    # _plot_tv3(filtered_df, woltka_meta_df, "Resolving Methanobrevibacter Smithii", "G000190095", "G000190135", "G000190155", W, H, ax)
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Resolving Methanobrevibacter Smithii", "G001477655", "G000016525", "G000190095", W, H, ax)

    plt.show()
    #
    # # Figure 1/2?, Load iMSMS show Dialister with 6 components
    # filtered_df = filter_df(df, woltka_meta_df, "Dialister")
    # # write_best_taxavec_genus(filtered_df, "Dialister_imsms", 5, 10)
    # W, H = load_best_taxavec_genus("Dialister_imsms", 5)
    #
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # _plot_tv3(filtered_df, woltka_meta_df, "Resolving Dialister", "G000435155", "G000160055", "G000433275", W, H, ax)
    # ax = fig.add_subplot(2, 2, 2, projection='3d')
    # _plot_tv3(filtered_df, woltka_meta_df, "Resolving Dialister", "G000434475", "G000242435", "G000194985", W, H, ax)
    # ax = fig.add_subplot(2, 2, 3, projection='3d')
    # _plot_tv3(filtered_df, woltka_meta_df, "Resolving Dialister", "G000435155", "G000160055", "G000438335", W, H, ax)
    #
    # plt.show()
    #
    # # Figure 1/2?, Load iMSMS show Bifidobacterium with 5/6/7 components
    # filtered_df = filter_df(df, woltka_meta_df, "Bifidobacterium")
    # write_best_taxavec_genus(filtered_df, "Bifidobacterium_imsms", 5, 10)
    # W, H = load_best_taxavec_genus("Bifidobacterium_imsms", 6)
    #
    # fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # _plot_tv3(filtered_df, woltka_meta_df, "Resolving Bifidobacterium", "G000010425", "G000007525", "G001025175", W, H, ax)
    #
    # plt.show()

def new_imsms_plots():
    woltka_meta_df = CSVTable("./woltka_metadata.tsv", delimiter="\t").load_dataframe()

    # Figure S1, Load new iMSMS samples.  Show that Akkermansia taxa vectors remain consistent across the samples
    NEW_IMSMS_SAMPLES = [
        "./dataset/biom/imsms-woltka-none/imsms-none-Sep2-2022-1of3.biom",
        "./dataset/biom/imsms-woltka-none/imsms-none-Sep2-2022-2of3.biom",
        "./dataset/biom/imsms-woltka-none/imsms-none-Sep2-2022-3of3.biom"
    ]
    dfs = [BiomTable(b).load_dataframe() for b in NEW_IMSMS_SAMPLES]
    df_imsms_new = pd.concat(dfs).fillna(0)

    bad_prefixes = [
        "11326.BLANK",
        "11326.NA",
        "11326.UCSF",
        "11326.COLUMN",
        "11326.Column",
        "11326.Q.DOD",
        "11326.Zymo.",
        "11326.Mag.Bead.Zymo",
        "11326.D.Twin",
        "11326.F.twin",
        "11326.Q.FMT"
    ]

    print(df_imsms_new.shape)
    bad_rows = df_imsms_new.index.str.startswith(bad_prefixes[0])
    for i in range(1, len(bad_prefixes)):
        bad_rows |= df_imsms_new.index.str.startswith(bad_prefixes[i])
    df_imsms_new = df_imsms_new[~bad_rows]
    print(df_imsms_new.shape)

    filtered_df = filter_df(df_imsms_new, woltka_meta_df, "Akkermansia")
    W, H = load_best_taxavec_genus("Akkermansia_imsms", 4)
    W, H = L1_normalize(W, H)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "KLE Clade", "G001580195", "G001647615", "G001578645", W, H, ax)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Muciniphila Clade", "G000020225", "G000723745", "G000436395", W, H, ax)
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Muciniphila/Glycaniphila Clade Overlap", "G000980515", "G900097105", "G001683795", W, H, ax)
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Multiple Clades", "G000437075", "G001917295", "G001580195", W, H, ax)
    plt.suptitle("New iMSMS")
    plt.show()


def ecoli_plots():
    # Figure 2, Load Celeste's E. coli data show E. coli and distinguishing reference overlaps
    woltka_meta_df = CSVTable("./woltka_metadata.tsv", delimiter="\t").load_dataframe()
    BIOM_TABLE=[
        "./dataset/biom/Celeste_Prep_1_1428_samples.biom",
        "./dataset/biom/Celeste_Prep_2_672_samples.biom",
        "./dataset/biom/Celeste_Prep_3_936_samples.biom",
        "./dataset/biom/Celeste_Prep_4_792_samples.biom"
    ]
    dfs = [BiomTable(b).load_dataframe() for b in BIOM_TABLE]
    df = pd.concat(dfs).fillna(0)

    # Filter to only the high count samples, hopefully removing stuff near the origin
    # that confuses nmf
    print(df)
    df['sum']= df.sum(axis=1)
    print(df['sum'])
    df = df[df['sum'] > 500000]
    df = df.drop(['sum'], axis=1)
    print(df)
    write_best_taxavec_genus(df, "Escherichia_Isolates", 10, 1, alpha_calculator=AlphaCalculator(0.01, 5000))
    W, H = load_best_taxavec_genus("Escherichia_Isolates", 10)

    print(W, H)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    _plot_tv3(df, woltka_meta_df, "E. Coli Nissle", "G000183345", "G000026345", "G000026325", W, H, ax)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    _plot_tv3(df, woltka_meta_df, "E. Coli Nissle", "G000299455", "G000008865", "G000006925", W, H, ax)
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_tv3(df, woltka_meta_df, "E. Coli Nissle", "G000012005", "G001283625", "G001941055", W, H, ax)
    plt.show()


if __name__ == "__main__":
    # imsms_plots()
    # new_imsms_plots()
    ecoli_plots()

