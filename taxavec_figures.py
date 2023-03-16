import json
import os

from detnmf.alpha_calculator import AlphaCalculator
from detnmf.beta_calculator import BetaCalculator
from detnmf.determinant_zero import DeterminantZeroCollapse, \
    DeterminantZeroException
from table_info import CSVTable, BiomTable
from woltka_metadata import list_woltka_refs
from plotting import plot_taxavec2, plot_taxavec3
from detnmf.detnmf import run_detnmf, L1_normalize, L2_residual, determinant, \
    run_caliper_nmf, guess_num_components
import plot_definitions
import math
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
import matplotlib
import matplotlib.patches as mpatches
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt


MIN_SAMPLES = 10
MIN_GENUS_COUNT = 500


def get_woltka_label(woltka_meta_df, col):
    series = woltka_meta_df[woltka_meta_df["#genome"] == col]["species"]

    if len(series) > 0:
        l = series.iloc[0]
        if l == "Escherichia coli":
            l = "E. coli"
        label = "\n"+ l + "\n" + col
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


def _plot_tv2(filtered_df, woltka_meta_df, title, c1,c2, W, H, subplot_ax, skipx=False):
    plot_taxavec2(filtered_df, title, c1, c2, get_woltka_label(woltka_meta_df, c1), get_woltka_label(woltka_meta_df, c2), W, H, subplot_ax, skipx=skipx)


def _plot_tv3(filtered_df, woltka_meta_df, title, c1,c2,c3, W, H, subplot_ax, skipxy=False):
    plot_taxavec3(filtered_df, title, c1, c2, c3, get_woltka_label(woltka_meta_df, c1), get_woltka_label(woltka_meta_df, c2), get_woltka_label(woltka_meta_df, c3), W, H, subplot_ax, skipxy=skipxy)


def calc_L1_resids(filtered_df, W, H):
    X = filtered_df.to_numpy()
    X2 = np.dot(W, H)

    resids = np.sum(np.abs(X - X2), axis=1)
    resids_pcts = resids / (np.sum(X, axis=1))

    return resids, resids_pcts


def calc_L1_resids_coexclusive(filtered_df, W, H):
    print(H)

    X = filtered_df.to_numpy()
    resids = []
    resids_pcts = []
    for sample_i in range(X.shape[0]):
        sample = X[sample_i, :]

        coefs = W[sample_i, :]
        max_coef_i = np.argmax(coefs)

        best_row = H[max_coef_i, :]
        proj_onto_row = np.dot(np.dot(sample, best_row) / np.dot(best_row, best_row), best_row)
        delta = sample - proj_onto_row
        L1_resid = np.sum(np.abs(delta))
        L1_resid_pct = L1_resid / np.sum(sample)
        resids.append(L1_resid)
        resids_pcts.append(L1_resid_pct)

    return np.array(resids), np.array(resids_pcts)


def _plot_fix_resids(ax, genus, l1_resids, vlines=None, x_label="Reads"):
    # sort the data in ascending order
    x = np.sort(l1_resids)

    # get the cdf values of y
    N = len(l1_resids)
    y = np.arange(N) / float(N)

    # plotting
    ax.set_xlabel(x_label)
    ax.set_ylabel('Fraction Samples Resid < X')
    ax.set_title(genus + ' Residuals')
    ax.plot(x, y, marker='o')
    ax.axhline(y=0.8, color='gray', linestyle=':')
    ax.axhline(y=0.9, color='gray', linestyle=':')
    if vlines is not None:
        for vline in vlines:
            ax.axvline(x=vline, color='gray', linestyle=":")


def get_filename(title, num_components):
    return "./taxavec_best_results/" + title + "_" + str(num_components)

def get_filename_W(title, num_components):
    return get_filename(title, num_components) + "_W.npy"

def get_filename_H(title, num_components):
    return get_filename(title, num_components) + "_H.npy"

def write_best_taxavec_genus(filtered_df, title, num_components, num_attempts, alpha_calculator=AlphaCalculator(0.05, 0, 5000), det0_handler=DeterminantZeroCollapse(), column_scale=False, verbose=False):
    X = filtered_df.to_numpy()

    best_resid = math.inf
    best_W = None
    best_H = None
    for attempt in range(num_attempts):
        try:
            if alpha_calculator is not None:
                # default values use the same object on all calls.
                alpha_calculator.reset()
            W, H = run_detnmf(X, num_components, alpha_calculator=alpha_calculator, iterations=50000, beta_calculator=BetaCalculator(0.0001, 2500), det0_handler=det0_handler, column_scale=column_scale, verbose=verbose)

            r, output_pt = DeterminantZeroCollapse.find_redundant_row(np.dot(H, H.T))
            while r is not None:
                print("Found row to collapse")
                print("Collapsing 1 component")
                W, H = DeterminantZeroCollapse.replace_row(W, H, r, output_pt)
                r, output_pt = DeterminantZeroCollapse.find_redundant_row(np.dot(H, H.T))

            # TODO FIXME HACK: use the full score, not the L2 resid (argh needs the internal scaling)
            L2_resid = L2_residual(X, W, H)
            if L2_resid < best_resid:
                best_resid = L2_resid
                best_W = W
                best_H = H
        except np.linalg.LinAlgError as e:
            print(e)
            continue
        except DeterminantZeroException as e:
            print(e)
            continue

    best_W, best_H = L1_normalize(best_W, best_H)

    # Det 0 handlers can change number of components. save file named by final number of components
    final_num_components = best_H.shape[0]
    np.save(get_filename_W(title, final_num_components), best_W)
    np.save(get_filename_H(title, final_num_components), best_H)
    return best_W, best_H


def load_best_taxavec_genus(title, num_components):
    W = np.load(get_filename_W(title, num_components))
    H = np.load(get_filename_H(title, num_components))
    return W, H


# Figure 1, Load iMSMS show Akkermansia with 4 components
def imsms_plots():
    woltka_meta_df = CSVTable("./woltka_metadata.tsv", delimiter="\t").load_dataframe()
    df = BiomTable("./dataset/biom/imsms-combined-none.biom").load_dataframe()

    filtered_df = filter_df(df, woltka_meta_df, "Akkermansia")
    # write_best_taxavec_genus(filtered_df, "Akkermansia_imsms", 4, 100)
    W, H = load_best_taxavec_genus("Akkermansia_imsms", 4)

    W, H = L1_normalize(W, H)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "KLE Clade", "G001580195", "G001647615", "G001578645", W, H, ax)
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Muciniphila Clade", "G000020225", "G000723745", "G000436395", W, H, ax)
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Muciniphila/Glycaniphila Clade Overlap", "G000980515", "G900097105", "G001683795", W, H, ax)
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Multiple Clades", "G000437075", "G001917295", "G001580195", W, H, ax)
    fig.tight_layout()
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(2, 2, 1)
    ax.scatter(W[:, 0], W[:, 1], alpha=.2)
    ax.set_xlabel("TV1")
    ax.set_ylabel("TV2")

    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(W[:, 2], W[:, 3], alpha=.2)
    ax.set_xlabel("TV3")
    ax.set_ylabel("TV4")

    resids, resids_pcts = calc_L1_resids(filtered_df, W, H)
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    _plot_fix_resids(ax, "Akkermansia", resids, x_label="Genus Akkermansia Reads")
    ax = fig.add_subplot(2, 2, 4)
    _plot_fix_resids(ax, "Akkermansia", resids_pcts, vlines=[0.05, 0.10], x_label="Fraction Per Sample Genus Akkermansia Reads")
    fig.tight_layout()
    plt.show()

    # Figure 1/2?, Load iMSMS show Methanobrevibacter with 2 or 3 components
    filtered_df = filter_df(df, woltka_meta_df, "Methanobrevibacter")
    # write_best_taxavec_genus(filtered_df, "Methanobrevibacter_imsms", 2, 20)
    # write_best_taxavec_genus(filtered_df, "Methanobrevibacter_imsms", 3, 5)
    n = 3
    W, H = load_best_taxavec_genus("Methanobrevibacter_imsms", n)

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

    resids, resids_pcts = calc_L1_resids(filtered_df, W, H)
    fig, axs = plt.subplots(1, 2, sharey=True)
    _plot_fix_resids(axs[0], "Methanobrevibacter (" + str(n) + ") ", resids)
    _plot_fix_resids(axs[1], "Methanobrevibacter (" + str(n) + ") ", resids_pcts, vlines=[0.05, 0.10])
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


def finrisk_plots():
    woltka_meta_df = CSVTable("./woltka_metadata.tsv", delimiter="\t").load_dataframe()

    # Figure S1, Load new iMSMS samples.  Show that Akkermansia taxa vectors remain consistent across the samples
    df = BiomTable("./dataset/biom/finrisk-combined-none.biom").load_dataframe()

    filtered_df = filter_df(df, woltka_meta_df, "Akkermansia")

    # W, H = write_best_taxavec_genus(filtered_df, "Akkermansia_finrisk", 4, 25)
    W, H = load_best_taxavec_genus("Akkermansia_imsms", 4)
    W_imsms, H_imsms = L1_normalize(W, H)
    W, H = load_best_taxavec_genus("Akkermansia_finrisk", 4)
    W_finrisk, H_finrisk = L1_normalize(W, H)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Muciniphila Clade (iMSMS model)", "G000020225", "G000723745", "G000436395", W_imsms, H_imsms, ax)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    _plot_tv3(filtered_df, woltka_meta_df, "Muciniphila Clade (Finrisk model)", "G000020225", "G000723745", "G000436395", W_finrisk, H_finrisk, ax)
    plt.suptitle("Finrisk")
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

    # df = CSVTable("./dataset/biom/celeste_ecoli_kraken.tsv", sep="\t", index_col="sampleid").load_dataframe()

    # # Let's do the most bog standard filtering and show that there's still tons of stuff
    # # that remains that is linearly dependent on E. coli.
    # print(df)
    # df['sum']= df.sum(axis=1)
    # print(df['sum'])
    # df = df[df['sum'] > 500000]
    # df = df.drop(['sum'], axis=1)
    # print(df)
    #
    # print("AND THEN")
    # Filter to only high count features, hopefully reducing the number of columns
    # to something more approachable.
    df_sum = df.sum(axis=0)
    df_sum_pct = df_sum / df_sum.sum()
    print(df_sum_pct)
    df = df.loc[:, df_sum_pct > 1/10000]
    print(df)


    # is_ecoli = pd.Series(np.zeros(df_sum_pct.shape), index=df_sum_pct.index)
    # is_ecoli_nan = pd.Series(np.full(df_sum_pct.shape, np.nan), index=df_sum_pct.index)
    #
    # is_ecoli.loc[["G000008865", "G000026325", "G000026345",
    #                     "G000183345", "G000299455", "G000759795",
    #                     "G001283625"]] = 1
    # is_ecoli_nan.loc[["G000008865", "G000026325", "G000026345",
    #               "G000183345", "G000299455", "G000759795",
    #               "G001283625"]] = 1
    # empress = pd.DataFrame([df_sum_pct, df_sum_pct > 0, df_sum_pct > 1/10000, is_ecoli, is_ecoli_nan]).T
    # empress.index.name = "Feature ID"
    # empress.columns = ["frac_reads", "one_read", "passes_abundance", "is_ecoli", "is_ecoli_nan"]
    # print(empress)
    # empress_ecoli = empress[["is_ecoli"]]
    # empress_ecoli = empress_ecoli[empress_ecoli["is_ecoli"] == 1]
    # empress_one_read = empress[["one_read"]]
    # empress_one_read = empress_one_read[empress["one_read"] == 1]
    # empress_pass_filter = empress[["passes_abundance"]]
    # empress_pass_filter = empress_pass_filter[empress_pass_filter["passes_abundance"] == 1]
    # print(empress_ecoli.shape)
    # print(empress_one_read.shape)
    # print(empress_pass_filter.shape)

    # empress_ecoli.to_csv("./dataset/newick/ecoli.tsv", sep='\t')
    # empress_one_read.to_csv("./dataset/newick/one_read.tsv", sep='\t')
    # empress_pass_filter.to_csv("./dataset/newick/pass_filter.tsv", sep='\t')

    # write_best_taxavec_genus(df, "Escherichia_Isolates_Filtered", 2, 3, alpha_calculator=AlphaCalculator(0.05, 0, 5000))
    W, H = load_best_taxavec_genus("Escherichia_Isolates_Filtered", 2)

    # write_best_taxavec_genus(df, "Escherichia_Isolates_Kraken_Filtered", 2, 3, alpha_calculator=AlphaCalculator(0.05, 0, 5000))
    # W, H = load_best_taxavec_genus("Escherichia_Isolates_Kraken_Filtered", 2)

    print(W, H)
    fig = plt.figure(figsize=(8, 6))
    # ax = fig.add_subplot(2, 2, 1, projection='3d')
    # _plot_tv3(df, woltka_meta_df, "E. Coli Nissle", "G000183345", "G000026345", "G000026325", W, H, ax)
    # # ax = fig.add_subplot(2, 2, 2, projection='3d')
    # _plot_tv3(df, woltka_meta_df, "E. Coli Nissle", "G000299455", "G000008865", "G000006925", W, H, ax)
    # ax = fig.add_subplot(2, 2, 3, projection='3d')
    # _plot_tv3(df, woltka_meta_df, "E. Coli Nissle", "G000012005", "G001283625", "G001941055", W, H, ax)

    colors = []
    for i in df.index:
        second = i.split(".")[1] # 0 for kraken, 1 for woltka, blah.
        if second.startswith("4"):
            colors.append("m")
        elif second.startswith("2"):
            colors.append("b")
        else:
            colors.append("gray")
    ax = fig.add_subplot(2, 2, 2)
    ax.scatter(W[:, 0], W[:, 1], c=colors, alpha=.2)
    cc = ["m","b","gray"]
    labels = ["1","2","Blank"]
    recs = []
    for i in range(0, 3):
        recs.append(mpatches.Rectangle((0, 0), 1, 1, fc=cc[i]))
    plt.legend(recs, labels, loc="upper right", title="Sample Type")

    ax.set_xlabel("TV1")
    ax.set_ylabel("TV2")
    plt.title("Reconstruction TV Loadings")

    resids, resids_pcts = calc_L1_resids(df, W, H)
    ax = fig.add_subplot(2, 2, 3)
    _plot_fix_resids(ax, "E. coli", resids, x_label="Reads")
    ax = fig.add_subplot(2, 2, 4)
    _plot_fix_resids(ax, "E. coli", resids_pcts, vlines=[0.05, 0.10], x_label="Fraction Per Sample Reads")
    fig.tight_layout()
    plt.show()

# for i in range(2, df.shape[1]):
    #     _plot_tv3(df, woltka_meta_df, "E. Coli Nissle", "G000183345", "G000026345", df.columns[i], W, H, None)
    distant_relatives = [
        "G001604445", "G001941055", "G001750165", "G000238795",
        # "G000415085", "G000415005", "G000648175", "G000007785",  # One enterococcus faecalis is probably fine.
        "G001543285", "G000648015", "G000018665", "G001447195"]
    close_relatives = ["G000012005", "G000694955", "G000247895", "G000168835"]  # everything else...
    fig = plt.figure(figsize=(8, 11))
    for ci in range(len(distant_relatives)):
        c = distant_relatives[ci]
        ax = fig.add_subplot(4, 2, ci+1)
        _plot_tv2(df, woltka_meta_df, "", "G000183345", c, W, H, ax, skipx=ci < 6)
        # _plot_tv3(df, woltka_meta_df, "Enterococcus", "G000648015", "G000415085", c, W, H, None)
    fig.tight_layout()
    plt.show()


# Overlap:
# Synergistales
# Ruminococcus Zagget 7

def bulk_plots(show_plots=False):
    PLOT=show_plots

    woltka_meta_df = CSVTable("./woltka_metadata.tsv", delimiter="\t").load_dataframe()
    df = BiomTable("./dataset/biom/imsms-combined-none.biom").load_dataframe()
    best_components = {}
    counter = 0
    result_rows = []

    fig = None  # plt.figure(figsize=(6, 4))
    resid_left = None  # fig.add_subplot(1, 2, 1)
    resid_right = None  # fig.add_subplot(1, 2, 2)

    with open("./taxavec_best_results/bulk_components.tsv", "a+") as component_file:
        component_file.seek(0)
        for line in component_file.readlines():
            line = line[:-1]  # stupid newlines.
            ss = line.split("\t")
            if len(ss) < 2:
                continue
            best_components[ss[0]] = int(ss[1])

        for genus in plot_definitions.imsms_plots:
            if genus != "Acinetobacter":
                continue
            filtered_df = filter_df(df, woltka_meta_df, genus)
            cols = plot_definitions.imsms_plots[genus]

            if genus not in best_components:
                # First have to guess the number of components,
                # we'll do this by binary searching to the largest number that doesn't crash
                X = filtered_df.to_numpy()
                approx_components = guess_num_components(X, filtered_df.shape[1], alpha_calculator=AlphaCalculator(0.05, 0, 5000), iterations=2500, beta_calculator=BetaCalculator(0.0001, 2500), min_det=1e-10)
                print("Approximate species in", genus, approx_components)
            else:
                approx_components = best_components[genus]

            title = genus + "_imsms_bulk"
            if not os.path.exists(get_filename_W(title, approx_components)):
                W, H = write_best_taxavec_genus(filtered_df, title, approx_components, 10, column_scale=True, verbose=False)
                # Some det0 handlers can collapse components
                approx_components = H.shape[0]
                if genus not in best_components or best_components[genus] != approx_components:
                    best_components[genus] = approx_components
                    component_file.write(genus + "\t" + str(approx_components) + "\n")
                    component_file.flush()
                    os.fsync(component_file)  # okay that's the dumbest api ever.

            W, H = load_best_taxavec_genus(title, approx_components)
            W, H = L1_normalize(W, H)

            if PLOT:
                _plot_tv3(filtered_df, woltka_meta_df, genus, cols[0], cols[1], cols[2], W, H, None)

            resids, resids_pcts = calc_L1_resids(filtered_df, W, H)
            coex_resids, coex_resids_pcts = calc_L1_resids_coexclusive(filtered_df, W, H)
            if resid_left is None:
                fig = plt.figure(figsize=(6, 4))
                ax = fig.add_subplot(1, 2, 1)
            else:
                ax = resid_left
            if PLOT:
                _plot_fix_resids(ax, genus, resids, x_label="Reads")
                _plot_fix_resids(ax, genus, coex_resids, x_label="Reads")
            if resid_right is None:
                ax = fig.add_subplot(1, 2, 2)
            else:
                ax = resid_right
            if PLOT:
                _plot_fix_resids(ax, genus, resids_pcts, vlines=[0.05, 0.10], x_label="Fraction Per Sample Reads")
                _plot_fix_resids(ax, genus, coex_resids_pcts, vlines=[0.05, 0.10], x_label="Fraction Per Sample Reads")

            if PLOT:
                if resid_left is None:
                    plt.show()

            resid_80 = np.sort(resids)[int(.8 * len(resids))]
            resid_80_pct = np.sort(resids_pcts)[int(.8 * len(resids_pcts))]
            coex_resid_80 = np.sort(coex_resids)[int(.8 * len(resids))]
            coex_resid_80_pct = np.sort(coex_resids_pcts)[int(.8 * len(resids_pcts))]

            row = [genus, H.shape[0], resid_80, resid_80_pct, coex_resid_80, coex_resid_80_pct]
            result_rows.append(row)
            print(genus,
                  "Num Components:", H.shape[0],
                  "80% resid reads", resid_80,
                  "80% resid frac", resid_80_pct,
                  "80% resid reads exclusive", coex_resid_80,
                  "80% resid frac exclusive", coex_resid_80_pct,
              )

    import csv
    with open('taxavec_best_results/results.csv', 'w', newline='') as csvfile:
        fieldnames = ["genus", "components", "resid_80_reads", "resid_80_frac", "coex_resids", "coex_resids_pcts"]
        writer = csv.writer(csvfile)
        writer.writerow(fieldnames)
        for r in result_rows:
            writer.writerow(r)


if __name__ == "__main__":
    # imsms_plots()
    # new_imsms_plots()
    # finrisk_plots()
    ecoli_plots()
    # bulk_plots(show_plots=True)

    # W, H = load_best_taxavec_genus("Acidaminococcus_imsms_bulk", 5)
    # print(H)