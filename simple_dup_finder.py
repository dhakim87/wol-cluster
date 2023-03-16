from common.union_join import SetJoiner
from detnmf.alpha_calculator import AlphaCalculator
from detnmf.beta_calculator import BetaCalculator
from detnmf.determinant_zero import DeterminantZeroCollapse
from detnmf.detnmf import run_detnmf, L1_normalize
from table_info import BiomTable, CSVTable
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from IPython.display import display
import test_linear_clustering
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pandas as pd
from plot_definitions import is_ms, is_new_sample_closure, build_new_samples_set, imsms_plots, imsms_qualitative_class
from coexclusion import counts_to_presence_absence, counts_to_dynamic_presence_absence, presence_absence_to_contingency, pairwise_eval
import os
import biom
from merge_similar_vectors import merge_similar_vectors
from woltka_metadata import list_genera, list_woltka_refs, filter_and_sort_df, list_extended_akkermansia
from plotting import plot_taxavec2, plot_taxavec3


from test_linear_clustering import recursive_subspacer, iterative_clustering, calc_projected, _fit_a_linear_subspace
from plotting import _get_color, _draw_subspace_line

from linear_subspace_clustering import linear_subspace_clustering, calc_subspace_bases
from cluster_manager import ClusterManager, ClusterState, RecursiveClusterer, OUTLIER_CLUSTER_ID
import numpy as np
import matplotlib
import json
import scipy
import data_transform as transformer
import math


def calc_L1_resids(filtered_df, W, H):
    X = filtered_df.to_numpy()
    X2 = np.dot(W, H)

    resids = np.sum(np.abs(X - X2), axis=1)
    resids_pcts = resids / (np.sum(X, axis=1))

    return resids, resids_pcts


def calc_L1_resids_coexclusive(filtered_df, W, H):
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


def pairwise_pearson(filtered_df, thresh=0.95):
    corr_mat = filtered_df.corr()
    starting_mat = corr_mat
    # print(corr_mat)

    corr_mat = corr_mat.where((np.tril(np.ones(corr_mat.shape)) - np.identity(corr_mat.shape[0])).astype(np.bool))
    # print(corr_mat)
    # print(corr_mat.max(axis=1))
    # corr_mat.max(axis=1).hist()
    # plt.show()
    # print(corr_mat.min())

    # foo = corr_mat.max(axis=1)
    # foo = foo.fillna(0)
    # foo = [int(x*1000) for x in sorted(foo)]
    # for i in range(len(foo)):
    #     print(i, foo[i])
    # exit(-1)

    to_collapse = (corr_mat.max(axis=1) > thresh)
    # print(to_collapse.sum())

    to_collapse_to = corr_mat.idxmax(axis=1)
    to_collapse_to = to_collapse_to.where(to_collapse)
    # print(to_collapse_to)

    converter = {}
    for i in range(len(to_collapse_to.index)):
        converter[to_collapse_to.index[i]] = i

    merger = SetJoiner(range(len(to_collapse_to.index)))
    for col1, col2 in to_collapse_to.items():
        if isinstance(col2, str):
            c1 = converter[col1]
            c2 = converter[col2]
            # print("merge", c1, c2)
            merger.merge(c1, c2)

    conv_sets = merger.get_sets()
    # print(conv_sets)
    sets = {}
    for conv_rep in conv_sets:
        rep = to_collapse_to.index[conv_rep]
        conv_rep_set = conv_sets[conv_rep]
        rep_set = set([to_collapse_to.index[r] for r in conv_rep_set])
        sets[rep] = rep_set

    # print(sets)
    out_cols = {}
    for rep in sets:
        out_col = filtered_df[list(sets[rep])].sum(axis=1)
        out_col.name = rep
        out_cols[rep] = out_col

    out_df = pd.DataFrame(out_cols)

    # sns.heatmap(starting_mat)
    # plt.show()
    #
    # out_corr_mat = out_df.corr()
    # sns.heatmap(out_corr_mat)
    # plt.show()

    return out_df, sets


def find_duplicates(filtered_df, thresh=0.5, title="", woltka_meta_df=None):
    fig = plt.figure(figsize=(12, 12))
    i = 0
    page = 1
    to_remove = []
    for c1 in range(len(filtered_df.columns)):
        col1 = filtered_df.columns[c1]
        best_sameness = None
        min_deltasum = 999999999999
        if woltka_meta_df is not None:
            name1 = str(woltka_meta_df[woltka_meta_df["#genome"] == col1]["genus"].iloc[0]) + "\n" + col1
        else:
            name1 = col1
        for c2 in range(c1):
            # is c1 a multiple of c2?
            col2 = filtered_df.columns[c2]
            if woltka_meta_df is not None:
                name2 = str(woltka_meta_df[woltka_meta_df["#genome"] == col2]["genus"].iloc[0]) + "\n" + col2
            else:
                name2 = col2

            c1sum = filtered_df[col1].sum()
            c2sum = filtered_df[col2].sum()
            multiplier = c1sum / c2sum
            c1predicted = filtered_df[col2] * multiplier
            delta = filtered_df[col1]-c1predicted
            deltasum = delta.abs().sum()
            deltasum /= c1sum
            if deltasum < min_deltasum:
                min_deltasum = deltasum
                best_sameness = c2

        # threshold is pretty arbitrary...
        if best_sameness is None or min_deltasum > thresh:
            print("No match for ", name1)
        else:
            col2 = filtered_df.columns[best_sameness]
            if woltka_meta_df is not None:
                name2 = str(woltka_meta_df[woltka_meta_df["#genome"] == col2]["genus"].iloc[0]) + "\n" + col2
            else:
                name2 = col2

            print(name1, "replicates", name2)
            to_remove.append(col1)
            # if name1 == name2:
            #     continue
            i += 1
            # if i <= 16:
            #     ax = fig.add_subplot(4, 4, i)
            #     ax.scatter(x=list(filtered_df[col1].values), y=list(filtered_df[col2].values), s=2)
            #     ax.set_xticks([])
            #     ax.set_yticks([])
            #     ax.set_xlabel(name1)
            #     ax.set_ylabel(name2)
            #
            # if i == 16:
            #     plt.savefig('carlos_figs/' + str(page) + '.png', bbox_inches='tight')
            #     page += 1
            #     plt.show()
            #     fig = plt.figure(figsize=(12, 12))
            #     i = 0
            print(i)

    plt.suptitle(title + " GOTU Dependencies")
    fig.tight_layout()
    plt.savefig('carlos_figs/' + str(page) + '.png', bbox_inches='tight')
    plt.show()

    reduced_dups = list(filtered_df.columns)
    for v in to_remove:
        reduced_dups.remove(v)
    print(title)
    for v in reduced_dups:
        print(v)

    return reduced_dups


def ez_plot(fdf, c1, c2, c3, W=None, H=None, ax=None):
    plot_taxavec3(fdf, "", c1, c2, c3, c1,c2,c3, W, H, subplot_ax=ax, linewidth=4)
    if ax is not None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])


def ez_nmf(fdf, n, column_scale):
    return run_detnmf(
        fdf.to_numpy(),
        n,
        alpha_calculator=AlphaCalculator(0.05, 0, 5000),
        iterations=50000,
        beta_calculator=BetaCalculator(0.0001, 2500),
        det0_handler=DeterminantZeroCollapse(),
        column_scale=column_scale
    )


if __name__ == "__main__":
    WOLTKA_METADATA_PATH="./woltka_metadata.tsv"
    woltka_meta_table = CSVTable(WOLTKA_METADATA_PATH, delimiter="\t")
    woltka_meta_df = woltka_meta_table.load_dataframe()

    BIOM_TABLE="./dataset/biom/IBD200.biom"

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
    filtered_df = filter_and_sort_df(df, woltka_meta_df, "Akkermansia", min_genus_count=0)
    print(filtered_df)

    filtered_df = filtered_df[[
        "G018847035", "G008421395", "G010223015", "G008422705",
        "G002885235", "G019132835", "G018847235", "G018779665",
        "G018379415", "G008421425", "G008421305", "G004557455"]]
    # "G017517385", "G900097105", "G018779965", "G001683795",
    # "G017398355", "G017546445", "G017435365", "G017521405",
    # "G002364575", "G017451815"]]

    # NMF is pretty shitty on this table, possibly because of the huge disparity in column counts.
    # We can scale columns prior to NMF then scale back.  Meh.
    # column min max normalization
    # pre_scaled = filtered_df
    # col_scaling = filtered_df.max()
    # df_norm = (filtered_df * 10000)/(col_scaling)
    #
    # filtered_df = df_norm

    W, H = ez_nmf(filtered_df, 9, column_scale=True)
    fig = plt.figure(figsize=(8, 10))
    ax = fig.add_subplot(3, 2, 1, projection='3d')
    ez_plot(filtered_df, "G018847035", "G008421395", "G010223015", W, H, ax)
    ax = fig.add_subplot(3, 2, 2, projection='3d')
    ez_plot(filtered_df, "G008422705", "G002885235", "G019132835", W, H, ax)
    ax = fig.add_subplot(3, 2, 3, projection='3d')
    ez_plot(filtered_df, "G018847235", "G018779665", "G018379415", W, H, ax)
    ax = fig.add_subplot(3, 2, 4, projection='3d')
    ez_plot(filtered_df, "G008421425", "G008421305", "G004557455", W, H, ax)

    # H = (H.T * col_scaling.to_numpy().reshape(-1,1) * 1/10000).T
    # W, H = L1_normalize(W, H)
    # filtered_df = pre_scaled

    r, rp = calc_L1_resids(filtered_df, W, H)
    r2, rp2 = calc_L1_resids_coexclusive(filtered_df, W, H)
    ax = fig.add_subplot(3, 2, 5)
    _plot_fix_resids(ax, "Akkermansia ", r)
    _plot_fix_resids(ax, "Akkermansia ", r2)
    ax = fig.add_subplot(3, 2, 6)
    _plot_fix_resids(ax, "Akkermansia ", rp, vlines=[0.05, 0.10], x_label="Fraction Reads")
    _plot_fix_resids(ax, "Akkermansia ", rp2, x_label="Fraction Reads")
    plt.suptitle("9 Strains of Akkermansia")
    plt.show()

    fig = plt.figure(figsize=(12, 12))
    i = 0
    page = 0
    for c1 in range(len(filtered_df.columns)):
        col1 = filtered_df.columns[c1]
        for c2 in range(c1):
            col2 = filtered_df.columns[c2]
            i += 1
            if i <= 16:
                ax = fig.add_subplot(4, 4, i)
                plot_taxavec2(filtered_df, "", col1, col2, col1, col2, W, H, ax)
                ax.set_xticks([])
                ax.set_yticks([])
            if i == 16:
                plt.savefig('carlos_figs/taxavec_' + str(page) + '.png', bbox_inches='tight')
                page += 1
                plt.show()
                fig = plt.figure(figsize=(12, 12))
                i = 0

    plt.savefig('carlos_figs/taxavec_' + str(page) + '.png', bbox_inches='tight')
    page += 1
    plt.show()
