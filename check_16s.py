# We examine prep 1 of Caitriona's matrix tube results and find
# that even 16S data built with sortmerna has the problem of redundant columns.
# due to ambiguous reads of 16S (and some potentially funky behavior of sortmerna...?)
from simple_dup_finder import pairwise_pearson
from table_info import BiomTable

from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict

import skbio.alignment
from skbio.sequence.distance import hamming

# Proof that higher score is better
# tab, score, s_e = skbio.alignment.local_pairwise_align_ssw(
# skbio.sequence.DNA("TACGTGTATTTA"),
# skbio.sequence.DNA("TACGTGTATTTA")
# )
# print(score, s_e)
# tab, score, s_e = skbio.alignment.local_pairwise_align_ssw(
#     skbio.sequence.DNA("TACGTTTTA"),
#     skbio.sequence.DNA("TACGTTTTA")
# )
# print(score, s_e)
# tab, score, s_e = skbio.alignment.local_pairwise_align_ssw(
#     skbio.sequence.DNA("TTTTTTTTT"),
#     skbio.sequence.DNA("TATAAAAAA")
# )
# print(score, s_e)

def read_fasta(f, parsekey=None):
    db = {}
    state = 0
    key = ""

    with open(f) as fasta:
        while (True):
            line = fasta.readline()
            if line == "":
                break
            if state == 0:
                if line.startswith(">"):
                    key = line[1:-1]
                    if parsekey is not None:
                        key = parsekey(key)
                    state = 1
            elif state == 1:
                val = line[:-1]
                db[key] = val
                state = 0
    return db


def read_sortmerna(f):
    db = {}
    with open(f) as links:
        while (True):
            line = links.readline()
            if line == "":
                break
            else:
                line = line[:-1]
            ss = line.split()
            db[ss[0]] = ss[1:]
    return db


df = BiomTable([
    "dataset/biom/16s/145131_otu_table.biom",
    # "dataset/biom/16s/145135_all.biom",
    # "dataset/biom/16s/145136_reference-hit.biom",
]).load_dataframe()

otu_metadata = pd.read_csv("./dataset/biom/16s/97_otu_taxonomy.txt", index_col=0, names=["id", "taxonomy"], sep="\t")
otu_metadata.index = otu_metadata.index.map(str)

db_16s = read_fasta("./dataset/biom/16s/97_otus.fasta")
all_reads = read_fasta("./dataset/biom/16s/sortmerna_picked_otus/145126_seqs.fasta",
                       parsekey=lambda x: x.split()[0])
read_map = read_sortmerna("./dataset/biom/16s/sortmerna_picked_otus/seqs_otus.txt")


def get_prefix(otu_metadata, otu, prefix):
    tax = otu_metadata.loc[otu, "taxonomy"]
    ss = tax.split("; ")
    for s in ss:
        if s.startswith(prefix):
            return s[3:]
    return "Unknown"

def get_taxa(otu_metadata, otu):
    return otu_metadata.loc[otu, "taxonomy"]

def get_species(otu_metadata, otu):
    return get_prefix(otu_metadata, otu, "s__")

def get_genus(otu_metadata, otu):
    return get_prefix(otu_metadata, otu, "g__")

def get_family(otu_metadata, otu):
    return get_prefix(otu_metadata, otu, "f__")

def get_order(otu_metadata, otu):
    return get_prefix(otu_metadata, otu, "o__")

def get_diff(otu_metadata, otu_a, otu_b):
    for prefix in ["k__", "p__", "c__", "o__", "f__", "g__", "s__"]:
        a = get_prefix(otu_metadata, otu_a, prefix)
        b = get_prefix(otu_metadata, otu_b, prefix)
        if a != b:
            if a == "" or b == "":
                return prefix + "Unknown"
            return prefix
    return "Same"

def sort_by_taxonomy(otu_metadata, df):
    mapper = {}
    for col in df.columns:
        tax = otu_metadata.loc[col, "taxonomy"]
        mapper[col] = tax
    cols_sorted = sorted(df.columns, key=lambda x: mapper[x])
    return df[cols_sorted]

def plot_heatmap(otu_metadata, df):
    fig = plt.figure(figsize=(12, 10), dpi=300) # Gotta tweak fontsize if you touch dpi
    pp_df_sorted = sort_by_taxonomy(otu_metadata, df)
    sns.heatmap(pp_df_sorted.corr())
    ticks = [x + 0.5 for x in range(len(pp_df_sorted.columns))]
    ylabels = [get_species(otu_metadata, gotu) for gotu in pp_df_sorted.columns]
    xlabels = [get_genus(otu_metadata, gotu) for gotu in pp_df_sorted.columns]
    plt.xticks(ticks=ticks, labels=xlabels, fontsize=6) # Gotta tweak fontsize if you touch dpi
    plt.yticks(ticks=ticks, labels=ylabels, fontsize=6) # Gotta tweak fontsize if you touch dpi
    plt.title("Merged GOTU Pearson Correlation")
    plt.subplots_adjust(left=0.2, bottom=.15)
    plt.show()


def print_tab(tab, hammings=None, indices=None):
    rowlen = 150
    length = len(tab.loc[0,:])

    if indices is None:
        indices = [0,1]
    for i in range(0, length, rowlen):
        s = i
        e = min(length-1, i + rowlen)
        for idx in indices:
            print(tab.loc[idx, s:e])
        if hammings is not None:
            print("".join(hammings[s:e]))
        if e != length-1:
            print()


def filter_gotus(df, otu_metadata, do_abundance=True, do_pairwise_pearson=True):
    print("Original shape")
    print(df.shape)
    if do_abundance:
        # Filter GOTUs by abundance
        df_frac = df.sum() / df.sum().sum()
        passing_gotus = list(df_frac[df_frac > 1/10000].index)
        df = df[passing_gotus]
        print("post abundance")
        print(df.shape)

    if do_pairwise_pearson:
        # plot_heatmap(otu_metadata, df)
        # Collapse GOTUs with pairwise pearson
        col_sums = df.sum()
        col_sums = col_sums.sort_values(ascending=False)
        df = df[col_sums.index]
        pre_pp = df
        df, sets = pairwise_pearson(df, thresh=0.95)
        print("post pearson")
        print(df.shape)

        counts = defaultdict(int)
        for rep in sets:
            for mem in sets[rep]:
                if mem == rep:
                    continue
                d = get_diff(otu_metadata, rep, mem)
                counts[d] += 1

                if d == "p__" or d == "p__Unknown":
                    # print(rep + " (likely real)")
                    # print(db_16s[rep])
                    # print(mem + " (likely redundant)")
                    # print(db_16s[mem])
                    rep_rna = skbio.sequence.DNA(db_16s[rep], validate=True)
                    mem_rna = skbio.sequence.DNA(db_16s[mem], validate=True)
                    tab, alignment, start_end = skbio.alignment.local_pairwise_align_ssw(rep_rna, mem_rna)
                    print(start_end)
                    memstart = start_end[1][0]
                    hammings = []
                    length = len(tab.loc[0,:])
                    hamlen = 50
                    for i in range(0, length-hamlen-1):
                        h = hamming(tab.loc[0, i:i+hamlen], tab.loc[1, i:i+hamlen])
                        h = int(h * hamlen)
                        if h < 10:
                            h = str(h)
                        else:
                            h = "-"
                        hammings.append(h)
                    # print(alignment)
                    # print(start_end)

                    print_tab(tab, hammings)

                    plt.scatter(pre_pp[rep], pre_pp[mem])
                    maxval = pre_pp[mem].max()
                    row = pre_pp[[rep,mem]][pre_pp[mem] == maxval]
                    plt.annotate(row.index[0], (row[rep], row[mem]))

                    print(row.index[0])
                    foo = 0
                    reads = set([])
                    for key in read_map[mem]:
                        if key.startswith(row.index[0]):
                            reads.add(all_reads[key])
                            foo += 1
                    print(foo)
                    print(len(reads), "unique")
                    mem_avg_score = 0
                    mem_avg_match_len = 0
                    rep_avg_score = 0
                    rep_avg_match_len = 0

                    for r in reads:
                        tab_mem, mem_score, mem_start_end = skbio.alignment.local_pairwise_align_ssw(
                            skbio.sequence.DNA(r),
                            mem_rna)
                        tab_rep, rep_score, rep_start_end = skbio.alignment.local_pairwise_align_ssw(
                            skbio.sequence.DNA(r),
                            rep_rna)
                        print("Full len:", len(r))
                        print(mem_start_end, rep_start_end)
                        print(mem_score, " vs", rep_score)
                        mem_avg_score += mem_score
                        rep_avg_score += rep_score
                        mem_avg_match_len += (mem_start_end[1][1] - mem_start_end[1][0])
                        rep_avg_match_len += (rep_start_end[1][1] - rep_start_end[1][0])
                        print("---")
                        print_tab(tab_mem)
                        print("vs")
                        print_tab(tab_rep)
                        print("---")

                    mem_avg_score /= len(reads)
                    rep_avg_score /= len(reads)
                    mem_avg_match_len /= len(reads)
                    rep_avg_match_len /= len(reads)

                    print("Stats:")
                    print("MemScore:", mem_avg_score)
                    print("RepScore:", rep_avg_score)
                    print("MemLen:", mem_avg_match_len)
                    print("RepLen:", rep_avg_match_len)

                    plt.xlabel(get_taxa(otu_metadata, rep) + str(rep))
                    plt.ylabel(get_taxa(otu_metadata, mem) + str(mem))
                    plt.show()
        print(counts)

        plot_heatmap(otu_metadata, df)

    return df

filter_gotus(df, otu_metadata, True, True)

