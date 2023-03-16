import pandas as pd

HACK_FOR_CARLOS=False


def list_extended_akkermansia():
    print("Applying hack to extend akkermansia genomes")
    # TODO FIXME HACK:  Support for Carlos' extended Akkermansia dataset
    mmdata = pd.read_csv("./dataset/biom/microbe_metadata.txt", sep='\t')
    akk_columns = []
    akk_genus = []
    akk_species = []
    for i, row in mmdata.iterrows():
        feat_id = row["Feature ID"]
        taxon = row["Taxon"]
        if taxon.startswith("Akkermansia"):
            akk_columns.append(feat_id)
            akk_genus.append("Akkermansia")
            akk_species.append(taxon)

    akk_df = pd.DataFrame({"#genome": akk_columns, "genus":akk_genus, "species":akk_species}, index=akk_columns)
    return akk_df


def list_genera(df, woltka_meta_df, min_sample_count=0, min_genus_count=0):
    woltka_included = woltka_meta_df[woltka_meta_df['#genome'].isin(df.columns)]
    if HACK_FOR_CARLOS:
        woltka_included = woltka_included[["#genome", "genus", "species"]]
        woltka_included = woltka_included.set_index("#genome")
        akk_df = list_extended_akkermansia()
        akk_df = akk_df[akk_df['#genome'].isin(df.columns)]
        woltka_included = woltka_included.drop(akk_df.index.values, axis=0, errors="ignore")
        woltka_included['#genome'] = woltka_included.index
        woltka_included.index.name=""
        akk_df.index.name=""
        woltka_included = woltka_included.merge(akk_df, how="outer")

    vcs = woltka_included['genus'].value_counts()
    genera = sorted(vcs[vcs>1].index.astype(str).tolist())

    filt_genera = ['all']
    for g in genera:
        filtered_df = df[list_woltka_refs(df, woltka_meta_df, g)['#genome']]
        filtered_df_sum = filtered_df.sum(axis=1)
        filtered_df = filtered_df[filtered_df_sum >= min_genus_count]
        if len(filtered_df) >= min_sample_count:
            filt_genera.append(g)
    return filt_genera


def list_woltka_refs(df, woltka_meta_df, genus=None):
    woltka_included = woltka_meta_df[woltka_meta_df['#genome'].isin(df.columns)]
    if HACK_FOR_CARLOS and (genus=='Akkermansia' or genus is None):
        woltka_included = woltka_included[["#genome", "genus", "species"]]
        woltka_included = woltka_included.set_index("#genome")
        akk_df = list_extended_akkermansia()
        akk_df = akk_df[akk_df['#genome'].isin(df.columns)]
        woltka_included = woltka_included.drop(akk_df.index.values, axis=0, errors="ignore")
        woltka_included['#genome'] = woltka_included.index
        woltka_included.index.name=""
        akk_df.index.name=""
        woltka_included = woltka_included.merge(akk_df, how="outer")

    if genus is None:
        refs = woltka_included
    else:
        refs = woltka_included[woltka_included['genus']==genus]        

    filtered_df = df[refs['#genome']]
    col_sums = filtered_df.sum()
    col_sums.name='total'
    refs = refs.join(col_sums, on='#genome')

    refs = refs.reset_index()
    refs = refs.sort_values(["total", "#genome"], ascending=False)[['total', '#genome','species']]
    return refs


def filter_and_sort_df(df, woltka_meta_df, genus, min_genus_count=0):
    if genus == 'all':
        refs_df = list_woltka_refs(df, woltka_meta_df)
    else:
        refs_df = list_woltka_refs(df, woltka_meta_df, genus)

    genomes = refs_df['#genome'].tolist()
    filtered_df = df[genomes]

    filtered_df_sum = filtered_df.sum(axis=1)
    filtered_df = filtered_df[filtered_df_sum >= min_genus_count]
    return filtered_df
