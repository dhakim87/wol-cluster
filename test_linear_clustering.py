import matplotlib
matplotlib.use('TkAgg')
from linear_subspace_clustering import linear_subspace_clustering, \
    _fit_a_linear_subspace
from table_info import BiomTable,CSVTable
import sklearn.cluster
from sklearn.metrics import silhouette_score
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import math
from plotting import _get_color, _draw_subspace_line

MIN_GENUS_COUNT = 500

# These clusters are the result of Xiaoyuan's SNP analysis.  She used MIDAS, metagenomic inter-species diversity analysis system
# May be able to do a comparison to show my tool is better when fitting low read counts.
# clustered_samples_2 = [
#     ("71801-0158", 2, 1),
#     ("71802-0150", 2, 1),
#     ("71701-0138", 2, 1),
#     ("71802-0005", 2, 1),
#     ("71701-0061", 2, 1),
#     ("71701-0014", 2, 1),
#     ("71801-0083", 2, 1),
#     ("71801-0072", 2, 1),
#     ("76401-0027", 2, 1),
#     ("71701-0013", 2, 1),
#     ("71601-0008", 2, 1),
#     ("71801-0031", 2, 1),
#     ("71402-0077", 2, 1),
#     ("71801-0013", 2, 1),
#     ("71402-0002", 2, 1),
#     ("71801-0012", 2, 1),
#     ("71701-0001", 4, 2),
#     ("71401-0093", 4, 2),
#     ("71401-0249", 4, 2),
#     ("71402-0106", 4, 2),
#     ("71801-0035", 4, 2),
#     ("71402-0073", 4, 2),
#     ("71401-0121", 4, 2),
#     ("71702-0167", 4, 2),
#     ("71502-0026", 4, 2),
#     ("71401-0260", 4, 2),
#     ("71701-0051", 4, 2),
#     ("71701-0004", 4, 2),
#     ("71402-0249", 4, 2),
#     ("71802-0057", 4, 2),
#     ("71702-0155", 4, 2),
#     ("71401-0197", 4, 2),
#     ("71801-0091", 1, 2),
#     ("71702-0050", 1, 2),
#     ("71701-0074", 1, 2),
#     ("71402-0126", 1, 2),
#     ("71801-0023", 1, 2),
#     ("71702-0151", 1, 2),
#     ("71702-0031", 1, 2),
#     ("71501-0039", 1, 2),
#     ("71501-0018", 1, 2),
#     ("76401-0043", 1, 2),
#     ("71802-0035", 1, 2),
#     ("71801-0053", 1, 2),
#     ("71701-0029", 1, 2),
#     ("71801-0069", 1, 2),
#     ("71802-0060", 1, 2),
#     ("71401-0080", 1, 2),
#     ("71401-0033", 1, 2),
#     ("71802-0003", 1, 2),
#     ("71601-0026", 1, 2),
#     ("71701-0045", 1, 2),
#     ("71602-0143", 1, 2),
#     ("76402-0001", 1, 2),
#     ("71801-0025", 1, 2),
#     ("71701-0166", 1, 2),
#     ("71801-0046", 1, 2),
#     ("71501-0044", 1, 2),
# ]
#
# clustered_samples = [
#     ("71801-0158", 2),
#     ("71802-0150", 2),
#     ("71701-0138", 2),
#     ("71802-0005", 2),
#     ("71701-0061", 2),
#     ("71701-0014", 2),
#     ("71801-0083", 2),
#     ("71801-0072", 2),
#     ("76401-0027", 2),
#     ("71701-0013", 2),
#     ("71601-0008", 2),
#     ("71801-0031", 2),
#     ("71402-0077", 2),
#     ("71801-0013", 2),
#     ("71402-0002", 2),
#     ("71801-0012", 2),
#     ("71701-0001", 1),
#     ("71401-0093", 1),
#     ("71401-0249", 1),
#     ("71402-0106", 1),
#     ("71801-0035", 1),
#     ("71402-0073", 1),
#     ("71401-0121", 1),
#     ("71702-0167", 1),
#     ("71502-0026", 1),
#     ("71401-0260", 1),
#     ("71701-0051", 1),
#     ("71701-0004", 1),
#     ("71402-0249", 1),
#     ("71802-0057", 1),
#     ("71702-0155", 1),
#     ("71401-0197", 1),
#     ("71801-0091", 1),
#     ("71702-0050", 1),
#     ("71701-0074", 1),
#     ("71402-0126", 1),
#     ("71801-0023", 1),
#     ("71702-0151", 1),
#     ("71702-0031", 1),
#     ("71501-0039", 1),
#     ("71501-0018", 1),
#     ("76401-0043", 1),
#     ("71802-0035", 1),
#     ("71801-0053", 1),
#     ("71701-0029", 1),
#     ("71801-0069", 1),
#     ("71802-0060", 1),
#     ("71401-0080", 1),
#     ("71401-0033", 1),
#     ("71802-0003", 1),
#     ("71601-0026", 1),
#     ("71701-0045", 1),
#     ("71602-0143", 1),
#     ("76402-0001", 1),
#     ("71801-0025", 1),
#     ("71701-0166", 1),
#     ("71801-0046", 1),
#     ("71501-0044", 1),
# ]

samples_to_highlight = [
    "71701-0061",
    "71402-0077",
    "71801-0077",
    "71701-0013",
    "71701-0014"
]

samples_to_highlight_2 = [
    "71702-0155",
    "71401-0260",
    "71701-0004",
    "71401-0080",
    "71501-0018"
]




def plot_scatter3(filtered_df, best_clustering, cluster_info_map, title, col1_index, col2_index, col3_index, target_cluster=None):
    
    subspace_bases = calc_subspace_bases(filtered_df, best_clustering, cluster_info_map)
    all_resid = calc_all_residuals(filtered_df, best_clustering, subspace_bases)
    all_proj = calc_projected(filtered_df, best_clustering, subspace_bases)
    df_sum = filtered_df.sum(axis=1)
    frac_resid = all_resid / df_sum

    print("Fractional Residuals")
    series = {"frac_resid": frac_resid, "cluster": best_clustering, "resid": all_resid}
    for label in cluster_info_map:
        cross_cluster_resid = calc_all_residuals(filtered_df, np.full(len(best_clustering), label), subspace_bases)
        series[label] = cross_cluster_resid
    clusters_df = pd.DataFrame(series)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.precision", 3):  # more options can be specified also
    #     print(clusters_df)

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    c1 = filtered_df.columns[col1_index]
    c2 = filtered_df.columns[col2_index]
    c3 = filtered_df.columns[col3_index]

    print("COMPUTING FOO")
    # foo = filtered_df.loc[[x[0] for x in clustered_samples]]

    maxx = max(filtered_df[c1])
    maxy = max(filtered_df[c2])
    maxz = max(filtered_df[c3])

    if target_cluster is not None:
        pidx = np.where(best_clustering == target_cluster)[0]
        cdf = filtered_df.iloc[pidx]
        maxx = max(cdf[c1])
        maxy = max(cdf[c2])
        maxz = max(cdf[c3])

    min_label = min(best_clustering)
    max_label = max(best_clustering)

    x = np.linspace(0, maxx, 10)
    y = np.linspace(0, maxy, 10)

    for label in cluster_info_map:
        # if target_cluster is not None and label != target_cluster:
        #     continue
        dim = cluster_info_map[label]['dim']
        if dim == 1:
            pidx = np.where(best_clustering == label)[0]
            subspace_basis_vectors = _fit_a_linear_subspace(filtered_df.T.to_numpy()[:,pidx], dim)
            u = subspace_basis_vectors[:,0][[col1_index,col2_index,col3_index]].T
            _draw_subspace_line(ax, u, maxx,maxy,maxz, label, min_label, max_label)
            print("BASIS")
            print(label)
            print(np.array(subspace_basis_vectors)[:,0])
            print(pd.Series(data=np.array(subspace_basis_vectors)[:,0], index=filtered_df.columns))

        if dim == 2:
            pidx = np.where(best_clustering == label)[0]
            subspace_basis_vectors = _fit_a_linear_subspace(filtered_df.T.to_numpy()[:, pidx], dim)
            print(subspace_basis_vectors)
            # the basis vectors are orthogonal
            # But when you project them down to this space, their projections may not be orthogonal
            # They could even be parallel, or even 0 (Take u,v = 0,0,1,-1 and 0,0,1,1 projected to first three dimensions)
            # If they are parallel, we should be drawing a line, not a plane
            # If they are 0, we should just draw a single point (happens to be at the origin
            u = subspace_basis_vectors[:,0][[col1_index,col2_index,col3_index]]
            v = subspace_basis_vectors[:,1][[col1_index,col2_index,col3_index]]

            npu = np.array(u)
            npv = np.array(v)
            lenu = np.linalg.norm(u)
            lenv = np.linalg.norm(v)
            if lenu < 0.01 and lenv < 0.01:
                # Both of the vectors are basically 0.  Can't draw anything.
                continue
            elif lenu < 0.01:
                _draw_subspace_line(ax, v, maxx,maxy,maxz, label, min_label, max_label)
                continue
            elif lenv < 0.01:
                _draw_subspace_line(ax, u, maxx,maxy,maxz, label, min_label, max_label)
                continue

            npu = npu / lenu
            npv = npv / lenv

            dotproduct = np.dot(npu,npv)
            if dotproduct > 0.9 or dotproduct < -0.9:
                print(dotproduct)
                # Vectors are basically parallel.  Can't draw a plane, maybe can draw a line.
                if dotproduct < 0:
                    npu = -npu
                npu = (npu + npv) / 2
                _draw_subspace_line(ax, npu, maxx, maxy, maxz, label, min_label, max_label)
                continue

            normal = np.cross(u.T, v.T)
            # Normal vector gives x*n0 + y*n1 + z*n2 = 0
            # Divide everything by n2
            # x * n0/n2 + y * n1/n2 + z = 0
            # z = -n0/n2 * x - n1 / n2 * y
            X, Y = np.meshgrid(x,y)
            Z = -normal[0]/normal[2] * X - normal[1] / normal[2] * Y
            rgba = _get_color(label, min_label, max_label)
            rgba = rgba[0],rgba[1],rgba[2],0.25
            surf = ax.plot_surface(X, Y, Z, color=rgba)

            # xs = []
            # ys = []
            # zs = []
            # for x in range(-10000,10000,100):
            #     for y in range(-10000, 10000, 100):
            #         xs.append(u[0] * x + v[0] * y)
            #         ys.append(u[1] * x + v[1] * y)
            #         zs.append(u[2] * x + v[2] * y)
            # ax.scatter(xs,ys,zs, c='black', marker='*')

    if target_cluster is None:
        if best_clustering is None:
            ax.scatter(filtered_df[c1], filtered_df[c2], filtered_df[c3])
            projected = all_proj
            ax.scatter(projected[c1], projected[c2], projected[c3], c=best_clustering, marker='x')
        else:
            # print("HELLO I AM HERE")
            # cmap = matplotlib.cm.get_cmap('Set1')
            # cdict = {x[0]: cmap(x[1] / (len(cluster_info_map)-1)) for x in clustered_samples_2}
            # mdict = {x[0]: ['x','o','v','D','D','D'][x[1] % 6] for x in clustered_samples_2}
            # ccol = foo.index.map(cdict)
            # mcol = foo.index.map(mdict)
            # print(ccol)
            # print(mcol)

            # foo1 = foo[mcol == 'o']
            # foo2 = foo[mcol == 'v']
            # foo3 = foo[mcol == 'D']

            ax.scatter(filtered_df[c1], filtered_df[c2], filtered_df[c3], c=best_clustering, cmap="Set1")
            projected = all_proj
            ax.scatter(projected[col1_index], projected[col2_index], projected[col3_index], c=best_clustering, marker='x', cmap='Set1')
            # ax.scatter(filtered_df[c1], filtered_df[c2], filtered_df[c3], c='k')
            # ax.scatter(foo1[c1], foo1[c2], foo1[c3], c=foo1.index.map(cdict), marker='o')
            # ax.scatter(foo2[c1], foo2[c2], foo2[c3], c=foo2.index.map(cdict), marker='v')
            # ax.scatter(foo3[c1], foo3[c2], foo3[c3], c=foo3.index.map(cdict), marker='*')

            # for i in range(projected.shape[1]):
            #     ax.plot3D([filtered_df[c1][i], projected[col1_index][i]], [filtered_df[c2][i], projected[col2_index][i]], [filtered_df[c3][i], projected[col3_index][i]], c='gray')

            # print(filtered_df[c1][0], filtered_df[c2][0], filtered_df[c3][0])
            # print(projected[col1_index][0], projected[col2_index][0], projected[col3_index][0])
            # print(
            #     math.sqrt(
            #         (filtered_df[c1][0] - projected[col1_index][0])**2 +
            #         (filtered_df[c2][0] - projected[col2_index][0])**2 +
            #         (filtered_df[c3][0] - projected[col3_index][0])**2
            #     )
            # )
    else:
        pidx = np.where(best_clustering == target_cluster)[0]
        cdf = filtered_df.iloc[pidx]
        rgba = _get_color(target_cluster, min_label, max_label)
        ax.scatter(cdf[c1], cdf[c2], cdf[c3], c=[rgba])
        projected = all_proj[:,pidx]
        print(all_proj.shape)
        ax.scatter(projected[col1_index], projected[col2_index], projected[col3_index], c=[rgba], marker='x')

    patches = []
    for label in range(min_label, max_label + 1):
        # if target_cluster is not None and label != target_cluster:
        #     continue
        rgba = _get_color(label, min_label, max_label)
        if label == -1:
            patch = matplotlib.patches.Patch(color=rgba, label="Outliers: " + str(len(best_clustering[best_clustering == -1])))
            patches.append(patch)
        else:
            patch = matplotlib.patches.Patch(
                color=rgba,
                label=str(label) +
                      ": dim=" + str(cluster_info_map[label]['dim']) +
                      "#fit=" + str(cluster_info_map[label]['num_points']-cluster_info_map[label]['total_outliers']))
            patches.append(patch)

    plt.title(title)
    ax.set_xlabel(c1)
    ax.set_ylabel(c2)
    ax.set_zlabel(c3)
    ax.set_xlim([0, maxx * 1.5 + 100])
    ax.set_ylim([0, maxy * 1.5 + 100])
    ax.set_zlim([0, maxz * 1.5 + 100])
    plt.legend(handles=patches)
    plt.show()


def plot_dim2_clusters(filtered_df, best_clustering, cluster_info_map):
    subspace_bases = calc_subspace_bases(filtered_df, best_clustering, cluster_info_map)
    data = filtered_df.T.to_numpy()
    min_label = min(best_clustering)
    max_label = max(best_clustering)

    for cluster_id in cluster_info_map:
        if cluster_info_map[cluster_id]['dim'] != 2:
            continue
        u = subspace_bases[cluster_id][:, 0]
        v = subspace_bases[cluster_id][:, 1]
        print(u)
        print(v)

        pidx = np.where(best_clustering == cluster_id)[0]

        out_pts = []
        for idx in pidx:
            vv = data[:, idx]
            proj_vv = np.matmul(subspace_bases[cluster_id].T, vv)
            out_pts.append(proj_vv)
        out_pts = np.array(out_pts)

        rgba = _get_color(cluster_id, min_label, max_label)
        plt.title("Cluster:" + str(cluster_id) + " #Pts:" + str(len(pidx)))
        plt.scatter(out_pts[:, 0], out_pts[:, 1], c=[rgba])
        plt.axis('equal')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()

        out_pts = []
        for idx in range(data.shape[1]):
            vv = data[:, idx]
            proj_vv = np.matmul(subspace_bases[cluster_id].T, vv)
            out_pts.append(proj_vv)
        out_pts = np.array(out_pts)

        rgba = _get_color(cluster_id, min_label, max_label)
        plt.title("Cluster:" + str(cluster_id) + " All Points")
        plt.scatter(out_pts[:, 0], out_pts[:, 1], c='black')
        plt.axis('equal')
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.show()




def plot_scatter(filtered_df, best_clustering, title, col1_index, col2_index):
    if col1_index == col2_index:
        print(title, "Plot Skipped (same columns)")
        return
    c1 = filtered_df.columns[col1_index]
    c2 = filtered_df.columns[col2_index]
    maxx = max(filtered_df[c1])
    maxy = max(filtered_df[c2])

    # plt.subplot(1,2,1)
    if best_clustering is None:
        plt.scatter(filtered_df[c1], filtered_df[c2])
    else:
        plt.scatter(filtered_df[c1], filtered_df[c2], c=best_clustering, cmap="Set1")

    # if best_clustering is not None:
    #     for line_index in range(len(best_clustering.cluster_centers_)):
    #         line = best_clustering.cluster_centers_[line_index]
    #         lx = line[col1_index]
    #         ly = line[col2_index]
    #         if lx < 0:
    #             lx = 0
    #         if ly < 0:
    #             ly = 0
    #
    #         if lx == 0 and ly == 0:
    #             x = 0
    #             y = 0
    #         elif lx == 0:
    #             x = 0
    #             y = maxy
    #         elif ly == 0:
    #             x = maxx
    #             y = 0
    #         else:
    #             x = maxx
    #             y = ly * maxx / lx
    #             if y > maxy:
    #                 x = lx * maxy / ly
    #                 y = maxy
    #
    #         cmap = matplotlib.cm.get_cmap('Set1')
    #         plt.plot([0, x], [0, y], c=cmap(line_index / (len(best_clustering.cluster_centers_)-1)))

    plt.axis('equal')
    plt.title(title)
    plt.xlabel(c1)
    plt.ylabel(c2)

    # plt.subplot(1,2,2)
    # centers_x = []
    # centers_y = []
    # if best_clustering is None:
    #     plt.scatter(simplex_df[c1], simplex_df[c2])
    # else:
    #     plt.scatter(simplex_df[c1], simplex_df[c2], c=best_clustering, cmap="Set1")

    # if best_clustering is not None:
    #     for cluster_index in range(len(best_clustering.cluster_centers_)):
    #         centers_x.append(best_clustering.cluster_centers_[cluster_index][col1_index])
    #         centers_y.append(best_clustering.cluster_centers_[cluster_index][col2_index])
    #
    #     plt.scatter(
    #         centers_x, centers_y,
    #         s=80, c=range(len(centers_x)),
    #         cmap="Set1", marker='X',
    #         edgecolors='black')
    #     plt.xlabel(c1)
    plt.show()


def _parse_sample_id(index: str):
    # print(index)
    # Input of form Q.71401.0009.2016.02.23
    # Output of form 71401-0009
    ss = index.split('.')
    if len(ss) < 3:
        print("BAD: ", index)
        raise ValueError("Can't convert sample id:", index)

    sample_id = ss[1] + "-" + ss[2]
    # print("GOOD: ", index, "->", sample_id)
    return sample_id


def _filter_to_samples(df):
    BAD_SAMPLE_PREFIXES = [
        'NA.',
        'UCSF.',
        'COLUMN.',
        'Column.',
        'Q.DOD',
        'Zymo.',
        'Mag.Bead.Zymo.'
    ]

    bad_rows = df.index.str.startswith(BAD_SAMPLE_PREFIXES[0])
    for i in range(1, len(BAD_SAMPLE_PREFIXES)):
        bad_rows |= df.index.str.startswith(BAD_SAMPLE_PREFIXES[i])
    return df[~bad_rows]


def calc_cluster_residuals(cluster_assignments, residuals, sums):
    labels = np.unique(cluster_assignments)

    output = {}
    num_outliers = {}

    fractional_residuals = residuals / sums
    outliers = {}

    for label in labels:
        idx = np.where(cluster_assignments == label)[0]
        # cluster_residuals = residuals[idx]
        # cluster_sums = sums[idx]
        cluster_fractional_residuals = fractional_residuals[idx]

        # dfdebug = pd.DataFrame()
        # dfdebug['sum'] = cluster_sums
        # dfdebug['frac'] = cluster_fractional_residuals
        # print(dfdebug)
        # print(np.quantile(cluster_fractional_residuals, [.50,.60,.70,.80,.90,1.0]))
        # print(np.median(cluster_fractional_residuals), np.mean(cluster_fractional_residuals), max(dfdebug['frac']))
        # dfdebug.hist(column='frac')
        # plt.show()

        # Sort of arbitrary what we threshold on.  Could pick mean or median residual, or look at quantiles
        # This picks the .8 quantile.
        output[label] = np.quantile(cluster_fractional_residuals, .8)
        num_outliers[label] = len(cluster_fractional_residuals[cluster_fractional_residuals > 0.10])
        outliers[label] = (cluster_fractional_residuals > 0.10).values

    return output, num_outliers, outliers


def calc_subspace_bases(filtered_df, cluster_assignments, cluster_info_map):
    data = filtered_df.T.to_numpy()
    subspace_dims = {k:cluster_info_map[k]['dim'] for k in cluster_info_map.keys()}
    labels = np.unique(cluster_assignments)

    vecs = {}
    for gg in labels:
        pidx = np.where(cluster_assignments == gg)[0]
        if gg == -1:
            continue
        subspace_basis_vectors = _fit_a_linear_subspace(data[:, pidx], subspace_dims[gg])
        vecs[gg] = subspace_basis_vectors
    return vecs


def calc_projected(filtered_df, cluster_assignments, subspace_bases):
    data = filtered_df.T.to_numpy()
    labels = np.unique(cluster_assignments)
    out_pts = np.copy(data)

    for gg in labels:
        pidx = np.where(cluster_assignments == gg)[0]
        if gg == -1:
            continue

        # These are orthonormal column vectors
        # TODO FIXME HACK:  Ack, this turns every face of the conic polyhedron into a plane doesn't it?
        #  I need to eventually retrieve basis vectors of the conic polyhedron in order to transform the data
        subspace_basis_vectors = subspace_bases[gg]
        for idx in pidx:
            vv = data[:, idx]
            proj_vv = np.matmul(subspace_basis_vectors, np.matmul(subspace_basis_vectors.T, vv))
            out_pts[:, idx] = proj_vv

    return out_pts


def calc_all_residuals(filtered_df, cluster_assignments, subspace_bases):
    data = filtered_df.T.to_numpy()
    nr_data_points = data.shape[1]
    labels = np.unique(cluster_assignments)
    residuals = np.zeros(nr_data_points)

    for gg in labels:
        pidx = np.where(cluster_assignments == gg)[0]
        if gg == -1:
            np.put(residuals, pidx, -1)
            continue

        # These are orthonormal column vectors
        # TODO FIXME HACK:  Ack, this turns every face of the conic polyhedron into a plane doesn't it?
        #  I need to eventually retrieve basis vectors of the conic polyhedron in order to transform the data
        subspace_basis_vectors = subspace_bases[gg]
        dists = []
        for idx in pidx:
            vv = data[:, idx]
            proj_vv = np.matmul(subspace_basis_vectors, np.matmul(subspace_basis_vectors.T, vv))
            dists.append(np.linalg.norm(vv - proj_vv))
        np.put(residuals, pidx, dists)

    return residuals


def calc_cluster_sizes(cluster_assignments):
    labels = np.unique(cluster_assignments)
    output = {}
    for label in labels:
        output[label] = len(np.where(cluster_assignments == label)[0])
    return output


def greedy_subspacer(df, nr_clusters, subspace_dim):
    df_sum = df.sum(axis=1)

    def subspace_helper(df, nr_clusters, subspace_dim):
        initial_clusters, initial_residuals = linear_subspace_clustering(df.T.to_numpy(), nr_clusters=nr_clusters, subspace_dim=subspace_dim)
        cluster_80_residuals, num_outliers, outliers = calc_cluster_residuals(initial_clusters, initial_residuals, df_sum)
        cluster_sizes = calc_cluster_sizes(initial_clusters)

        total_outliers = 0
        if len(cluster_80_residuals) != nr_clusters:
            print("Found fewer clusters than requested")
            return("Fail", None, None, None, None, None)
        for c in cluster_80_residuals:
            if cluster_80_residuals[c] > 0.10:
                return ("Fail", None, None, None, None, None)
            if cluster_sizes[c] < 10:
                return ("Fail", None, None, None, None, None)
            total_outliers += num_outliers[c]
        return "Pass", initial_clusters, initial_residuals, cluster_80_residuals, cluster_sizes, total_outliers

    passfail, initial_clusters, initial_residuals, cluster_80_residuals, cluster_sizes, total_outliers = subspace_helper(df, nr_clusters, subspace_dim)
    if passfail == "Fail":
        return "Fail", None

    # We have a good solution, let's try subdividing each cluster if possible.
    subcluster_info = {"clusters": initial_clusters, "num_clusters": len(cluster_80_residuals)}
    for cluster_key in cluster_80_residuals:
        idx = np.where(initial_clusters == cluster_key)[0]
        cluster_df = df.iloc[idx]
        # Try dividing it into d-1 clusters of dimension d-1 where d is the dimension of the subspace for that cluster
        if subspace_dim > 1:
            possible = []
            for num_clusters in range(1, max(subspace_dim, 5)):  # Small fudge factor to deal with co (hyper)planar subspace division, we always try fitting at least 4 things of the smaller dimension
                cl_passfail, cl_initial_clusters, cl_initial_residuals, cl_cluster_80_residuals, cl_cluster_sizes, cl_total_outliers = subspace_helper(cluster_df, num_clusters, subspace_dim-1)

                if cl_passfail == "Fail":
                    continue
                else:
                    # Found a potentially better solution, figure out how to choose between this one and all other ones
                    CLUSTER_SCORE_MULTIPLIER = 10
                    possible.append((num_clusters*CLUSTER_SCORE_MULTIPLIER + cl_total_outliers, cl_total_outliers, num_clusters))
                    if num_clusters == 1 and cl_total_outliers < CLUSTER_SCORE_MULTIPLIER:
                        break

            if len(possible) > 0:
                # print("SUBDIVIDING CLUSTER: ", cluster_key)
                # for p in possible:
                #     print(p)

                # Oof, picking by score seems to due terribly!
                # possible.sort(key=lambda x: x[0])
                chosen = possible[0]
                print("RECURSE", cluster_key, "Subclusters:", chosen[2], "Expected outliers:", chosen[1])
                passfail, greedy_cluster_info = greedy_subspacer(cluster_df, chosen[2], subspace_dim-1)
                subcluster_info[cluster_key] = greedy_cluster_info
        if cluster_key not in subcluster_info or subcluster_info[cluster_key] is None:
            # Couldn't subdivide cluster_key, use the subspace we already found
            subcluster_info[cluster_key] = {
                "dim": subspace_dim,
                "percentile_80": cluster_80_residuals[cluster_key],
                "num_points": cluster_sizes[cluster_key],
                "total_outliers": total_outliers
            }
    return ("Pass", subcluster_info)


def recursive_subspacer(df, nr_clusters, subspace_dim):
    df_sum = df.sum(axis=1)

    initial_clusters, initial_residuals = linear_subspace_clustering(df.T.to_numpy(), nr_clusters=nr_clusters, subspace_dim=subspace_dim)
    cluster_80_residuals, num_outliers, outliers = calc_cluster_residuals(initial_clusters, initial_residuals, df_sum)
    cluster_sizes = calc_cluster_sizes(initial_clusters)

    print("Attempt:", nr_clusters, subspace_dim)
    total_outliers = 0
    if len(cluster_80_residuals) != nr_clusters:
        print("FAIL ", "Found fewer clusters than requested")
        return("Fail", None)

    for c in cluster_80_residuals:
        if cluster_80_residuals[c] > 0.10:
            print("FAIL", cluster_80_residuals[c])
            return ("Fail", None)
        if cluster_sizes[c] < 10:
            print("FAIL", cluster_sizes[c])
            return ("Fail", None)
        total_outliers += num_outliers[c]

    # We have a good solution, let's try subdividing each cluster if possible.
    subcluster_info = {"clusters": initial_clusters, "num_clusters": len(cluster_80_residuals)}
    for cluster_key in cluster_80_residuals:
        idx = np.where(initial_clusters == cluster_key)[0]
        cluster_df = df.iloc[idx]
        # Try dividing it into d-1 clusters of dimension d-1 where d is the dimension of the subspace for that cluster
        if subspace_dim > 1:
            for num_clusters in range(1, max(subspace_dim, 5)):  # Small fudge factor to deal with co (hyper)planar subspace division, we always try fitting at least 5 things of the smaller dimension
                print(subspace_dim)
                print(cluster_df.shape)
                passfail, cluster_info = recursive_subspacer(cluster_df, num_clusters, subspace_dim-1)
                if passfail == "Fail":
                    # print("FAIL", num_clusters, subspace_dim-1)
                    continue
                else:
                    # Found a potentially better solution, figure out how to choose between this one and all other ones
                    # For now, just pick it.
                    subcluster_info[cluster_key] = cluster_info
                    break

        if cluster_key not in subcluster_info:
            # Couldn't subdivide cluster_key, use the subspace we already found
            subcluster_info[cluster_key] = {
                "dim": subspace_dim,
                "percentile_80": cluster_80_residuals[cluster_key],
                "num_points": cluster_sizes[cluster_key],
                "total_outliers": num_outliers[cluster_key],
                "outliers": outliers[cluster_key],
                "basis": "SomeBasecase"  # This is where we need to report info like number of dimensions of subspace, and basis vectors of the cone
            }
    return ("Pass", subcluster_info)

def write_final_assignments(cluster_info_tree):
    cluster_id_gen = [0]
    cluster_info_map = {}

    def write_assignment_helper(cluster_info_tree, cluster_id_gen, cluster_info_map):
        # start a root of tree
        cur = cluster_info_tree

        # if a leaf, return new cluster id
        if 'num_clusters' not in cur:
            cluster_id = cluster_id_gen[0]
            cluster_id_gen[0] += 1
            cluster_info_map[cluster_id] = cur

            leaf_val = np.zeros(cur['num_points'])
            np.put(leaf_val, np.where(cur['outliers'] == True), -1)
            np.put(leaf_val, np.where(cur['outliers'] == False), cluster_id)
            return leaf_val

        # if not a leaf, rewrite assignment
        else:
            # copy the cluster assignments
            new_clusters = np.copy(cur['clusters'])

            for i in range(cur['num_clusters']):
                indices_to_rewrite = np.where(cur['clusters'] == i)[0]
                # print("Indices To Rewrite:", len(indices_to_rewrite))

                child = write_assignment_helper(cur[i], cluster_id_gen, cluster_info_map)
                if isinstance(child, int):
                    # child was a leaf, write this value over all indices to rewrite, and return
                    new_clusters[indices_to_rewrite] = child
                else:
                    # Copy new values over into new clusters
                    np.put(new_clusters, indices_to_rewrite, child)
            return new_clusters

    final_assignment = write_assignment_helper(cluster_info_tree, cluster_id_gen, cluster_info_map)

    return final_assignment, cluster_info_map


def run_clustering(filtered_df, initial_nr_clusters, initial_dim):
    passfail, cluster_info = recursive_subspacer(filtered_df, initial_nr_clusters, initial_dim)
    if passfail == 'Fail':
        print("Couldn't find any valid clustering")
        return None

    final_assignment, cluster_info_map = write_final_assignments(cluster_info)
    return final_assignment, cluster_info_map


def merge(
        outer_assignment, inner_assignment,
        outer_map, inner_map):

    start_key = max(outer_map.keys())

    result_map = {}
    for key in outer_map:
        result_map[key] = outer_map[key]
    for key in inner_map:
        if key == -1:
            raise Exception("I didn't think this could happen")
        result_map[key + start_key + 1] = inner_map[key]

    to_put = np.copy(inner_assignment)
    to_put = np.add(inner_assignment, start_key + 1, out=to_put, where=inner_assignment >= 0)

    result_assignment = np.copy(outer_assignment)
    np.put(result_assignment, np.where(result_assignment == -1), to_put)
    return result_assignment, result_map


def sort_clusters(final_assignment, cluster_info_map):
    keys = [(k, cluster_info_map[k]['dim'], cluster_info_map[k]['num_points']-cluster_info_map[k]['total_outliers']) for k in cluster_info_map.keys()]
    keys.sort(key=lambda x:(x[1],-x[2]))
    final_pos = {}
    for i in range(len(keys)):
        final_pos[keys[i][0]] = i
    sorted_info_map = {}
    for k in final_pos:
        sorted_info_map[final_pos[k]] = cluster_info_map[k]

    final_pos[-1] = -1
    final_assignment = np.vectorize(final_pos.get)(final_assignment)

    return final_assignment, sorted_info_map


def iterative_clustering(filtered_df, initial_dim):
    result = run_clustering(filtered_df, 1, initial_dim)
    if result is None:
        return None
    else:
        final_assignment, cluster_info_map = result
        max_iter = 5
        for i in range(1, max_iter+1):
            outlier_count = len(final_assignment[final_assignment == -1])
            print("Iteration:", i, "Outlier count:", outlier_count)
            if outlier_count < 10:
                print("Outlier count < 10, terminating")
                break
            remaining_df = filtered_df[final_assignment == -1]
            remaining_result = run_clustering(remaining_df, 1, initial_dim)
            if remaining_result is None:
                print("Couldn't run clustering on remaining outliers")
                break
            remaining_assignment, remaining_info_map = remaining_result
            final_assignment, cluster_info_map = merge(final_assignment, remaining_assignment, cluster_info_map, remaining_info_map)

    final_assignment, cluster_info_map = sort_clusters(final_assignment, cluster_info_map)

    return final_assignment, cluster_info_map


def main():
    if __name__ == '__main__':
        woltka_none_table = BiomTable("./dataset/biom/combined-none.biom")
        woltka_species_table = BiomTable("./dataset/biom/combined-species.biom")

        USE_SPECIES=False

        if USE_SPECIES:
            woltka_table = woltka_species_table
        else:
            woltka_table = woltka_none_table

        metadata_table = CSVTable("./woltka_metadata.tsv", delimiter="\t")

        woltka_table = woltka_table.load_dataframe()
        metadata_table = metadata_table.load_dataframe()

        sil_map = {}
        all_genera = metadata_table['genus'].unique()
        all_genera = [str(g) for g in all_genera]
        all_genera.sort()

        start_genera = 'Butyricimonas'
        skip_genera = set(["Bacteroides", "Clostridium"])
        for genus in ['Butyricimonas']: # Acidaminococcus
            # if genus <= start_genera:
            #     continue
            # if genus in skip_genera:
            #     continue
            print("CLUSTERING GENUS: ", genus)
            genus_ids = metadata_table[metadata_table['genus'] == genus].sort_values(["species","#genome"])[['#genome', 'genus', 'species', 'species_taxid']]
            if USE_SPECIES:
                genus_ids['#genome'] = genus_ids['species_taxid'].astype(str)
            genus_ids = genus_ids[genus_ids['#genome'].isin(woltka_table.columns)]

            genus_ids['rowid'] = range(genus_ids.shape[0])
            if USE_SPECIES:
                genus_ids = genus_ids.drop_duplicates(subset=["#genome"])
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.precision", 3):  # more options can be specified also
                print(genus_ids)

            genus_ids = genus_ids['#genome'].to_list()
            if len(genus_ids) < 2:
                print(genus, "Not enough species in dataset: ", len(genus_ids))
                continue

            df = woltka_table[genus_ids]
            df = _filter_to_samples(df)
            df = df.rename(mapper=_parse_sample_id)

            print(df)
            df_sum = df.sum(axis=1)
            filtered_df = df[df_sum >= MIN_GENUS_COUNT]
            small_sums = df_sum[df_sum >= MIN_GENUS_COUNT]

            # print(filtered_df)
            # Start from 1 cluster of max dim, get a fit
            if len(filtered_df) <= 10:
                print(genus, "Not enough reads to do clustering\n")
                continue

            # result = iterative_clustering(filtered_df, len(genus_ids))
            result = iterative_clustering(filtered_df, 5)
            if result is None:
                print(genus, "Couldn't run clustering\n")
                continue
            else:
                final_assignment, cluster_info_map = result

            print(genus)
            print("Dimension of initial space", len(genus_ids))
            print("Num Clusters:", len(cluster_info_map))
            cluster_data = [{'id':k, 'dim':cluster_info_map[k]['dim'], 'good_fit':cluster_info_map[k]['num_points']-cluster_info_map[k]['total_outliers']} for k in cluster_info_map]
            print("Cluster dimensions", cluster_data)
            print("Num Points:", filtered_df.shape[0])
            print("Num Outliers:", len(final_assignment[final_assignment == -1]))
            print()

            # Acidaminococcus: High overlap groups (0,1,2), (3,4), (5)

            # for col1_index in range(len(filtered_df.columns) - 1):
            #     plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, col1_index, col1_index+1, col1_index+2)

            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 8, 23, 129)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 8, 23, 130)
            # plot_scatter(filtered_df, final_assignment, genus, 8, 129)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 1, 2, 3)
            plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 1, 2, 3)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 7, 8, 9)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 2, 6, 7)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 3, 4, 5)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 2, 3, 4)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 2, 3, 6)
            # plot_dim2_clusters(filtered_df, final_assignment, cluster_info_map)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 1, 2)
            # plot_scatter(filtered_df, final_assignment, genus, 3, 4)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 3, 5)

            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 1, 2, target_cluster=0)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 1, 2, target_cluster=1)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 1, 2, target_cluster=2)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 1, 2, target_cluster=0)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 1, 2, target_cluster=1)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 1, 2, target_cluster=2)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 3, 4)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 3, 4, target_cluster=0)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 3, 4, target_cluster=1)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 3, 4, target_cluster=2)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 4, 5)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 4, 5, target_cluster=0)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 4, 5, target_cluster=1)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 4, 5, target_cluster=2)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 0, 4, 5, target_cluster=1)

            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 3, 4, 5)
            # plot_scatter3(filtered_df, final_assignment, cluster_info_map, genus, 3, 4, 6)

            for col1_index in range(len(filtered_df.columns) - 1):
                plot_scatter(filtered_df, final_assignment, genus, col1_index, col1_index+1)

            # for col1_index in range(len(filtered_df.columns)):
            #     for col2_index in range(len(filtered_df.columns)):
            #         if col2_index <= col1_index:
            #             continue
            #         print(col1_index, col2_index)
            #         plot_scatter(filtered_df, final_assignment, genus, col1_index, col2_index)


main()
