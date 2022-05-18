import numpy as np
import sklearn.cluster

import math

# A direct port of Jeffrey Ho's Linear_Subspace_clustering.m
# Ported by Daniel Hakim


def linear_subspace_clustering(data, nr_clusters, subspace_dim):
    # Inputs:
    #   data: dimension by number of data points 2D array.
    #       dimension is the feature dimension.  Points are column vectors.
    #   nr_clusters: The number of clusters
    #   subspace_dim: The dimension of the subspaces which define the clusters

    # Each cluster is defined by a linear subspace of known dimension (subspace_dim) which contains the origin

    # Output:
    # Returns cluster_assignments, cluster_assignments is a 1-by-number of data points array
    # Each element has an integer value from 1 to number of clusters indicating the cluster assignment.
    # (It may be that the number of clusters assigned by the algorithm is smaller than nr_clusters)

    dim = data.shape[0]
    NN = data.shape[1]

    l2_magnitude = np.sqrt(np.sum(data**2, axis=0))
    origin_points = np.where(l2_magnitude == 0)[0]

    if len(origin_points) > 0:
        raise(Exception("Clustering cannot assign clusters to points at the origin.  Data should be filtered beforehand.  Origin point indices:" + str(origin_points)))

    data_normalized = data/l2_magnitude
    # Use spherical metric to define affinity.
    # This is just the dot product between unit vectors
    dot_products = np.matmul(data_normalized.T, data_normalized)

    # Floating point spits out NaNs when the dot product is 1 without clipping.
    dot_products = np.clip(-1, 1, dot_products)
    chord_distances = np.arccos(dot_products)

    # TODO FIXME HACK:  I'm not sure what the matlab code is trying to do
    #  when computing sigma.  And I'm not sure that the numpy sort code matches
    #  the matlab sort code.
    # print(chord_distances)
    # print("------------------")
    SS = np.sort(chord_distances)
    sort_ss = np.sort(chord_distances[1,:])
    # print(SS)
    # print("+++++++++++++++")
    # print(sort_ss)

    # How much does choice of sigma matter?  Do we care about it as long as it is non zero?
    nn_idx = max([math.floor(NN * .75), 1])
    sigma = sort_ss[nn_idx]
    # print(sigma)
    if sigma == 0:
        print("Oof, couldn't find nonzero sigma, trying a failsafe")
        sort_ss2 = [s for s in sort_ss if s != 0]
        sigma = sort_ss2[len(sort_ss2)//2]
        # raise Exception("Uh oh, I don't understand what sigma is supposed to be, but I calculated it as 0")

    # TODO FIXME HACK:  I think this is a mechanism for normalizing distances
    #  into affinities, so that 0 distance becomes 1 and larger distances go towards 0.
    affinity_matrix = np.exp(-chord_distances / sigma)
    cluster_assignments = _spectral_clustering(affinity_matrix, nr_clusters)

    max_number_iterations = min(math.ceil(NN/10), 50)
    cluster_assignments, residuals = _k_subspace_clustering(data, cluster_assignments, subspace_dim, max_number_iterations)
    return cluster_assignments, residuals


def _spectral_clustering(affinity_matrix, nr_clusters):
    # The affinity matrix is assumed to be symmetric
    # print("AFFINITY")
    # print(affinity_matrix)
    row_sum = np.sum(affinity_matrix, 0)

    # TODO FIXME HACK:  Since every vector's affinity to itself is 1,
    #  how can row_sum ever be close to zero?
    idx = np.where(row_sum < 0.0001)[0]
    if len(idx) != 0:
        for i in idx:
            row_sum[i] = 0.0001

    row_sum = 1 / np.sqrt(row_sum)
    # print(row_sum)

    DD = np.diag(row_sum)
    # print(DD)
    LL = np.matmul(np.matmul(DD, affinity_matrix), DD)
    # print(LL)

    # TODO FIXME HACK:  Uhh...  is LL supposed to be a lower left matrix?  Because it isn't...
    # Eigenvalues (ee) are returned in an arbitrary order
    # Eigenvectors are columns of VV
    ee, VV = np.linalg.eig(LL)

    # print(ee)
    # print(VV)
    # Sort eigenvalues and eigenvectors in decreasing order
    idx = ee.argsort()[::-1]
    # print(idx)

    ee = ee[idx]
    VV = VV[:, idx]

    # Take first nr_clusters + 1 eigenvalues and eigenvectors.
    ee = ee[0:nr_clusters+1]
    VV = VV[:, 0:nr_clusters+1]

    # print("Largest Eigens:")
    # print(ee)
    # print(VV)

    dd = np.diag(ee)

    # TODO FIXME HACK:  This isn't magnitude, this is magnitude squared.
    # Col magnitudes are all 1, because they're unit eigenvectors.  Good.
    # col_magnitudes = np.sum(VV**2, axis=0)
    # print(col_magnitudes)
    row_magnitudes = np.sum(VV**2, axis=1)
    row_magnitudes = row_magnitudes.reshape(row_magnitudes.shape[0], 1)  # This is a column vector.
    # print(row_magnitudes)

    # TODO FIXME HACK:  I don't have intuition for this.
    #  What does dividing eigenvectors by row magnitude squared do for us?
    VV = np.divide(VV, row_magnitudes)

    # TODO FIXME HACK:
    #  This seems to be necessary, is this just floating point error?  Where are complex eigenvectors sneaking in?
    VV = np.real_if_close(VV)

    MAX_KMEANS_ITER = 300
    clusterer = sklearn.cluster.KMeans(n_clusters=nr_clusters, max_iter=MAX_KMEANS_ITER)
    clusterer.fit(VV)
    if clusterer.n_iter_ == MAX_KMEANS_ITER:
        raise Exception("k=" + str(nr_clusters) + ": Convergence Failed")

    labels = clusterer.labels_
    # print(labels)
    # print(clusterer.cluster_centers_)
    return labels


def calc_residuals(data, cluster_assignments, subspace_dimension, target_cluster=None):
    nr_data_points = data.shape[1]
    labels = np.unique(cluster_assignments)
    nr_clusters = len(labels)
    residuals = np.zeros(nr_data_points)

    if target_cluster is not None:
        labels = [target_cluster]
        
    for gg in range(len(labels)):
        pidx = np.where(cluster_assignments == labels[gg])[0]
        
        if isinstance(subspace_dimension, int):
            dim = subspace_dimension
        else:
            dim = subspace_dimension[labels[gg]]
        
        subspace_basis_vectors = _fit_a_linear_subspace(data[:, pidx], dim)
        dists = []
        for idx in pidx:
            vv = data[:, idx]
            proj_vv = np.matmul(subspace_basis_vectors, np.matmul(subspace_basis_vectors.T, vv))
            dists.append(np.linalg.norm(vv - proj_vv))
        np.put(residuals, pidx, dists)

    return residuals    


def _k_subspace_clustering(data, cluster_assignments, subspace_dimension, max_number_iterations):
    # print("Initial Cluster Assignments")
    # print(cluster_assignments)
    nr_data_points = data.shape[1]

    # estimate subspaces ......
    for KK in range(max_number_iterations):
        # TODO FIXME HACK:  Appears this can collapse the number of labels every iteration
        #  is that okay??
        labels = np.unique(cluster_assignments)
        nr_clusters = len(labels)
        dists_to_subspaces = np.zeros((nr_clusters, nr_data_points))

        for gg in range(nr_clusters):
            pidx = np.where(cluster_assignments == labels[gg])[0]

            # These are orthonormal column vectors
            # TODO FIXME HACK:  Ack, this turns every face of the conic polyhedron into a plane doesn't it?
            #  I need to eventually retrieve basis vectors of the conic polyhedron in order to transform the data
            # print("BIG DATA\n", data)
            subspace_basis_vectors = _fit_a_linear_subspace(data[:, pidx], subspace_dimension)

            for tt in range(nr_data_points):
                vv = data[:, tt]
                proj_vv = np.matmul(subspace_basis_vectors, np.matmul(subspace_basis_vectors.T, vv))
                dists_to_subspaces[gg,tt] = np.linalg.norm(vv - proj_vv)

        cluster_assignments = np.argmin(dists_to_subspaces, axis=0)

    residuals_1 = calc_residuals(data, cluster_assignments, subspace_dimension)
    # residuals = np.min(dists_to_subspaces, axis=0)  # This is the second to last residual, which is very wrong if the clustering isn't stable.

    # print(dists_to_subspaces.shape)
    # print(dists_to_subspaces)
    # print(residuals)
    return cluster_assignments, residuals_1


def _fit_a_linear_subspace(data, dim_subspace):
    UU, SV, VV = np.linalg.svd(data)

    idx = SV.argsort()[::-1]
    UU = UU[:, idx]
    subspace_basis = UU[:, 0:dim_subspace]
    return subspace_basis


if __name__ == "__main__":
    # 6 points spread across two rays out from the origin.
    # Must ensure none of the points are all 0 before passing in.
    toy_data = [[1.01, 1, 1], [2,2.02,2], [1,0,0.01], [3.03,3,3], [2,0.02,0], [3,0,0.03]]
    toy_data = np.array(toy_data).T

    linear_subspace_clustering(toy_data, 2, 1)
