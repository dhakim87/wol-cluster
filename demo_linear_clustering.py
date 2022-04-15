import math

from collections import defaultdict
import pandas as pd
import numpy as np
from linear_subspace_clustering import linear_subspace_clustering, \
    _fit_a_linear_subspace


def demo_linear_clustering():
    # features are column vectors in this csv
    data, cluster_assignments = _get_data()
    nr_data = data.shape[1]

    for KK in range(1):
        nr_clusters = 3
        subspace_dim = 2

        cluster_assignments_1, residuals_1 = linear_subspace_clustering(data, nr_clusters, subspace_dim)

        print("TRIAL " + str(KK))

        counter = defaultdict(int)
        for i in range(len(cluster_assignments)):
            left = cluster_assignments[i]
            right = cluster_assignments_1[i]
            counter[(left, right)] += 1
        print(counter)

        for c_index in range(nr_clusters):
            pidx = np.where(cluster_assignments_1 == c_index)[0]
            subspace_basis_vectors = _fit_a_linear_subspace(data[:, pidx], subspace_dim)
            print(subspace_basis_vectors)
            # vv = np.array([-22 + 20, 62 + 20, -10 + 80])
            # vv = np.array([0,1,0])
            # proj_vv = np.matmul(subspace_basis_vectors, np.matmul(subspace_basis_vectors.T, vv))
            # print(vv)
            # print(proj_vv)


def _get_data():
    v1 = np.array([.2,.2,.8,.3,.6,.2,.1,.8])
    v2 = np.array([.8,.3,.1,.1,.8,.9,.2,.6])
    v3 = np.array([.4,.6,.1,.1,.2,.3,.4,.5])

    # v1 = np.array([1,0,0])
    # v2 = np.array([.5,.5,0])
    # v3 = np.array([0,0,1])

    vs = [v1,v2,v3]
    vs = [v / np.linalg.norm(v) for v in vs]

    Data = []
    cluster_assignments = []

    cluster = 0
    for first in range(3):
        for second in range(3):
            if second <= first:
                continue
            for num_pts in range(400):
                count_first = np.random.random() * 1000
                count_second = np.random.random() * 1000

                data_vector = vs[first] * count_first + vs[second] * count_second

                # Maybe should add some noise here.
                Data.append(data_vector)
                cluster_assignments.append(cluster)
            cluster += 1

    return np.array(Data).T, np.array(cluster_assignments)


if __name__ == "__main__":
    demo_linear_clustering()