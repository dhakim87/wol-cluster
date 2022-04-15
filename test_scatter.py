import matplotlib
matplotlib.use('TkAgg')
from linear_subspace_clustering import linear_subspace_clustering, \
    _fit_a_linear_subspace
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd


def test():
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(range(21), range(21), range(21), c=[-1,0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], cmap="Set1")

    # patches = []
    # for label in range(min(best_clustering), max(best_clustering) + 1):
    #     cmap = matplotlib.cm.get_cmap('Set1')
    #     rgba = cmap(label)
    #     if label == -1:
    #         patch = matplotlib.patches.Patch(color=rgba, label="Outliers")
    #         patches.append(patch)
    #     else:
    #         patch = matplotlib.patches.Patch(
    #             color=rgba,
    #             label=str(label) +
    #                   ": dim=" + str(cluster_info_map[label]['dim']) +
    #                   "#fit=" + str(cluster_info_map[label]['num_points']-cluster_info_map[label]['total_outliers']))
    #         patches.append(patch)
    #
    # plt.legend(handles=patches)
    plt.show()

if __name__ == '__main__':
    test()