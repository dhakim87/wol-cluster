import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def _get_color(label, min_label, max_label):
    cmap = matplotlib.cm.get_cmap('Set1')
    if min_label != max_label:
        rgba = cmap((label - min_label) / (max_label - min_label))
    else:
        rgba = cmap(0)
    return rgba


def _draw_subspace_line(ax, u, maxx, maxy, maxz, label, min_label, max_label):
    rgba = _get_color(label, min_label, max_label)
    mx = maxx / u[0]
    my = maxy / u[1]
    mz = maxz / u[2]
    multiplier = min(abs(mx), abs(my), abs(mz))

    if multiplier == -mx or multiplier == -my or multiplier == -mz:
        multiplier = -multiplier
    ax.plot3D([0, u[0] * multiplier], [0, u[1] * multiplier], [0,u[2] * multiplier], color=rgba)


def plot_taxavec3(filtered_df, title, c1, c2, c3, xlabel, ylabel, zlabel, W, H, subplot_ax=None):
    if subplot_ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    else:
        ax = subplot_ax

    maxx = max(filtered_df[c1])
    maxy = max(filtered_df[c2])
    maxz = max(filtered_df[c3])

    col1_index = filtered_df.columns.get_loc(c1)
    col2_index = filtered_df.columns.get_loc(c2)
    col3_index = filtered_df.columns.get_loc(c3)

    for label in range(H.shape[0]):
        u = H[label,[col1_index, col2_index, col3_index]]
        rgba = _get_color(label, 0, H.shape[0]-1)
        max_W = np.max(W[:, label])
        ax.plot3D([0, u[0] * max_W], [0, u[1] * max_W], [0,u[2] * max_W], color=rgba)
        # _draw_subspace_line(ax,u,maxx,maxy,maxz,label,0,H.shape[0]-1)

    ax.scatter(filtered_df[c1], filtered_df[c2], filtered_df[c3], cmap="Set1")

    # patches = []
    # for label in range(H.shape[0]):
    #     rgba = _get_color(label, 0, H.shape[0]-1)
    #     patch = matplotlib.patches.Patch(
    #         color=rgba,
    #         label=str(label) + "#fit=" + str(cluster_counts.get(label, 0)))
    #     patches.append(patch)

    plt.title(title)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_xlim([0, maxx * 1.5 + 100])
    ax.set_ylim([0, maxy * 1.5 + 100])
    ax.set_zlim([0, maxz * 1.5 + 100])

    # if subspace_bases is not None:
    #     plt.legend(handles=patches)

    if subplot_ax is None:
        plt.show()