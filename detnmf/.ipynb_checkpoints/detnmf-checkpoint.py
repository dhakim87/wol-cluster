import numpy as np


def L2_residual(X, W, H):
    return np.linalg.norm(X - np.dot(W,H))

def determinant(H):
    HHT = np.dot(H, H.T)
    return np.linalg.det(HHT)

def nmf_H(X, W, H):
    WTX = np.dot(W.T, X)
    WTWH = np.dot(np.dot(W.T, W), H)
    WTXoverWTWH = WTX/WTWH  # Ah, elementwise division I think, check for Nans after (maybe infinities as well?)
    WTXoverWTWH[np.isnan(WTXoverWTWH)] = 0
    return WTXoverWTWH


def nmf_W(X, W, H):
    XHT = np.dot(X, H.T)
    WHHT = np.dot(np.dot(W,H), H.T)
    XHToverWHHT = XHT/WHHT  # Ah, elementwise division I think, check for Nans after (maybe infinities as well?)
    XHToverWHHT[np.isnan(XHToverWHHT)] = 0
    return XHToverWHHT


def det_nmf_H(X,W,H, alpha_scale):
    WTXoverWTWH = nmf_H(X,W,H)
    HHT = np.dot(H, H.T)  # Hmm, could reuse this if we like
    detHHT = np.linalg.det(HHT)
    HHTinvH = np.dot(np.linalg.inv(HHT), H)
    WTW = np.dot(W.T, W)
    alpha = WTW.mean() * alpha_scale
    WTWH = np.dot(WTW, H)  # Hmm, could reuse this if we like


    HHTinvHoverWTWH = HHTinvH / WTWH  # Elementwise division, check for Nans

    HHTinvHoverWTWH[np.isnan(HHTinvHoverWTWH)] = 0
    # print("ACK")
    # print(WTXoverWTWH)
    # print(HHTinvH)
    # print(HHTinvH)
    # print(WTWH)
    # print(HHTinvHoverWTWH)

    # return WTXoverWTWH - (alpha * detHHT * HHTinvHoverWTWH)
    return WTXoverWTWH - (alpha * detHHT * HHTinvHoverWTWH)


# def foo_check_normalize(W, H):
#     L2s = np.linalg.norm(H, axis=1)
#     foos = []
#     for k in range(H.shape[0]):
#         total = 0
#         for m in range(H.shape[1]):
#             total += H[k,m] * H[k,m]
#         foos.append(np.sqrt(total))
#
#     print("FOO CHECK")
#     print(L2s)
#     print(foos)


def L2_normalize(W, H):
    L2s = np.linalg.norm(H, axis=1)
    L2s[L2s == 0] = 1

    H = (H.T / L2s).T
    W = W * L2s
    return W, H

def L1_normalize(W, H):
    L1s = H.sum(axis=1)
    L1s[L1s == 0] = 1  # Don't fuck with all zero components.

    H = (H.T / L1s).T
    W = W * L1s
    return W, H


def run_detnmf(X, components, alpha_scale = 0.01, W=None, H=None, iterations=5000, beta=0.0001):
    # For the purposes of rescaling to avoid numerical stability issues, we may think of X as the initial W
    # and H as the axes of the space of X with num components equal to the number of columns
    # If we center X, we can help avoid WTW going to 0 or infinity, we just need to rescale W at the end by the same value.
    Xmean = X.mean()
    X = X / (Xmean)

    if W is not None and H is not None:
        print("Initial Scoring")
        print(L2_residual(X,W,H), determinant(H))

    if W is None:
        W = np.random.rand(X.shape[0], components)
    if H is None:
        H = np.random.rand(components, X.shape[1])

    for iter in range(iterations):
        # Adding beta prevents individual elements of H from hitting exactly 0 and getting stuck.  
        H = (H + beta) * det_nmf_H(X, W, H, alpha_scale=alpha_scale)
        W = W * nmf_W(X, W, H)
        W, H = L2_normalize(W,H)

        if iter % 100 == 0:
            WTW_print = np.dot(W.T, W)
            alpha_print = WTW_print.mean() * alpha_scale
            print(iter, "L2Resid=", L2_residual(X,W,H), "Det=", determinant(H), "alpha=", alpha_print)

    W = W * Xmean
    return W,H


def run_nmf(X, components, iterations):
    W = np.random.rand(X.shape[0], components)
    H = np.random.rand(components, X.shape[1])

    for iter in range(iterations):
        H = H * nmf_H(X, W, H)
        W = W * nmf_W(X, W, H)
        W, H = L2_normalize(W,H)

    return W,H


def run_caliper_nmf(X, components, W=None, H=None, alpha=0.95, iterations=50000):
    if W is not None and H is not None:
        print("Initial Scoring")
        print(L2_residual(X,W,H), determinant(H))

    if W is None:
        W = np.random.rand(X.shape[0], components)
    if H is None:
        H = np.random.rand(components, X.shape[1])

    for iter in range(iterations):
        H = H * nmf_H(X, W, H)
        W = W * nmf_W(X, W, H)
        W, H = L2_normalize(W,H)

        H_mean = H.sum(axis=0)
        for r in range(H.shape[0]):
            H[r,:] = alpha * H[r,:] + (1-alpha) * H_mean

        if iter % 100 == 0:
            print(iter, L2_residual(X,W,H), determinant(H))
    return W, H


def main():
    import pandas as pd
    # from sklearn.decomposition import NMF
    df = pd.read_csv("./three_cols.tsv", sep="\t", index_col=0)
    # df = df[df.sum(axis=1) > 2000000]
    # outliers = df[df[df.columns[2]] < 100]
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #     print(outliers)

    # X = df.to_numpy()

    # X[:,0] = X[:,0] / X[:,0].sum()
    # X[:,1] = X[:,1] / X[:,1].sum()
    # X[:,2] = X[:,2] / X[:,2].sum()

    pts = []
    v1 = np.array([10,10,10])
    v2 = np.array([10,2,5])
    v3 = v2
    # v3 = np.array([4,5,12])

    for i in range(100):
        pts.append(v1 * i)
        pts.append(v2 * i)
        pts.append(v3 * i)
        for j in range(3):
            pts.append(v1 * np.random.randint(25) + v2 * np.random.randint(50) + v3 * np.random.randint(25))
    X = np.array(pts)
    components = 2

    # model = NMF(n_components=components, max_iter=200)
    # W = model.fit_transform(X)
    # H = model.components_

    W,H = None, None
    # print("NMF")
    # W,H = run_nmf(X, components, iterations=15000)
    # print("CaliperNMF")
    # W,H = run_caliper_nmf(X, components, alpha=0.99, W=W, H=H)
    print("detNMF")
    W,H = run_detnmf(X, components, alpha_scale=0.05, iterations=50000,W=W, H=H)

    W, H = L1_normalize(W,H)
    W1,H1 = W,H
    W2, H2 = L2_normalize(W,H)
    print(np.rint(np.dot(W,H)))
    print(np.linalg.det(np.dot(H,H.T)))
    print(np.rint(H*100))

    print(L2_residual(X,W,H))
    print(determinant(H))
    print(L2_residual(X,W,H) / determinant(H))

    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(X[:,0], X[:,1], X[:,2], alpha=0.5)

    maxx = X[:,0].max()
    maxy = X[:,1].max()
    maxz = X[:,2].max()
    maxall = max(maxx, maxy, maxz)
    for i in range(H1.shape[0]):
        start = [0,0,0]
        end1 = H1[i,:]
        end2 = H2[i,:]
        ax.plot3D([start[0], end1[0] * maxall], [start[1], end1[1] * maxall], [start[2], end1[2] * maxall], color='red')
        ax.plot3D([start[0], end2[0] * maxall], [start[1], end2[1] * maxall], [start[2], end2[2] * maxall], color='green')
    plt.show()

if __name__ == "__main__":
    main()