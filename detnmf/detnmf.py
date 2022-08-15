import numpy as np


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


def det_nmf_H(X,W,H,alpha):
    WTXoverWTWH = nmf_H(X,W,H)
    HHT = np.dot(H, H.T)  # Hmm, could reuse this if we like
    detHHT = np.linalg.det(HHT)
    HHTinvH = np.dot(np.linalg.inv(HHT), H)
    WTWH = np.dot(np.dot(W.T, W), H)  # Hmm, could reuse this if we like

    HHTinvHoverWTWH = HHTinvH / WTWH  # Elementwise division, check for Nans
    HHTinvHoverWTWH[np.isnan(HHTinvHoverWTWH)] = 0

    return WTXoverWTWH - (alpha * detHHT * HHTinvHoverWTWH)


# TODO FIXME HACK:  Does the determinant still minimize area if we use 
# L1 normalization rather than L2 normalization?
def L1_normalize(W, H):
    L1s = H.sum(axis=1)

    L1s[L1s == 0] = 1  # Don't fuck with all zero components or you get nans

    H = (H.T / L1s).T
    W = W * L1s

    return W, H

def run_detnmf(X, components, alpha, iterations=1000):
    W = np.random.rand(X.shape[0], components)
    H = np.random.rand(components, X.shape[1])

    for iter in range(iterations):
        if iter % 100 == 0:
            print(iter)
        H = H * det_nmf_H(X, W, H, alpha)
        W = W * nmf_W(X, W, H)
        W, H = L1_normalize(W,H)  # Have to normalize frequently or you'll get overflow/underflow issues.  

    return W,H

def run_nmf(X, components, iterations=1000):
    W = np.random.rand(X.shape[0], components)
    H = np.random.rand(components, X.shape[1])

    for iter in range(1000):
        H = H * nmf_H(X, W, H)
        W = W * nmf_W(X, W, H)

def main():
    X = np.array([[100, 100, 0], [200, 200, 0], [300, 300, 0], [100, 0, 0], [200, 0, 0], [300, 0, 0]])
    components = 2

    # print("NMF")
    # run_nmf(X)
    print("detNMF")
    W,H = run_detnmf(X, components, alpha=1000)
    print(np.rint(np.dot(W,H)))
    print(np.linalg.det(np.dot(H,H.T)))
    print(np.rint(H*100))

if __name__ == "__main__":
    main()
