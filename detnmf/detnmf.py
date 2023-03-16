import numpy as np

from detnmf.determinant_zero import DeterminantZeroException


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


def det_nmf_H(X,W,H, iteration, alpha=None, alpha_scale=None, alpha_calculator=None):
    WTXoverWTWH = nmf_H(X,W,H)
    HHT = np.dot(H, H.T)  # Hmm, could reuse this if we like
    WTW = np.dot(W.T, W)
    if alpha is None:
        if alpha_scale is not None:
            alpha = WTW.mean() * alpha_scale
        elif alpha_calculator is not None:
            alpha = alpha_calculator.calculate(WTW, iteration)
        else:
            raise Exception("No Alpha Set")
    if alpha == 0:
        return WTXoverWTWH

    detHHT = np.linalg.det(HHT)
    if detHHT < 1e-30:
        raise DeterminantZeroException("Det = " + str(detHHT))
    HHTinvH = np.dot(np.linalg.inv(HHT), H)
    WTWH = np.dot(WTW, H)  # Hmm, could reuse this if we like

    HHTinvHoverWTWH = HHTinvH / WTWH  # Elementwise division, check for Nans
    HHTinvHoverWTWH[np.isnan(HHTinvHoverWTWH)] = 0

    # print("ACK")
    # print(WTXoverWTWH)
    # print(HHTinvH)
    # print(HHTinvH)
    # print(WTWH)
    # print(HHTinvHoverWTWH)

    return WTXoverWTWH - (alpha * detHHT * HHTinvHoverWTWH)


def L2_normalize(W, H):
    L2s = np.linalg.norm(H, axis=1)
    L2s[L2s == 0] = 1

    H = (H.T / L2s).T
    W = W * L2s
    return W, H


def L1_normalize(W, H):
    L1s = H.sum(axis=1)
    L1s[L1s == 0] = 1  # Don't modify all zero components.

    H = (H.T / L1s).T
    W = W * L1s
    return W, H


def run_detnmf(X, components, alpha=None, alpha_scale=None, alpha_calculator=None, W=None, H=None, iterations=5000, beta=None, beta_calculator=None, det0_handler=None, column_scale=False, verbose=True):
    # For the purposes of rescaling to avoid discarding low abundance columns
    if column_scale:
        # Technically not min max scaling since we have non negative matrices and want 0 to remain 0.
        col_scaling = X.max(axis=0)
        # If any columns have max 0, set them to max 1 so we don't rescale them and get nans.
        col_scaling[col_scaling == 0] = 1
        X = X / col_scaling

    # For the purposes of rescaling to avoid numerical stability issues, we may think of X as the initial W
    # and H as the axes of the space of X with num components equal to the number of columns
    # If we center X, we can help avoid WTW going to 0 or infinity, we just need to rescale W at the end by the same value.
    Xmean = X.mean()
    X = X / Xmean

    if W is not None and H is not None:
        print("Initial Scoring")
        print(L2_residual(X,W,H), determinant(H))

    if W is None:
        W = np.random.rand(X.shape[0], components)
    if H is None:
        H = np.random.rand(components, X.shape[1])

    if beta is None:
        beta = 0
    for iter in range(iterations):
        try:
            # Adding beta prevents individual elements of H from hitting exactly 0 and getting stuck.
            if beta_calculator is not None:
                beta = beta_calculator.calculate(iter)

            H = (H + beta) * det_nmf_H(X, W, H, iter, alpha=alpha, alpha_scale=alpha_scale, alpha_calculator=alpha_calculator)
            W = W * nmf_W(X, W, H)
            W, H = L2_normalize(W,H)
            H[H < 0.0001] = 0.0001

            if iter % 100 == 0:
                alpha_print = alpha
                if alpha_print is None and alpha_scale is not None:
                    WTW_print = np.dot(W.T, W)
                    alpha_print = WTW_print.mean() * alpha_scale
                if alpha_print is None and alpha_calculator is not None:
                    WTW_print = np.dot(W.T, W)
                    alpha_print = alpha_calculator.calculate(WTW_print, iter)
                print(iter, "L2Resid=", L2_residual(X,W,H), "Det=", determinant(H), "alpha=", alpha_print)
        except DeterminantZeroException as exc:
            if det0_handler is not None:
                W, H = det0_handler.handle(W, H, exc)
                continue
            else:
                raise(exc)

    W = W * Xmean

    if column_scale:
        H = (H.T * col_scaling.reshape(-1,1)).T
        W, H = L1_normalize(W, H)
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


def guess_num_components(X, max_components, alpha=None, alpha_scale=None, alpha_calculator=None, W=None, H=None, iterations=5000, beta=None, beta_calculator=None, verbose=True, min_det = 1e-15):
    max_success = 1
    min_fail = max_components + 1
    while max_success < min_fail - 1:
        success = True
        n_components = (max_success + min_fail) // 2
        try:
            W, H = run_detnmf(X, n_components, alpha, alpha_scale, alpha_calculator, W, H, iterations, beta, beta_calculator, None, verbose)
            if determinant(H) < min_det:
                success = False
        except DeterminantZeroException as e:
            success = False
        if success:
            max_success = n_components
        else:
            min_fail = n_components
        print("Success", success, "Max Success:", max_success, "Min Fail", min_fail)

    print("Final Guess:", max_success)
    return max_success

