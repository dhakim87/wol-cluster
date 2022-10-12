import numpy as np
import scipy
import scipy.optimize

from determinant_zero import DeterminantZeroCollapse, \
    DeterminantZeroException

if __name__=="__main__":
    H = np.array(
        [
            [.7,.2,0,0,0,0],
            [.2,.8,0,0,0,0],
            [.5,.5,.2,0,0,0],
            [0,0,1,0,0,0]
        ]
    )

    H = np.array(
        [
            [1,1,1,0,1],
            [0,1,1,1,0],
            [1,1,1,1,0],
            [0,1,1,0,1]
        ]
    )

    print(np.dot(H, H.T))

    W = np.random.rand(2,4)
    X = np.dot(W, H)

    handler = DeterminantZeroCollapse()

    # Nice, looks like we can run nnls on H or HHT, and HHT should be much much smaller.
    # rx, xx = DeterminantZeroCollapse.find_redundant_row(H)
    # ry, yy = DeterminantZeroCollapse.find_redundant_row(np.dot(H, H.T))
    #
    # print(rx, ry, rx==ry)
    # print(xx, yy, xx==yy)

    X = np.dot(W, H)
    W, H = handler.handle(W, H, DeterminantZeroException())
    X_p = np.dot(W,H)
    print(X - X_p)
    print(np.linalg.det(np.dot(H, H.T)))

