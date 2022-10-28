import numpy as np
import scipy

class DeterminantZeroException(Exception):
    pass

class DeterminantZeroReraise:
    def handle(self, W, H, exc):
        raise exc


class DeterminantZeroCollapse:
    def handle(self, W, H, exc):
        # Gradient descent has left us with redundant components.  Redundant components are those
        # which are fully inside of the conic polyhedron formed by the other components.
        # Frequently, but not always, the redundancy will result from two duplicate components.
        # We can use NNLS to identify components which are redundant, then collapse one.

        # Since HHT is usually much smaller than H, we can also find the row in HHT with better numerical stability
        r, output_pt = DeterminantZeroCollapse.find_redundant_row(np.dot(H, H.T))
        if r is None:
            print("Oops, we couldn't find a row to collapse to fix things")
            raise(exc)
        print("Collapsing 1 component")
        new_W, new_H = DeterminantZeroCollapse.replace_row(W, H, r, output_pt)
        return new_W, new_H

    @staticmethod
    def find_redundant_row(H):
        # TODO FIXME HACK: Can det HHT ever be zero without there being a redundant row?
        #  Probably would mean there's underflow from a huge number of very close together vectors.
        min_resid = 10000
        min_r = 0
        approx_output_pt = None
        for r in range(H.shape[0]):
            H_prime = np.vstack((H[:r, :], H[r+1:,]))
            # TODO FIXME HACK: nnls itself can fail with a RuntimeError("too many iterations").  Ridiculous.
            try:
                output_pt, l2_resid = scipy.optimize.nnls(H_prime.T, H[r,:])
            except RuntimeError as e:
                # Couldn't run nnls, argh.
                return None, None

            if l2_resid < min_resid:
                min_resid = l2_resid
                min_r = r
                approx_output_pt = output_pt
            if l2_resid < 0.001:
                # print("Row", r, H[r,:], "is redundant")
                # print(H_prime, "*", output_pt)
                return r, output_pt
        # print("Oops, determinant was zero but there's no redundant rows...")
        # print("Best candidate row:", r)
        # print("L2 resid: ", min_resid)
        return None, None

    @staticmethod
    def replace_row(W, H, r, row_conversion):
        # W is samples x components
        # H is components x OTUs
        # To remove a row of H is to remove a component,
        # thus a column of W must also be removed

        # X = W * H
        # H[row,:] = H * row_conv
        H_prime = np.vstack((H[:r, :], H[r+1:,]))
        W_prime = np.hstack((W[:, :r], W[:, r+1:]))

        W_col = W[:, r].reshape(W.shape[0], 1)
        H_conv = row_conversion.reshape(1, W_prime.shape[1])
        W_offset = np.dot(W_col, H_conv)
        return W_prime + W_offset, H_prime