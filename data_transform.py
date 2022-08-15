import numpy as np
import pandas as pd
import scipy


def L1_normalize(conic_basis):
    basis = (conic_basis / conic_basis.sum(axis=0))
    return basis

def L2_normalize(conic_basis):
    sum = conic_basis.sum(axis=0)
    sumsq = sqrt(sum * sum)
    basis = conic_basis / sumsq
    return basis

def merge_basis_vectors(conic_basis, SAMENESS_THRESH=0.9):
    # TODO FIXME HACK:  We may want to pass in number of points on each basis vector for
    # weighting purposes
    
    conic_basis = L2_normalize(conic_basis)
    
    # Find all nearly parallel/anti-parallel vectors, have to collapse these or we'll have 
    # ambiguous assignment
    sames = {}
    
    def find_root(sames, i):
        while i in sames and sames[i] != i:
            i = sames[i]
        return i
    
    for i in range(conic_basis.shape[1]):
        arr = []
        for j in range(conic_basis.shape[1]):
            dp = np.dot(conic_basis[:,i], conic_basis[:,j])
            if j > i and abs(dp) > SAMENESS_THRESH:
                print("Bases: ", i, j, " are nearly identical")
                left = find_root(sames, i)
                right = find_root(sames, j)
                final = min(left, right)
                sames[i] = final
                sames[j] = final
            arr.append(dp)
        # print(arr)

    final_sets = {}
    for i in range(conic_basis.shape[1]):
        final = find_root(sames, i)
        if final not in final_sets:
            final_sets[final] = []
        final_sets[final].append(i)

    final_basis_vecs = []
    for final in final_sets:
        vecs = full_basis[:, final_sets[final]]
        final_basis_vecs.append(vecs.mean())
        
    return pd.DataFrame(final_basis_vecs)


# See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
# for a way to project onto a conic polyhedron
def transform(df, conic_basis):
    conic_basis = conic_basis.sort_index(axis=0)
    df = df.sort_index(axis=1)
    
    setA = set(conic_basis.index)
    setB = set(df.columns)
    if setA != setB:
        raise Exception("Conic basis index must equal df columns but does not")
    
    # L1 normalize the basis to keep read count equal in the transformed space
    basis = L1_normalize(conic_basis).to_numpy()

    # print(conic_basis.index)
    # print(df.columns)
    
    out_pts = []
    max_resid = 0
    for i in range(df.shape[0]):
        pt = df.iloc[i].T.to_numpy()
        output_pt_nnls, l2_resid = scipy.optimize.nnls(basis, pt)

        nnls_proj = np.matmul(basis, output_pt_nnls)
        l1_resid = np.sum(np.abs(pt - nnls_proj))
        worst_axis = np.argmax(np.abs(pt - nnls_proj))
        
        output_pt = output_pt_nnls
        output_pt = pd.DataFrame(output_pt).T
        output_pt.index = [df.index[i]]
        output_pt["L1_resid"] = [l1_resid]
        output_pt["L2_resid"] = [l2_resid]
        output_pt["WorstAxis"] = [conic_basis.index[worst_axis]]
        max_resid = max(max_resid, l1_resid)
        # if pt.sum() > 0: 
        #     max_resid_pct = max(max_resid_pct, abs(l1_resid)/pt.sum())
        out_pts.append(output_pt)        
        
    print("MAX RESIDUAL: ", max_resid)

    output_df = pd.concat(out_pts)
    output_df = output_df.fillna(0)
    return output_df
