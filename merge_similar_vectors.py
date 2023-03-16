import numpy as np
import math


# We need the ability to merge similar vectors that come back from clustering.
def merge_similar_vectors(vectors, weights, same_thresh_degrees=5):
    def kruskal_top(groups, i):
        if i not in groups:
            return i
        cur = i
        while groups[cur] != cur:
            cur = groups[cur]
        return cur

    # How to pick same_thresh:
    # 1 degree off: dot product .9998
    # 5 degrees off: dot product .996
    # 15 degrees off: dot product .966
    # x degrees off: dot product = cos(x * pi/180)
    dot_thresh = np.cos(same_thresh_degrees * math.pi/180)

    # L2 normalize and ensure all vectors are positive
    # (or at least as positive as we can make them)
    vectors = np.copy(vectors)
    for col in range(vectors.shape[1]):
        l2 = np.linalg.norm(vectors[:, col])
        vectors[:, col] /= l2
        if np.sum(vectors[:,col]) < 0:
            vectors[:, col] = -vectors[:, col]

    # Find all nearly parallel/anti-parallel vectors, probably have to collapse these or we'll have
    # extreme numerical instabilities.
    print("Basis Shape:", vectors.shape)
    print("Weights:", weights)
    sames = {}
    for i in range(vectors.shape[1]):
        arr = []
        for j in range(vectors.shape[1]):
            # TODO: Ack, basis is L1 normalized, needs to be L2 normalized to make the sameness thresh meaningful
            dp = np.dot(vectors[:,i], vectors[:,j])
            if j > i and dp > dot_thresh or dp < -dot_thresh:
                print("Bases: ", i, j, " are nearly identical")
                a = kruskal_top(sames,i)
                b = kruskal_top(sames,j)
                chosen = min(a,b)
                sames[i] = chosen
                sames[j] = chosen
            arr.append(dp)

    final_groups = {}
    for i in range(vectors.shape[1]):
        if i not in sames:
            sames[i] = i
        group_id = kruskal_top(sames, i)
        if group_id not in final_groups:
            final_groups[group_id] = set()
        final_groups[group_id].add(i)

    print(final_groups)

    out_vecs = []
    for group_id in final_groups:
        group = final_groups[group_id]
        # Build final vector as weighted average of vectors in group
        total_vec = np.zeros(vectors.shape[0])
        total_weight = 0
        for col in group:
            w = weights[col]
            if len(group) == 1:
                w = 1
            total_vec += vectors[:, col] * w
            total_weight += w
        if total_weight == 0:
            print(group_id, "has 0 weight")
            continue
        total_vec /= total_weight

        # Then L1 normalize output vector
        total_vec = total_vec / np.sum(np.abs(total_vec))
        out_vecs.append(total_vec)

    return np.stack(out_vecs, axis=-1)
