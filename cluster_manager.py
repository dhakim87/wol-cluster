from linear_subspace_clustering import linear_subspace_clustering, calc_residuals, calc_subspace_bases
import numpy as np
import pandas as pd
import json
from detnmf.detnmf import run_detnmf, run_caliper_nmf


OUTLIER_CLUSTER_ID = -1
class ClusterManager:
    
    def __init__(self, df, dim):
        self.data = df.T.to_numpy()
        self.data_cols = df.columns
        self.cluster_state = ClusterState(self, -np.ones(df.shape[0]), {-1:dim})
        self.cluster_id_gen = -1
        self.initial_dim = dim
        self.saved_subspace_bases = None
        
    @classmethod
    def apply_bases_from_detnmf(clz, df, n_components, alpha_scale=0.05, iterations=5000):
        W,H = run_detnmf(df.to_numpy(), n_components, alpha_scale = alpha_scale, iterations=iterations)
        
        subspace_bases = {}
        for subspace in range(H.shape[0]):
            conic_basis = H[subspace,:].T
            conic_basis = pd.DataFrame(conic_basis)
            conic_basis.index = df.columns.tolist()
            conic_basis = conic_basis.to_numpy()
            subspace_bases[subspace] = conic_basis
                
        cm = clz(df, df.shape[1])
        cm.cluster_state.cluster_dims={k:subspace_bases[k].shape[1] for k in subspace_bases}
        cm.cluster_state.cluster_dims[OUTLIER_CLUSTER_ID] = cm.initial_dim
        cm.get_reassign_nearest(subspace_bases=subspace_bases).apply()
        cm.cluster_id_gen = max(cm.cluster_state.cluster_dims.keys())
        cm.saved_subspace_bases = subspace_bases
        return cm
        
        
    @classmethod
    def apply_bases_from_file(clz, file, df):
        with open(file) as infile:
            data_transform = json.load(infile)
        subspace_bases = {}
        key = list(data_transform.keys())[0]
        if "all" in data_transform:
            key = "all"
        for subspace in range(len(data_transform[key])):
            conic_basis = pd.read_json(data_transform[key][subspace])
            conic_basis = conic_basis.loc[df.columns.tolist()]
            subspace_bases[subspace] = conic_basis
                
        if df.columns.tolist() != conic_basis.index.tolist():
            raise Exception("MISMATCHED ROWS/COLUMNS")

        cm = clz(df, df.shape[1])
        cm.cluster_state.cluster_dims={k:subspace_bases[k].shape[1] for k in subspace_bases}
        cm.cluster_state.cluster_dims[OUTLIER_CLUSTER_ID] = cm.initial_dim
        cm.get_reassign_nearest(subspace_bases=subspace_bases).apply()
        cm.cluster_id_gen = max(cm.cluster_state.cluster_dims.keys())
        cm.saved_subspace_bases = subspace_bases
        return cm
    
    def calc_subspace_bases(self):
        # When bases came from a file or from detnmf, we just return those subspace bases.  
        # otherwise, we recalculate them based on the clustering
        if self.saved_subspace_bases is not None:
            return self.saved_subspace_bases
        else:
            return calc_subspace_bases(self.data, self.cluster_state.clusters, self.cluster_state.cluster_dims)
            
    
    def get_merge_clusters(self, i, j, new_dim):
        new_state = self.cluster_state.copy()
        new_id = self.gen_cluster_id()
        new_state.cluster_dims[new_id] = new_dim
        
        pidx = np.where(new_state.clusters == i)[0]
        pidx2 = np.where(new_state.clusters == j)[0]
        
        new_state.clusters[pidx] = new_id
        new_state.clusters[pidx2] = new_id
        
        del new_state.cluster_dims[i]
        del new_state.cluster_dims[j]

        return new_state
        
    def get_split_cluster(self, i, num_subclusters):
        new_state = self.cluster_state.copy()
        
        idx = np.where(new_state.clusters == i)[0]
        cluster_data = self.data[:, idx]
        new_dim = max(new_state.cluster_dims[i] - 1, 1)
        
        new_clusters, _ = linear_subspace_clustering(
            cluster_data, 
            nr_clusters=num_subclusters, 
            subspace_dim=new_dim
        )
        
        new_labels = np.unique(new_clusters)
        mapper = {x: self.gen_cluster_id() for x in new_labels}
        
        for key in mapper:
            pidx = np.where(new_clusters == key)[0]
            new_clusters[pidx] = mapper[key]
        
        np.put(new_state.clusters, idx, new_clusters)
        del new_state.cluster_dims[i]
        for key in mapper:
            new_state.cluster_dims[mapper[key]] = new_dim
            
        if OUTLIER_CLUSTER_ID not in new_state.cluster_dims:
            new_state.cluster_dims[OUTLIER_CLUSTER_ID] = self.initial_dim
        
        return new_state
    
    def get_delete_cluster(self, cluster_id):
        if cluster_id == OUTLIER_CLUSTER_ID:
            return self.cluster_state
        
        new_state = self.cluster_state.copy()
        new_state.clusters[new_state.clusters == cluster_id] = OUTLIER_CLUSTER_ID
        del new_state.cluster_dims[cluster_id]
        return new_state
    
    def apply_active_clustering(self, new_df, dim_penalty=0, outlier_thresh=None):
        subspace_bases = calc_subspace_bases(self.data, self.cluster_state.clusters, self.cluster_state.cluster_dims)
        
        new_filt_df = new_df[self.data_cols]

        names = []
        closests = []
        for index, pt in new_filt_df.iterrows():
            closest = self._find_closest(subspace_bases, pt, self.cluster_state.cluster_dims, dim_penalty=dim_penalty, outlier_thresh=outlier_thresh)
            names.append(pt.name)
            closests.append(closest)
        
        return pd.Series(data=closests, index=names)        
        
    def _find_closest(self, subspace_bases, pt, cluster_dims, dim_penalty=0, outlier_thresh=None):
        closest_cluster = OUTLIER_CLUSTER_ID
        min_real_dist = float('inf')
        min_heuristic_dist = float('inf')
        for cluster_id in subspace_bases:
            proj_pt = np.matmul(subspace_bases[cluster_id], np.matmul(subspace_bases[cluster_id].T, pt))
            dist = np.linalg.norm(pt - proj_pt)
            heuristic_dist = dist + dim_penalty * cluster_dims[cluster_id]

            if heuristic_dist < min_heuristic_dist:
                closest_cluster = cluster_id
                min_real_dist = dist
                min_heuristic_dist = heuristic_dist

        if outlier_thresh is None:
            return closest_cluster
        else:
            num_reads = np.sum(pt)
            if num_reads == 0 or min_real_dist / num_reads > outlier_thresh:
                return OUTLIER_CLUSTER_ID
            else:
                return closest_cluster
        
    # How should we manage rules about outliers?  Pass a validation function here?
    def get_reassign_nearest(self, dim_penalty=0, outlier_thresh=None, subspace_bases=None):
        new_state = self.cluster_state.copy()

        if subspace_bases is None:
            subspace_bases = calc_subspace_bases(self.data, new_state.clusters, new_state.cluster_dims)

        for sample_index in range(self.data.shape[1]):
            pt = self.data[:, sample_index]
            closest = self._find_closest(subspace_bases, pt, new_state.cluster_dims, dim_penalty=dim_penalty, outlier_thresh=outlier_thresh)
            new_state.clusters[sample_index] = closest

        return new_state
        
    
    #TODO pass ruleset instead of idx
    def get_refine_outliers(self, outlier_idx):
        new_state = self.cluster_state.copy()
        
        outlier_cluster_id = -1
        new_state.clusters[outlier_idx] = outlier_cluster_id
        
        final_labels = np.unique(new_state.clusters)
        
        to_del = []
        for key in new_state.cluster_dims:
            if key not in final_labels:
                to_del.append(key)
        for key in to_del:
            del new_state.cluster_dims[key]
        
        new_state.cluster_dims[outlier_cluster_id] = self.initial_dim
        
        return new_state
    
    def gen_cluster_id(self):
        self.cluster_id_gen += 1
        return self.cluster_id_gen
    
    def max_cluster_id(self):
        return self.cluster_id_gen
    
    def finalize(self):
        cluster_dims = self.cluster_state.cluster_dims
        cluster_counts = self.cluster_state.get_cluster_counts()

        print("START FINALIZE:", cluster_counts)
        keys = [(k, cluster_dims[k], cluster_counts.get(k,0)) for k in cluster_dims.keys()]
        keys.sort(key=lambda x:(x[1],-x[2]))
        final_pos = {}
        for i in range(len(keys)):
            final_pos[keys[i][0]] = i

        final_pos[OUTLIER_CLUSTER_ID] = OUTLIER_CLUSTER_ID
        self.cluster_state.clusters = np.vectorize(final_pos.get)(self.cluster_state.clusters)
        sorted_cluster_dims = {}
        for k in final_pos:
            sorted_cluster_dims[final_pos[k]] = cluster_dims[k]
        self.cluster_state.cluster_dims = sorted_cluster_dims
        self.cluster_id_gen = max(sorted_cluster_dims.keys())

    
class ClusterState:
    def __init__(self, manager, clusters, cluster_dims):
        self.manager = manager
        self.clusters = clusters
        self.cluster_dims = cluster_dims
    
    def get_cluster_resid(self, i=None):
        resid = calc_residuals(self.manager.data, self.clusters, self.cluster_dims, target_cluster=i)
        if i is None:
            return resid
        else:
            pidx = np.where(self.clusters == i)[0]
            return resid[pidx]
        
    def get_cluster_counts(self):
        labels, counts = np.unique(self.clusters, return_counts=True)
        label_counts = {}
        max_id = self.manager.max_cluster_id()
        
        # initialize for all labels
        for i in range(0, max_id + 1):
            label_counts[i] = 0
        # Fill in labels that occurred
        for i in range(len(labels)):
            label_counts[labels[i]] = counts[i]
        # Special case for outlier label
        if OUTLIER_CLUSTER_ID in self.cluster_dims and OUTLIER_CLUSTER_ID not in label_counts:
            label_counts[OUTLIER_CLUSTER_ID] = 0
        return label_counts
    
    def apply(self):
        # TODO: Consider stating which clusters are modified so that validation can verify only those clusters.
        self.manager.cluster_state = self
    
    def copy(self):
        return ClusterState(self.manager, self.clusters.copy(), self.cluster_dims.copy())
    
    def num_clusters(self):
        count = len(self.cluster_dims)
        if OUTLIER_CLUSTER_ID in self.cluster_dims:
            count -= 1
        return count

    
    
class ClusterRule:
    def __init__():
        pass
    
    def validate(cluster):
        pass
    
    
class RecursiveClusterer:
    def __init__(self):
        self.MIN_CLUSTER_COUNT=10 #Minimum of 10 points per cluster
        self.MAX_80_RESID=.10 #80th percentile of fractional residuals must be < .10 * number of reads in that sample
        self.MAX_OUTLIER_RESID=.10
    
    def is_valid(self, data, cluster_state, target_nr_clusters):
        labels, counts = np.unique(cluster_state.clusters, return_counts=True)
        
        for i in range(len(labels)):
            l = labels[i]
            cluster_count = counts[i]
            if cluster_count < self.MIN_CLUSTER_COUNT and l != OUTLIER_CLUSTER_ID:
                print("FAIL", l, cluster_count)
                return False
        
        resid = cluster_state.get_cluster_resid()
        sums = np.sum(data, axis=0)
        
        cluster_80_residuals = {}

        fractional_residuals = resid / sums

        for label in labels:
            if label == OUTLIER_CLUSTER_ID:
                continue
                
            idx = np.where(cluster_state.clusters == label)[0]
            cluster_fractional_residuals = fractional_residuals[idx]
            
            # Sort of arbitrary what we threshold on.  Could pick mean or median residual, or look at quantiles
            # This picks the .8 quantile.
            cluster_80_residuals[label] = np.quantile(cluster_fractional_residuals, .8)
            
        if len(cluster_80_residuals) != target_nr_clusters:
            # Too few clusters
            print("TOO FEW", cluster_80_residuals, target_nr_clusters)
            return False

        for c in cluster_80_residuals:
            if cluster_80_residuals[c] > 0.10:
                print("HIGH RESID", c, cluster_80_residuals[c])
                return False
        
        return True
    
    def find_outlier_idx(self, data, cluster_state):
        labels, counts = np.unique(cluster_state.clusters, return_counts=True)
        resid = cluster_state.get_cluster_resid()
        sums = np.sum(data, axis=0)        
        fractional_residuals = resid / sums
        return fractional_residuals > self.MAX_OUTLIER_RESID
    
    def run(self, cm):
        total_clusters = 0
        good = self.is_valid(cm.data, cm.cluster_state, total_clusters)
        
        if not good:
            raise Exception("Initial cluster_manager state is invalid")
        
        for refine_iter in range(5):
            split_cluster = -1
            if np.sum(cm.cluster_state.clusters == OUTLIER_CLUSTER_ID) < self.MIN_CLUSTER_COUNT:
                # can't split outliers when there are too few outliers to make a cluster
                split_cluster = 0

            while split_cluster <= cm.max_cluster_id():
                if split_cluster not in cm.cluster_state.clusters:
                    split_cluster += 1
                    continue

                cluster_dim = cm.cluster_state.cluster_dims[split_cluster]
                if cluster_dim == 1:
                    split_cluster += 1
                    continue

                for num_subclusters in range(1, max(cluster_dim, 5)):  # Small fudge factor to deal with co (hyper)planar subspace division, we always try fitting at least 5 things of the smaller dimension
                    cs = cm.get_split_cluster(split_cluster, num_subclusters)
                    if split_cluster != OUTLIER_CLUSTER_ID:
                        offset = 1
                    else:
                        offset = 0
                    if self.is_valid(cm.data, cs, total_clusters + num_subclusters - offset):
                        print("SPLIT: ", split_cluster, "->", num_subclusters, " pieces")
                        cs.apply()
                        print(np.unique(cm.cluster_state.clusters))
                        total_clusters = total_clusters + num_subclusters - offset
                        break

                split_cluster += 1

            # Refine all clusters by gathering up their
            # outliers and assigning them to OUTLIER_CLUSTER_ID
            outlier_idx = self.find_outlier_idx(cm.data, cm.cluster_state)
            cs = cm.get_refine_outliers(outlier_idx)
            if self.is_valid(cm.data, cs, total_clusters):
                print("REFINE: Outliers:", outlier_idx.sum())
                cs.apply()
            else:
                break
        
        cm.finalize()