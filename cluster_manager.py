from linear_subspace_clustering import linear_subspace_clustering, calc_residuals
import numpy as np

OUTLIER_CLUSTER_ID = -1
class ClusterManager:
    
    def __init__(self, df, dim):
        self.data = df.T.to_numpy()
        self.cluster_state = ClusterState(self, -np.ones(df.shape[0]), {-1:dim})
        self.cluster_id_gen = -1
        self.initial_dim = dim
    
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
        new_dim = new_state.cluster_dims[i] - 1
        
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
    
    def get_reassign_outliers(self):
        new_state = self.cluster_state.copy()
        
        
    
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

        keys = [(k, cluster_dims[k], cluster_counts[k]) for k in cluster_dims.keys()]
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
        for i in range(len(labels)):
            label_counts[labels[i]] = counts[i]
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