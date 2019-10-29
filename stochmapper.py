"""
@author: Mathieu Carriere
All rights reserved
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.base            import BaseEstimator, TransformerMixin
from sklearn.preprocessing   import LabelEncoder
from sklearn.cluster         import DBSCAN, AgglomerativeClustering
from sklearn.metrics         import pairwise_distances
from scipy.spatial.distance  import directed_hausdorff
from scipy.sparse            import csgraph
from scipy.stats             import entropy
from sklearn.neighbors       import KernelDensity, kneighbors_graph, radius_neighbors_graph, NearestNeighbors
from ot.bregman              import sinkhorn2

try:
    import gudhi as gd

except ImportError:
    print("Gudhi not found: StochasticMapperComplex will not work")

class EntropyRegularizedWasserstein(BaseEstimator, TransformerMixin):

    def __init__(self, epsilon=1e-4, num_bins=10, bnds=(0,1)):
        self.epsilon = epsilon
        self.n_bins, self.bnds = num_bins, bnds

    def compute_distance(self, d1, d2):
        h1, e1 = np.histogram(d1, bins=self.n_bins, range=self.bnds)
        h2, e2 = np.histogram(d2, bins=self.n_bins, range=self.bnds)
        h1 = h1/np.sum(h1)
        h2 = h2/np.sum(h2)
        c1 = (e1[:-1] + e1[1:])/2
        c2 = (e2[:-1] + e2[1:])/2
        return sinkhorn2(a=h1, b=h2, M=np.abs(np.reshape(c1,[-1,1])-np.reshape(c2,[1,-1])), reg=self.epsilon)
    
    def compute_matrix(self, D):
        num_dist = len(D)
        H, C = [], []
        for i in range(num_dist):
            h, e = np.histogram(D[i], bins=self.n_bins, range=self.bnds)
            c = (e[:-1] + e[1:])/2
            h = h/np.sum(h)
            H.append(h)
            C.append(c)
        M = np.zeros([num_dist, num_dist])
        for i in range(num_dist):
            print(i)
            for j in range(i+1, num_dist):
                M[i,j] = sinkhorn2(a=H[i], b=H[j], M=np.abs(np.reshape(C[i],[-1,1])-np.reshape(C[j],[1,-1])), reg=self.epsilon, numItermax=10)
                M[j,i] = M[i,j]
        return M

class KullbackLeiblerDivergence(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_bins=10, bnds=(0,1)):
        self.n_bins, self.bnds = num_bins, bnds
        
    def compute_distance(self, d1, d2):
        h1, _ = np.histogram(d1, bins=self.n_bins, range=self.bnds)
        h2, _ = np.histogram(d2, bins=self.n_bins, range=self.bnds)
        h1 = h1/np.sum(h1)
        h2 = h2/np.sum(h2)
        return entropy(h1, h2)
    
    def compute_matrix(self, D):
        num_dist = len(D)
        H = []
        for i in range(num_dist):
            h, _ = np.histogram(D[i], bins=self.n_bins, range=self.bnds)
            h = h/np.sum(h)
            H.append(h)
        M = np.zeros([num_dist, num_dist])
        for i in range(num_dist):
            for j in range(i+1, num_dist):
                M[i,j] = entropy(H[i], H[j])
                M[j,i] = M[i,j]
        return M
    
class EuclideanDistance(BaseEstimator, TransformerMixin):
    
    def __init__(self, num_bins=10, bnds=(0,1)):
        self.n_bins, self.bnds = num_bins, bnds
        
    def compute_distance(self, d1, d2):
        h1, _ = np.histogram(d1, bins=self.n_bins, range=self.bnds)
        h2, _ = np.histogram(d2, bins=self.n_bins, range=self.bnds)
        h1 = h1/np.sum(h1)
        h2 = h2/np.sum(h2)
        return np.linalg.norm(h1-h2)
    
    def compute_matrix(self, D):
        num_dist = len(D)
        H = []
        for i in range(num_dist):
            h, _ = np.histogram(D[i], bins=self.n_bins, range=self.bnds)
            h = h/np.sum(h)
            H.append(np.reshape(h, [1,-1]))
        return pairwise_distances(np.vstack(H))
    
class AgglomerativeCover(BaseEstimator, TransformerMixin):

    def __init__(self, n_patches=10, threshold=1.):
        self.n_patches, self.threshold = n_patches, threshold

    def compute_cover(self, M):

        # Partition data with agglomerative clustering
        agg = AgglomerativeClustering(n_clusters=self.n_patches, linkage="single", affinity="precomputed").fit(M)
        binned_data = {i: [] for i in range(self.n_patches)}
        for idx, lab in enumerate(agg.labels_):
            binned_data[lab].append(idx)

        # Thicken clusters so that they overlap
        for i in range(self.n_patches):
            pts_cluster, pts_others = np.reshape(np.argwhere(np.array(agg.labels_) == i), [-1]), np.reshape(np.argwhere(np.array(agg.labels_) != i), [-1])
            pts_in_offset = pts_others[np.reshape(np.argwhere(M[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
            for p in pts_in_offset:
                binned_data[i].append(p)

        return binned_data

class VoronoiCover(BaseEstimator, TransformerMixin):

    def __init__(self, n_patches=10, threshold=1.):
        self.n_patches, self.threshold = n_patches, threshold

    def compute_cover(self, M):

        # Partition data with Voronoi cover
        germs = np.random.choice(M.shape[0], self.n_patches, replace=False)
        labels = np.argmin(M[germs,:], axis=0)
        binned_data = {i: [] for i in range(self.n_patches)}
        for i in range(M.shape[0]):
            binned_data[labels[i]].append(i)

        # Thicken clusters so that they overlap
        for i in range(self.n_patches):
            pts_cluster, pts_others = np.reshape(np.argwhere(labels == i), [-1]), np.reshape(np.argwhere(labels != i), [-1])
            pts_in_offset = pts_others[np.reshape(np.argwhere(M[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
            for p in pts_in_offset:
                binned_data[i].append(p)

        return binned_data

class StochasticMapperComplex(BaseEstimator, TransformerMixin):
    """
    This is a class for computing Mapper simplicial complexes on point clouds or distance matrices. 
    """
    def __init__(self, distributions, colors, cover=AgglomerativeCover, distance=EntropyRegularizedWasserstein, inp="point cloud", clustering=DBSCAN(), mask=0):
        """
        Constructor for the MapperComplex class.

        Attributes:
            inp (string): either "point cloud" or "distance matrix". Specifies the type of input data.
            distributions (list (of length num_points) of lists of floats): Probability distributions associated to each input points.
            distance (function): Distance used to compute probability distributions.
            distance_params (dict): Dictionary that contains distance parameters.
            colors (numpy array of shape (num_points) x (num_colors)): functions used to color the nodes of the output Mapper simplicial complex. More specifically, coloring is done by computing the means of these functions on the subpopulations corresponding to each node.
            clustering (class): clustering class (default sklearn.cluster.DBSCAN()). Common clustering classes can be found in the scikit-learn library (such as AgglomerativeClustering for instance).
            mask (int): threshold on the size of the Mapper nodes (default 0). Any node associated to a subpopulation with less than **mask** points will be removed.

            mapper_ (gudhi SimplexTree): Mapper simplicial complex computed after calling the fit() method
            node_info_ (dictionary): various information associated to the nodes of the Mapper. 
        """
        self.distributions, self.distance, self.colors, self.clustering = distributions, distance, colors, clustering
        self.cover, self.input, self.mask = cover, inp, mask

    def fit(self, X, y=None):
        """
        Fit the MapperComplex class on a point cloud or a distance matrix: compute the Mapper and store it in a simplex tree called mapper_

        Parameters:
            X (numpy array of shape (num_points) x (num_coordinates) if point cloud and (num_points) x (num_points) if distance matrix): input point cloud or distance matrix.
            y (n x 1 array): point labels (unused).
        """
        num_pts, num_colors = X.shape[0], self.colors.shape[1]

        # Compute pairwise distances in codomain
        if type(self.distance) is np.ndarray:
            codomain_distances = self.distance
        else:
            codomain_distances = np.zeros([num_pts, num_pts])
            for i in range(num_pts):
                #print(i)
                for j in range(i+1, num_pts):
                    codomain_distances[i,j] = self.distance.compute_distance(self.distributions[i], self.distributions[j])
                    #print(self.distributions[i], self.distributions[j])
                    #print(codomain_distances[i,j])
                    codomain_distances[j,i] = codomain_distances[i,j]

        # Compute cover
        binned_data = self.cover.compute_cover(codomain_distances)

        # Initialize the cover map, that takes a point and outputs the clusters to which it belongs
        cover, clus_base = [[] for _ in range(num_pts)], 0

        # Initialize attributes
        self.mapper_, self.node_info_ = gd.SimplexTree(), {}

        # For each patch
        for preimage in binned_data:

            # Apply clustering on the corresponding subpopulation
            idxs = np.array(binned_data[preimage])
            if len(idxs) > 1:
                clusters = self.clustering.fit_predict(X[idxs,:]) if self.input == "point cloud" else self.clustering.fit_predict(X[idxs,:][:,idxs])
            elif len(idxs) == 1:
                clusters = np.array([0])
            else:
                continue

            # Collect various information on each cluster
            num_clus_pre = np.max(clusters) + 1
            for clus_i in range(num_clus_pre):
                node_name = clus_base + clus_i
                subpopulation = idxs[clusters == clus_i]
                self.node_info_[node_name] = {}
                self.node_info_[node_name]["indices"] = subpopulation
                self.node_info_[node_name]["size"] = len(subpopulation)
                self.node_info_[node_name]["colors"] = np.mean(self.colors[subpopulation,:], axis=0)
                self.node_info_[node_name]["patch"] = preimage

            # Update the cover map
            for pt in range(clusters.shape[0]):
                node_name = clus_base + clusters[pt]
                if clusters[pt] != -1 and self.node_info_[node_name]["size"] >= self.mask:
                    cover[idxs[pt]].append(node_name)

            clus_base += np.max(clusters) + 1

        # Insert the simplices of the Mapper complex 
        for i in range(num_pts):
            self.mapper_.insert(cover[i], filtration=-3)
        self.mapper_.initialize_filtration()

        return self
