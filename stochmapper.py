"""
@author: Mathieu Carriere
All rights reserved
"""
import matplotlib.pyplot as plt
import numpy as np
import itertools

from sklearn.base             import BaseEstimator, TransformerMixin
from sklearn.preprocessing    import LabelEncoder
from sklearn.cluster          import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics          import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance   import directed_hausdorff
from scipy.sparse             import csgraph
from scipy.stats              import entropy
from sklearn.neighbors        import KernelDensity, kneighbors_graph, radius_neighbors_graph, NearestNeighbors
from ot.bregman               import sinkhorn2, barycenter
from ot.lp                    import wasserstein_1d

try:
    import gudhi as gd
    import sklearn_tda as sktda
except ImportError:
    print("Gudhi not found: StochasticMapperComplex will not work")

def infer_distributions_from_neighborhood(real, X, threshold=1., domain="point cloud"):
    num_pts, distributions = len(X), []
    pdist = X if domain == "distance matrix" else pairwise_distances(X)
    for i in range(num_pts):
        distrib = np.squeeze(np.argwhere(pdist[i,:] <= threshold))
        np.random.shuffle(distrib)
        distributions.append([real[n] for n in distrib])
    return distributions

class Histogram(BaseEstimator, TransformerMixin):

    def __init__(self, num_bins=10, bnds=(0,1)):
        self.n_bins, self.bnds = num_bins, bnds

    def compute_histograms(self, D):
        num_dist = len(D)
        H, C = [], []
        for i in range(num_dist):
            h, e = np.histogram(D[i], bins=self.n_bins, range=self.bnds)
            c = (e[:-1] + e[1:])/2
            h = h/np.sum(h)
            H.append(h)
            C.append(c)
        return H, C

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
        H, C = Histogram(num_bins=self.n_bins, bnds=self.bnds).compute_histograms(D)
        M = np.zeros([num_dist, num_dist])
        for i in range(num_dist):
            for j in range(i+1, num_dist):
                M[i,j] = sinkhorn2(a=H[i], b=H[j], M=np.abs(np.reshape(C[i],[-1,1])-np.reshape(C[j],[1,-1])), reg=self.epsilon, numItermax=10)
                M[j,i] = M[i,j]
        return M

class Wasserstein1D(BaseEstimator, TransformerMixin):

    def __init__(self, p=1, num_bins=10, bnds=(0,1)):
        self.p = p
        self.n_bins, self.bnds = num_bins, bnds

    def compute_distance(self, d1, d2):
        h1, e1 = np.histogram(d1, bins=self.n_bins, range=self.bnds)
        h2, e2 = np.histogram(d2, bins=self.n_bins, range=self.bnds)
        h1 = h1/np.sum(h1)
        h2 = h2/np.sum(h2)
        c1 = (e1[:-1] + e1[1:])/2
        c2 = (e2[:-1] + e2[1:])/2
        return wasserstein_1d(x_a=c1, x_b=c2, a=h1, b=h2, p=self.p)
    
    def compute_matrix(self, D):
        num_dist = len(D)
        H, C = Histogram(num_bins=self.n_bins, bnds=self.bnds).compute_histograms(D)
        M = np.zeros([num_dist, num_dist])
        for i in range(num_dist):
            for j in range(i+1, num_dist):
                M[i,j] = wasserstein_1d(x_a=C[i], x_b=C[j], a=H[i], b=H[j], p=self.p)
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
        H, _ = Histogram(num_bins=self.n_bins, bnds=self.bnds).compute_histograms(D)
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
        H, _ = Histogram(num_bins=self.n_bins, bnds=self.bnds).compute_histograms(D)
        return euclidean_distances(np.vstack(H))

class VoronoiCover(BaseEstimator, TransformerMixin):

    def __init__(self, n_patches=10, threshold=1.):

        self.n_patches = n_patches
        self.threshold = threshold
        self.mode = "metric"

    def compute_cover(self, D):

        # Partition data with Voronoi cover
        germs = np.random.choice(D.shape[0], self.n_patches, replace=False)
        labels = np.argmin(D[germs,:], axis=0)
        binned_data = {}
        for i, l in enumerate(labels):
            try:
                binned_data[l].append(i)
            except KeyError:
                binned_data[l] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            for i in binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(labels == i), [-1]), np.reshape(np.argwhere(labels != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(D[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    binned_data[i].append(p)

        return binned_data

class EuclideanKMeansCover(BaseEstimator, TransformerMixin):

    def __init__(self, n_patches=10, 
                       threshold=1., distances=None,
                       histo=True, num_bins=10, bnds=(0,1)):

        self.n_patches = n_patches
        self.threshold, self.distances = threshold, distances
        self.histo, self.n_bins, self.bnds = histo, num_bins, bnds
        self.mode = "embedding"

    def compute_cover(self, D):

        # Compute histograms
        if self.histo:
            num_dist = len(D)
            Y, _ = Histogram(num_bins=self.n_bins, bnds=self.bnds).compute_histograms(D)
            Y = np.vstack(Y)
        else:
            Y = D

        # Euclidean KMeans on histograms
        km = KMeans(n_clusters=self.n_patches).fit(Y)
        binned_data = {}
        for i, l in enumerate(km.labels_):
            try:
                binned_data[l].append(i)
            except KeyError:
                binned_data[l] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            DY = self.distances if self.distances is not None else euclidean_distances(Y)
            for i in binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(km.labels_ == i), [-1]), np.reshape(np.argwhere(km.labels_ != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(DY[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    binned_data[i].append(p)

        return binned_data

class WassersteinKMeansCover(BaseEstimator, TransformerMixin):

    def __init__(self, n_patches=10, epsilon=1e-4, p=1, 
                       threshold=1., distances=None,
                       histo=True, num_bins=10, bnds=(0,1)):

        self.n_patches, self.epsilon, self.p = n_patches, epsilon, p
        self.threshold, self.distances = threshold, distances
        self.histo, self.n_bins, self.bnds = histo, num_bins, bnds
        self.mode = "embedding"

    def compute_cover(self, D):
        
        # Compute histograms
        if self.histo:
            num_dist = len(D)
            Y, C = Histogram(num_bins=self.n_bins, bnds=self.bnds).compute_histograms(D)
            Y = np.vstack(Y)
        else:
            Y, C = D[0], D[1]

        cost = np.abs(np.reshape(C[i],[-1,1])-np.reshape(C[j],[1,-1]))
        curr_patches = Y[np.random.choice(np.arange(len(Y)), size=self.n_patches, replace=False), :]
        criterion = np.inf
        while criterion > self.tolerance:
            dists = np.zeros([len(curr_patches), len(Y)])
            for i in range(len(curr_patches)):
                for j in range(len(Y)):
                    dists[i,j] = wasserstein_1d(x_a=C[0], x_b=C[0], a=curr_patches[i,:], b=Y[j,:], p=1)
            Q = np.argmin(dists.T, axis=1)
            new_curr_patches = []
            for t in range(self.n_patches):
                if len(np.argwhere(Q==t)) > 0:
                    new_curr_patches.append(np.reshape(barycenter(A=Y[np.argwhere(Q==t)[:,0],:].T, M=cost, reg=self.epsilon), [1,-1]))
                else:
                    new_curr_patches.append(curr_patches[t:t+1,:])
            new_curr_patches = np.vstack(new_curr_patches)
            criterion = np.sqrt(np.square(new_curr_patches - curr_patches).sum(axis=1)).max(axis=0)
            curr_patches = new_curr_patches

        binned_data = {}
        for i in range(len(Q)):
            try:
                binned_data[Q[i]].append(i)
            except KeyError:
                binned_data[Q[i]] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            if self.distances is not None:
                DY = self.distances
            else:
                DY = DY = np.zeros([len(Y), len(Y)])
                for i in range(len(Y)):
                    for j in range(i+1, len(Y)):
                        DY[i,j] = wasserstein_1d(x_a=C[0], x_b=C[0], a=Y[i,:], b=Y[j,:], p=1)
                        DY[j,i] = DY[i,j]
            for i in binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(Q == i), [-1]), np.reshape(np.argwhere(Q != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(DY[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    binned_data[i].append(p)

        return binned_data


class kPDTMCover(BaseEstimator, TransformerMixin):

    def __init__(self, n_patches=10, h=10, tol=1e-4, 
                       threshold=1., distances=None,
                       histo=True, num_bins=10, bnds=(0,1)):

        self.n_patches, self.h, self.tolerance = n_patches, h, tol
        self.threshold, self.distances = threshold, distances
        self.histo, self.n_bins, self.bnds = histo, num_bins, bnds
        self.mode = "embedding"

    def compute_cover(self, D):
        
        # Compute histograms
        if self.histo:
            num_dist = len(D)
            Y, _ = Histogram(num_bins=self.n_bins, bnds=self.bnds).compute_histograms(D)
            Y = np.vstack(Y)
        else:
            Y = D

        curr_patches = Y[np.random.choice(np.arange(len(Y)), size=self.n_patches, replace=False), :]
        criterion = np.inf
        while criterion > self.tolerance:
            dists = euclidean_distances(curr_patches, Y)
            means, variances = [], []
            for t in range(self.n_patches):
                ball_idxs = np.argpartition(dists[t,:], self.h)[:self.h]
                M = np.reshape(Y[ball_idxs,:].mean(axis=0), [1,-1])
                V = np.square(euclidean_distances(M, Y[ball_idxs,:])).sum()
                means.append(M)
                variances.append(V)
            Q = np.argmin(np.square(euclidean_distances(Y, np.vstack(means))) + np.reshape(np.array(variances), [1,-1]), axis=1)
            new_curr_patches = []
            for t in range(self.n_patches):
                if len(np.argwhere(Q==t)) > 0:
                    new_curr_patches.append(np.reshape(Y[np.argwhere(Q==t)[:,0],:].mean(axis=0), [1,-1]))
                else:
                    new_curr_patches.append(curr_patches[t:t+1,:])
            new_curr_patches = np.vstack(new_curr_patches)
            criterion = np.sqrt(np.square(new_curr_patches - curr_patches).sum(axis=1)).max(axis=0)
            curr_patches = new_curr_patches

        binned_data = {}
        for i in range(len(Q)):
            try:
                binned_data[Q[i]].append(i)
            except KeyError:
                binned_data[Q[i]] = [i]

        # Thicken clusters so that they overlap
        DY = self.distances if self.distances is not None else euclidean_distances(Y)
        for i in binned_data.keys():
            pts_cluster, pts_others = np.reshape(np.argwhere(Q == i), [-1]), np.reshape(np.argwhere(Q != i), [-1])
            pts_in_offset = pts_others[np.reshape(np.argwhere(DY[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
            for p in pts_in_offset:
                binned_data[i].append(p)

        return binned_data

class StochasticMapperComplex(BaseEstimator, TransformerMixin):

    def __init__(self, filters, colors, codomain="distributions", infer_distributions=True, threshold=1.,  
                       cover=VoronoiCover, distance=EntropyRegularizedWasserstein, 
                       domain="point cloud", clustering=DBSCAN(), 
                       mask=0):
        
        self.filters, self.codomain, self.infdist, self.threshold = filters, codomain, infer_distributions, threshold
        self.cover, self.distance = cover, distance
        self.domain, self.clustering = domain, clustering
        self.mask, self.colors = mask, colors

    def fit(self, X, y=None):

        num_pts, num_colors = len(X), self.colors.shape[1]

        if self.codomain == "distributions":
            if self.infdist:
                # self.filters is supposed to be a list containing a single realization of each conditional
                distributions = infer_distributions_from_neighborhood(self.filters, X, self.threshold, self.domain)
            else:
                # self.filters is supposed to be a list of lists containing the distribution of each conditional
                distributions = self.filters
        if self.codomain == "distance matrix":
            # self.filters is supposed to be a square array of pairwise distances
            codomain_distances = self.filters


        if self.cover.mode == "metric":
            if self.codomain is not "distance matrix":
                codomain_distances = self.distance.compute_matrix(distributions)

        # Compute cover
        if self.cover.mode == "metric":
            binned_data = self.cover.compute_cover(codomain_distances)
        if self.cover.mode == "embedding":
            binned_data = self.cover.compute_cover(distributions)

        # Initialize the cover map, that takes a point and outputs the clusters to which it belongs
        cover, clus_base = [[] for _ in range(num_pts)], 0

        # Initialize attributes
        self.mapper_, self.node_info_ = gd.SimplexTree(), {}

        # For each patch
        for preimage in binned_data:

            # Apply clustering on the corresponding subpopulation
            idxs = np.array(binned_data[preimage])
            if len(idxs) > 1:
                clusters = self.clustering.fit_predict(X[idxs,:]) if self.domain == "point cloud" else self.clustering.fit_predict(X[idxs,:][:,idxs])
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

class MeanStochasticMapperComplex(BaseEstimator, TransformerMixin):

    def __init__(self, filters, colors, infer_distributions=True, threshold=1.,  
                       resolution=10, gain=.3,
                       domain="point cloud", clustering=DBSCAN(), 
                       mask=0):
        
        self.filters, self.infdist, self.threshold = filters, infer_distributions, threshold
        self.resolution, self.gain = resolution, gain
        self.domain, self.clustering = domain, clustering
        self.mask, self.colors = mask, colors

    def fit(self, X, y=None):

        num_pts, num_colors = len(X), self.colors.shape[1]

        if self.infdist:
            # self.filters is supposed to be a list containing a single realization of each conditional
            distributions = infer_distributions_from_neighborhood(self.filters, X, self.threshold, self.domain)
        else:
            # self.filters is supposed to be a list of lists containing the distribution of each conditional
            distributions = self.filters

        flt = np.reshape([np.mean(distrib) for distrib in distributions], [-1,1])

        mapper = sktda.MapperComplex(filters=flt, filter_bnds=np.array([[np.nan, np.nan]]), 
                                     resolutions=np.array([self.resolution]), gains=np.array([self.gain]), 
                                     clustering=self.clustering, colors=self.colors, mask=self.mask, inp=self.domain).fit(X)

        return mapper
