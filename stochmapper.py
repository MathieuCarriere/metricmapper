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
from scipy.sparse             import csgraph, lil_matrix, csr_matrix
from scipy.sparse.csgraph     import connected_components
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

    def __init__(self, num_bins=10, bnds=(np.nan, np.nan)):
        self.n_bins, self.bnds = num_bins, bnds

    def fit(self, X, y=None):
        if np.isnan(self.bnds[0]):
            min_dist = np.min([np.min(d) for d in X])
        if np.isnan(self.bnds[1]):
            max_dist = np.max([np.max(d) for d in X])
        self.bnds = tuple(np.where(np.isnan(np.array(self.bnds)), np.array([min_dist, max_dist]), self.bnds))
        return self

    def transform(self, X, y=None):
        num_dist = len(X)
        H, C = [], []
        for i in range(num_dist):
            h, e = np.histogram(X[i], bins=self.n_bins, range=self.bnds)
            c = (e[:-1] + e[1:])/2
            h = h/np.sum(h)
            H.append(h)
            C.append(c)
        return H, C

class Wasserstein1D(BaseEstimator, TransformerMixin):

    def __init__(self, C=[], p=1):
        self.C, self.p = C, p
        self.mode = "embedding_histogram"

    def compute_matrix(self, X):
        num_dist = len(X)
        M = np.zeros([num_dist, num_dist])
        for i in range(num_dist):
            for j in range(i+1, num_dist):
                M[i,j] = wasserstein_1d(x_a=self.C[i], x_b=self.C[j], a=X[i], b=X[j], p=self.p)
                M[j,i] = M[i,j]
        return M

class KullbackLeiblerDivergence(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.mode = "embedding"
        
    def compute_matrix(self, X):
        num_dist = len(X)
        M = np.zeros([num_dist, num_dist])
        for i in range(num_dist):
            for j in range(i+1, num_dist):
                M[i,j] = entropy(X[i], X[j])
                M[j,i] = M[i,j]
        return M
    
class EuclideanDistance(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        self.mode = "embedding"
    
    def compute_matrix(self, X):
        return euclidean_distances(X)

class VoronoiCover(BaseEstimator, TransformerMixin):

    def __init__(self, n_patches=10, threshold=1.):

        self.n_patches = n_patches
        self.threshold = threshold
        self.mode = "metric"

    def fit(self, D, y=None):

        # Partition data with Voronoi cover
        self.germs = np.random.choice(D.shape[0], self.n_patches, replace=False)
        labels = np.argmin(D[self.germs,:], axis=0)
        self.binned_data = {}
        for i, l in enumerate(labels):
            try:
                self.binned_data[l].append(i)
            except KeyError:
                self.binned_data[l] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            for i in self.binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(labels == i), [-1]), np.reshape(np.argwhere(labels != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(D[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    self.binned_data[i].append(p)

        return self

    def predict(self, D, y=None):
        labels = np.argmin(D[self.germs,:], axis=0)
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
                       threshold=1.):

        self.n_patches = n_patches
        self.threshold = threshold
        self.mode = "embedding"

    def fit(self, X, y=None):

        # Euclidean KMeans on histograms
        self.km = KMeans(n_clusters=self.n_patches).fit(X)
        self.binned_data = {}
        for i, l in enumerate(self.km.labels_):
            try:
                self.binned_data[l].append(i)
            except KeyError:
                self.binned_data[l] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            DX = euclidean_distances(X)
            for i in self.binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(self.km.labels_ == i), [-1]), np.reshape(np.argwhere(self.km.labels_ != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(DX[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    self.binned_data[i].append(p)

        return self
    
    def predict(self, X, y=None):

        L = self.km.predict(X)

        binned_data = {}
        for i, l in enumerate(L):
            try:
                binned_data[l].append(i)
            except KeyError:
                binned_data[l] = [i]

        if self.threshold > 0:
            DX = euclidean_distances(X)
            for i in binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(L == i), [-1]), np.reshape(np.argwhere(L != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(DX[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    binned_data[i].append(p)
        
        return binned_data

class WassersteinKMeansCover(BaseEstimator, TransformerMixin):

    def __init__(self, C=[], n_patches=10, epsilon=1e-4, p=1, tol=1e-8,
                       threshold=1.):

        self.C, self.n_patches, self.epsilon, self.p, self.tolerance = C, n_patches, epsilon, p, tol
        self.threshold = threshold
        self.mode = "embedding_histogram"

    def fit(self, X, y=None):

        cost = np.abs(np.reshape(self.C[0],[-1,1])-np.reshape(self.C[0],[1,-1]))
        self.curr_patches = X[np.random.choice(np.arange(len(X)), size=self.n_patches, replace=False), :]
        criterion = np.inf
        while criterion > self.tolerance:
            dists = np.zeros([len(self.curr_patches), len(X)])
            for i in range(len(self.curr_patches)):
                for j in range(len(X)):
                    dists[i,j] = wasserstein_1d(x_a=self.C[0], x_b=self.C[0], a=self.curr_patches[i,:], b=X[j,:], p=1)
            Q = np.argmin(dists.T, axis=1)
            new_curr_patches = []
            for t in range(self.n_patches):
                if len(np.argwhere(Q==t)) > 0:
                    new_curr_patches.append(np.reshape(barycenter(A=X[np.argwhere(Q==t)[:,0],:].T, M=cost, reg=self.epsilon), [1,-1]))
                else:
                    new_curr_patches.append(self.curr_patches[t:t+1,:])
            new_curr_patches = np.vstack(new_curr_patches)
            criterion = np.sqrt(np.square(new_curr_patches - self.curr_patches).sum(axis=1)).max(axis=0)
            self.curr_patches = new_curr_patches

        self.binned_data = {}
        for i in range(len(Q)):
            try:
                self.binned_data[Q[i]].append(i)
            except KeyError:
                self.binned_data[Q[i]] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            DX = np.zeros([len(X), len(X)])
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    DX[i,j] = wasserstein_1d(x_a=C[0], x_b=C[0], a=X[i,:], b=X[j,:], p=1)
                    DX[j,i] = DX[i,j]
            for i in self.binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(Q == i), [-1]), np.reshape(np.argwhere(Q != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(DX[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    self.binned_data[i].append(p)

        return self

    def predict(self, X, y=None):
        
        dists = np.zeros([len(self.curr_patches), len(X)])
        for i in range(len(self.curr_patches)):
            for j in range(len(X)):
                dists[i,j] = wasserstein_1d(x_a=self.C[0], x_b=self.C[0], a=self.curr_patches[i,:], b=X[j,:], p=1)

        L = np.argmin(dists.T, axis=1)

        binned_data = {}
        for i in range(len(L)):
            try:
                binned_data[L[i]].append(i)
            except KeyError:
                binned_data[L[i]] = [i]

        if self.threshold > 0:
            DX = np.zeros([len(X), len(X)])
            for i in range(len(X)):
                for j in range(i+1, len(X)):
                    DX[i,j] = wasserstein_1d(x_a=self.C[0], x_b=self.C[0], a=X[i,:], b=X[j,:], p=1)
                    DX[j,i] = DX[i,j]
            for i in binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(L == i), [-1]), np.reshape(np.argwhere(L != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(DX[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    binned_data[i].append(p)

        return binned_data 


class kPDTMCover(BaseEstimator, TransformerMixin):

    def __init__(self, n_patches=10, h=10, tol=1e-4, 
                       threshold=1.):

        self.n_patches, self.h, self.tolerance = n_patches, h, tol
        self.threshold = threshold
        self.mode = "embedding"

    def fit(self, X, y=None):

        self.curr_patches = X[np.random.choice(np.arange(len(X)), size=self.n_patches, replace=False), :]
        criterion = np.inf
        while criterion > self.tolerance:
            dists = euclidean_distances(self.curr_patches, X)
            self.means, self.variances = [], []
            for t in range(self.n_patches):
                ball_idxs = np.argpartition(dists[t,:], self.h)[:self.h]
                M = np.reshape(X[ball_idxs,:].mean(axis=0), [1,-1])
                V = np.square(euclidean_distances(M, X[ball_idxs,:])).sum()
                self.means.append(M)
                self.variances.append(V)
            Q = np.argmin(np.square(euclidean_distances(X, np.vstack(self.means))) + np.reshape(np.array(self.variances), [1,-1]), axis=1)
            new_curr_patches = []
            for t in range(self.n_patches):
                if len(np.argwhere(Q==t)) > 0:
                    new_curr_patches.append(np.reshape(X[np.argwhere(Q==t)[:,0],:].mean(axis=0), [1,-1]))
                else:
                    new_curr_patches.append(self.curr_patches[t:t+1,:])
            new_curr_patches = np.vstack(new_curr_patches)
            criterion = np.sqrt(np.square(new_curr_patches - self.curr_patches).sum(axis=1)).max(axis=0)
            self.curr_patches = new_curr_patches

        self.binned_data = {}
        for i in range(len(Q)):
            try:
                self.binned_data[Q[i]].append(i)
            except KeyError:
                self.binned_data[Q[i]] = [i]

        # Thicken clusters so that they overlap
        DX = euclidean_distances(X)
        for i in self.binned_data.keys():
            pts_cluster, pts_others = np.reshape(np.argwhere(Q == i), [-1]), np.reshape(np.argwhere(Q != i), [-1])
            pts_in_offset = pts_others[np.reshape(np.argwhere(DX[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
            for p in pts_in_offset:
                self.binned_data[i].append(p)

        return self

    def predict(self, X, y=None):

        L = np.argmin(np.square(euclidean_distances(X, np.vstack(self.means))) + np.reshape(np.array(self.variances), [1,-1]), axis=1)

        binned_data = {}
        for i in range(len(L)):
            try:
                binned_data[L[i]].append(i)
            except KeyError:
                binned_data[L[i]] = [i]

        if self.threshold > 0:
            DX = euclidean_distances(X)
            for i in binned_data.keys():
                pts_cluster, pts_others = np.reshape(np.argwhere(L == i), [-1]), np.reshape(np.argwhere(L != i), [-1])
                pts_in_offset = pts_others[np.reshape(np.argwhere(DX[pts_cluster,:][:,pts_others].min(axis=0) <= self.threshold), [-1])]
                for p in pts_in_offset:
                    binned_data[i].append(p)

        return binned_data


class GraphClustering(BaseEstimator, TransformerMixin):

    def __init__(self, A):
        self.adjacency = csr_matrix(A)

    def fit(self, X, y=None):
        return self

    def fit_predict(self, X, y=None):
        _, labels = connected_components(self.adjacency[X,:][:,X], directed=False)
        return labels
    

class StochasticMapperComplex(BaseEstimator, TransformerMixin):

    def __init__(self, filters, colors, codomain="distributions", infer_distributions=True, threshold=1., num_bins=10, bnds=(np.nan, np.nan),
                       correct_Rips=False, delta=1., n_subdivisions=1,
                       cover=VoronoiCover(), distance=EuclideanDistance(), 
                       domain="point cloud", clustering=DBSCAN(), 
                       mask=0):
        
        self.filters, self.codomain, self.infdist, self.threshold = filters, codomain, infer_distributions, threshold
        self.n_bins, self.bnds = num_bins, bnds
        self.correct_Rips, self.delta, self.n_subdivisions = correct_Rips, delta, n_subdivisions
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
            if (self.cover.mode == "embedding" or self.cover.mode == "embedding_histogram") or (self.cover.mode == "metric" and (self.distance.mode == "embedding" or self.distance.mode == "embedding_histogram")):
                # Compute histograms if necessary
                embeddings, c_embeddings = Histogram(num_bins=self.n_bins, bnds=self.bnds).fit_transform(distributions)
                embeddings = np.vstack(embeddings)

        if self.codomain == "distance matrix":
            # self.filters is supposed to be a square array of pairwise distances
            codomain_distances = self.filters

        if self.codomain == "vectors":
            # self.filters is supposed to be an array of vectors
            embeddings = self.filters


        if self.cover.mode == "metric":
            if self.codomain is not "distance matrix":
                if self.distance.mode == "embedding" or self.distance.mode == "embedding_histogram":
                    if self.distance.mode == "embedding_histogram":
                        self.distance.C = c_embeddings
                    codomain_distances = self.distance.compute_matrix(embeddings)
                if self.distance.mode == "distributions":
                    codomain_distances = self.distance.compute_matrix(distributions)

        # Compute cover
        if self.cover.mode == "metric":
            self.cover.fit(codomain_distances)
        if self.cover.mode == "embedding":
            self.cover.fit(embeddings)
        if self.cover.mode == "embedding_histogram":
            self.cover.C = c_embeddings
            self.cover.fit(embeddings)

        binned_data = self.cover.binned_data

        if self.correct_Rips and (self.cover.mode == "embedding" or self.cover.mode == "embedding_histogram"):

            DX = np.triu(euclidean_distances(X), k=1)
            edges, ndiv = np.argwhere((DX <= self.delta) & (DX > 0)), self.n_subdivisions
            new_points = []
            AA = lil_matrix((num_pts + edges.shape[0] * ndiv, num_pts + edges.shape[0] * ndiv))
            for i in range(len(edges)):
                new_points.append(np.linspace(embeddings[edges[i,0],:], embeddings[edges[i,1],:], ndiv+2)[1:-1,:])
                AA[num_pts + ndiv*i, edges[i,0]], AA[edges[i,0], num_pts + ndiv*i], AA[num_pts + ndiv*i + ndiv-1, edges[i,1]], AA[edges[i,1], num_pts + ndiv*i + ndiv-1] = 1, 1, 1, 1
                if ndiv > 1:
                    AA[num_pts + ndiv*i, num_pts + ndiv*i + 1], AA[num_pts + ndiv*i + 1, num_pts + ndiv*i] = 1, 1
                    AA[num_pts + ndiv*i + ndiv-1, num_pts + ndiv*i + ndiv-2], AA[num_pts + ndiv*i + ndiv-2, num_pts + ndiv*i + ndiv-1] = 1, 1
                    for k in range(1, ndiv-1):
                        AA[num_pts + ndiv*i + k, num_pts + ndiv*i + k-1] = 1
                        AA[num_pts + ndiv*i + k-1, num_pts + ndiv*i + k] = 1
                        AA[num_pts + ndiv*i + k, num_pts + ndiv*i + k+1] = 1
                        AA[num_pts + ndiv*i + k+1, num_pts + ndiv*i + k] = 1

            new_points = np.vstack(new_points)
            new_binned_data = self.cover.predict(new_points)
            for i in binned_data.keys():
                binned_data[i] = binned_data[i] + [num_pts + pt for pt in new_binned_data[i]]
            num_pts += edges.shape[0] * ndiv
            self.clustering = GraphClustering(AA)
            self.domain = "indices"
            self.colors = np.vstack([self.colors, np.zeros([num_pts + edges.shape[0] * ndiv, self.colors.shape[1]])])

        # Initialize the cover map, that takes a point and outputs the clusters to which it belongs
        cover, clus_base = [[] for _ in range(num_pts)], 0

        # Initialize attributes
        self.mapper_, self.node_info_ = gd.SimplexTree(), {}

        # For each patch
        for preimage in binned_data:

            # Apply clustering on the corresponding subpopulation
            idxs = np.array(binned_data[preimage])
            if len(idxs) > 1:
                if self.domain == "point cloud":
                    clusters = self.clustering.fit_predict(X[idxs,:])
                if self.domain == "distance matrix":
                    clusters = self.clustering.fit_predict(X[idxs,:][:,idxs])
                if self.domain == "indices":
                    clusters = self.clustering.fit_predict(idxs)
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
