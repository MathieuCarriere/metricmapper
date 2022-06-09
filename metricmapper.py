"""
@author: Mathieu Carriere
All rights reserved
"""
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
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
except ImportError:
    print("Gudhi not found: MetricMapperComplex will not work")


def estimate_scale(X, N=100, inp="point cloud", beta=0., C=10.):
    """
    Compute estimated scale of a point cloud or a distance matrix.

    Parameters:
        X (numpy array of shape (num_points) x (num_coordinates) if point cloud and (num_points) x (num_points) 
            if distance matrix): input point cloud or distance matrix.
        N (int): subsampling iterations (default 100). 
        inp (string): either "point cloud" or "distance matrix". Type of input data (default "point cloud").
        beta (double): exponent parameter (default 0.).
        C (double): constant parameter (default 10.).

    Returns:
        delta (double): estimated scale that can be used with eg agglomerative clustering.
    """
    num_pts = X.shape[0]
    delta, m = 0., int(  num_pts / np.exp((1+beta) * np.log(np.log(num_pts)/np.log(C)))  )
    for _ in range(N):
        subpop = np.random.choice(num_pts, size=m, replace=False)
        if inp == "point cloud":
            d, _, _ = directed_hausdorff(X, X[subpop,:])
        if inp == "distance matrix":
            d = np.max(np.min(X[:,subpop], axis=1), axis=0)
        delta += d/N
    return delta

def mapper2networkx(M, get_attrs=False):
    """
    Turn the 1-skeleton of M (computed after calling fit() method) into a networkx graph.
    This function requires networkx (https://networkx.org/documentation/stable/install.html).

    Parameters:
        M (MetricMapperComplex): simplicial complex
        get_attrs (bool): if True, the color functions will be used as attributes for the networkx graph.

    Returns:
        G (networkx graph): graph representing the 1-skeleton of the cover complex.
    """
    st = M.mapper_
    G = nx.Graph()
    for (splx,_) in st.get_skeleton(1):
        if len(splx) == 1:
            G.add_node(splx[0])
        if len(splx) == 2:
            G.add_edge(splx[0], splx[1])
    if get_attrs:
        attrs = {k: {"attr_name": M.node_info[k]["colors"]} for k in G.nodes()}
        nx.set_node_attributes(G, attrs)
    return G


########################################
# Kernel for Nadaraya-Watson estimator #
########################################

class GaussianKernel(BaseEstimator, TransformerMixin):
    """
    This class implements the usual Gaussian kernel.
    """
    def __init__(self, h=1.):
        """
        Constructor for the GaussianKernel class.

        Attributes:
            h (float): bandwidth of the Gaussian kernel.
        """
        self.h = h

    def compute_kernel_matrix(self, X):
        """
        Method for computing the kernel matrix associated to a data set.

        Parameters:
            X (n x d numpy array): numpy array containing the data points coordinates.

        Returns:
            K (n x n numpy array): kernel matrix.
        """
        DX = euclidean_distances(X)
        K = np.exp(-np.square(DX)/(self.h*self.h))
        return K





###########################################################################
# Conditional probability distribution inference from single observations #
###########################################################################

def infer_distributions_from_neighborhood(real, X, threshold=1., domain="point cloud"):
    """
    This function infers, for each data point, the associated conditional probability distribution by using the values of the other points that are included in a ball of a given radius centered on the data point. Returns lists of values sampled from the probability distributions on which means or histograms can be computed later.

    Parameters:
        real (list of float): single observations for each data point. The size of the list must be equal to the number n of data points.
        X (n x d numpy array or n x n numpy array): numpy array containing the data points coordinates (if domain = "point cloud") or the pairwise distances between points (if domain = "distance matrix").
        threshold (float): radius of the balls centered on the points.
        domain (string): type of input data. Either "point cloud" or "distance matrix".

    Returns:
        distributions (list of lists of float): list (of size equal to the number of points n) of list of values sampled from the conditional probability distributions. 
    """
    num_pts, distributions = len(X), []
    pdist = X if domain == "distance matrix" else pairwise_distances(X)
    for i in range(num_pts):
        distrib = np.squeeze(np.argwhere(pdist[i,:] <= threshold))
        np.random.shuffle(distrib)
        distributions.append([real[n] for n in distrib])
    return distributions

def infer_distributions_from_Nadaraya_Watson(real, X, kernel=GaussianKernel(h=1), means=False, num_bins=100, bnds=(0,1)):
    """
    This function infers, for each data point, the associated conditional probability distribution by using the values given by a Nadaraya-Watson kernel estimator computed on the other points. Directly returns either the means or the histograms of the probability distributions.

    Parameters:
        real (list of float): single observations for each data point. The size of the list must be equal to the number n of data points.
        X (n x d numpy array): numpy array containing the data points coordinates.
        kernel (class): kernel to use for the Nadaraya-Watson estimator. Must have a "compute_kernel_matrix" method. 
        means (bool): boolean specifying whether only the means of the conditional probability distributions are returned (True) or their histograms instead (False).
        num_bins (int): number of bins of the histograms. Used only if means = False.
        bnds (tuple of int): inf and sup limits of the histograms. Used only if means = False. If one of the two values is numpy.nan, it will be estimated from data.

    Returns:
        output (list): list of mean values (if means = True) or list containing a numpy array containing the histograms (of shape n x num_bins) and the bin edges.
    """
    num_pts, dims = len(X), []
    kmat  = kernel.compute_kernel_matrix(X)
    if not means:
        bnds = tuple(np.where(np.isnan(np.array(bnds)), np.array([min(real), max(real)]), np.array(bnds)))
        bins = np.linspace(bnds[0], bnds[1], num=num_bins+1)
        digits = np.digitize(np.array(real), bins)
        idxs = [np.argwhere(digits==i+1)[:,0] for i in range(num_bins)]
    output = []
    for i in range(num_pts):
        if means:
            output.append( np.multiply(np.array(real), kmat[i,:]).sum() / kmat[i,:].sum()  )
        else:
            proba = kmat[i,:] / kmat[i,:].sum()
            hist = np.zeros([num_bins])
            for j in range(num_bins):
                hist[j] = proba[idxs[j]].sum()
            output.append(hist)
    if not means:
        output = [np.vstack(output), bins]
    return output





##################################################
# Euclidean embeddings for sampled distributions #
##################################################

class Histogram(BaseEstimator, TransformerMixin):
    """
    This class computes histograms associated to samplings of probability distributions.
    """
    def __init__(self, num_bins=10, bnds=(np.nan, np.nan)):
        """
        Constructor for the Histogram class.
  
        Parameters:
            num_bins (int): number of bins of the histograms.
            bnds (tuple of int): inf and sup limits of the histograms. If one of the two values is numpy.nan, it will be estimated from data.
        """
        self.num_bins, self.bnds = num_bins, bnds

    def fit(self, X, y=None):
        """
        Fit the Histogram class on a list of probability distribution samplings. The minimum and maximum values of the distributions are computed and used as histogram limits if those are not specified.

        Parameters:
            X (list of lists of float): samplings of the probability distributions.
            y (n x 1 array): probability distribution labels (unused).
        """
        if np.isnan(self.bnds[0]):
            min_dist = np.min([np.min(d) for d in X])
        if np.isnan(self.bnds[1]):
            max_dist = np.max([np.max(d) for d in X])
        self.bnds = tuple(np.where(np.isnan(np.array(self.bnds)), np.array([min_dist, max_dist]), self.bnds))
        return self

    def transform(self, X, y=None):
        """
        Compute the histograms associated to the probability distribution samplings.

        Parameters:
            X (list of lists of float): samplings of the probability distributions.
            y (n x 1 array): probability distribution labels (unused).

        Returns:
            H (list of numpy array of shape num_bins): histograms associated to the probability distributions.
            C (list of float): centers of each bin in the histograms. This is useful for computing, e.g., Wasserstein distances between the histograms.
        """
        num_dist = len(X)
        H, C = [], []
        for i in range(num_dist):
            h, e = np.histogram(X[i], bins=self.num_bins, range=self.bnds)
            c = (e[:-1] + e[1:])/2
            h = h/np.sum(h)
            H.append(h)
            C.append(c)
        return H, C





##########################
# Metrics for input data #
##########################

class Wasserstein1D(BaseEstimator, TransformerMixin):
    """
    This class is a wrapper for the Wasserstein distance between 1D probability distributions, implemented in POT (https://pot.readthedocs.io/en/stable/all.html#module-ot).
    """
    def __init__(self, C=[], p=1):
        """
        Constructor for the Wasserstein1D class.

        Parameters:
            C (list of float): source Dirac locations.
            p (int): power for the Wasserstein distance.
        """
        self.C, self.p = C, p
        self.mode = "embedding_histogram"

    def compute_matrix(self, X):
        """
        Compute the Wasserstein distance matrix.

        Parameters:
            X (n x m numpy array): numpy array containing the histograms. The second dimension must be equal to the size of C.

        Returns:
            M (n x n numpy array): distance matrix.
        """
        num_dist = len(X)
        M = np.zeros([num_dist, num_dist])
        for i in range(num_dist):
            for j in range(i+1, num_dist):
                M[i,j] = wasserstein_1d(x_a=self.C[i], x_b=self.C[j], a=X[i], b=X[j], p=self.p)
                M[j,i] = M[i,j]
        return M

class KullbackLeiblerDivergence(BaseEstimator, TransformerMixin):
    """
    This class computes the Kullback-Leibler divergences between 1D probability distributions.
    """
    def __init__(self):
        """
        Constructor for the Wasserstein1D class.
        """
        self.mode = "embedding"
        
    def compute_matrix(self, X):
        """
        Compute the Kullback-Leibler divergence matrix.

        Parameters:
            X (n x m numpy array): numpy array containing the histograms.

        Returns:
            M (n x n numpy array): divergence matrix.
        """
        num_dist = len(X)
        M = np.zeros([num_dist, num_dist])
        for i in range(num_dist):
            for j in range(i+1, num_dist):
                M[i,j] = entropy(X[i], X[j])
                M[j,i] = M[i,j]
        return M
    
class EuclideanDistance(BaseEstimator, TransformerMixin):
    """
    This class computes the Euclidean distances between points.
    """  
    def __init__(self):
        """
        Constructor for the EuclideanDistance class.
        """
        self.mode = "embedding"
    
    def compute_matrix(self, X):
        """
        Compute the Euclidean distance matrix.

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates.

        Returns:
            M (n x n numpy array): distance matrix.
        """
        return euclidean_distances(X)





################################################
# Covers for domains included in metric spaces #
################################################

class HypercubeCover(BaseEstimator, TransformerMixin):
    """
    This class is for computing standard hypercube covers of Euclidean spaces. 
    """
    def __init__(self, bnds=[], resolutions=[], gains=[], cover_mode="", cover=[]):
        """
        Constructor for the VoronoiCover class.

        Parameters:
            bnds (num_filt x 2 numpy array): array of boundaries (one for each filter). Only used if cover_mode == "implicit".
            resolutions (num_filt numpy array): array of resolutions (one for each filter). Only used if cover_mode == "implicit".
            gains (num_filt numpy array): array of gains (one for each filter). Only used if cover_mode == "implicit".
            cover_mode (string): Either "implicit" or "explicit". Whether the cover is defined with boundaries, resolutions and gains, or explicitly. 
            covers ([list (size num_filt) of left endpoints, list (size num_filt) of right endpoints]): Cover. Only used if cover_mode == "explicit".

        """
        self.bnds = bnds
        self.resolutions = resolutions
        self.gains = gains 
        self.mode = "embedding"
        self.can_refine = True
        self.cover_mode = cover_mode
        self.cover = cover

    def _compute_cover_1(self, interval_inds, intersec_inds, X):
        num_pts, num_filters = X.shape[0], X.shape[1]
        binned_data = {}
        for i in range(num_pts):
            list_preimage = []
            for j in range(num_filters):
                a, b = interval_inds[i,j]-1, intersec_inds[i,j]-1
                list_preimage.append([a])
                if b == a:
                    list_preimage[j].append(a+1)
                if b == a-1:
                    list_preimage[j].append(a-1)
            list_preimage = list(itertools.product(*list_preimage))
            for pre_idx in list_preimage:
                try:
                    binned_data[pre_idx].append(i)
                except KeyError:
                    binned_data[pre_idx] = [i]
        return binned_data

    def _compute_cover_2(self, X):
        num_pts, num_filters = X.shape[0], X.shape[1]
        binned_data = {}
        for i in range(num_pts):
            list_preimage = []
            for j in range(num_filters):
                fval = X[i,j]
                start, end = int(min(np.argwhere(np.array(self.r_int[j]) >= fval))), int(max(np.argwhere(np.array(self.l_int[j]) <= fval)))
                list_preimage.append(list(range(start, end+1)))
            list_preimage = list(itertools.product(*list_preimage))
            for pre_idx in list_preimage:
                try:
                    binned_data[pre_idx].append(i)
                except KeyError:
                    binned_data[pre_idx] = [i]
        return binned_data

    def fit(self, X, y=None):
        """
        Fit the HypercubeCover class on a dataset, which is encoded in the dictionary self.binned_data.

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates (in the codomain).
            y (n x 1 array): point labels (unused).
        """
        num_pts, num_filters = X.shape[0], X.shape[1]

        if self.cover_mode == "implicit":
            if np.all(self.gains < .5):
                self.I_int, self.i_int = [], [] 
                interval_inds, intersec_inds = np.empty(X.shape), np.empty(X.shape)
                for i in range(num_filters):
                    f, r, g = X[:,i], self.resolutions[i], self.gains[i]
                    min_f, max_f = self.bnds[i,0], np.nextafter(self.bnds[i,1], np.inf)
                    interval_endpoints, l = np.linspace(min_f, max_f, num=r+1, retstep=True)
                    intersec_endpoints = []
                    for j in range(1, len(interval_endpoints)-1):
                        intersec_endpoints.append(interval_endpoints[j] - g*l / (2 - 2*g))
                        intersec_endpoints.append(interval_endpoints[j] + g*l / (2 - 2*g))
                    self.I_int.append(interval_endpoints)
                    self.i_int.append(intersec_endpoints)
                    interval_inds[:,i] = np.digitize(f, interval_endpoints)
                    intersec_inds[:,i] = 0.5 * (np.digitize(f, intersec_endpoints) + 1)
                self.binned_data = self._compute_cover_1(interval_inds, intersec_inds, X)       

            else:
                self.l_int, self.r_int = [], []
                for i in range(num_filters):
                    L, R = [], []
                    f, r, g = X[:,i], self.resolutions[i], self.gains[i]
                    min_f, max_f = self.bnds[i,0], np.nextafter(self.bnds[i,1], np.inf)
                    interval_endpoints, l = np.linspace(min_f, max_f, num=r+1, retstep=True)
                    for j in range(len(interval_endpoints)-1):
                        L.append(interval_endpoints[j]   - g*l / (2 - 2*g))
                        R.append(interval_endpoints[j+1] + g*l / (2 - 2*g))
                    self.l_int.append(L)
                    self.r_int.append(R)
                self.binned_data = self._compute_cover_2(X)

        elif self.cover_mode == "explicit":
            self.l_int, self.r_int = self.cover[0], self.cover[1]
            self.binned_data = self._compute_cover_2(X)

        return self

    def predict(self, X, y=None):
        """
        Predict the cover elements of a given dataset.

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates (in the codomain).
            y (n x 1 array): point labels (unused).

        Returns:
            binned_data (dictionary): predictions. The keys of this dictionary are the cover elements, and their associated values are the indices of their corresponding data points.
        """
        num_pts, num_filters = X.shape[0], X.shape[1]
        if self.cover == "implicit" and np.all(self.gains < .5):
            # Compute which points fall in which patch or patch intersections 
            interval_inds, intersec_inds = np.empty(X.shape), np.empty(X.shape)
            for i in range(num_filters):
                f, interval_endpoints, intersec_endpoints = X[:,i], self.I_int[i], self.i_int[i]
                interval_inds[:,i] = np.digitize(f, interval_endpoints)
                intersec_inds[:,i] = .5 * (np.digitize(f, intersec_endpoints) + 1)
            binned_data = self._compute_cover_1(interval_inds, intersec_inds, X)       
        else:
            binned_data = self._compute_cover_2(X)
        return binned_data

    def _compute_intersections(self, hypercubes, max_number):
        intersecs = [[], [], []]
        for d in range(2, max_number+1):
            for sigma in itertools.combinations(hypercubes, d):
                S = np.hstack([np.vstack( [np.array(interval)[1:3][np.newaxis,:] for interval in cube] ) for cube in sigma])
                I = np.hstack([np.vstack( [np.array(interval)[0:1][np.newaxis,:] for interval in cube] ) for cube in sigma])
                SX, SY = S[:,0::2].max(axis=1), S[:,1::2].min(axis=1)
                L = SY-SX
                if (L>=0).sum() == SX.shape[0]:
                    intersecs[0].append(SX[:,np.newaxis])
                    intersecs[1].append(SY[:,np.newaxis])
                    intersecs[2].append([tuple(I[:,s]) for s in range(I.shape[1])])
        return [np.hstack(intersecs[0]), np.hstack(intersecs[1]), intersecs[2]]

    def _check_intersections(self, XXs, YYs, IXs, IYs):
        Xs, Ys = np.minimum(XXs, YYs), np.maximum(XXs, YYs)
        d = Xs.shape[0]
        Ls = np.maximum(Xs[:,np.newaxis,:], IXs[:,:,np.newaxis])
        LI = np.equal(Ls, IXs[:,:,np.newaxis])
        Rs = np.minimum(Ys[:,np.newaxis,:], IYs[:,:,np.newaxis])
        RI = np.equal(Rs, IYs[:,:,np.newaxis])
        LI, RI = ((XXs[:,np.newaxis,:] < Ls) | (XXs[:,np.newaxis,:] > Rs)), ((YYs[:,np.newaxis,:] < Ls) | (YYs[:,np.newaxis,:] > Rs))
        LLI, RRI = (LI[0,:,:] | LI[1,:,:]), (RI[0,:,:] | RI[1,:,:])
        Es = (LLI & RRI)
        C1s = np.sum(Rs-Ls>=0, axis=0)
        return (  (C1s == d) & Es  )

    def _refine(self, embedding_edges, num_pts):
        num_filters = embedding_edges[0].shape[1]
        if self.cover_mode == "implicit":
            if np.all(self.gains < .5):
                cubes = []
                for i in range(num_filters):
                    #print(self.i_int[i], self.I_int[i])
                    left   = [[0, self.I_int[i][0], self.i_int[i][1]]]
                    middle = [[j/2+1, self.i_int[i][j], self.i_int[i][j+3]] for j in range(0, len(self.i_int[i])-3, 2)]
                    right  = [[self.resolutions[i]-1, self.i_int[i][len(self.i_int[i])-2], self.I_int[i][-1]]]
                    #print(left, middle, right)
                    cubes.append(left + middle + right)
            else:
                cubes = [[ [j, self.l_int[i][j], self.r_int[i][j]] for j in range(self.resolutions[i])] for i in range(num_filters)]
        elif self.cover_mode == "explicit":
            cubes = [[ [j, self.l_int[i][j], self.r_int[i][j]] for j in range(len(self.l_int[i]))] for i in range(num_filters)]
        hypercubes = itertools.product(*cubes)
        intersecs = self._compute_intersections([hc for hc in hypercubes], 2 ** num_filters)
        E = self._check_intersections( embedding_edges[0].T, embedding_edges[1].T, intersecs[0], intersecs[1] ).T
        count, newpts = num_pts, [[] for e in range(len(embedding_edges[0]))]
        for i in range(E.shape[0]):
            sigma_indx, bindict = np.argwhere(E[i,:])[:,0], {}
            for j in sigma_indx[::-1]:
                sigma = intersecs[2][j]
                known_sigma = bindict.keys()
                new = True
                for prev_sigma in known_sigma:
                    if all([cov in list(prev_sigma) for cov in sigma]):
                        new, p = False, bindict[prev_sigma]
                        for cov in sigma:
                            try:
                                self.binned_data[cov].append(p)
                            except KeyError:
                                self.binned_data[cov] = [p]
                if new:
                    bindict[tuple(sigma)] = count
                    newpts[i].append(count)
                    for cov in sigma:
                        try:
                            self.binned_data[cov].append(count)
                        except KeyError:
                            self.binned_data[cov] = [count]
                    count += 1
        for k in self.binned_data.keys():
            self.binned_data[k] = list(np.unique(self.binned_data[k]))
        return newpts

class VoronoiCover(BaseEstimator, TransformerMixin):
    """
    This class is for computing the Voronoi cover of a metric space. It randomly picks a specified number of germs, computes the associated Voronoi partition, and thicken the cells by a specified amount to create a cover. 
    """
    def __init__(self, num_patches=10, threshold=1.):
        """
        Constructor for the VoronoiCover class.

        Parameters:
            num_patches (int): number of Voronoi cells.
            threshold (float): thickening amount for each cell. 
        """
        self.num_patches = num_patches
        self.threshold = threshold
        self.mode = "metric"
        self.can_refine = False

    def fit(self, D, y=None):
        """
        Fit the VoronoiCover class on a distance matrix. The Voronoi partition is computed and thickened, and then encoded in the dictionary self.binned_data.

        Parameters:
            D (n x n numpy array): distance matrix.
            y (n x 1 array): point labels (unused).
        """
        # Partition data with Voronoi cover
        self.germs = np.random.choice(D.shape[0], self.num_patches, replace=False)
        labels = np.argmin(D[self.germs,:], axis=0)
        self.binned_data = {}
        for i, l in enumerate(labels):
            try:
                self.binned_data[l].append(i)
            except KeyError:
                self.binned_data[l] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            DX = D[:, self.germs]
            Dm = np.reshape(DX.min(axis=1), [-1,1])
            Di = np.argwhere( (DX <= Dm + 2*self.threshold) & (DX > Dm) )
            for i in range(len(Di)):
                try:
                    self.binned_data[Di[i,1]].append(Di[i,0])
                except KeyError:
                    self.binned_data[Di[i,1]] = [Di[i,0]]

        return self

    def predict(self, D, y=None):
        """
        Predict the cover elements of a given distance matrix.

        Parameters:
            D (n x n numpy array): distance matrix.
            y (n x 1 array): point labels (unused).

        Returns:
            binned_data (dictionary): predictions. The keys of this dictionary are the cover elements, and their associated values are the indices of their corresponding data points.
        """
        labels = np.argmin(D[self.germs,:], axis=0)
        binned_data = {}
        for i, l in enumerate(labels):
            try:
                binned_data[l].append(i)
            except KeyError:
                binned_data[l] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            DX = D[:, self.germs]
            Dm = np.reshape(DX.min(axis=1), [-1,1])
            Di = np.argwhere( (DX <= Dm + 2*self.threshold) & (DX > Dm) )
            for i in range(len(Di)):
                try:
                    binned_data[Di[i,1]].append(Di[i,0])
                except KeyError:
                    binned_data[Di[i,1]] = [Di[i,0]]

        return binned_data

class EuclideanKMeansCover(BaseEstimator, TransformerMixin):
    """
    This class is for computing the K-means cover of a metric space. It computes the standard K-means partition, and thicken the cells by a specified amount to create a cover. 
    """
    def __init__(self, num_patches=10, threshold=1.):
        """
        Constructor for the EuclideanKMeansCover class.

        Parameters:
            num_patches (int): number of cells.
            threshold (float): thickening amount for each cell. 
        """
        self.num_patches = num_patches
        self.threshold = threshold
        self.mode = "embedding"
        self.can_refine = False

    def fit(self, X, y=None):
        """
        Fit the EuclideanKMeansCover class on a dataset. The K-means partition is computed and thickened, and then encoded in the dictionary self.binned_data.

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates.
            y (n x 1 array): point labels (unused).
        """
        # Euclidean KMeans on histograms
        self.km = KMeans(n_clusters=self.num_patches).fit(X)
        self.binned_data = {}
        for i, l in enumerate(self.km.labels_):
            try:
                self.binned_data[l].append(i)
            except KeyError:
                self.binned_data[l] = [i]

        # Thicken clusters so that they overlap
        if self.threshold > 0:
            DX = euclidean_distances(X, self.km.cluster_centers_)
            Dm = np.reshape(DX.min(axis=1), [-1,1])
            Di = np.argwhere( (DX <= Dm + 2*self.threshold) & (DX > Dm) )
            for i in range(len(Di)):
                try:
                    self.binned_data[Di[i,1]].append(Di[i,0])
                except KeyError:
                    self.binned_data[Di[i,1]] = [Di[i,0]]

        return self
    
    def predict(self, X, y=None):
        """
        Predict the cover elements of a given dataset.

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates.
            y (n x 1 array): point labels (unused).

        Returns:
            binned_data (dictionary): predictions. The keys of this dictionary are the cover elements, and their associated values are the indices of their corresponding data points.
        """
        L = self.km.predict(X)

        binned_data = {}
        for i, l in enumerate(L):
            try:
                binned_data[l].append(i)
            except KeyError:
                binned_data[l] = [i]

        if self.threshold > 0:
            DX = euclidean_distances(X, self.km.cluster_centers_)
            Dm = np.reshape(DX.min(axis=1), [-1,1])
            Di = np.argwhere( (DX <= Dm + 2*self.threshold) & (DX > Dm) )
            for i in range(len(Di)):
                try:
                    binned_data[Di[i,1]].append(Di[i,0])
                except KeyError:
                    binned_data[Di[i,1]] = [Di[i,0]]
        
        return binned_data

class WassersteinKMeansCover(BaseEstimator, TransformerMixin):
    """
    This class is for computing the K-means cover of a metric space with Wasserstein distances. It computes the standard K-means partition with Wasserstein distances, and thicken the cells by a specified amount to create a cover. 
    """
    def __init__(self, num_patches=10, threshold=1., p=1, C=[], epsilon=1e-4, tol=1e-8):
        """
        Constructor for the WassersteinKMeansCover class.

        Parameters:
            num_patches (int): number of cells.
            threshold (float): thickening amount for each cell. 
            C (list of floats): source Dirac locations.
            p (int): power for the Wasserstein distance.
            epsilon (float): entropy regularization (see https://pot.readthedocs.io/en/stable/all.html#module-ot).
            tol (float): tolerance for K-means: the algorithm stops when the distance between centroids becomes less than this value.
        """
        self.C, self.num_patches, self.epsilon, self.p, self.tolerance = C, num_patches, epsilon, p, tol
        self.threshold = threshold
        self.mode = "embedding_histogram"
        self.can_refine = False

    def fit(self, X, y=None):
        """
        Fit the WassersteinKMeansCover class on a dataset of histograms. The Wasserstein K-means partition is computed and thickened, and then encoded in the dictionary self.binned_data.

        Parameters:
            X (n x m numpy array): numpy array containing the histograms. The second dimension must be equal to the size of C.
            y (n x 1 array): point labels (unused).
        """
        cost = np.abs(np.reshape(self.C[0],[-1,1])-np.reshape(self.C[0],[1,-1]))
        self.curr_patches = X[np.random.choice(np.arange(len(X)), size=self.num_patches, replace=False), :]
        criterion = np.inf
        while criterion > self.tolerance:
            dists = np.zeros([len(self.curr_patches), len(X)])
            for i in range(len(self.curr_patches)):
                for j in range(len(X)):
                    dists[i,j] = wasserstein_1d(x_a=self.C[0], x_b=self.C[0], a=self.curr_patches[i,:], b=X[j,:], p=1)
            Q = np.argmin(dists, axis=0)
            new_curr_patches = []
            for t in range(self.num_patches):
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
            DX = dists.T
            Dm = np.reshape(DX.min(axis=1), [-1,1])
            Di = np.argwhere( (DX <= Dm + 2*self.threshold) & (DX > Dm) )
            for i in range(len(Di)):
                try:
                    self.binned_data[Di[i,1]].append(Di[i,0])
                except KeyError:
                    self.binned_data[Di[i,1]] = [Di[i,0]]

        return self

    def predict(self, X, y=None):
        """
        Predict the cover elements of a given dataset of histograms.

        Parameters:
            X (n x m numpy array): numpy array containing the histograms. The second dimension must be equal to the size of C.
            y (n x 1 array): point labels (unused).

        Returns:
            binned_data (dictionary): predictions. The keys of this dictionary are the cover elements, and their associated values are the indices of their corresponding data points.
        """
        dists = np.zeros([len(self.curr_patches), len(X)])
        for i in range(len(self.curr_patches)):
            for j in range(len(X)):
                dists[i,j] = wasserstein_1d(x_a=self.C[0], x_b=self.C[0], a=self.curr_patches[i,:], b=X[j,:], p=1)

        L = np.argmin(dists, axis=0)

        binned_data = {}
        for i in range(len(L)):
            try:
                binned_data[L[i]].append(i)
            except KeyError:
                binned_data[L[i]] = [i]

        if self.threshold > 0:
            DX = dists.T
            Dm = np.reshape(DX.min(axis=1), [-1,1])
            Di = np.argwhere( (DX <= Dm + 2*self.threshold) & (DX > Dm) )
            for i in range(len(Di)):
                try:
                    binned_data[Di[i,1]].append(Di[i,0])
                except KeyError:
                    binned_data[Di[i,1]] = [Di[i,0]]

        return binned_data 


class kPDTMCover(BaseEstimator, TransformerMixin):
    """
    This class is for computing the distance-to-measure variation of the K-means cover described in https://hal.archives-ouvertes.fr/hal-02266408, called the K-PDTM cover. It computes the K-PDTM partition, and thicken the cells by a specified amount to create a cover. 
    """
    def __init__(self, num_patches=10, threshold=1., h=10, tol=1e-4):
        """
        Constructor for the kPDTMCover class.

        Parameters:
            num_patches (int): number of cells.
            threshold (float): thickening amount for each cell. 
            h (int): number of nearest neighbors for the distance-to-measure computation (see https://hal.archives-ouvertes.fr/hal-02266408).
            tol (float): tolerance for K-PDTM: the algorithm stops when the distance between centroids becomes less than this value.
        """
        self.num_patches, self.h, self.tolerance = num_patches, h, tol
        self.threshold = threshold
        self.mode = "embedding"
        self.can_refine = False

    def fit(self, X, y=None):
        """
        Fit the kPDTMCover class on a dataset. The K-PDTM partition is computed and thickened, and then encoded in the dictionary self.binned_data.

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates.
            y (n x 1 array): point labels (unused).
        """
        self.curr_patches = X[np.random.choice(np.arange(len(X)), size=self.num_patches, replace=False), :]
        criterion = np.inf
        while criterion > self.tolerance:
            dists = euclidean_distances(self.curr_patches, X)
            self.means, self.variances = [], []
            for t in range(self.num_patches):
                ball_idxs = np.argpartition(dists[t,:], self.h)[:self.h]
                M = np.reshape(X[ball_idxs,:].mean(axis=0), [1,-1])
                V = np.square(euclidean_distances(M, X[ball_idxs,:])).sum()
                self.means.append(M)
                self.variances.append(V)
            dists = np.square(euclidean_distances(X, np.vstack(self.means))) + np.reshape(np.array(self.variances), [1,-1])
            Q = np.argmin(dists, axis=1)
            new_curr_patches = []
            for t in range(self.num_patches):
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
        if self.threshold > 0:
            DX = dists
            Dm = np.reshape(DX.min(axis=1), [-1,1])
            Di = np.argwhere( (DX <= Dm + 2*self.threshold) & (DX > Dm) )
            for i in range(len(Di)):
                try:
                    self.binned_data[Di[i,1]].append(Di[i,0])
                except KeyError:
                    self.binned_data[Di[i,1]] = [Di[i,0]]

        return self

    def predict(self, X, y=None):
        """
        Predict the cover elements of a given dataset.

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates.
            y (n x 1 array): point labels (unused).

        Returns:
            binned_data (dictionary): predictions. The keys of this dictionary are the cover elements, and their associated values are the indices of their corresponding data points.
        """
        dists = np.square(euclidean_distances(X, np.vstack(self.means))) + np.reshape(np.array(self.variances), [1,-1])
        L = np.argmin(dists, axis=1)

        binned_data = {}
        for i in range(len(L)):
            try:
                binned_data[L[i]].append(i)
            except KeyError:
                binned_data[L[i]] = [i]

        if self.threshold > 0:
            DX = dists
            Dm = np.reshape(DX.min(axis=1), [-1,1])
            Di = np.argwhere( (DX <= Dm + 2*self.threshold) & (DX > Dm) )
            for i in range(len(Di)):
                try:
                    binned_data[Di[i,1]].append(Di[i,0])
                except KeyError:
                    binned_data[Di[i,1]] = [Di[i,0]]

        return binned_data





#########################################
# Clustering methods in the input space #
#########################################

class GraphClustering(BaseEstimator, TransformerMixin):
    """
    This class is for computing graph clustering with connected components.
    """
    def __init__(self, A):
        """
        Constructor for the GraphClustering class.

        Parameters:
            A (n x n numpy array): adjacency matrix of the graph built on top of the points.
        """
        self.adjacency = csr_matrix(A)

    def fit(self, X, y=None):
        """
        Fit the GraphClustering class on a dataset (this function actually does nothing but is useful when GraphClustering is included in a scikit-learn Pipeline).

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates.
            y (n x 1 array): point labels (unused).
        """
        return self

    def fit_predict(self, X, y=None):
        """
        Predict the clusters of a given dataset.

        Parameters:
            X (n x d numpy array): numpy array containing the point coordinates.
            y (n x 1 array): point labels (unused).

        Returns:
            labels (list): list (of size equal to the number of points) of cluster elements.
        """
        _, labels = connected_components(self.adjacency[X,:][:,X], directed=False)
        return labels





#################
# Metric Mapper #
#################

class MetricMapperComplex(BaseEstimator, TransformerMixin):
    """
    This class if for computing Mappers with codomain (i.e., filter domain) included in a metric space, using cover strategies for metric spaces instead of the standard hypercube covers.
    """
    def __init__(self, filters, colors, codomain="distributions", infer_distributions=True, mode="NN", threshold=1., kernel=GaussianKernel(h=1.), num_bins=10, bnds=(np.nan, np.nan),
                       correct_Rips=False, delta=1., correct_mode="uniform_refinement", num_subdivisions=1,
                       cover=VoronoiCover(), distance=EuclideanDistance(), 
                       domain="point cloud", clustering=DBSCAN(), 
                       mask=0):
        """
        Constructor for the MetricMapperComplex class.

        Parameters:
            filters (n x n numpy array or n x d numpy array or list of (lists of) floats): filter information. It is either a pairwise distance matrix (if codomain = "distance matrix"), a matrix containing the filter value coordinates (if codomain = "vectors"), or a list of probability distribution samplings or single observations (if codomain = "distributions").
            colors (n x num_colors numpy array): color functions used to visualize the Mapper nodes. 
            codomain (string): specifies filter domain. Either "distance matrix" when only distance matrices in the codomain are known, "vectors" when the codomain is Euclidean space, or "distributions" when the codomain is the space of conditional probability distributions.
            infer_distributions (bool): whether to infer conditional probability distributions from single observations. Used only if codomain = "distributions".
            mode (string): specifies how to infer distributions. Either using balls ("NN") or Nadaraya-Watson kernel estimator ("NW"). Used only if codomain = "distributions" and "infdist" = True.
            threshold (float): radius of the balls centered on the points for inferring distributions. Used only if codomain = "distributions", "infdist" = True and mode = "NN".
            kernel (class): kernel to use for the Nadaraya-Watson estimator. Must have a "compute_kernel_matrix" method. Used only if codomain = "distributions", "infdist" = True and mode = "NW".
            num_bins (int): number of bins of the histograms. Used only if codomain = "distributions", "infdist" = True and mode = "NW". 
            bnds (tuple of int): inf and sup limits of the histograms. If one of the two values is numpy.nan, it will be estimated from data. Used only if codomain = "distributions", "infdist" = True and mode = "NW".
            correct_Rips (bool): whether to subdivide Rips complex.
            delta (float): neighborhood parameter used for computing Rips complex. Used only if correct_Rips = True.
            correct_mode (string): either "uniform_refinement" or "cover_refinement". Whether to refine each edge of the Rips complex or only those that are intersection-crossing.
            num_subdivisions (int): number of subdivisions on each edge of the Rips complex. Used only if correct_Rips = True and correct_mode == "uniform_refinement".
            cover (class): cover method to use.
            distance (class): distances to use. Used if cover.mode="metric" and codomain is not "distance matrix".
            domain (string): specifies the input data. Either "point cloud" or "distance matrix".
            clustering (class): clustering class (default sklearn.cluster.DBSCAN()). Common clustering classes can be found in the scikit-learn library (such as AgglomerativeClustering for instance).
            mask (int): threshold on the size of the Mapper nodes. Any node associated to a subpopulation with number of points less than this value will be removed.
        """
        self.filters, self.codomain, self.infdist, self.threshold, self.kernel, self.mode = filters, codomain, infer_distributions, threshold, kernel, mode
        self.num_bins, self.bnds = num_bins, bnds
        self.correct_Rips, self.delta, self.correct_mode, self.num_subdivisions = correct_Rips, delta, correct_mode, num_subdivisions
        self.cover, self.distance = cover, distance
        self.domain, self.clustering = domain, clustering
        self.mask, self.colors = mask, colors

    def fit(self, X, y=None):
        """
        Fit the MetricMapperComplex class on a dataset: compute the Mapper and store it in a simplex tree called mapper_

        Parameters:
            X (numpy array of shape n x d if point cloud and n x n if distance matrix): input point cloud or distance matrix.
            y (n x 1 array): point labels (unused).
        """
        num_pts, num_colors = len(X), self.colors.shape[1]

        if self.codomain == "distributions":
            if self.infdist:
                # self.filters is supposed to be a list containing a single realization of each conditional
                if self.mode == "NN":
                    distributions = infer_distributions_from_neighborhood(self.filters, X, self.threshold, self.domain)
                elif self.mode == "NW":
                    [embeddings, c_embeddings] = infer_distributions_from_Nadaraya_Watson(self.filters, X, kernel=self.kernel, means=False, num_bins=self.num_bins, bnds=self.bnds)
            else:
                # self.filters is supposed to be a list of lists containing the distribution of each conditional
                distributions = self.filters
            if (self.cover.mode == "embedding" or self.cover.mode == "embedding_histogram") or (self.cover.mode == "metric" and (self.distance.mode == "embedding" or self.distance.mode == "embedding_histogram")):
                # Compute histograms if necessary
                if self.infdist == False or (self.infdist ==True and self.mode == "NN"):
                    embeddings, c_embeddings = Histogram(num_bins=self.num_bins, bnds=self.bnds).fit_transform(distributions)
                    embeddings = np.vstack(embeddings)
        elif self.codomain == "distance matrix":
            # self.filters is supposed to be a square array of pairwise distances
            codomain_distances = self.filters

        elif self.codomain == "vectors":
            # self.filters is supposed to be an array of vectors
            embeddings = self.filters


        if self.cover.mode == "metric":
            if self.codomain != "distance matrix":
                if self.distance.mode == "embedding" or self.distance.mode == "embedding_histogram":
                    if self.distance.mode == "embedding_histogram":
                        self.distance.C = c_embeddings
                    codomain_distances = self.distance.compute_matrix(embeddings)
                elif self.distance.mode == "distributions":
                    codomain_distances = self.distance.compute_matrix(distributions)

        # Compute cover
        if self.cover.mode == "metric":
            self.cover.fit(codomain_distances)
        elif self.cover.mode == "embedding":
            self.cover.fit(embeddings)
        elif self.cover.mode == "embedding_histogram":
            self.cover.C = c_embeddings
            self.cover.fit(embeddings)

        binned_data = self.cover.binned_data

        if self.correct_Rips and (self.cover.mode == "embedding" or self.cover.mode == "embedding_histogram"):

            DX = np.triu(euclidean_distances(X), k=1) if self.domain == "point cloud" else X
            edges = np.argwhere((DX <= self.delta) & (DX > 0))

            if self.correct_mode == "uniform_refinement":
                ndiv = self.num_subdivisions
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
                self.colors = np.vstack([self.colors, np.zeros([num_pts + edges.shape[0] * ndiv, self.colors.shape[1]])])

            elif self.correct_mode == "cover_refinement" and self.cover.can_refine:
                new_points = self.cover._refine( [embeddings[edges[:,0],:], embeddings[edges[:,1],:]], num_pts )
                npts = sum([len(gr) for gr in new_points])
                AA = lil_matrix((num_pts + npts, num_pts + npts))
                for idx, gr in enumerate(new_points):
                    if len(gr) > 0:
                        idxs = [edges[idx,0]] + gr + [edges[idx,1]]
                        for idd in range(len(idxs)-1):
                            AA[idxs[idd], idxs[idd+1]] = 1
                    else:
                        AA[edges[idx,0], edges[idx,1]] = 1
                binned_data = self.cover.binned_data
                num_pts += npts
                self.colors = np.vstack([self.colors, np.zeros([num_pts + npts, self.colors.shape[1]])])

            self.clustering = GraphClustering(AA)
            self.domain = "indices"

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
