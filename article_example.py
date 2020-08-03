import numpy as np
from metricmapper import *

X = np.array( [[np.cos(0*2*np.pi/5), np.sin(0*2*np.pi/5)], 
               [np.cos(1*2*np.pi/5), np.sin(1*2*np.pi/5)], 
               [np.cos(2*2*np.pi/5), np.sin(2*2*np.pi/5)], 
               [np.cos(3*2*np.pi/5), np.sin(3*2*np.pi/5)], 
               [np.cos(4*2*np.pi/5), np.sin(4*2*np.pi/5)]] )

F = np.array( [[1.5, 1.5],
               [4.5, 1.5],
               [4.1, 3.5],
               [3.5, 4.1],
               [1.5, 5.5]] )

HC1 = HypercubeCover(cover_mode="explicit", cover=[[[0.,3.], [0.,3.]], [[4.,7.], [4.,7.]]])
HC2 = HypercubeCover(cover_mode="implicit", bnds=np.array([[0.,7.],[0.,7.]]), resolutions=np.array([2,2]), gains=np.array([.25,.25]))

MMC1 = MetricMapperComplex(filters=F, colors=F, codomain="vectors", correct_Rips=True, delta=2*np.pi/5, correct_mode="cover_refinement",                       cover=HC1)
MMC2 = MetricMapperComplex(filters=F, colors=F, codomain="vectors", correct_Rips=True, delta=2*np.pi/5, correct_mode="cover_refinement",                       cover=HC2)
MMC3 = MetricMapperComplex(filters=F, colors=F, codomain="vectors", correct_Rips=True, delta=2*np.pi/5, correct_mode="uniform_refinement", num_subdivisions=1, cover=HC1)

MMC1.fit(X)
print(list(MMC1.mapper_.get_filtration()))
MMC2.fit(X)
print(list(MMC2.mapper_.get_filtration()))
MMC3.fit(X)
print(list(MMC3.mapper_.get_filtration()))
