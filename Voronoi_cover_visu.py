import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances

mx, Mx, my, My = -1, 1, -1, 1
step = .01
num_germs = 5
germs = np.hstack([np.random.uniform(low=mx, high=Mx, size=(num_germs,1)), np.random.uniform(low=my, high=My, size=(num_germs,1))])
threshold = 0.01 * np.unique(np.sort(pairwise_distances(germs)))[1]
print(threshold)


x = np.arange(mx, Mx, step)
y = np.arange(my, My, step)
xx, yy = np.meshgrid(x, y)
X = np.reshape(np.concatenate([xx[:,:,np.newaxis], yy[:,:,np.newaxis]], axis=2), [-1,2])
Ax, Ay, Bx, By = -0.01, 0., 0.01, 0.
epsilon = 0.001

V = np.abs(np.sqrt((xx-Ax)**2 + (yy-Ay)**2) - np.sqrt((xx-Bx)**2 + (yy-By)**2))
W = np.abs(np.sqrt((xx-Ax)**2 + (yy-Ay)**2) - np.sqrt((xx-Bx)**2 + (yy-By)**2)) <= epsilon

DX = pairwise_distances(X, germs)
Dm = np.reshape(DX.min(axis=1), [-1,1])
Di = np.argwhere( (DX <= Dm + 2*threshold) & (DX >= Dm) )

binned_data={}
for i in range(len(Di)):
	try:
		binned_data[Di[i,1]].append(Di[i,0])
	except KeyError:
		binned_data[Di[i,1]] = [Di[i,0]]
plt.figure()
for k in binned_data.keys():
	if k == 0:
		plt.scatter(X[binned_data[k]][:,0] + np.random.uniform(low=-1e-2, high=1e-2, size=(len(binned_data[k]))), 
                            X[binned_data[k]][:,1] + np.random.uniform(low=-1e-2, high=1e-2, size=(len(binned_data[k]))), 
                            s=10)
plt.show()

#plt.figure()
#h = plt.contourf(x,y,V)
#plt.show()

#plt.figure()
#plt.imshow(W)
#plt.show()

