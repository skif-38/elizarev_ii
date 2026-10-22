import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

data = np.loadtxt('buildings.dat', skiprows=1)

model = DBSCAN(eps=300, min_samples=2)
model.fit(data)

print(data[model.labels_ == -1])
plt.scatter(data[:, 0], data[:, 1], c=model.labels_)
plt.show()