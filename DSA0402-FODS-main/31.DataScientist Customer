import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
np.random.seed(0)
num_samples = 300
num_features = 2
data = np.random.rand(num_samples, num_features) * 10
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters,n_init=10)
kmeans.fit(data)
cluster_labels = kmeans.labels_
plt.figure(figsize=(8, 6))
for cluster_id in range(num_clusters):
    plt.scatter(
        data[cluster_labels == cluster_id, 0],
        data[cluster_labels == cluster_id, 1],
        label=f'Cluster {cluster_id + 1}'
    )
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=200,
    marker='X',
    c='black',
    label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Customer Segmentation using K-Means Clustering')
plt.legend()
plt.show()
