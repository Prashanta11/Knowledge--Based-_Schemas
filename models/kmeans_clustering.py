import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Load preprocessed data
data = pd.read_csv('data/preprocessed_stroke_data.csv')

# Select features for clustering
features = data[['age', 'avg_glucose_level', 'bmi']]

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(features)

# Add cluster labels to the data
data['cluster'] = clusters

# Reduce dimensions for visualization
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(features)

# Plot the clusters
plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=clusters, cmap='viridis')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('K-Means Clustering')
plt.colorbar(label='Cluster')
plt.show()