# Import required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt

# Step 1: Load the dataset
file_path = 'D:/SEM 5/dwdm lab/apriori/employee_data.csv'
data = pd.read_csv(file_path)

# Step 2: Select relevant features for clustering
features = data[['Age', 'Salary', 'Years at Company']]

# Step 3: Standardize the data
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 4: Visualize the dendrogram to determine the optimal number of clusters
plt.figure(figsize=(10, 7))
plt.title('Dendrogram for Hierarchical Clustering')
dendrogram = sch.dendrogram(sch.linkage(scaled_features, method='ward'))
plt.xlabel('Employees')
plt.ylabel('Euclidean Distances')
plt.show()

# Perform agglomerative hierarchical clustering
n_clusters = 3  # Set based on dendrogram analysis
hc = AgglomerativeClustering(n_clusters=n_clusters, metric='euclidean', linkage='ward')
data['Cluster'] = hc.fit_predict(scaled_features)


# Step 6: Calculate the silhouette score
sil_score = silhouette_score(scaled_features, data['Cluster'])
print(f"Silhouette Score: {sil_score:.3f}")

# Step 7: Display the data with assigned clusters
print(data[['Employee ID', 'Department', 'Age', 'Salary', 'Years at Company', 'Cluster']])

# Step 8: Visualize the clusters (2D visualization using first two features)
plt.figure(figsize=(8, 6))
plt.scatter(scaled_features[:, 0], scaled_features[:, 1], c=data['Cluster'], cmap='viridis', s=50)
plt.title('Cluster Visualization')
plt.xlabel('Feature 1 (e.g., Age)')
plt.ylabel('Feature 2 (e.g., Salary)')
plt.colorbar(label='Cluster')
plt.show()
