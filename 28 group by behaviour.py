import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Load and Explore Data
# Load your customer data (replace 'your_data.csv' with your actual data file)
df = pd.read_csv('your_data.csv')

# Display the first few rows of the DataFrame
print(df.head())

# Explore the data and check for missing values
print(df.info())

# Step 2: Preprocess Data
# Handle missing values if needed
df = df.dropna()

# Select relevant features for clustering (adjust as needed)
features = df[['purchase_history', 'browsing_behavior', 'age', '...']]

# Standardize numerical features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Step 3: Determine the Number of Clusters (K)
# Use the elbow method to find the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.show()

# Step 4: Apply K-Means Clustering
# Choose the optimal K based on the elbow method or other criteria
optimal_k = 3  # Adjust based on your analysis

# Apply K-Means clustering
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(features_scaled)

# Step 5: Analyze and Interpret Clusters
# Analyze cluster characteristics
cluster_means = df.groupby('cluster').mean()
print(cluster_means)

# Step 6: Visualize Results
# Visualize the clusters (consider using PCA or t-SNE for dimensionality reduction)
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=df['cluster'], cmap='viridis', alpha=0.5)
plt.title('Customer Segmentation')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Step 7: Interpret and Share Insights
# Interpret the results, understand the characteristics of each cluster, and share insights with the marketing team.
