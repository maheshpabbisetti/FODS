import numpy as np
from sklearn.cluster import KMeans

# Assuming you have a customer dataset with shopping-related features (replace this with your dataset)
# Here, I'll create a sample dataset for demonstration purposes
customer_data = np.array([
    [5, 20, 30],   # Customer 1
    [15, 25, 10],  # Customer 2
    [10, 15, 5],   # Customer 3
    [25, 10, 20],  # Customer 4
    [30, 5, 25]    # Customer 5
])

# Number of clusters (replace this with the number of segments you want)
num_clusters = 3

def train_kmeans_model(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans

def assign_to_cluster(new_customer_features, kmeans_model):
    cluster_label = kmeans_model.predict([new_customer_features])
    return cluster_label[0]

if __name__ == "__main__":
    # Train the K-Means model
    kmeans_model = train_kmeans_model(customer_data, num_clusters)

    # Get user input for new customer features
    new_customer_features = [float(feature) for feature in input("Enter new customer features separated by commas: ").split(',')]

    # Assign the new customer to a cluster
    assigned_cluster = assign_to_cluster(new_customer_features, kmeans_model)

    # Display the result
    print(f"The new customer is assigned to Cluster {assigned_cluster + 1}")
