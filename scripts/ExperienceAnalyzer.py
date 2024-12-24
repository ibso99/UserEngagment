import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import seaborn as sns

class ExperienceAnalyzer:
    """
    Analyzes user experience based on network parameters and handset type.
    """

    def __init__(self, df):
        """
        Initializes the analyzer with the input DataFrame.

        Args:
            df: pandas DataFrame with the required columns.
        """
        self.df = df.copy()
        self.user_agg = None
        self.experience_cluster_centers_ = None

    def aggregate_user_data(self):
        """
        Aggregates network parameters and handset type per customer,
        handling missing values using mean/mode.
        """
        # Handle missing values (replace with mean/mode)
        fill_values = {
            'TCP DL Retrans. Vol (Bytes)': self.df['TCP DL Retrans. Vol (Bytes)'].mean(),
            'Avg RTT DL (ms)': self.df['Avg RTT DL (ms)'].mean(),
            'Avg Bearer TP DL (kbps)': self.df['Avg Bearer TP DL (kbps)'].mean(),
            'Handset Type': self.df['Handset Type'].mode()[0]
        }
        self.df.fillna(fill_values, inplace=True)

        # Aggregate data per customer
        self.user_agg = self.df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Handset Type': 'first',
            'Avg Bearer TP DL (kbps)': 'mean'
        }).reset_index()

        # Print aggregated user data 
        print("\nAggregated User Data:")
        # print(self.user_agg)

    def top_bottom_frequent_values(self, column, n=10):
        """
        Finds top n, bottom n, and most frequent values for a given column.

        Args:
            column: Name of the column.
            n: Number of top/bottom values to find.
        """
        print(f"\nTop {n} {column} values:")
        print(self.df[column].nlargest(n))
        print(f"\nBottom {n} {column} values:")
        print(self.df[column].nsmallest(n))
        print(f"\nMost frequent {n} {column} values:")
        print(self.df[column].value_counts().head(n))

    def analyze_throughput_per_handset(self):
        """
        Analyzes the distribution of average throughput per handset type.
        """
        throughput_per_handset = self.df.groupby('Handset Type')['Avg Bearer TP DL (kbps)'].mean()
        print("\nAverage Throughput per Handset Type:")
        print(throughput_per_handset)
        # Visualize distribution (e.g., bar plot)
        throughput_per_handset.plot(kind='bar')
        plt.xlabel('Handset Type')
        plt.ylabel('Average Throughput (kbps)')
        plt.title('Average Throughput per Handset Type')
        plt.show()

    def analyze_retransmission_per_handset(self):
        """
        Analyzes the average TCP retransmission per handset type.
        """
        retransmission_per_handset = self.df.groupby('Handset Type')['TCP DL Retrans. Vol (Bytes)'].mean()
        print("\nAverage TCP Retransmission per Handset Type:")
        print(retransmission_per_handset)
        # Visualize distribution (e.g., bar plot)
        retransmission_per_handset.plot(kind='bar')
        plt.xlabel('Handset Type')
        plt.ylabel('Average TCP Retransmission (Bytes)')
        plt.title('Average TCP Retransmission per Handset Type')
        plt.show()

    def cluster_users(self, n_clusters=3):
        """
        Performs K-Means clustering on user experience data.

        Args:
            n_clusters: Number of clusters.
        """
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.user_agg[['TCP DL Retrans. Vol (Bytes)',
                                                            'Avg RTT DL (ms)',
                                                            'Avg Bearer TP DL (kbps)']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_agg['Cluster'] = kmeans.fit_predict(normalized_data)
        self.experience_cluster_centers_ = kmeans.cluster_centers_

    def calculate_experience_score(self):
        """
        Calculates experience score based on Euclidean distance 
        from the best experience cluster.
        """
        if self.experience_cluster_centers_ is None:
            raise ValueError("Experience clustering must be performed first. "
                             "Call cluster_users() before calculating experience score.")

        # Select the best experience cluster center (e.g., the first one) 
        best_experience_center = self.experience_cluster_centers_[0] 

        # Normalize experience data
        scaler = StandardScaler()
        normalized_experience = scaler.fit_transform(self.user_agg[['TCP DL Retrans. Vol (Bytes)', 
                                                                    'Avg RTT DL (ms)', 
                                                                    'Avg Bearer TP DL (kbps)']])

        # Calculate Euclidean distances to the best experience cluster
        self.user_agg['Experience_Score'] = euclidean_distances(normalized_experience, 
                                                                best_experience_center.reshape(1, -1))[:, 0]

    def describe_clusters(self):
        """
        Provides a brief description of each cluster.
        """
        print("\nCluster Descriptions:")
        for i in range(len(self.user_agg['Cluster'].unique())):
            cluster_data = self.user_agg[self.user_agg['Cluster'] == i]
            print(f"Cluster {i+1}:")
            print(f"  - Avg TCP Retransmission: {cluster_data['TCP DL Retrans. Vol (Bytes)'].mean()}")
            print(f"  - Avg RTT DL: {cluster_data['Avg RTT DL (ms)'].mean()}")
            print(f"  - Avg Throughput: {cluster_data['Avg Bearer TP DL (kbps)'].mean()}")
            # Analyze dominant handset types in each cluster
            dominant_handset = cluster_data['Handset Type'].value_counts().idxmax()
            print(f"  - Dominant Handset Type: {dominant_handset}") 
            # Provide descriptive labels based on cluster characteristics
            if i == 0: 
                cluster_label = "High Performance"  # Example label
            elif i == 1:
                cluster_label = "Moderate Performance"
            else:
                cluster_label = "Low Performance"
            print(f"  - Cluster Label: {cluster_label}")
            print()

    def visualize_clusters(self):
        """
        Visualizes user experience clusters.
        """
        sns.scatterplot(x='Avg Bearer TP DL (kbps)', y='Avg RTT DL (ms)', hue='Cluster', 
                        data=self.user_agg)
        plt.title("User Experience Clusters")
        plt.xlabel("Average Throughput (kbps)")
        plt.ylabel("Average RTT DL (ms)")
        plt.show()

    def find_optimal_k(self, max_k=10):
        """
        Finds the optimal number of clusters using the elbow method.

        Args:
            max_k: Maximum number of clusters to evaluate.
        """
        inertia = []
        for k in range(1, max_k + 1):
            scaler = StandardScaler()
            normalized_data = scaler.fit_transform(self.user_agg[['TCP DL Retrans. Vol (Bytes)', 
                                                            'Avg RTT DL (ms)', 
                                                            'Avg Bearer TP DL (kbps)']])
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(normalized_data)
            inertia.append(kmeans.inertia_)
        plt.plot(range(1, max_k + 1), inertia)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.show()