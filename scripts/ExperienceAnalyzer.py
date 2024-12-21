import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
        print(self.user_agg)

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