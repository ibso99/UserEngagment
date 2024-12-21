import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

class UserEngagementAnalyzer:
    """
    Analyzes user engagement based on session frequency, duration, and traffic.
    """

    def __init__(self, df):
        """
        Initializes the analyzer with the input DataFrame.

        Args:
            df: pandas DataFrame with the required columns.
        """
        self.df = df

    def calculate_engagement_metrics(self):
        """
        Calculates engagement metrics: Session Frequency, Total Session Duration, Total Traffic.
        """
        self.df['Session_Frequency'] = self.df.groupby('MSISDN/Number')['Bearer Id'].transform('nunique')
        self.df['Total_Session_Duration'] = self.df['Dur. (ms)']
        self.df['Total_Traffic'] = self.df['Total DL (Bytes)'] + self.df['Total UL (Bytes)']

    def aggregate_user_data(self):
        """
        Aggregates engagement metrics per user.
        """
        self.user_agg = self.df.groupby('MSISDN/Number').agg({
            'Session_Frequency': 'first',
            'Total_Session_Duration': 'sum',
            'Total_Traffic': 'sum'
        }).reset_index()

    def find_top_customers(self, n=10):
        """
        Finds the top n customers per engagement metric.

        Args:
            n: Number of top customers to find.
        """
        print("\nTop {} Customers per Metric:".format(n))
        print("Top {} by Session Frequency:".format(n), 
              self.user_agg.nlargest(n, 'Session_Frequency'))
        print("Top {} by Total Session Duration:".format(n), 
              self.user_agg.nlargest(n, 'Total_Session_Duration'))
        print("Top {} by Total Traffic:".format(n), 
              self.user_agg.nlargest(n, 'Total_Traffic'))

    def cluster_users(self, n_clusters=3):
        """
        Performs K-Means clustering on user engagement data.

        Args:
            n_clusters: Number of clusters.
        """
        scaler = StandardScaler()
        normalized_data = scaler.fit_transform(self.user_agg[['Session_Frequency', 
                                                            'Total_Session_Duration', 
                                                            'Total_Traffic']])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_agg['Cluster'] = kmeans.fit_predict(normalized_data)

    def analyze_clusters(self):
        """
        Analyzes and prints cluster statistics.
        """
        cluster_stats = self.user_agg.groupby('Cluster').agg({
            'Session_Frequency': ['min', 'max', 'mean', 'sum'],
            'Total_Session_Duration': ['min', 'max', 'mean', 'sum'],
            'Total_Traffic': ['min', 'max', 'mean', 'sum']
        })
        print("\nCluster Statistics:")
        print(cluster_stats)

    def visualize_clusters(self):
        """
        Visualizes user engagement clusters.
        """
        sns.scatterplot(x='Total_Traffic', y='Total_Session_Duration', hue='Cluster', 
                        data=self.user_agg)
        plt.title("User Engagement Clusters")
        plt.xlabel("Total Traffic (Bytes)")
        plt.ylabel("Total Session Duration (ms)")
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
            normalized_data = scaler.fit_transform(self.user_agg[['Session_Frequency', 
                                                            'Total_Session_Duration', 
                                                            'Total_Traffic']])
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(normalized_data)
            inertia.append(kmeans.inertia_)
        plt.plot(range(1, max_k + 1), inertia)
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.show()
