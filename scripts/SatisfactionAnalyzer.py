import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import psycopg2 

class SatisfactionAnalyzer:
    """
    Analyzes user satisfaction based on engagement and experience scores.
    """

    def __init__(self, df):
        """
        Initializes the analyzer with the input DataFrame.

        Args:
            df: pandas DataFrame with the required columns.
        """
        self.df = df.copy()
        self.engagement_cluster_centers_ = None
        self.experience_cluster_centers_ = None

    def aggregate_user_data(self):
        """
        Aggregates network parameters and handset type per customer,
        handling missing values using mean/mode.
        """
        required_columns = [
            'TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)',
            'Handset Type', 'MSISDN/Number', 'Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)'
        ]

        # Check for missing columns and handle them
        for col in required_columns:
            if col not in self.df.columns:
                raise KeyError(f"Column '{col}' is missing from the DataFrame.")

        # Calculate additional required columns
        self.df['Session_Frequency'] = self.df.groupby('MSISDN/Number')['MSISDN/Number'].transform('count')
        self.df['Total_Session_Duration'] = self.df.groupby('MSISDN/Number')['Dur. (ms)'].transform('sum')
        self.df['Total_Traffic'] = self.df['Total UL (Bytes)'] + self.df['Total DL (Bytes)']

        fill_values = {
            'TCP DL Retrans. Vol (Bytes)': self.df['TCP DL Retrans. Vol (Bytes)'].mean(),
            'Avg RTT DL (ms)': self.df['Avg RTT DL (ms)'].mean(),
            'Avg Bearer TP DL (kbps)': self.df['Avg Bearer TP DL (kbps)'].mean(),
            'Handset Type': self.df['Handset Type'].mode()[0]
        }
        self.df.fillna(fill_values, inplace=True)

        self.user_agg = self.df.groupby('MSISDN/Number').agg({
            'TCP DL Retrans. Vol (Bytes)': 'mean',
            'Avg RTT DL (ms)': 'mean',
            'Handset Type': 'first',
            'Avg Bearer TP DL (kbps)': 'mean',
            'Session_Frequency': 'first',
            'Total_Session_Duration': 'first',
            'Total_Traffic': 'first'
        }).reset_index()

        # Print aggregated user data 
        # print("\nAggregated User Data:")
        # print(self.user_agg)

    def perform_engagement_clustering(self, n_clusters=3):
            """
            Performs K-Means clustering on user engagement data.

            Args:
                n_clusters: Number of clusters.
            """
            scaler = StandardScaler()
            normalized_engagement = scaler.fit_transform(self.user_agg[['Session_Frequency', 
                                                                        'Total_Session_Duration', 
                                                                        'Total_Traffic']])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.user_agg['Engagement_Cluster'] = kmeans.fit_predict(normalized_engagement)
            self.engagement_cluster_centers_ = kmeans.cluster_centers_

    def perform_experience_clustering(self, n_clusters=3):
            """
            Performs K-Means clustering on user experience data.

            Args:
                n_clusters: Number of clusters.
            """
            scaler = StandardScaler()
            normalized_experience = scaler.fit_transform(self.user_agg[['TCP DL Retrans. Vol (Bytes)',
                                                                        'Avg RTT DL (ms)',
                                                                        'Avg Bearer TP DL (kbps)']])
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            self.user_agg['Experience_Cluster'] = kmeans.fit_predict(normalized_experience)
            self.experience_cluster_centers_ = kmeans.cluster_centers_

    
    def calculate_engagement_score(self):
        """
        Calculates engagement score based on Euclidean distance 
        from the least engaged cluster.
        """
        if self.engagement_cluster_centers_ is None:
            raise ValueError("Engagement clustering must be performed first. "
                             "Call perform_engagement_clustering() before "
                             "calculating engagement score.")

        # Select the least engaged cluster center (e.g., the first one) 
        least_engaged_center = self.engagement_cluster_centers_[0] 

        # Normalize engagement data
        scaler = StandardScaler()
        normalized_engagement = scaler.fit_transform(self.user_agg[['Session_Frequency', 
                                                                    'Total_Session_Duration', 
                                                                    'Total_Traffic']])

        # Calculate Euclidean distances to the least engaged cluster
        self.user_agg['Engagement_Score'] = euclidean_distances(normalized_engagement, 
                                                                least_engaged_center.reshape(1, -1))[:, 0]
    def calculate_experience_score(self):
            """
            Calculates experience score based on Euclidean distance 
            from the worst experience cluster.
            """
            if self.experience_cluster_centers_ is None:
                raise ValueError("Experience clustering must be performed first. "
                                "Call perform_experience_clustering() before "
                                "calculating experience score.")

            # Select the worst experience cluster center (e.g., the last one)
            worst_experience_center = self.experience_cluster_centers_[-1] 

            # Normalize experience data
            scaler = StandardScaler()
            normalized_experience = scaler.fit_transform(self.user_agg[['TCP DL Retrans. Vol (Bytes)', 
                                                                        'Avg RTT DL (ms)', 
                                                                        'Avg Bearer TP DL (kbps)']])

            # Calculate Euclidean distances to the worst experience cluster
            self.user_agg['Experience_Score'] = euclidean_distances(normalized_experience, 
                                                                    worst_experience_center.reshape(1, -1))[:, 0]

    def calculate_satisfaction_score(self):
            """
            Calculates satisfaction score as the average of engagement and experience scores.
            """
            self.user_agg['Satisfaction_Score'] = (self.user_agg['Engagement_Score'] + 
                                               self.user_agg['Experience_Score']) / 2

    def find_top_satisfied_customers(self, n=10):
        """
        Finds the top n most satisfied customers.
        """
        print("\nTop {} Most Satisfied Customers:".format(n))
        print(self.user_agg.nlargest(n, 'Satisfaction_Score'))

    def build_regression_model(self):
        """
        Builds a linear regression model to predict satisfaction score.
        """
        X = self.user_agg[['Session_Frequency', 'Total_Session_Duration', 
                           'Total_Traffic', 'TCP DL Retrans. Vol (Bytes)', 
                           'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)']]
        y = self.user_agg['Satisfaction_Score']
        model = LinearRegression()
        model.fit(X, y)
        print("Regression Model Coefficients:", model.coef_)
        print("Regression Model Intercept:", model.intercept_)

    def cluster_satisfaction(self, n_clusters=2):
        """
        Performs K-Means clustering on engagement and experience scores.
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.user_agg['Satisfaction_Cluster'] = kmeans.fit_predict(
            self.user_agg[['Engagement_Score', 'Experience_Score']]
        )

    def analyze_satisfaction_clusters(self):
        """
        Aggregates average satisfaction and experience scores per cluster.
        """
        cluster_stats = self.user_agg.groupby('Satisfaction_Cluster').agg({
            'Satisfaction_Score': ['mean'],
            'Experience_Score': ['mean']
        })
        print("\nCluster Statistics:")
        print(cluster_stats)
        return cluster_stats


    def export_to_postgresql(self, host, user, password, database, table_name, port=5432):
            """
            Exports the final table to a PostgreSQL database.

            Args:
                host: PostgreSQL host address.
                user: PostgreSQL username.
                password: PostgreSQL password.
                database: PostgreSQL database name.
                table_name: Name of the table to create in the database.
                port: PostgreSQL port number (default is 5432).
            """
            try:
                conn = psycopg2.connect(
                    host=host,
                    user=user,
                    password=password,
                    dbname=database,
                    port=port
                )
                cursor = conn.cursor()

                # Create table in PostgreSQL
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    MSISDN_Number VARCHAR(255),
                    Engagement_Score FLOAT,
                    Experience_Score FLOAT,
                    Satisfaction_Score FLOAT,
                    Cluster INT
                )
                """
                cursor.execute(create_table_query)

                # Insert data into PostgreSQL
                for index, row in self.user_agg.iterrows():
                    msisdn = row['MSISDN/Number']
                    engagement_score = row['Engagement_Score']
                    experience_score = row['Experience_Score']
                    satisfaction_score = row['Satisfaction_Score']
                    cluster = row['Satisfaction_Cluster']
                    insert_query = f"""
                    INSERT INTO {table_name} (MSISDN_Number, Engagement_Score, Experience_Score, Satisfaction_Score, Cluster)
                    VALUES (%s, %s, %s, %s, %s)
                    """
                    cursor.execute(insert_query, (msisdn, engagement_score, experience_score, satisfaction_score, cluster))

                conn.commit()
                cursor.close()
                conn.close()
                print(f"Data successfully exported to {table_name} table in {database} database.")

            except psycopg2.Error as err:
                print(f"Error: {err}")
