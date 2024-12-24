import os, sys
current_dir = os.getcwd()
# print(current_dir)

parent_dir = os.path.dirname(current_dir)
# print(parent_dir)

sys.path.insert(0,parent_dir)

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.DataPipline import DataPipeline
from scripts.SatisfactionAnalyzer import SatisfactionAnalyzer

class SatisfactionDashboard:
    def __init__(self, query):
        self.query = query
        self.analyzer = None

    def load_data(self):
        # Load your data using the provided SQL query
        telecom_df = DataPipeline.load_data_using_sqlalchemy(query=self.query)
        # Create an instance of the SatisfactionAnalyzer class
        self.analyzer = SatisfactionAnalyzer(telecom_df)
        # Perform the analysis
        self.analyzer.aggregate_user_data()
        self.analyzer.perform_engagement_clustering(n_clusters=3)
        self.analyzer.perform_experience_clustering(n_clusters=3)
        self.analyzer.calculate_engagement_score()
        self.analyzer.calculate_experience_score()
        self.analyzer.calculate_satisfaction_score()

    def display_dashboard(self):
        st.title('Satisfaction Analysis Dashboard')

        # Display aggregated user data
        st.header('Aggregated User Data')
        st.dataframe(self.analyzer.user_agg)

        # Plot engagement scores
        st.markdown("## Engagement Scores")
        fig, ax = plt.subplots()
        sns.histplot(self.analyzer.user_agg['Engagement_Score'], kde=True, ax=ax)
        ax.set_title('Distribution of Engagement Scores')
        st.pyplot(fig)

        # Plot experience scores
        st.markdown("## Experience Scores")
        fig, ax = plt.subplots()
        sns.histplot(self.analyzer.user_agg['Experience_Score'], kde=True, ax=ax)
        ax.set_title('Distribution of Experience Scores')
        st.pyplot(fig)

        # Plot satisfaction scores
        st.markdown("## Satisfaction Scores")
        fig, ax = plt.subplots()
        sns.histplot(self.analyzer.user_agg['Satisfaction_Score'], kde=True, ax=ax)
        ax.set_title('Distribution of Satisfaction Scores')
        st.pyplot(fig)

        # Display top 10 most satisfied customers
        st.markdown("## Top 10 Most Satisfied Customers")
        top_satisfied_customers = self.analyzer.user_agg.nlargest(10, 'Satisfaction_Score')
        st.dataframe(top_satisfied_customers)

        # Analyze satisfaction clusters
        st.markdown("## Satisfaction Clusters Analysis")
        self.analyzer.cluster_satisfaction(n_clusters=2)
        self.analyzer.analyze_satisfaction_clusters()
        cluster_stats = self.analyzer.user_agg.groupby('Satisfaction_Cluster').agg({
            'Satisfaction_Score': ['mean'],
            'Experience_Score': ['mean']
        }).reset_index()
        st.dataframe(cluster_stats)

         # Plot top 10 most satisfied customers
        st.markdown("## Top 10 Most Satisfied Customers")
        top_customers = self.analyzer.find_top_satisfied_customers(n=10)
        fig, ax = plt.subplots()
        sns.barplot(x='Satisfaction_Score', y='MSISDN/Number', data=top_customers, ax=ax)
        ax.set_title('Top 10 Most Satisfied Customers')
        st.pyplot(fig)

        # Plot satisfaction clusters analysis
        st.markdown("## Satisfaction Clusters Analysis")
        cluster_counts = self.analyzer.user_agg['Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig = px.pie(cluster_counts, names='Cluster', values='Count', title='Satisfaction Clusters Distribution')
        st.plotly_chart(fig)