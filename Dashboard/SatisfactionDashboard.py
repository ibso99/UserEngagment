import os, sys
current_dir = os.getcwd()
# print(current_dir)

parent_dir = os.path.dirname(current_dir)
# print(parent_dir)

sys.path.insert(0, parent_dir)

import streamlit as st
import pandas as pd
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
        self.analyzer.cluster_satisfaction(n_clusters=3)
        self.analyzer.analyze_satisfaction_clusters()

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

        # Plot satisfaction scores using a pie chart
        st.markdown("## Satisfaction Scores")
        satisfaction_counts = self.analyzer.user_agg['Satisfaction_Score'].value_counts().reset_index()
        satisfaction_counts.columns = ['Satisfaction_Score', 'Count']
        fig, ax = plt.subplots()
        ax.pie(satisfaction_counts['Count'], labels=satisfaction_counts['Satisfaction_Score'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('husl', len(satisfaction_counts)))
        ax.set_title('Satisfaction Scores Distribution')
        st.pyplot(fig)

        # Display top 10 most satisfied customers
        st.markdown("## Top 10 Most Satisfied Customers")
        top_satisfied_customers = self.analyzer.user_agg.nlargest(10, 'Satisfaction_Score')
        st.dataframe(top_satisfied_customers)

        # Plot top 10 most satisfied customers using a bar plot
        st.markdown("## Top 10 Most Satisfied Customers")
        top_customers = self.analyzer.find_top_satisfied_customers(n=10)
        fig, ax = plt.subplots()
        sns.barplot(x='Satisfaction_Score', y='MSISDN/Number', data=top_customers, ax=ax, palette='viridis')
        ax.set_title('Top 10 Most Satisfied Customers')
        st.pyplot(fig)

        # Analyze satisfaction clusters
        st.markdown("## Satisfaction Clusters Analysis")
        self.analyzer.cluster_satisfaction(n_clusters=2)
        self.analyzer.analyze_satisfaction_clusters()
        cluster_stats = self.analyzer.user_agg.groupby('Satisfaction_Cluster').agg({
            'Satisfaction_Score': ['mean'],
            'Experience_Score': ['mean']
        }).reset_index()
        st.dataframe(cluster_stats)

        # Plot satisfaction clusters analysis using Matplotlib
        st.markdown("## Satisfaction Clusters Analysis")
        cluster_counts = self.analyzer.user_agg['Satisfaction_Cluster'].value_counts().reset_index()
        cluster_counts.columns = ['Cluster', 'Count']
        fig, ax = plt.subplots()
        ax.pie(cluster_counts['Count'], labels=cluster_counts['Cluster'], autopct='%1.1f%%', startangle=90, colors=sns.color_palette('husl', len(cluster_counts)))
        ax.set_title('Satisfaction Clusters Distribution')
        st.pyplot(fig)