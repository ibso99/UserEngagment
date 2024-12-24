import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.ExperienceAnalyzer import ExperienceAnalyzer
from scripts.DataPipline import DataPipeline

class ExperienceDashboard:
    def __init__(self, query):
        self.query = query
        self.analyzer = None

    def load_data(self):
        # Load your data using the provided SQL query
        telecom_df = DataPipeline.load_data_using_sqlalchemy(query=self.query)
        # Create an instance of the ExperienceAnalyzer class
        self.analyzer = ExperienceAnalyzer(telecom_df)
        # Perform the analysis
        self.analyzer.aggregate_user_data()
        self.analyzer.cluster_users(n_clusters=3)
        self.analyzer.calculate_experience_score()

    def display_dashboard(self):
        st.title('Experience Analysis Dashboard')

        # Display aggregated user data
        st.header('Aggregated User Data')
        st.dataframe(self.analyzer.user_agg)

        # Plot experience scores
        st.markdown("## Experience Scores")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.analyzer.user_agg['Experience_Score'], kde=True, ax=ax)
        ax.set_title('Distribution of Experience Scores')
        st.pyplot(fig)

        # Display top 10 best experience customers
        st.markdown("## Top 10 Best Experience Customers")
        top_experience_customers = self.analyzer.user_agg.nlargest(10, 'Experience_Score')
        st.dataframe(top_experience_customers)

        # Plot top 10 best experience customers using a bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='MSISDN/Number', y='Experience_Score',  
                    data=top_experience_customers, ax=ax, 
                    palette='viridis')
        ax.set_title('Top 10 Best Experience Customers')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig)