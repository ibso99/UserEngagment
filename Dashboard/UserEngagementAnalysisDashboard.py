import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.UserEngagmentAnalyzer import UserEngagementAnalyzer
from scripts.DataPipline import DataPipeline

class UserEngagementDashboard:
    def __init__(self, query):
        self.query = query
        self.analyzer = None

    def load_data(self):
        # Load your data using the provided SQL query
        telecom_df = DataPipeline.load_data_using_sqlalchemy(query=self.query)
        # Create an instance of the UserEngagementAnalyzer class
        self.analyzer = UserEngagementAnalyzer(telecom_df)
        # Perform the analysis
        self.analyzer.aggregate_user_data()
        self.analyzer.cluster_users(n_clusters=3)
        self.analyzer.calculate_engagement_score()

    def display_dashboard(self):
        st.title('User Engagement Analysis Dashboard')

        # Display aggregated user data
        st.header('Aggregated User Data')
        st.dataframe(self.analyzer.user_agg)

        # Plot engagement scores
        st.markdown("## Engagement Scores")
        fig, ax = plt.subplots()
        sns.histplot(self.analyzer.user_agg['Engagement_Score'], kde=True, ax=ax)
        ax.set_title('Distribution of Engagement Scores')
        st.pyplot(fig)

        # Display top 10 most engaged customers
        st.markdown("## Top 10 Most Engaged Customers")
        top_engaged_customers = self.analyzer.user_agg.nlargest(10, 'Engagement_Score')
        st.dataframe(top_engaged_customers)