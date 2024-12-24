# import streamlit as st
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from scripts.UserEngagmentAnalyzer import UserEngagementAnalyzer
# from scripts.DataPipline import DataPipeline

# class UserEngagementDashboard:
#     def __init__(self, query):
#         self.query = query
#         self.analyzer = None

#     def load_data(self):
#         # Load your data using the provided SQL query
#         telecom_df = DataPipeline.load_data_using_sqlalchemy(query=self.query)
#         # Create an instance of the UserEngagementAnalyzer class
#         self.analyzer = UserEngagementAnalyzer(telecom_df)
#         # Perform the analysis
#         self.analyzer.aggregate_user_data()
#         self.analyzer.cluster_users(n_clusters=3)
#         self.analyzer.calculate_engagement_score()

#     def display_dashboard(self):
#         st.title('User Engagement Analysis Dashboard')

#         # Display aggregated user data
#         st.header('Aggregated User Data')
#         st.dataframe(self.analyzer.user_agg)

#         # Plot engagement scores
#         st.markdown("## Engagement Scores")
#         fig, ax = plt.subplots()
#         sns.histplot(self.analyzer.user_agg['Engagement_Score'], kde=True, ax=ax)
#         ax.set_title('Distribution of Engagement Scores')
#         st.pyplot(fig)

#         # Display top 10 most engaged customers
#         st.markdown("## Top 10 Most Engaged Customers")
#         top_engaged_customers = self.analyzer.user_agg.nlargest(10, 'Engagement_Score')
#         st.dataframe(top_engaged_customers)

#         # Plot top 10 most engaged customers using a bar plot with rotated x-axis labels
#         fig, ax = plt.subplots(figsize=(10, 6))
#         sns.barplot(x='MSISDN/Number', y='Engagement_Score',  data=top_engaged_customers, ax=ax, palette='viridis')
#         ax.set_title('Top 10 Most Engaged Customers')
#         ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
#         st.pyplot(fig)



import streamlit as st
import pandas as pd
import plotly.express as px
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

        # Commented out the plot for the distribution of engagement scores
        # st.markdown("## Engagement Scores")
        # fig, ax = plt.subplots()
        # sns.histplot(self.analyzer.user_agg['Engagement_Score'], kde=True, ax=ax)
        # ax.set_title('Distribution of Engagement Scores')
        # st.pyplot(fig)

        # Display top 10 most engaged customers
        st.markdown("## Top 10 Most Engaged Customers")
        top_engaged_customers = self.analyzer.user_agg.nlargest(10, 'Engagement_Score')
        st.dataframe(top_engaged_customers)

        # Plot top 10 most engaged customers using an interactive bar plot
        fig = px.bar(top_engaged_customers, x='MSISDN/Number', y='Engagement_Score',  title='Top 10 Most Engaged Customers', color='Engagement_Score', color_continuous_scale=px.colors.sequential.Viridis)
        fig.update_layout(xaxis_title='Engagement Score', yaxis_title='MSISDN/Number', xaxis_tickangle=-45)
        st.plotly_chart(fig)