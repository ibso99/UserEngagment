import os, sys
current_dir = os.getcwd()
# print(current_dir)

parent_dir = os.path.dirname(current_dir)
# print(parent_dir)

sys.path.insert(0,parent_dir)

import streamlit as st
from Dashboard.SatisfactionDashboard import SatisfactionDashboard
from Dashboard.UserEngagementAnalysisDashboard import UserEngagementDashboard
from Dashboard.ExperienceAnalysisDashboard import ExperienceDashboard
from Dashboard.DataOverviewDashboard import DataOverviewDashboard

query = "SELECT * FROM xdr_data;"

st.sidebar.title("Navigation")
dashboard_selection = st.sidebar.radio("Go to", ["Data Overview", "Satisfaction Analysis", "User Engagement Analysis", "Experience Analysis"])

if dashboard_selection == "Data Overview":
    dashboard = DataOverviewDashboard(query=query)
elif dashboard_selection == "Satisfaction Analysis":
    dashboard = SatisfactionDashboard(query=query)
elif dashboard_selection == "User Engagement Analysis":
    dashboard = UserEngagementDashboard(query=query)
elif dashboard_selection == "Experience Analysis":
    dashboard = ExperienceDashboard(query=query)

dashboard.load_data()
dashboard.display_dashboard()