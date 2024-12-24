import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
from streamlit_option_menu import option_menu

# Adjust the system path to include the parent directory
current_dir = os.getcwd()
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Import dashboards
from Dashboard.SatisfactionDashboard import SatisfactionDashboard
from Dashboard.UserEngagementAnalysisDashboard import UserEngagementDashboard
from Dashboard.ExperienceAnalysisDashboard import ExperienceDashboard
from Dashboard.DataOverviewDashboard import DataOverviewDashboard

st.set_page_config(page_title="Dashboard", layout="wide")
# SQL query to fetch data
query = "SELECT * FROM xdr_data;"

# Sidebar menu
# st.sidebar.title("User Engagement Analysis Menu")
# dashboard_selection = st.sidebar.radio("Go to", ["Data Overview", "Satisfaction Analysis", "User Engagement Analysis", "Experience Analysis"])

with st.sidebar:
    selected = option_menu(
        menu_title="Main Menu",
        options=["Data Overview", "Satisfaction Analysis", "User Engagement Analysis", "Experience Analysis", "Contact Us"],
        icons=["chart-bar", "smile", "users", "star", "envelope"],
        menu_icon="cast",
        default_index=0,
        # orientation="horizontal",
    )

# Display the selected dashboard
if selected == "Data Overview":
    st.header('Data Overview Dashboard')
    dashboard = DataOverviewDashboard(query=query)
    dashboard.load_data()
    dashboard.display_dashboard()

elif selected == "Experience Analysis":
    st.subheader("Experience Analysis")
    dashboard = ExperienceDashboard(query=query)
    dashboard.load_data()
    dashboard.display_dashboard()

elif selected == "User Engagement Analysis":
    st.subheader("User Engagement Analysis")
    dashboard = UserEngagementDashboard(query=query)
    dashboard.load_data()
    dashboard.display_dashboard()

elif selected == "Satisfaction Analysis":
    st.subheader("Satisfaction Analysis")
    dashboard = SatisfactionDashboard(query=query)
    dashboard.load_data()
    dashboard.display_dashboard()