import os, sys
current_dir = os.getcwd()
# print(current_dir)

parent_dir = os.path.dirname(current_dir)
# print(parent_dir)

sys.path.insert(0,parent_dir)

import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scripts.DataPipline import DataPipeline

class DataOverviewDashboard:
    def __init__(self, query):
        self.query = query
        self.df = None

    def load_data(self):
        # Load your data using the provided SQL query
        self.df = DataPipeline.load_data_using_sqlalchemy(query=self.query)

    def display_dashboard(self):
        st.title('Data Overview Dashboard')

        # Display raw data
        st.header('Raw Data')
        st.dataframe(self.df)

        # Univariate Analysis
        st.markdown("## Univariate Analysis")
        required_columns = [
            'TCP DL Retrans. Vol (Bytes)', 'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)',
            'Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)', 'Social Media DL (Bytes)',
            'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)',
            'Total_Duration', 'Total_Data'
        ]

        num_cols = len(required_columns)
        cols_per_row = 3
        num_rows = (num_cols + cols_per_row - 1) // cols_per_row

        fig, axes = plt.subplots(num_rows, cols_per_row, figsize=(15, 5 * num_rows))
        axes = axes.flatten()

        for i, col in enumerate(required_columns):
            if col in self.df.columns:
                sns.histplot(self.df[col], kde=True, ax=axes[i])
                axes[i].set_title(f'Distribution of {col}')
            else:
                axes[i].set_visible(False)

        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        st.pyplot(fig)

        # Bivariate Analysis
        st.markdown("## Bivariate Analysis")
        numerical_cols = self.df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 1:
            col1 = st.selectbox('Select first column for bivariate analysis', numerical_cols)
            col2 = st.selectbox('Select second column for bivariate analysis', numerical_cols)
            if col1 and col2:
                st.markdown(f"### {col1} vs {col2}")
                fig, ax = plt.subplots()
                sns.scatterplot(x=self.df[col1], y=self.df[col2], ax=ax)
                ax.set_title(f'{col1} vs {col2}')
                st.pyplot(fig)