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
import plotly.express as px
from scripts.DataPipline import DataPipeline

class DataOverviewDashboard:
    def __init__(self, query):
        self.query = query
        self.df = None

    def load_data(self):
        # Load your data using the provided SQL query
        self.df = DataPipeline.load_data_using_sqlalchemy(query=self.query)
        # Clean the data
        self.df = DataPipeline.Data_Cleaning(self.df)

    def display_dashboard(self):
        st.title('Data Overview Dashboard')

        # Display raw data
        st.header('Raw Data')
        st.dataframe((self.df).head(100))

        # Top 10 Handsets Analysis
        st.markdown("## EDA Analysis Plots")
        top_10_handsets = self.df['Handset Type'].value_counts().nlargest(10)
        top_10_handsets_df = top_10_handsets.reset_index()
        top_10_handsets_df.columns = ['Handset Type', 'Count']

        # Plot the barplot with vibrant colors using Plotly
        fig1 = px.bar(top_10_handsets_df, x='Count', y='Handset Type', 
                      title='Top 10 Handset Types', color='Handset Type', 
                      color_discrete_sequence=px.colors.qualitative.Plotly)
        
        # Top 3 Handset Manufacturers Analysis
        top_3_manufacturers = self.df['Handset Manufacturer'].value_counts().nlargest(3)
        top_3_manufacturers_df = top_3_manufacturers.reset_index()
        top_3_manufacturers_df.columns = ['Handset Manufacturer', 'Count']

        # Plot the barplot for top 3 manufacturers using Plotly
        fig2 = px.bar(top_3_manufacturers_df, x='Handset Manufacturer', y='Count', title='Top 3 Handset Manufacturers', 
                      color='Handset Manufacturer', color_discrete_sequence=px.colors.qualitative.Plotly)

        # Top 5 Handsets for Top 3 Manufacturers
        grouped_data = self.df.groupby(['Handset Manufacturer', 'Handset Type']).size().reset_index(name='Count')
        top_manufacturers = grouped_data.groupby('Handset Manufacturer')['Count'].sum().nlargest(3).index.tolist()
        filtered_data = grouped_data[grouped_data['Handset Manufacturer'].isin(top_manufacturers)]
        top_handsets = filtered_data.groupby('Handset Manufacturer').apply(lambda x: x.nlargest(5, 'Count')).reset_index(drop=True)

        # Plot for top 5 handsets for each manufacturer using Plotly
        fig3 = px.bar(top_handsets, x='Count', y='Handset Type', color='Handset Manufacturer', 
                      title='Top 5 Handsets for Top 3 Manufacturers', 
                      barmode='group', color_discrete_sequence=px.colors.qualitative.Plotly)

        # Display the interactive plots side by side
        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            st.plotly_chart(fig2, use_container_width=True)
        with col3:
            st.plotly_chart(fig3, use_container_width=True)

        # Univariate Analysis
        st.markdown("## Univariate Analysis")
        required_columns = [
            'TCP DL Retrans. Vol (Bytes)',
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
        numerical_cols =   required_columns = [
             'Avg RTT DL (ms)', 'Avg Bearer TP DL (kbps)',
            'Dur. (ms)', 'Total UL (Bytes)', 'Total DL (Bytes)', 'Social Media DL (Bytes)',
            'Social Media UL (Bytes)', 'Google DL (Bytes)', 'Google UL (Bytes)',
            'Email DL (Bytes)', 'Email UL (Bytes)', 'Youtube DL (Bytes)', 'Youtube UL (Bytes)',
            'Netflix DL (Bytes)', 'Netflix UL (Bytes)', 'Gaming DL (Bytes)', 'Gaming UL (Bytes)',
            'Other DL (Bytes)', 'Other UL (Bytes)', 'Total UL (Bytes)', 'Total DL (Bytes)',
            'Total_Duration', 'Total_Data'
        ]
        if len(numerical_cols) > 1:
            col1 = st.selectbox('Select first column for bivariate analysis', numerical_cols)
            col2 = st.selectbox('Select second column for bivariate analysis', numerical_cols)
            if col1 and col2:
                st.markdown(f"### {col1} vs {col2}")
                fig, ax = plt.subplots()
                sns.scatterplot(x=self.df[col1], y=self.df[col2], ax=ax)
                ax.set_title(f'{col1} vs {col2}')
                st.pyplot(fig)