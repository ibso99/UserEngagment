import pandas as pd
import datetime

class Pipeline:
        ''' Data Engineering Pipe line for loading cleaning and performing 
        aggregate values'''
        #################################################
        def Data_Loader(data_path):
            return pd.read_csv(data_path)
        #################################################
        def Data_Cleaning(df):
            # 1. Handle Missing Values
            missing_values_count = df.isnull().sum()
            columns_with_missing_values = missing_values_count[missing_values_count > 0].index.tolist()

            df = df.dropna(subset=['MSISDN/Number'], how='any')  # Drop rows with missing MSISDN/Number

            # 2. Data Type Conversion
            try:
                df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
                df['End'] = pd.to_datetime(df['End'], errors='coerce')
            except ValueError:
                print("Error: Unable to convert 'Start' or 'End' columns to datetime.")

            # 3. Handle Duplicate Rows
            #    - Identify and remove duplicate rows
            num_duplicates = df.duplicated().sum()
            if num_duplicates > 0:
                print(f"Found {num_duplicates} duplicate rows. Removing duplicates.")
                df = df.drop_duplicates()

            # 4. Handle Outliers (
            # Q1 = df['Dur. (ms)'].quantile(0.25)
            # Q3 = df['Dur. (ms)'].quantile(0.75)
            # IQR = Q3 - Q1
            # lower_bound = Q1 - 1.5 * IQR
            # upper_bound = Q3 + 1.5 * IQR
            # df = df[(df['Dur. (ms)'] >= lower_bound) & (df['Dur. (ms)'] <= upper_bound)]

            # 5. Clean Specific Columns (if needed)
            #    - Example: Clean 'Last Location Name' (remove leading/trailing spaces, standardize)
            df['Last Location Name'] = df['Last Location Name'].str.strip() 

            return df

        #################################################
        def aggregate_user_behavior(df):
            """
            Aggregates user behavior data per user based on the provided DataFrame columns.

            Args:
                df: pandas DataFrame with the specified columns.

            Returns:
                pandas DataFrame with aggregated data per user:
                    - User (assuming 'MSISDN/Number' represents the user)
                    - Number_of_Sessions
                    - Total_Session_Duration
                    - Total_Download
                    - Total_Upload
            """

            # 'MSISDN/Number' better represents the user that Bearer Id 
            df = df.rename(columns={'MSISDN/Number': 'User'}) 

            # Group by User and aggregate
            aggregated_df = df.groupby('User').agg({
                'Bearer Id': 'count',  # Number of Sessions
                'Dur. (ms)': 'sum',    # Total Session Duration
                'Total DL (Bytes)': 'sum',  # Total Download
                'Total UL (Bytes)': 'sum'   # Total Upload
            }).reset_index()

            # Rename columns
            aggregated_df.columns = ['User', 'Number_of_Sessions', 
                                    'Total_Session_Duration', 'Total_Download', 
                                    'Total_Upload']

            return aggregated_df
        