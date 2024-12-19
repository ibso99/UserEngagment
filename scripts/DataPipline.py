import pandas as pd
import numpy as np
import datetime

class DataPipeline:
        ''' Data Engineering Pipe line for loading cleaning and performing 
        aggregate values'''
        #################################################

        def Data_Loader(data_path):

            return pd.read_csv(data_path)
        
        ################################################

        def Data_Cleaning(df):
            """
            Cleans the DataFrame by:
                1. Dropping columns with more than 80% missing values.
                2. Dropping rows with missing 'Bearer Id' or 'MSISDN/Number'.
                3. Imputing mean values for numerical columns.
                4. Dropping rows with missing values in categorical columns.
                5. Converting 'Start' and 'End' columns to datetime.
                6. Removing duplicate rows.
                7. Cleaning 'Last Location Name' (optional).

            Args:
                df: pandas DataFrame to be cleaned.

            Returns:
                Cleaned pandas DataFrame.
            """

            # 1. Drop columns with more than 80% missing values
            threshold = 0.8 * len(df)
            df = df.dropna(axis=1, thresh=threshold)

            # 2. Drop rows with missing 'Bearer Id' or 'MSISDN/Number'
            df = df.dropna(subset=['Bearer Id', 'MSISDN/Number'])

            # 3. Impute mean values for numerical columns
            numeric_cols = df.select_dtypes(include=np.number).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

            # 4. Drop rows with missing values in categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            df = df.dropna(subset=categorical_cols)

            # 5. Convert 'Start' and 'End' columns to datetime
            try:
                df['Start'] = pd.to_datetime(df['Start'], errors='coerce')
                df['End'] = pd.to_datetime(df['End'], errors='coerce')
            except ValueError:
                print("Error: Unable to convert 'Start' or 'End' columns to datetime.")

            # 6. Remove duplicate rows
            df = df.drop_duplicates()

            # 7. Clean 'Last Location Name' (optional)
            df['Last Location Name'] = df['Last Location Name'].str.strip() 

            return df

        #################################################
        def aggregate_user_behavior(df):
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
        

        