import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine

import os
import psycopg2 
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

# load environmemt variable from .dotenv
load_dotenv()

DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

class DataPipeline:
        ''' Data Engineering Pipe line for loading cleaning and performing 
        aggregate values'''
        #################################################

        def load_data_using_sqlalchemy(query):
            try:
                connection_string = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

                engine = create_engine(connection_string)

                df = pd.read_sql_query(query, engine)

                return df 
            except Exception as e:
                print(f'An Error Occured:{e}')
                return None
            
        def load_data_from_postgres(query):
            try:
                connection = psycopg2.connect(
                    host = DB_HOST,
                    port = DB_PORT,
                    database=DB_NAME,
                    user = DB_USER,
                    password = DB_PASSWORD
                )

                df = pd.read_sql_query(query, connection)

                connection.close()
                return df
            
            except Exception as e:
                print(f"an error occured: {e}")
            return None
        
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
       

        def excute_telecom_queries(db_url):
            engine = create_engine(db_url)

            # 1 Count of unique IMSIs
            unique_imsi_count = pd.read_sql_query("""
        SELECT COUNT(DISTINCT "IMSI") AS unique_imsi_count 
                                                FROM xdr_data;

        """, engine)
            # 2 Average Duration of Calls
            average_duration = pd.read_sql_query("""
        SELECT AVG("Dur. (ms)") AS average_duration
                                                FROM xdr_data
                                                WHERE "Dur. (ms)" IS NOT NULL; """, engine)
            # 3. Total Data Usage per User
            total_data_usage = pd.read_sql_query("""
        SELECT "IMSI",
                                                SUM("Total UL (Bytes)") AS total_ul_bytes,
                                                SUM("Total DL (Bytes)") AS total_dl_bytes
                                                FROM xdr_data
                                                GROUP BY "IMSI"
                                                ORDER BY total_dl_bytes DESC
                                                LIMIT 10; """, engine)
            # Average RTT by Last Location Name 
            avg_rtt_by_location = pd.read_sql_query("""
        SELECT "Last Location Name",
                                                    AVG("Avg RTT DL (ms)") AS avg_rtt_dl,
                                                    AVG("Avg RTT UL (ms)") AS avg_rtt_ul
                                                    FROM xdr_data
                                                    GROUP BY "Last Location Name" 
                                                    HAVING COUNT(*) > 10 
                                                    ORDER BY avg_rtt_dl DESC;""", engine)
            return {
                'unique_imsi_count': unique_imsi_count,
                'average_duration' : average_duration,
                'total_data_usage' : total_data_usage,
                'avg_rtt_by_location': avg_rtt_by_location,
            }
        

        