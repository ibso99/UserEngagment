import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

class EDA_AND_STAT_ANALYZDER:

    def EDA(df):
        # 1. User Segmentation
        df['Total_Duration'] = df['Dur. (ms)'] 
        df['Total_Data'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
        df['Decile'] = pd.qcut(df['Total_Duration'], q=10, labels=False) 
        decile_data = df.groupby('Decile')['Total_Data'].sum()
        print("\nTotal Data per Decile:")
        print(decile_data)

        # 2. Basic Metrics
        print("\nBasic Metrics:")
        print(df.describe()) 

        # 3. Univariate Analysis
        print("\nDispersion Parameters:")
        print(df.agg(['mean', 'std', 'min', 'max', 'median', 'skew', 'kurtosis'])) 

    def Univariate_graph(df):

        # 4. Graphical Univariate Analysis
        df = df.rename(columns={'MSISDN/Number': 'User'})
        df['Total_Session_Duration'] = df['Dur. (ms)']
        df['Total_Download'] = df['Total DL (Bytes)'] + df['Total UL (Bytes)']
        df['Total_Upload']=df['Total UL (Bytes)']

        fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 10))
        sns.histplot(df['Total_Session_Duration'], ax=axes[0, 0])
        sns.boxplot(df['Total_Session_Duration'], ax=axes[0, 1])
        sns.histplot(df['Total_Download'], ax=axes[1, 0])
        sns.boxplot(df['Total_Download'], ax=axes[1, 1])
        sns.histplot(df['Total_Upload'], ax=axes[2, 0])
        sns.boxplot(df['Total_Upload'], ax=axes[2, 1])
        plt.tight_layout()
        plt.show()

    def Bivariate_graph(df):
        # 5. Bivariate Analysis
        sns.scatterplot(x='Total DL (Bytes)', y='Total UL (Bytes)', data=df)
        plt.show()

    def Correlation(df):
        # 6. Correlation Analysis
        corr_matrix = df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 
                        'Other DL (Bytes)']].corr()
        sns.heatmap(corr_matrix, annot=True)
        plt.show()

    def Correlation(df):
        # 7. Dimensionality Reduction (PCA)
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(df[['Social Media DL (Bytes)', 'Google DL (Bytes)', 'Email DL (Bytes)', 
                                        'Youtube DL (Bytes)', 'Netflix DL (Bytes)', 'Gaming DL (Bytes)', 
                                        'Other DL (Bytes)']])
        print("\nPCA Results:")
        print(f"Explained Variance Ratio: {pca.explained_variance_ratio_}")
        plt.scatter(pca_result[:, 0], pca_result[:, 1])
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
