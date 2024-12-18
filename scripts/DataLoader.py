import pandas as pd

class DataLoader:
    
    def Data_Loader(data_path):
        return pd.read_csv(data_path)