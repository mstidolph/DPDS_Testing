"""
DPDS Individual Function Testing - German Data
"""


# Initial setup
import sys
sys.path.append("../")
# Import main packages
import pandas as pd
import numpy as np
# Import provenance tracker 
from prov_acquisition.prov_libraries.provenance_tracker import ProvenanceTracker
# Import functions for testing
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    Normalizer, 
    OrdinalEncoder, 
    OneHotEncoder
)
from sklearn.model_selection import train_test_split


# Import data
# Load once then comment to reduce runtime for repeated tests
# german = pd.read_csv("../data/german_credit.csv")



# Testing function
def main() -> None:
    
    """ 
    Data Loading 
    """
    
    # Select data to be used for testing
    df = german.copy()
    
    # Create empty datarame to avoid data loading error
    # Can be ignored if loading multiple dataframes for join testing
    df2 = pd.DataFrame()
    
    
    """
    Pre-Tracking Data Editing 
    
    Edit data to meet requirements of certain functions in brackets
    """
    
    # Change to string type (string modification and extraction, encoding)
    df = df.astype("str")
    # Add test string to extract (string extraction, encoding)
    df['Purpose'] = 'test ' + df['Purpose']
    
    # Add null values to allow imputation (imputation)
    # for i in df.columns:
    #     df.loc[df.sample(frac = 0.2).index, i] = np.nan
    
    # Add duplicate rows (drop duplicates)
    # df = pd.concat([df, df.head(10)], ignore_index=True)
    
    # Create new dataframe to join (merge)
    # df2['Credit Amount'] = df['Credit Amount'].unique()
    # df2['val'] = np.linspace(1, len(df2), len(df2))
    
    # Create new dataframe to join (join)
    # df2['val'] = 2 * np.linspace(1, len(df), len(df))
    
    # Split data to concatenate (concatenate and append)
    # df2 = df.tail(100)
    # df = df.head(len(df) - 100)
    
    
    """
    Tracker Setup 
    """
    
    # Initialise tracker and link to data
    tracker = ProvenanceTracker()
    df, df2 = tracker.subscribe([df, df2])
    
    
    """ 
    Testing Functions
    
    Comment/uncomment desired functions
    """
    
    # Value transformation
    # df['Occupation'] = df['Occupation'].map(lambda x: x * 2 + 1)
    
    # Rename column/row
    # df = df.rename(index = {0: "A", 1: "B", 2: "C"})  # row
    # df = df.rename(columns={"Purpose": "A", "Occupation": "B", "Creditability": "C"}) # column
    
    # Filter
    # df = df.loc[df['Creditability'] < 1]
    
    # Drop instance
    # df = df.drop([0, 1, 2, 3, 4]) # head
    # df = df.drop([995, 996, 997, 998, 999]) # tail
    # df = df.drop([17, 302, 443, 692, 904]) # random
    
    # Add instance
    # df = df.append({"Purpose": 2, "Occupation": 4, "Creditability": 1}, ignore_index=True)
    
    # Drop feature
    # df = df.drop(columns=["Purpose", "Occupation", "Creditability"])
    
    # Add feature
    # df['new_col'] = df.apply(lambda x: x['Purpose'] + x['Occupation'], axis=1)
    
    # Change datatype
    # df = df.astype("str")
    
    # Create datetime object
    # df['Occupation'] = pd.to_datetime(df['Occupation'])
    
    # Change index
    # df = df.set_index(pd.Series(range(1000, 1150)))
    
    # Add to string
    # df['Purpose'] = 'cred' + df['Purpose']
    
    # String Extract
    # df['str_test'] = df['Purpose'].str.extract('([a-zA-z ]+)')
    
    # Impute
    # df = df.fillna(df.median(numeric_only=True)) # median
    # df = df.fillna(df.mean(numeric_only=True)) # mean
    # df = df.fillna(df.mode(numeric_only=True)) # mode
    # df = df.fillna(-1) # set value
    # df = df.fillna(method = 'ffill') # forward fill
    # df = df.fillna(method = 'bfill') # backward fill
    # df = df.fillna(df.min(numeric_only=True)) # min value
    # df = df.fillna(df.max(numeric_only=True)) # max value
    
    # Remove duplicates
    # df = df.drop_duplicates()
    
    # Sort 
    # df = df.sort_values(by = 'Occupation') # ascending
    # df = df.sort_values(by = 'Occupation', ascending = False) # descending
    
    # Transpose
    # df = df.T
    
    # Group and aggregate
    # df = df.groupby(by = 'Occupation').mean(numeric_only=True) # mean
    # df = df.groupby(by = 'Occupation').sum(numeric_only=True) # sum
    # df = df.groupby(by = 'Occupation').median(numeric_only=True) # median
    # df = df.groupby(by = 'Occupation').min(numeric_only=True) # min
    # df = df.groupby(by = 'Occupation').max(numeric_only=True) # max
    
    # Scaling / Normalisation
    # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # scaler = RobustScaler()
    # scaler = Normalizer()
    # num_feat = list(df.dtypes[df.dtypes != "object"].index)
    # df[num_feat] = scaler.fit_transform(df[num_feat])
        
    # One hot encoder
    df_dum = pd.get_dummies(df['Purpose'])
    df = df.join(df_dum)
    df = df.drop(['Purpose'], axis = 1)
    
    # Categorical encoder
    # enc = OrdinalEncoder()
    # df['Purpose'] = enc.fit_transform(df['Purpose'].values.reshape(-1, 1))
    
    # Reorder columns
    # df = df[['Purpose'] + [col for col in df.columns if col != 'Purpose']]
    
    # Join
    # df = df.merge(df2, on = ['Credit Amount']) # merge
    # df = df.join(df2, rsuffix = "r_") # join
    # df = df.append(df2) # append
    # df = pd.concat([df, df2]) # concatenate
    
    # Test-train split
    # df, df2 = train_test_split(df, test_size = 0.3)
    
    
    """ 
    Results Verification
    """
    
    # Print modified dataframe and shape to verify successful operation
    print(df)
    print("Output Shape: ", df.shape)



# Execute function
if __name__ == '__main__':
    main()

