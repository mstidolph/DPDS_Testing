"""
DPDS Individual Function Testing - Census Data
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
# census = pd.read_csv("../data/adult.csv")



# Testing function
def main() -> None:
    
    """ 
    Data Loading 
    """
    
    # Select data to be used for testing
    df = census.copy()
    
    # Create empty datarame to avoid data loading error
    # Can be ignored if loading multiple dataframes for join testing
    df2 = pd.DataFrame()
    
    
    """
    Pre-Tracking Data Editing 
    
    Edit data to meet requirements of certain functions in brackets
    """
    
    # Change to string type (string modification and extraction)
    # df = df.astype("str")
    # Add test string to extract (string extraction)
    # df['capital.gain'] = 'test ' + df['capital.gain']
    
    # Add null values to allow imputation (imputation)
    # for i in df.columns:
    #     df.loc[df.sample(frac = 0.2).index, i] = np.nan
    
    # Add duplicate rows (drop duplicates)
    # df = pd.concat([df, df.head(10)], ignore_index=True)
    
    # Create new dataframe to join (merge)
    # df2['education'] = df['education'].unique()
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
    # df['age'] = df['age'].map(lambda x: x * 2 + 1)
    
    # Rename column/row
    # df = df.rename(index = {0: "A", 1: "B", 2: "C"})  # row
    # df = df.rename(columns={"capital.gain": "A", "age": "B", "capital.loss": "C"}) # column
    
    # Filter
    # df = df.loc[df['capital.gain'] < 4]
    
    # Drop instance
    # df = df.drop([0, 1, 2, 3, 4]) # head
    # df = df.drop([32556, 32557, 32558, 32559, 32560]) # tail
    # df = df.drop([3982, 8920, 11266, 19765, 26731]) # random
    
    # Add instance
    # df = df.append({"capital.gain": 2, "age": 4, "capital.loss": 1}, ignore_index=True)
    
    # Drop feature
    # df = df.drop(columns=["capital.gain", "age", "capital.loss"])
    
    # Add feature
    # df['new_col'] = df.apply(lambda x: x['capital.gain'] + x['age'], axis=1)
    
    # Change datatype
    # df = df.astype("str")
    
    # Create datetime object
    # df['age'] = pd.to_datetime(df['age'])
    
    # Change index
    # df = df.set_index(pd.Series(range(1000, 1150)))
    
    # Add to string
    # df['capital.gain'] = 'cred' + df['capital.gain']
    
    # String Extract
    # df['str_test'] = df['capital.gain'].str.extract('([a-zA-z ]+)')
    
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
    # df = df.sort_values(by = 'age') # ascending
    # df = df.sort_values(by = 'age', ascending = False) # descending
    
    # Transpose
    # df = df.T
    
    # Group and aggregate
    # df = df.groupby(by = 'age').mean(numeric_only=True) # mean
    # df = df.groupby(by = 'age').sum(numeric_only=True) # sum
    # df = df.groupby(by = 'age').median(numeric_only=True) # median
    # df = df.groupby(by = 'age').min(numeric_only=True) # min
    # df = df.groupby(by = 'age').max(numeric_only=True) # max
    
    # Scaling / Normalisation
    # scaler = StandardScaler()
    # scaler = MinMaxScaler()
    # scaler = RobustScaler()
    # scaler = Normalizer()
    # num_feat = list(df.dtypes[df.dtypes != "object"].index)
    # df[num_feat] = scaler.fit_transform(df[num_feat])
        
    # One hot encoder
    df_dum = pd.get_dummies(df['workclass'])
    df = df.join(df_dum)
    df = df.drop(['workclass'], axis = 1)
    
    # Categorical encoder
    # enc = OrdinalEncoder()
    # df['education'] = enc.fit_transform(df['education'].values.reshape(-1, 1))
    
    # Reorder columns
    # df = df[['education'] + [col for col in df.columns if col != 'education']]
    
    # Join
    # df = df.merge(df2, on = ['education']) # merge
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

