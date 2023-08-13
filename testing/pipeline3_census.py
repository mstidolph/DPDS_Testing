"""
DPDS Custome Pipeline 3 - Census Data
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
    OrdinalEncoder
)


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
    
    # Create second and third empty datarames
    df2 = pd.DataFrame()
    df3 = pd.DataFrame()
    
    
    """
    Pre-Tracking Data Editing 
    """
    
    # Add null values to allow imputation 
    for i in df.columns:
        if i != "income":
            df.loc[df.sample(frac = 0.2).index, i] = np.nan
    
    # Create new lookup dataframe to merge
    df3['education'] = df['education'].unique()
    df3['val'] = np.linspace(1, len(df3), len(df3))
    
    # Split data to simulate test and training split
    df2 = df.tail(100)
    df = df.head(len(df) - 100)


    """
    Tracker Setup 
    """
    
    # Initialise tracker and link to data
    tracker = ProvenanceTracker()
    df, df2, df3 = tracker.subscribe([df, df2, df3])
    
    
    """ 
    Pipeline Testing
    """
    
    # Join test and training
    df = df.append(df2) 
    
    # Imputation
    df = df.fillna(method = 'ffill') 
    
    # Value transformation
    df['age'] = df['age'].map(lambda x: x * 100)
    
    # Merge with lookup
    df = df.merge(df3, on = ['education'])

   
    # Add feature
    df['ratio'] = df.apply(lambda x: x['capital.gain'] / x['age'], axis=1)
    
    # Scaling
    scaler = StandardScaler()
    num_feat = list(df.dtypes[df.dtypes != "object"].index)
    df[num_feat] = scaler.fit_transform(df[num_feat])
    
    # One hot encoder
    dum = pd.get_dummies(df['workclass'])
    df = df.join(dum)
    
    # Rename columns
    df = df.rename(columns={"fnlwgt": "A", "relationship": "B"})
    
    # Categorical encoder
    enc = OrdinalEncoder()
    df['income'] = enc.fit_transform(df['income'].values.reshape(-1, 1))
    
    # Drop feature
    df = df.drop(columns=["capital.gain", "age", "workclass"])
    
    
    """ 
    Results Verification
    """
    
    # Print modified dataframe and shape to verify successful operation
    print(df)
    print("Output Shape: ", df.shape)



# Execute function
if __name__ == '__main__':
    main()

