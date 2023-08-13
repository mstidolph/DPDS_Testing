"""
DPDS Custome Pipeline 2 - German Data
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
# german = pd.read_csv("../data/german_credit.csv")



# Testing function
def main() -> None:
    
    """ 
    Data Loading 
    """
    
    # Select data to be used for testing
    df = german.copy()
    
    # Create second empty dataframe
    df2 = pd.DataFrame()
    
    
    """
    Pre-Tracking Data Editing 
    """
    
    # Convert class feature to categorical to allow encoding
    df['Creditability'] = df['Creditability'].astype("str")
    # Add to string to ensure encoding occurs
    df['Creditability'] = 'test ' + df['Creditability']
    
    # Add null values to allow imputation 
    for i in df.columns:
        if i != "Creditability":
            df.loc[df.sample(frac = 0.2).index, i] = np.nan
    
    # Split data to simulate test and training split
    df2 = df.tail(100)
    df = df.head(len(df) - 100)


    """
    Tracker Setup 
    """
    
    # Initialise tracker and link to data
    tracker = ProvenanceTracker()
    df, df2 = tracker.subscribe([df, df2])
    
    
    """ 
    Pipeline Testing
    """
    
    # Join test and training
    df = df.append(df2) 
    
    # Imputation
    df = df.fillna(method = 'ffill') 
    
    # Value transformation
    df['Occupation'] = df['Occupation'].map(lambda x: x * 100)
   
    # Add feature
    df['ratio'] = df.apply(lambda x: x['Purpose'] / x['Occupation'], axis=1)
    
    # Drop feature
    df = df.drop(columns=["Purpose", "Occupation", "Telephone"])
    
    # Scaling 
    scaler = StandardScaler()
    num_feat = list(df.dtypes[df.dtypes != "object"].index)
    df[num_feat] = scaler.fit_transform(df[num_feat])
        
    # Categorical encoder
    enc = OrdinalEncoder()
    df['Creditability'] = enc.fit_transform(df['Creditability'].values.reshape(-1, 1))
    
    
    """ 
    Results Verification
    """
    
    # Print modified dataframe and shape to verify successful operation
    print(df)
    print("Output Shape: ", df.shape)



# Execute function
if __name__ == '__main__':
    main()

