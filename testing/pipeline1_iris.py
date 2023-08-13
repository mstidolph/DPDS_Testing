"""
DPDS Custome Pipeline 1 - Iris Data
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
# iris = pd.read_csv("../data/iris.csv")



# Testing function
def main() -> None:
    
    """ 
    Data Loading 
    """
    
    # Select data to be used for testing
    df = iris.copy()
    
    # Create empty datarame to avoid data loading error
    df2 = pd.DataFrame()
    
    
    """
    Pre-Tracking Data Editing 
    """
    
    # Add null values to allow imputation 
    for i in df.columns:
        if i != "Species":
            df.loc[df.sample(frac = 0.2).index, i] = np.nan


    """
    Tracker Setup 
    """
    
    # Initialise tracker and link to data
    tracker = ProvenanceTracker()
    df, df2 = tracker.subscribe([df, df2])
    
    
    """ 
    Pipeline Testing
    """
    
    # Drop feature
    df = df.drop(columns=["SepalWidthCm", "SepalLengthCm", "PetalLengthCm"])
    
    # Imputation
    df = df.fillna(method = 'ffill') 
    
    # Scaling 
    scaler = StandardScaler()
    num_feat = list(df.dtypes[df.dtypes != "object"].index)
    df[num_feat] = scaler.fit_transform(df[num_feat])
        
    # Categorical encoder
    enc = OrdinalEncoder()
    df['Species'] = enc.fit_transform(df['Species'].values.reshape(-1, 1))
    
    
    """ 
    Results Verification
    """
    
    # Print modified dataframe and shape to verify successful operation
    print(df)
    print("Output Shape: ", df.shape)



# Execute function
if __name__ == '__main__':
    main()

