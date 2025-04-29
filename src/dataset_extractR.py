# 
#/ Takes a sci-kit dataset and converts it into the standardised format for the Lab ML Project
#? USAGE: Import the required function from this module (get_breast_cancer_dataset or get_diabetes_dataset) into your code

# === Module Imports ===
from sklearn.datasets import load_diabetes, load_breast_cancer

import pandas as pd

# === Code ===
def _load_dataset(x: pd.DataFrame, y: pd.DataFrame, metadata: list = []):
    """
    Takes the data from a Sci-kit dataset (targets [y] and all other data [x]) and converts it into the Palmer format.
    """
    original_col_names = list(x) # Gets a list of all the column names
    col_names = {col: f'X_{col}' for col in original_col_names if col not in metadata} # Converts all columns not in the metadata list into X_<col_name> (i.e. feature columns)
    x = x.rename(columns = col_names) # Renames the feature columns to X_<col_name>
    data = pd.concat([x, y], axis = 1) # Merges the targets into the rest of the data
    data.insert(0, 'id', [str(i) for i in data.index]) # Adds the id column
    data = data.set_index('id') # Sets the index to the id (rather than just a numerical index) #! This, in the case of these datasets, is the same thing anyway. It's here for parity with *actual* datasets
        
    return data

def get_breast_cancer_dataset():
    """
    Gets the Sci-kit toy breast_cancer (classification) dataset and converts it into the Palmer format.

    Description
    -----------
    Loads and converts the Sci-kit breast cancer dataset (source: https://doi.org/10.24432/C5DW2B). This dataset
    consists of 569 instances each with 30 features (10 attributes with mean, standard error, and worst) and a
    single classification target (0 or 1 for benign or malignant respectively).

    Returns
    -------
    DataFrame
        A dataframe of breast cancer data in the Palmer format ([id, [features], target])

    """
    x, y = load_breast_cancer(as_frame = True, return_X_y = True) # Gets the breast cancer dataset as a DateFrame
    data = _load_dataset(x, y) # Convert it into the Palmer format

    return data

def get_diabetes_dataset(metadata = []):
    """
    Gets the Sci-kit toy diabetes (regression) dataset and converts it into the Palmer format, ensuring that the
    specified columns in `metadata` are treated as such.

    Description
    -----------
    Loads and converts the Sci-kit diabetes dataset (source: https://www4.stat.ncsu.edu/~boos/var.select/diabetes.html).
    This dataset consists of 442 instances each with 10 features (mean-centred and scaled) and a single
    regression target.

    By default, no `metadata` are specified, meaning that all columns will be converted into features (X_). However, a
    suggested list of metadata columns would be `['age', 'sex', 'bmi', 'bp']`.  

    Parameters
    --------- 
    metadata : list[str], optional
        A list of column names to be considered as metadata, rather than features, when loading (default [], meaning all columns are features)

    Returns
    -------
    DataFrame
        A dataframe of diabetes data in the Palmer format ([id, <[metadata]>, [features], target])
    """
    x, y = load_diabetes(as_frame = True, return_X_y = True) # Gets the diabetes dataset as a DateFrame
    data = _load_dataset(x, y, metadata) # Convert it into the Palmer format

    return data