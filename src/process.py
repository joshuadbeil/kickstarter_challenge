import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# SHOULD CHANGE THIS SO IT TREATS TIME DIFFERENTLY
def num_cat_features(df, target=''):
    """Create two lists, each containing the column names of numerical (including. time) and categorical (object) data

    Args:
        target (str, optional): Define the name of the target variable to be ignored from the lists. Defaults to ''.

    Returns:
        numerical_features, categorical_features: two lists containing the names of the numerical and categorical columns.
            (optional: without the target variable)
    """

    numerical_features = [col for col in df.columns[df.dtypes != object] if col != target]
    categorical_features = [col for col in df.select_dtypes(include=['object', 'bool']) if col != target]    
    return numerical_features, categorical_features


def create_preprocessor(numerical_features=[], categorical_features=[]):
    """Generate a very generic scaler object to preprocess features when passing into an estimator object.

    Args:
        numerical_features (list, optional): List of numerical features to be preprocessed by the object when passed to an estimator. Can be empty.
        categorical_features (list, optional): List of categorical features to be preprocessed by the object when passed to an estimator. can be empty.

    Returns:
        ColumnTransformer(): returns a ColumnTransformer() object:
            - impute numerical NaN with median()
            - impute categorical/string NaN with 'missing'
            - use StandardScaler() on numerical features
            - Use OneHotEncoding to create dummies for categorical features
    """
    numerical_pipeline = Pipeline([
        ('imputer_num', SimpleImputer(strategy='median')),
        ('std_scaler', StandardScaler())
    ])

    # Pipeline for categorical features 
    categorical_pipeline = Pipeline([
        ('imputer_cat', SimpleImputer(strategy='constant', fill_value='missing')),
        ('1hot', OneHotEncoder(handle_unknown='ignore',drop='first'))
    ])

    return ColumnTransformer([
        ('num', numerical_pipeline, numerical_features),
        ('cat', categorical_pipeline, categorical_features)
    ])


def model_process_pipeline(model_dict, preprocessor, prefix='scaled'):
    """Takes a dictionary of ML models and returns a dictionary of preprocessed ML models

    Args:
        model_dict (Dictionary):
            - key should be an abbreviation of the model, eg. 'DT' for DecisionTreeClassifier()
            - Model is expected to be the model class, eg. DecisionTreeClassifier()
        preprocessor (sklearn.compose.ColumnTransformer): Object to apply a set of rules how to scale each column of the DataFrame
        prefix (str, optional): prefix for the preprocessed model key. also used to pass the preprocessor in the pipeline. Defaults to 'scaled'.

    Returns:
        Dictionary: Dictionary of Pipeline objects containing the specified models and the specified scaler.
    """
    return {f'{prefix}{model_name}': Pipeline([('scaled' , preprocessor),(model_name ,model)]) for (model_name, model) in model_dict.items()}