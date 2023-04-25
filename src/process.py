import pandas as pd
import numpy as np
import time

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, cross_validate, KFold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# SHOULD CHANGE THIS SO IT TREATS TIME DIFFERENTLY
def num_cat_features(df, target=''):
    """Create two lists, each containing the column names of numerical (including. time) and categorical (object) data

    Args:
        target (str, optional): Define the name of the target variable to be ignored from the lists. Defaults to ''.

    Returns:
        numerical_features, categorical_features: two lists containing the names of the numerical and categorical columns.
            (optional: without the target variable)
    """

    numerical_features = [col for col in df.select_dtypes(exclude=['object', 'bool']) if col != target]    
    categorical_features = [col for col in df.select_dtypes(include=['object', 'bool']) if col != target]
    print("Categorical Features:", categorical_features, "\nNumerical Features:", numerical_features)

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


def model_cv_scores(X_train, y_train, model_dict, kfolds=5, RSEED=42, shuffle=True, **kwargs):
    """Perform Cross Validation on the train datasets 

    Args:
        X_train (pd.DataFrame or np.array): Features of the train dataset
        y_train (pd.Series or np.array): Target variable of the train dataset
        model_dict (Dictionary): A dictionary of estimators to be fitted
        RSEED (int, optional): RSEED for the KFold pick/shuffle. Defaults to 42.
        shuffle (bool, optional): Whether or not the KFolds are to be shuffled. Defaults to True
        n_jobs (int, optional): Number of processing cores to use for the fitting computation. See sklearn glossary

    Returns:
        Dictionary: Dictionary containing predicted y values for each of the models in model_dict
    """
    predicted_y_dict = {}
    for model_name, model in model_dict.items():
        kfold = KFold(n_splits=kfolds, random_state=RSEED, shuffle=shuffle)
        start_time = time.time()
        predicted_y_dict[model_name] = cross_val_predict(model, X_train, y_train, cv=kfold, **kwargs)
        end_time = time.time()
        print(f"{model_name} - Time taken: {end_time - start_time:.2f} seconds")
    return predicted_y_dict


def model_selection_search(X_train, y_train, model_dict, parameter_dict, search_method=GridSearchCV, **kwargs):
    
    best_models = {}
    for model_name, model in model_dict.items():
        start_time = time.time()
        search_model = search_method(model, parameter_dict[model_name], **kwargs)
        search_model.fit(X_train, y_train)
        best_models[f'best{model_name}'] = search_model
        end_time = time.time()
        print(f"{model_name} - Time taken: {end_time - start_time:.2f} seconds")

    return best_models


def model_test_predict(X_train, X_test, y_train, model_dict):

    pred_ytest_dict = {}
    for model_name, model in model_dict.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        pred_ytest_dict[model_name] = model.predict(X_test)
        end_time = time.time()
        print(f"{model_name} - Time taken: {end_time - start_time:.2f} seconds")

    return pred_ytest_dict