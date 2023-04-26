import sys
import os
sys.path.append('../')
sys.path.append(os.path.abspath('../models'))

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import make_scorer, accuracy_score, fbeta_score, recall_score, precision_score
from sklearn.metrics import classification_report, confusion_matrix

import models.lib as modlib
import src.process as process
import src.cleaning as cleaning
import src.visualization as vs

import pickle

import warnings
warnings.filterwarnings('ignore')

RSEED = 42

# import data
processed_dir = os.path.join(os.path.dirname(__file__), '..', 'data/processed')
data = pd.read_csv(os.path.join(processed_dir, 'kickstarter_clean.csv'))


# converting some datatypes as they are categorical and dropping a few more columns that contain "future" information
data['day_hour_launch'] = data['day_hour_launch'].astype(str)
data['day_hour_deadline'] = data['day_hour_deadline'].astype(str)
data = data.drop(['staff_pick','usd_pledged','pledge_per_backer'], axis=1)


# splitting into train and test set
y = data['state']
X = data.drop('state', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X,y, stratify = y, test_size = 0.2, random_state = RSEED)

## in order to exemplify how the predict will work.. we will save the y_train
print("Saving test data in the data folder")
X_test.to_csv("data/X_test.csv", index=False)
y_test.to_csv("data/y_test.csv", index=False)


print("Generating Pipelines for further feature engineering")
target = 'state'
num_features, cat_features = process.num_cat_features(data, target=target)
models = {'XGB':  XGBClassifier(seed=RSEED)}
preprocessor = process.create_preprocessor(num_features, cat_features)
scaled_models = process.model_process_pipeline(models, preprocessor, prefix='scaled')


# model
print(f"Training data on {models.keys()}")
pred_ytest_dict, fitted_models_dict = process.model_test_predict(X_train, X_test, y_train, scaled_models)

for model_name, predictions in pred_ytest_dict.items():
    print(f"Metrics for y_test predictions using the {model_name} model:")
    print("------\n")
    print(classification_report(y_test, predictions))
    print("------\n")
    print(confusion_matrix(y_test, predictions))

# for model_name, predictions in pred_ytest_dict.items():
#     print(f"{model_name} ", "Accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
#     print(f"{model_name} ", "F-score on the testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta = 0.5)))
#     print(f"{model_name} ", "Recall score on the testing data: {:.4f}".format(recall_score(y_test, predictions)))
#     print(f"{model_name} ", "Precision on the testing data: {:.4f}".format(precision_score(y_test, predictions)))
#     print("------\n")

# export data
models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
model_file = pd.read_csv(os.path.join(processed_dir, 'kickstarter_xgb.sav'))


print("Saving model in the model folder")
xgb = fitted_models_dict['scaledXGB']
pickle.dump(xgb, open(model_file, 'wb'))