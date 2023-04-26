
import sys
# setting path
sys.path.append('../')

import src.process as process

# models
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB,GaussianNB

RSEED = 42

def models():
    model_dict = {
        'DT':   DecisionTreeClassifier(random_state=RSEED),
        # 'RFC':  RandomForestClassifier(random_state=RSEED),
        'XGB':  XGBClassifier(seed=RSEED),
        'ABC':  AdaBoostClassifier(random_state=RSEED),
        'LR':   LogisticRegression(random_state=RSEED),

        # # takes way too long to compute:
        # 'SVC':  SVC(random_state=RSEED),
        # 'KNN':  KNeighborsClassifier(),

        # # might try later
        # 'MNB': MultinomialNB(),
        # 'GNB': GaussianNB(),
    }
    return model_dict


def random_grids():

    random_grids = {

        'scaledRFC': {
            'RFC__bootstrap': [True, False],
            'RFC__max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, None],
            'RFC__max_features': ['auto', 'sqrt'],
            'RFC__min_samples_leaf': [1, 2, 4],
            'RFC__min_samples_split': [2, 5, 10],
            'RFC__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
            },

        'scaledXGB': { 
            'XGB__learning_rate': [0.0001,0.001, 0.01, 0.1, 1] ,
            'XGB__max_depth': range(3,21,3),
            'XGB__gamma': [i/10.0 for i in range(0,5)],
            'XGB__colsample_bytree': [i/10.0 for i in range(3,10)],
            'XGB__reg_alpha': [1e-5, 1e-2, 0.1, 1, 10, 100],
            'XGB__reg_lambda': [1e-5, 1e-2, 0.1, 1, 10, 100]
            },
    }
    return random_grids


####################################################################################


def main():

    preprocessor = process.create_preprocessor(numerical_features=[], categorical_features=[])
    scaled_models = process.model_process_pipeline(models, preprocessor, prefix='scaled')

if __name__ == "__main__":
    main()
