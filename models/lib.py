
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

RSEED = 42

def models():
    model_dict = {
        'DT':   DecisionTreeClassifier(random_state=RSEED),
        'RFC':  RandomForestClassifier(random_state=RSEED),
        'XGB':  XGBClassifier(seed=RSEED),
        'ABC':  AdaBoostClassifier(random_state=RSEED),
        'KNN':  KNeighborsClassifier(),
        'LR':   LogisticRegression(random_state=RSEED),
        # takes way too long to compute:
        # 'SVC':  SVC(random_state=RSEED),
    }
    return model_dict


def main():

    preprocessor = process.create_preprocessor(numerical_features=[], categorical_features=[])
    scaled_models = process.model_process_pipeline(models, preprocessor, prefix='scaled')

if __name__ == "__main__":
    main()
