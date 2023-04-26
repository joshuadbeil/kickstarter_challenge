import sys
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, fbeta_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv)) 

#in an ideal world this would validated
model = sys.argv[1]
X_test_path = sys.argv[2]
y_test_path = sys.argv[3]

# load the model from disk
loaded_model = pickle.load(open(model, 'rb'))
X_test = pd.read_csv(X_test_path)
y_test = pd.read_csv(y_test_path)

X_test['day_hour_launch'] = X_test['day_hour_launch'].astype(str)
X_test['day_hour_deadline'] = X_test['day_hour_deadline'].astype(str)

y_test_pred = loaded_model.predict(X_test)

print("Accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, y_test_pred)))
print("F-score on the testing data: {:.4f}".format(fbeta_score(y_test, y_test_pred, beta = 0.5)))
print("Recall score on the testing data: {:.4f}".format(recall_score(y_test, y_test_pred)))
print("Precision on the testing data: {:.4f}".format(precision_score(y_test, y_test_pred)))
print("------\n")
