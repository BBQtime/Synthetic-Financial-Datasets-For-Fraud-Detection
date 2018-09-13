# Import models and utility functions
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns


# ignore warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import data
file = 'fraud.csv'
df = pd.read_csv(file)
print(df.columns)
df = df.rename(columns={'oldbalanceOrg': 'Old_Balance_Orig',
                        'newbalanceOrig': 'New_Balance_Orig',
                        'oldbalanceDest': 'Old_Balance_Dest',
                        'newbalanceDest': 'New_Balance_Dest',
                        'nameOrig': 'Name_Orig',
                        'nameDest': 'Name_Dest'})
print(df.head())
print(df.info())


# EDA
print('\n The types of fraudulent transactions are {}'.format(
    list(df.loc[df.isFraud == 1].type.drop_duplicates().values)))
#
dfFraudTransfer = df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')]
dfFraudCashout = df.loc[(df.isFraud == 1) & (df.type == 'CASH_OUT')]

print('\n No.fraudulent in TRANSFERs = {}'.
      format(len(dfFraudTransfer)))
# The Number of

print('\n No.fraudulent in CASH_OUTs = {}'.
      format(len(dfFraudCashout)))

# data import and cleaning
X = df.loc[(df.type == 'TRANSFER')]
#X = df.loc[(df.type == 'CASH_OUT')]
y = X['isFraud']
del X['isFraud']

# Eliminate columns shown to be irrelevant for analysis in the EDA
X = X.drop(['Name_Orig', 'Name_Dest', 'isFlaggedFraud', 'type'], axis=1)
print(X.head())

# decision tree model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

#X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=1)

dt = DecisionTreeClassifier(max_depth=2, random_state=1)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)
acc = accuracy_score(y_test, y_pred)


print('size of X_train, X_test, y_train, y_test')
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

acc = accuracy_score(y_test, y_pred)
print('accuracy of decision tree result',acc)

from sklearn.metrics import f1_score
#f1 score is good to evaluate unbalanced data
print('F1 macro score')
print(f1_score(y_test, y_pred, average='macro')  )
print('F1 micro score')
print(f1_score(y_test, y_pred, average='micro')  )

#confusion matrix of decision tree result with .2 random test dataset
from sklearn.metrics import confusion_matrix
print('confusion matrix of decision tree with .2 random test data:')
print(confusion_matrix(y_test, y_pred))


#Cross validation accuracy score wiht cv = 5
from sklearn.model_selection import cross_val_score
clf = DecisionTreeClassifier(max_depth=2, random_state=1)
scores = cross_val_score(clf, X_train, y_train, cv=5)
print('Cross validation confisuion matrix wiht cv = 5')
print([s for s in scores])

#Cross validation confisuion matrix wiht cv = 5
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(clf, X_train, y_train,cv=5)
conf_mat = confusion_matrix(y_train,y_pred)
print('Cross validation confisuion matrix wiht cv = 5')
print(conf_mat)