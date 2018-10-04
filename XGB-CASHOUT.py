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
from sklearn.metrics import average_precision_score,recall_score
from time import time

# ignore warning
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# import data
file = 'D:/python/fraud.csv'
df = pd.read_csv(file)
print(df.columns)
df = df.rename(columns={'oldbalanceOrg': 'Old_Balance_Orig',
                        'newbalanceOrig': 'New_Balance_Orig',
                        'oldbalanceDest': 'Old_Balance_Dest',
                        'newbalanceDest': 'New_Balance_Dest',
                        'nameOrig': 'Name_Orig',
                        'nameDest': 'Name_Dest'})


#clean illogical data
# prepared dataset
df['day'] =pd.cut(df['step'],[0,24,48,72,96,120,144,168,192,216,240,264,288,312,336,360,384,408,432,456,480,504,528,552,576,600,624,648,672,
                      696,720,744], labels=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31])

df['day'] = df['day'].astype(int)
df['Error_Orig']=df['Old_Balance_Orig']-df['New_Balance_Orig']-df['amount']
df['Error_Dest']=df['Old_Balance_Dest']-df['New_Balance_Dest']+df['amount']
print(df.head())

# data import and cleaning
df=df.drop(['step','isFlaggedFraud'], axis=1)
sns.heatmap(df.corr())
plt.show()


print(df.shape)
print(df.loc[(df.isFraud==1)& (df.Old_Balance_Orig<df.amount)].shape)
#print(df.loc[(df.isFraud==1)& (df.Old_Balance_Orig<df.amount)].index)
df = df.drop(df.loc[(df.isFraud==1)& (df.Old_Balance_Orig<df.amount)].index) # 29

df = df.drop(df.loc[(df.isFraud==1)& (df.amount==0)].index) # 16
#print(df.shape,a.shape,b.shape,df.shape[0]-a.shape[0]-b.shape[0] )

# confirm
print('\n The fraud case with amount equals to zero{}'. format(
    len(df.loc[(df.isFraud==1) & (df.amount==0)])))


print('\n The types of fraudulent transactions are {}'.format(
    list(df.loc[df.isFraud == 1].type.drop_duplicates().values)))
#
dfFraudTransfer = df.loc[(df.isFraud == 1) & (df.type == 'TRANSFER')]
dfFraudCashout = df.loc[(df.isFraud == 1) & (df.type == 'CASH_OUT')]

print('\n No.fraudulent in TRANSFERs = {}'.
      format(len(dfFraudTransfer)))
 # origin fraud in transfer is 4097

print('\n No.fraudulent in CASH_OUTs = {}'. 
      format(len(dfFraudCashout)))
#origin fraud in cashout is 4116



X = df.loc[(df.type == 'CASH_OUT')]
X['Error_Orig']=X['Old_Balance_Orig']-X['New_Balance_Orig']-X['amount']
X['Error_Dest']=X['Old_Balance_Dest']-X['New_Balance_Dest']+X['amount']
y = X['isFraud']
del X['isFraud']

print(df.describe())


#EDA

X.hist(figsize=(20,20))
plt.show()
y.hist(figsize=(20,20))
plt.show()

# Eliminate columns shown to be irrelevant for analysis in the EDA
X = X.drop(['Name_Orig', 'Name_Dest', 'type'], axis=1)
print(X.head())
print(X.info())

print('\nerror in the originates account history is {}'.format(len(X.loc[X.Error_Orig!=0])))

print('\nerror in the balance recipients history is {}'.format(len(X.loc[X.Error_Dest!=0])))

print('\nerror in both is {}'.format(len(X.loc[(X['Error_Orig']!=0) & (X['Error_Dest']!=0)])))

print('\nerror in both is {}'.format(len(X.loc[(X['Error_Orig']==0) & (X['Error_Dest']==0)])))

print('\n fraud without error in victim account is {}'.format(len(X.loc[(X['Error_Orig']==0) & (y==1)])))

print('\n OLD BALANCE PROBLEM in victim account is {}'.format(len(X.loc[(X['Old_Balance_Orig']<X['amount'])])))

print('\n OLD BALANCE PROBLEM in victim account is {}'.format(len(X.loc[(X['Old_Balance_Orig']==0) & (y==1)&(X['amount']==0)])))

print('\n OLD BALANCE PROBLEM in victim account is {}'.format(len(X.loc[(X['Old_Balance_Orig']<X['amount'])&(y==1)])))




# decision tree model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2)

# X_train, X_test, y_train, y_test = train_test_split(
#    X, y, test_size=0.2, random_state=1)

# weight calculation
weights = (y == 0).sum() / (1.0 * (y == 1).sum())

# XGBoost
clf = xgb.XGBClassifier(max_depth=3, scale_pos_weight=weights, n_jobs=4)
probabilities = clf.fit(X_train, y_train).predict_proba(X_test)

# auprc equals to 0.7346
y_pred=clf.predict(X_test)
acc=accuracy_score(y_test,y_pred)
print('accuracy of XGBoost result', acc)
#acc is 0.96782122905
print('AUPRC = {}'.format(
    average_precision_score(y_test, probabilities[:, 1])))
# recall score
print('Recall:{0:2f}'.format(recall_score(y_test,y_pred)))

from sklearn.metrics import f1_score
# f1 score is good to evaluate unbalanced data
print('F1 macro score')
print(f1_score(y_test, y_pred, average='macro'))
print('F1 micro score')
print(f1_score(y_test, y_pred, average='micro'))



# confusion matrix of decision tree result with .2 random test dataset
from sklearn.metrics import confusion_matrix
print('confusion matrix of decision tree with .2 random test data:')
print(confusion_matrix(y_test, y_pred))


# Cross validation accuracy score wiht cv = 5
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
clf = xgb.XGBClassifier(max_depth=3, scale_pos_weight=weights, n_jobs=4)
kfold = StratifiedKFold(n_splits=10, random_state=7)
scores = cross_val_score(clf, X_train, y_train, cv=kfold)
#print('Cross validation confisuion matrix wiht cv = 5')
#print([s for s in scores])

print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))



# Cross validation confisuion matrix wiht cv = 5
from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(clf, X_train, y_train, cv=kfold)
conf_mat = confusion_matrix(y_train, y_pred)
print('Cross validation confisuion matrix wiht cv = 5')
print(conf_mat)


# Cross validation accuracy score wiht cv = 5
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
clf = xgb.XGBClassifier(max_depth=3, scale_pos_weight=weights, n_jobs=4)
kfold = StratifiedKFold(n_splits=10, random_state=7)
scores = cross_val_score(clf, X, y, cv=kfold)
#print('Cross validation confisuion matrix wiht cv = 5')
#print([s for s in scores])

print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean()*100, scores.std()*100))



# Cross validation confisuion matrix wiht cv = 5
from sklearn.model_selection import cross_val_predict
y_pred_all= cross_val_predict(clf, X, y, cv=kfold)
conf_mat = confusion_matrix(y, y_pred_all)
print('Cross validation confisuion matrix wiht cv = 5')
print(conf_mat)




