"""
Created on Thu May  7 15:23:13 2020
@author: DESHMUKH
ADA BOOSTING AND EXTREAM GRADIENT BOOSTING
"""
#conda install -c conda-forge xgboost
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score,classification_report
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier,plot_importance
pd.set_option('display.max_columns',None)

# ========================================================================================
# Business Problem - Perform AdaBoost and Extreme Gradient Boosting for the wbcd dataset.
# ========================================================================================

wbcd = pd.read_csv('wbcd.csv')
wbcd.head()
wbcd = wbcd.drop(['id'],axis=1)
wbcd.isnull().sum()
wbcd.head()

# converting B to Benign and M to Malignant 
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'B','Benign ',wbcd['diagnosis'])
wbcd['diagnosis'] = np.where(wbcd['diagnosis'] == 'M','Malignant ',wbcd['diagnosis'])

# Summary
wbcd.describe()

# Boxplot
wbcd.boxplot(notch=True, patch_artist=True, grid=False);plt.xticks(fontsize=4,rotation = 30)

# Histrogram
wbcd.hist(grid=False)

# Checking Percentage of Output classes with the help Value Count.
(wbcd['diagnosis'].value_counts())/len(wbcd)*100  # Benign-63%,Maliganat-37%

##################################### - Splitting data - ######################################

# Splitting in X and y
X = wbcd.iloc[:,1:]
y = wbcd.iloc[:,0]

# Splitting in train and test
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.30, random_state = False)

###################################### - Ada Boosting - ######################################

# Ada boosting 
ada = AdaBoostClassifier(n_estimators=300,learning_rate=0.9)
ada.fit(X_train,y_train)

# Accuracy
#ada.score(X_train,y_train)
#ada.score(X_test,y_test)   

# Prediction on Train & Test Data
ada_pred_train = ada.predict(X_train)
ada_pred_test = ada.predict(X_test)

# Accuracy of Train and Test
accuracy_score(y_train,ada_pred_train) # 1 Train
accuracy_score(y_test,ada_pred_test) # 0.99 Test

# Confusion matrix of Train and Test
## Train
confusion_matrix_train = pd.crosstab(y_train,ada_pred_train,rownames=['Actual'],colnames= ['Predictions Tarin']) 
sns.heatmap(confusion_matrix_train, annot = True, cmap = 'Blues',fmt='g')

## Test
confusion_matrix_test = pd.crosstab(y_test,ada_pred_test,rownames=['Actual'],colnames= ['Predictions Test']) 
sns.heatmap(confusion_matrix_test, annot = True, cmap = 'Reds',fmt='g')

# Classification Report of test
print(classification_report(y_test,ada_pred_test))

################################# - Extreme Gradient Boosting - #################################

# Extreme Gradient Boosting
egb = XGBClassifier(max_depth=6, subsample=.8, n_estimators=350, learning_rate=1, min_child_weight=1)
egb.fit(X_train,y_train)

# Accuracy
#egb.score(X_train,y_train) 
#egb.score(X_test,y_test) 

# Prediction on Train & Test Data
egb_pred_train = egb.predict(X_train)
egb_pred_test = egb.predict(X_test)

# Accuracy of Train and Test
accuracy_score(y_train,egb_pred_train) # 1 Train
accuracy_score(y_test,egb_pred_test) # 0.98 Test

# Confusion matrix of Train and Test
## Train
confusion_matrix_train = pd.crosstab(y_train,egb_pred_train,rownames=['Actual'],colnames= ['Predictions Train']) 
sns.heatmap(confusion_matrix_train, annot = True, cmap = 'Blues',fmt='g')

## Test
confusion_matrix_test = pd.crosstab(y_test,egb_pred_test,rownames=['Actual'],colnames= ['Predictions Test']) 
sns.heatmap(confusion_matrix_test, annot = True, cmap = 'Reds',fmt='g')

# Important Fetures Plot
plot_importance(egb, grid=False)

# Classification Report of test
print(classification_report(y_test,egb_pred_test))


                          # ---------------------------------------------------- #

