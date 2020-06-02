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

# ==============================================================================================
# Business Problem - Perform AdaBoost and Extreme Gradient Boosting for the Diabetes_RF dataset.
# ==============================================================================================

diabetes = pd.read_csv('Diabetes_RF.csv',skipinitialspace = True)
diabetes.head()
diabetes.isnull().sum()
diabetes.columns = diabetes.columns.str.replace(' ', '_')
diabetes.head()

# Summary
diabetes.describe()

# Boxplot
diabetes.boxplot(notch=True, patch_artist=True, grid=False);plt.xticks(fontsize=4,rotation = 30)

# Histrogram
diabetes.hist(grid=False)

# Checking Percentage of Output classes with the help Value Count.
(diabetes['Class_variable'].value_counts())/len(diabetes)*100  # No-65%,Yes-35%

##################################### - Spliting data - ######################################

# Splitting into X and y
X = diabetes.iloc[:,:8]
y = diabetes.iloc[:,8]

# Splitting into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.30, random_state = False)

###################################### - Ada Boosting - ######################################

# Ada Boosting 
ada = AdaBoostClassifier(n_estimators=200,learning_rate=1)
ada.fit(X_train,y_train)

# Accuracy
#ada.score(X_train,y_train)
#ada.score(X_test,y_test)    

# Prediction on Train & Test Data
ada_pred_train = ada.predict(X_train)
ada_pred_test = ada.predict(X_test)

# Accuracy of Train and Test
accuracy_score(y_train,ada_pred_train) 
accuracy_score(y_test,ada_pred_test) 

# Confusion matrix of Train and Test
## Train
confusion_matrix_train = pd.crosstab(y_train,ada_pred_train,rownames=['Actual'],colnames= ['Predictions Train']) 
sns.heatmap(confusion_matrix_train, annot = True, cmap = 'Blues',fmt='g')

## Test
confusion_matrix_test = pd.crosstab(y_test,ada_pred_test,rownames=['Actual'],colnames= ['Predictions Test']) 
sns.heatmap(confusion_matrix_test, annot = True, cmap = 'Reds',fmt='g')

# Classification Report of test
print(classification_report(y_test,ada_pred_test))

################################ - Extreme Gradient Boosting - ###############################

# Extreme Gradient Boosting
egb = XGBClassifier(max_depth=4, subsample=1, n_estimators=365, learning_rate=0.02, min_child_weight=1)
egb.fit(X_train,y_train)

# Accuracy
#egb.score(X_train,y_train) 
#egb.score(X_test,y_test)

# Prediction on Train & Test Data
egb_pred_train = egb.predict(X_train)
egb_pred_test = egb.predict(X_test)

# Accuracy of Train and Test
accuracy_score(y_train,egb_pred_train) 
accuracy_score(y_test,egb_pred_test) 

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

