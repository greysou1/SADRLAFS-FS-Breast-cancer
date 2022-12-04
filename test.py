# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from scipy.io import loadmat
# from sklearn.naive_bayes import GaussianNB
# from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
# from xgboost.sklearn import XGBClassifier




data_folder = 'data/'
dataset = pd.read_csv(data_folder + 'train.csv')
# dataset = pd.read_csv(data_folder + 'phpDYCOet.csv')
# dataset = pd.read_csv(data_folder + 'train_Amazon.csv')
# dataset = pd.read_csv(data_folder + 'train_cs.csv')

dataset = pd.read_csv(data_folder + 'breast_cancer_transcript_expression_with_label.csv')
#
# dataset.drop(dataset.columns[[0]],axis=1,inplace=True)

# rem = ['Id']
# dataset.drop(rem,axis=1,inplace=True)

r, c = dataset.shape
array = dataset.values

# Y = dataset.iloc[:,0]
# X = dataset.iloc[:,1:c]
X = dataset.iloc[:,0:(c-1)]
Y = dataset.iloc[:,(c-1)]

X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)

Forder =loadmat("IGorderCarto.mat")
order = Forder["Forder"].squeeze(0)-1
X_train = X_train.iloc[:, order]
X_val = X_val.iloc[:, order]

N_feature = X_train.shape[1] # feature number
N_sample = X_train.shape[0] # feature length,i.e., sample number
np.random.seed(0)

optimal_set = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1,
 0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0])
# optimal_set = np.array([1, 0, 0, 1, 1, 1, 0, 0, 1, 0 ,0 ,0 ,0 ,0 ,1 ,1 ,0 ,0 ,0 ,0 ,0 ,0 ,1 ,1 ,0 ,1 ,1 ,0, 0, 0, 0, 1, 1, 1, 1, 1, 0,
#  0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0,
#  1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1,
#  0, 1, 0, 0, 0, 0, 0])
model = RandomForestClassifier(n_jobs=-1,n_estimators=100, random_state=0)
model.fit(X_train.iloc[:, optimal_set==1], Y_train)
Y_pred = model.predict(X_val.iloc[:, optimal_set==1])
accuracy_RF = accuracy_score(Y_val, Y_pred)
macroF1_RF = f1_score(Y_val, Y_pred, average='macro')
print(accuracy_RF,macroF1_RF)
output =[]
name = []
name.append("result types")
output.append("Over_Carto_BY")

# model1= XGBClassifier(
#                 colsample_bytree= 0.7,
#                 subsample= 0.7,
#                 learning_rate= 0.075,
#                 # objective= 'binary:logistic',
#                 objective= 'multi:softmax',
#                 max_depth= 4,
#                 min_child_weight= 1,
#                 n_estimators= 4,
#                 seed=0
#             )

model2 =DecisionTreeClassifier()
# model1.fit(X_train.iloc[:, optimal_set==1], Y_train)
# accuracy_SVM = model1.score(X_val.iloc[:, optimal_set==1], Y_val)
# Y_pred = model1.predict(X_val.iloc[:, optimal_set==1])
# macroF1_SVM = f1_score(Y_val, Y_pred, average='macro')
# name.append("accuracy_SVM")
# output.append(accuracy_SVM)
# name.append("macroF1_SVM")
# output.append(macroF1_SVM)

model2.fit(X_train.iloc[:, optimal_set==1], Y_train)
accuracy_DT = model2.score(X_val.iloc[:, optimal_set==1], Y_val)
Y_pred = model2.predict(X_val.iloc[:, optimal_set==1])
macroF1_DT = f1_score(Y_val, Y_pred, average='macro')
name.append("accuracy_DT")
output.append(accuracy_DT)
name.append("macroF1_DT")
output.append(macroF1_DT)

out = dict(zip(name, output))
out = pd.DataFrame([out])
# out.to_csv(data_folder + 'result.csv')#,mode='a')

print(out)