# Data_Preprocessing
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn import svm
from sklearn.tree import DecisionTreeClassifier




data_folder = 'data/'
# dataset = pd.read_csv(data_folder + 'train.csv')
# dataset = pd.read_csv(data_folder + 'phpDYCOet.csv')
# dataset = pd.read_csv(data_folder + 'train_Amazon.csv')
# dataset = pd.read_csv(data_folder + 'train_cs.csv')
dataset = pd.read_csv(data_folder + 'Glycation.csv')

#
dataset.drop(dataset.columns[[0]],axis=1,inplace=True)

# rem = ['Id']
# dataset.drop(rem,axis=1,inplace=True)

r, c = dataset.shape
array = dataset.values

# Y = dataset.iloc[:,0]
# X = dataset.iloc[:,1:c]
X = dataset.iloc[:,0:(c-1)]
Y = dataset.iloc[:,(c-1)]
#
#
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)

model = RandomForestClassifier(n_jobs=-1,n_estimators=2, random_state=0)
# model =svm.SVC(C=0.8, kernel='rbf', gamma=20,decision_function_shape='ovr')
#

N_feature = X_train.shape[1] # feature number
N_sample = X_train.shape[0] # feature length,i.e., sample number

np.random.seed(0)



# selected feature set, initialized to be empty
n_selected_features = 221
F = []

count = 0

while count < n_selected_features:

    max_acc = 0

    for i in range(N_feature):

        if i not in F:

            F.append(i)
            print(F)
            model.fit(X_train.iloc[:, F], Y_train)
            acc = model.score(X_val.iloc[:, F], Y_val)

            F.pop()

            # record the feature which results in the largest accuracy

            if acc > max_acc:

                max_acc = acc

                idx = i

    F.append(idx)
    count += 1

output =[]
name = []
name.append("result types")
output.append("SFS_Glycation_BY")
optimal_set = F


model2 =DecisionTreeClassifier(random_state=0)#criterion='entropy', min_samples_leaf=3,random_state=0)



model2.fit(X_train.iloc[:, optimal_set], Y_train)
accuracy_DT = model2.score(X_val.iloc[:, optimal_set], Y_val)

name.append("accuracy_DT")
output.append(accuracy_DT)



out = dict(zip(name, output))
out = pd.DataFrame([out])
out.to_csv(data_folder + 'resultcom.csv',mode='a')