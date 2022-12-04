import json, csv
import pandas as pd
import numpy as np
import scipy as sp


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVR
from sklearn import metrics 
from sklearn import model_selection

from FS_models import FeatureSelection

import sys
 
file_path = 'feature_selection_lazy.out'
sys.stdout = open(file_path, "w")

print('Loading data ...')
dataset = pd.read_csv('data/breast_cancer_transcript_expression_with_label.csv')
print(f'Loaded {dataset.shape = }')
dataset = dataset.set_index('Unnamed: 0').T
print(f'Transpose {dataset.shape = }')

r, c = dataset.shape
array = dataset.values
X = dataset.iloc[:,0:(c-1)]
Y = dataset.iloc[:,(c-1)]

io = open("IGorderbreastcancer.txt","r")
order = np.array(json.load(io), dtype=np.int32)

X = X.iloc[:, order]
Y.loc[(Y == 'Positive')]= 1
Y.loc[(Y == 'Negative')]= 0

Y = Y.astype('float')
X = X.astype('float')

print('Data loaded and ordered.')

size_list = [50, 250, 1000, 2500, 10000]
csv_header = ['model', 'org_feature_size', 'final_feature_size', 'accuracy', 'features_selected']
csv_path = 'outputs/FRL_outputs.csv'
with open(csv_path, 'w') as f:
    writer = csv.writer(f)
    # write the header
    writer.writerow(csv_header)

for size in size_list:
    print(('\n')+('='*20)+' Feature size = '+str(size)+' '+('='*20))
    X_new = X.iloc[:, 0:size]

    FS = FeatureSelection(X_new, Y)

    for FS_model in FS.models:
        print(FS_model())
