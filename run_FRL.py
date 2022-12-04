import json
import numpy as np
import pandas as pd
from sklearn import preprocessing
import Representation_learning
from sklearn.neighbors import LocalOutlierFactor as LOF
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score,precision_score,recall_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import Net, DQN

print('Loading data ...')
dataset = pd.read_csv('data/breast_cancer_transcript_expression_with_label.csv')
print(f'Loaded {dataset.shape = }')
dataset = dataset.set_index('Unnamed: 0').T
print(f'Transpose {dataset.shape = }')
print('Data loaded.')

r, c = dataset.shape
array = dataset.values
print(f'{dataset.shape = }')
# Y = dataset.iloc[:,0]
# X = dataset.iloc[:,1:c]
X = dataset.iloc[:,0:(c-1)]
Y = dataset.iloc[:,(c-1)]
print(f'{X.shape = }, {Y.shape = }')
io = open("IGorderbreastcancer.txt","r")
order = np.array(json.load(io), dtype=np.int32)

print(f'{order.shape = }')
X = X.iloc[:, order]
X = X.iloc[:, 0:50]
print(f'{X.shape = }')

X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)
X_train = X_train.apply(pd.to_numeric) 
X_val = X_val.apply(pd.to_numeric) 
print(X_train.shape, X_val.shape, Y_train.shape, Y_val.shape)
print('train_test_split done ...')

print('Random forest and DQN initialising ...')
model = RandomForestClassifier(n_jobs=-1,n_estimators=100, random_state=0)
#model =DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, random_state=0)

N_feature = X_train.shape[1] # feature number
N_sample = X_train.shape[0] # feature length,i.e., sample number

BATCH_SIZE = 32
LR = 0.01
#EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100 # After how much time you refresh target network
MEMORY_CAPACITY = 400 # The size of experience replay buffer
EXPLORE_STEPS = 1000# How many exploration steps you'd like, should be larger than MEMORY_CAPACITY20
N_ACTIONS = 2
N_STATES = 480+N_feature

np.random.seed(0)
torch.manual_seed(0)    # reproducible

dqn = DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS)

Fstate = np.random.randint(2, size=N_feature)
while sum(Fstate) < 2:
    Fstate = np.random.randint(2, size=N_feature)

print('done.')

# Fstate = np.zeros(N_feature)
# Fstate[0] = 1
print('Pre-processing data ...')
X_selected = X_train.iloc[:, Fstate == 1]
X_array = np.array(X_selected)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
X_array = min_max_scaler.fit_transform(X_array)
print('done.')

print('Representation learning ...')
X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
s = Representation_learning.representation_training(X_tensor)
s = s.detach().numpy().reshape(-1)
position = np.arange(1, N_feature+1)
print(s.shape, position.shape)
import pandas as pd 
pd.DataFrame(s).to_csv("breast_cancer_representation_learning.csv")
print('done.')

# # load representation learning 
# df = pd.read_csv("breast_cancer_representation_learning.csv")
# s = np.array(df.to_dict('split')['data'])
# position = np.arange(1, N_feature+1)
# print(s.shape, position.shape)

print('One hot encoding data ...')
onehot_encoded = OneHotEncoder(sparse=False).fit_transform(position.reshape(-1, 1))
s = np.append(s, onehot_encoded[0])  
print(s.shape)  
print('done.')

result = []
T = N_feature
# dqn.EPSILON_STEP_SIZE = (dqn.EPSILON_MAX - dqn.EPSILON)/((EXPLORE_STEPS-1)*T)
for i in range(EXPLORE_STEPS):
    # print('='* 20)
    # print('i = {}/{}'.format(i+1, EXPLORE_STEPS), end='\r'))
    # print('{}/{}'.format(i+1, EXPLORE_STEPS), end='\r')

    t = i % T
    # print(f'{s.shape = }')
    Faction = dqn.choose_action(s)
    Fstate[t] = Faction
    if sum(Fstate) < 1:
        Faction = 1
        Fstate[t] = Faction

    # print('1')
    X_selected = X_train.iloc[:, Fstate == 1]
    X_array = np.array(X_selected)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_array = min_max_scaler.fit_transform(X_array)
    X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
    s_ = Representation_learning.representation_training(X_tensor)
    s_ = s_.detach().numpy().reshape(-1)
    if t == T-1:
        s_ = np.append(s_, onehot_encoded[0])
        # print(f'{s_.shape = }')
        # s_ = np.append(s_, 1)
    else:
        s_ = np.append(s_, onehot_encoded[t + 1])
        # print(f'{s_.shape = }')
        # s_ = np.append(s_, t + 2)

    # print('2')
    model.fit(X_train.iloc[:, Fstate == 1], Y_train)
    accuracy = model.score(X_val.iloc[:, Fstate == 1], Y_val)
    Y_pred = model.predict(X_val.iloc[:, Fstate == 1])
    macroF1 = f1_score(Y_val, Y_pred, average='macro')
    precision = precision_score(Y_val, Y_pred, average='macro')
    recall = recall_score(Y_val, Y_pred, average='macro')
    # print('3')
    corr = X_val.corr().abs()
    try:
        ave_corr = (corr.iloc[:, t].sum())/ (X_val.shape[1])
    except:
        # print(t)
        continue

    # ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
    r = (accuracy - ave_corr)
    # r = accuracy
    # print('4')
    dqn.store_transition(s, Faction, r, s_)

    if dqn.memory_counter > MEMORY_CAPACITY:
        dqn.learn()
    print('{}/{}: r = {:.2f}, accuracy = {:.2f}%'.format(i+1, EXPLORE_STEPS, r, accuracy), end='\r')
    s = s_
    result.append([accuracy, precision, recall, macroF1, Fstate])
    # print('5')

print('done')    

output =[]
name = []
name.append("result types")
output.append("over_Gly")

max_accuracy = 0
optimal_set = []
for i in range(len(result)):
    name.append("Accuracy of the {}-th explore step".format(i))
    output.append(result[i][0])

    if result[i][0] > max_accuracy:
        max_accuracy = result[i][0]
        optimal_set = result[i][4]
        Mmacro_f1 = result[i][3]
        Mprecision = result[i][1]
        Mrecall = result[i][2]

print("The maximum accuracy is: {}, the optimal selection for each feature is:{}".format(max_accuracy, optimal_set))

name.append("feature subset")
output.append(optimal_set)
name.append("max_accuracy")
output.append(max_accuracy)
name.append("precision")
output.append(Mprecision)
name.append("recall_RF")
output.append(Mrecall)
name.append("macro_f1")
output.append(Mmacro_f1)


out = dict(zip(name, output))
out = pd.DataFrame([out])
out.to_csv('data/breast_cancer_result_predictor.csv', mode='a')