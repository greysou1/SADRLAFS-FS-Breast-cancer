# Data_Preprocessing
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
from sklearn.metrics import accuracy_score, f1_score,precision_score,recall_score
# from sklearn.naive_bayes import GaussianNB
# from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
# from xgboost.sklearn import XGBClassifier
# import matplotlib.pyplot as plt


data_folder = 'data/'
# data_folder = 'C:/Users/12873/PycharmProjects/FSRL/data/'
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
X_train, X_val, Y_train, Y_val = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)
#
# X_train.to_csv(data_folder + 'X_trainBank.csv')
# temp = {'ID':Y_train.index, 'class': Y_train.values}
# Y_train = pd.DataFrame(temp)
# Y_train.to_csv(data_folder + 'Y_trainBank.csv')

model = RandomForestClassifier(n_jobs=-1,n_estimators=100, random_state=0)
#model =DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, random_state=0)

# Forder = loadmat("IGorderGlycation.mat")
# order = Forder["Forder"].squeeze(0)-1
# X_train = X_train.iloc[:, order]
# X_val = X_val.iloc[:, order]

N_feature = X_train.shape[1] # feature number
N_sample = X_train.shape[0] # feature length,i.e., sample number


# DQN
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
BATCH_SIZE = 32
LR = 0.01
#EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100 # After how much time you refresh target network
MEMORY_CAPACITY = 400 # The size of experience replay buffer
EXPLORE_STEPS = 1000# How many exploration steps you'd like, should be larger than MEMORY_CAPACITY20
N_ACTIONS = 2
N_STATES = 480+N_feature
#

class Net(nn.Module):

    def __init__(self, N_STATES, N_ACTIONS):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 100)
        self.fc1.weight.data.normal_(0, 0.1)  # initialization, set seed to ensure the same result
        self.out = nn.Linear(100, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)  # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_value = self.out(x)
        return action_value


class DQN(object):

    def __init__(self, N_STATES, N_ACTIONS):
        self.eval_net, self.target_net = Net(N_STATES, N_ACTIONS), Net(N_STATES, N_ACTIONS)

        self.learn_step_counter = 0
        self.EPSILON = 0.9
        # self.EPSILON_MAX = 1.0
        # self.EPSILON_STEP_SIZE = 0.001
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        if np.random.uniform() < self.EPSILON:
            action_value = self.eval_net.forward(x)
            action = torch.max(action_value, 1)[1].data.numpy()
            action = action[0]

        else:
            action = np.random.randint(0, N_ACTIONS)
            # if self.EPSILON <= self.EPSILON_MAX:
            #     self.EPSILON = self.EPSILON + self.EPSILON_STEP_SIZE
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % MEMORY_CAPACITY  # If full, restart from the beginning
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES + 1])
        b_r = torch.FloatTensor(b_memory[:, N_STATES + 1:N_STATES + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


np.random.seed(0)
torch.manual_seed(0)    # reproducible



dqn = DQN(N_STATES=N_STATES, N_ACTIONS=N_ACTIONS)
# # The element in the result list consists two parts,
# # i.e., accuracy and the action list (action 1 means selecting corresponding feature, 0 means deselection).
#
Fstate = np.random.randint(2, size=N_feature)
while sum(Fstate) < 2:
    Fstate = np.random.randint(2, size=N_feature)

# Fstate = np.zeros(N_feature)
# Fstate[0] = 1
X_selected = X_train.iloc[:, Fstate == 1]
X_array = np.array(X_selected)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
X_array = min_max_scaler.fit_transform(X_array)
X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
s = Representation_learning.representation_training(X_tensor)
s = s.detach().numpy().reshape(-1)
position = np.arange(1, N_feature+1)
onehot_encoded = OneHotEncoder(sparse=False).fit_transform(position.reshape(-1, 1))
s = np.append(s, onehot_encoded[0])
# s = np.append(s,1)

result = []
T = N_feature
# dqn.EPSILON_STEP_SIZE = (dqn.EPSILON_MAX - dqn.EPSILON)/((EXPLORE_STEPS-1)*T)
for i in range(EXPLORE_STEPS):
    t = i % T
    Faction = dqn.choose_action(s)
    Fstate[t] = Faction
    if sum(Fstate) < 1:
        Faction = 1
        Fstate[t] = Faction

    X_selected = X_train.iloc[:, Fstate == 1]
    X_array = np.array(X_selected)
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    X_array = min_max_scaler.fit_transform(X_array)
    X_tensor = torch.FloatTensor(X_array).unsqueeze(0).unsqueeze(0)
    s_ = Representation_learning.representation_training(X_tensor)
    s_ = s_.detach().numpy().reshape(-1)
    if t == T-1:
        s_ = np.append(s_, onehot_encoded[0])
        # s_ = np.append(s_, 1)
    else:
        s_ = np.append(s_, onehot_encoded[t + 1])
        # s_ = np.append(s_, t + 2)

    model.fit(X_train.iloc[:, Fstate == 1], Y_train)
    accuracy = model.score(X_val.iloc[:, Fstate == 1], Y_val)
    Y_pred = model.predict(X_val.iloc[:, Fstate == 1])
    macroF1 = f1_score(Y_val, Y_pred, average='macro')
    precision = precision_score(Y_val, Y_pred, average='macro')
    recall = recall_score(Y_val, Y_pred, average='macro')

    corr = X_val.corr().abs()
    ave_corr = (corr.iloc[:, t].sum())/ (X_val.shape[1])

    # ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
    r = (accuracy - ave_corr)
    # r = accuracy

    dqn.store_transition(s, Faction, r, s_)

    if dqn.memory_counter > MEMORY_CAPACITY:
        dqn.learn()
    print(f'{i+1}/{EXPLORE_STEPS}: {r = }, {accuracy = }', end='\r')
    s = s_
    result.append([accuracy, precision, recall, macroF1, Fstate])

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
out.to_csv(data_folder + 'result_predictor.csv', mode='a')
