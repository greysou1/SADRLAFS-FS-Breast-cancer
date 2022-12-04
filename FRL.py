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

class FRL:
    def __init__(self, X_train, Y_train, 
                BATCH_SIZE=32, 
                LR=0.01, 
                EPSILON = 0.9, 
                GAMMA = 0.9, 
                TARGET_REPLACE_ITER = 100,
                MEMORY_CAPACITY = 400,
                EXPLORE_STEPS = 1000,
                N_ACTIONS = 2,
                TOP_N=50):
        self.X_train = X_train
        self.Y_train = Y_train
        self.N_feature = self.X_train.shape[1] # feature number
        self.N_sample = self.X_train.shape[0] # feature length,i.e., sample number                
        self.TOP_N = TOP_N
        self.BATCH_SIZE = 32
        self.LR = 0.01
        self.EPSILON = 0.9
        self.GAMMA = 0.9
        self.TARGET_REPLACE_ITER = 100 # After how much time you refresh target network
        self.MEMORY_CAPACITY = 400 # The size of experience replay buffer
        self.EXPLORE_STEPS = 1000 # How many exploration steps you'd like, should be larger than MEMORY_CAPACITY20
        self.N_ACTIONS = 2
        self.N_STATES = 480+self.N_feature

        print('Random forest and DQN initialising ...')
        self.model = RandomForestClassifier(n_jobs=-1,n_estimators=100, random_state=0)
        #model =DecisionTreeClassifier(criterion='entropy', min_samples_leaf=3, random_state=0)

        

        
        np.random.seed(0)
        torch.manual_seed(0)    # reproducible

        self.dqn = DQN(N_STATES=self.N_STATES, N_ACTIONS=N_ACTIONS)

        self.Fstate = np.random.randint(2, size=self.N_feature)
        while sum(self.Fstate) < 2:
            self.Fstate = np.random.randint(2, size=self.N_feature)

        print('done.')

        # Fstate = np.zeros(N_feature)
        # Fstate[0] = 1
        print('Pre-processing data ...')
        self.X_selected = self.X_train.iloc[:, self.Fstate == 1]
        self.X_array = np.array(self.X_selected)
        self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
        self.X_array = self.min_max_scaler.fit_transform(self.X_array)
        print('done.')

        print('Representation learning ...')
        self.X_tensor = torch.FloatTensor(self.X_array).unsqueeze(0).unsqueeze(0)
        self.s = Representation_learning.representation_training(self.X_tensor)
        self.s = self.s.detach().numpy().reshape(-1)
        self.position = np.arange(1, self.N_feature+1)
        print(self.s.shape, self.position.shape)
        import pandas as pd 
        pd.DataFrame(self.s).to_csv("breast_cancer_representation_learning.csv")
        print('done.')

        print('One hot encoding data ...')
        self.onehot_encoded = OneHotEncoder(sparse=False).fit_transform(self.position.reshape(-1, 1))
        self.s = np.append(self.s, self.onehot_encoded[0])  
        print(self.s.shape)  
        print('done.')

    def fit(self):
        result = []
        T = self.N_feature
        # dqn.EPSILON_STEP_SIZE = (dqn.EPSILON_MAX - dqn.EPSILON)/((EXPLORE_STEPS-1)*T)
        for i in range(self.EXPLORE_STEPS):
            # print('='* 20)
            # print('i = {}/{}'.format(i+1, EXPLORE_STEPS), end='\r'))
            # print('{}/{}'.format(i+1, EXPLORE_STEPS), end='\r')

            t = i % T
            # print(f'{s.shape = }')
            self.Faction = self.dqn.choose_action(s)
            self.self.Fstate[t] = self.Faction
            if sum(self.Fstate) < 1:
                self.Faction = 1
                self.Fstate[t] = self.Faction

            # print('1')
            self.X_selected = self.X_train.iloc[:, self.Fstate == 1]
            self.X_array = np.array(self.X_selected)
            self.min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
            self.X_array = self.min_max_scaler.fit_transform(self.X_array)
            self.X_tensor = torch.FloatTensor(self.X_array).unsqueeze(0).unsqueeze(0)
            self.s_ = Representation_learning.representation_training(self.X_tensor)
            self.s_ = self.s_.detach().numpy().reshape(-1)
            if t == T-1:
                self.s_ = np.append(self.s_, self.onehot_encoded[0])
                # print(f'{s_.shape = }')
                # s_ = np.append(s_, 1)
            else:
                self.s_ = np.append(self.s_, self.onehot_encoded[t + 1])
                # print(f'{s_.shape = }')
                # s_ = np.append(s_, t + 2)

            # print('2')
            self.model.fit(self.X_train.iloc[:, self.Fstate == 1], self.Y_train)
            accuracy = self.model.score(self.X_val.iloc[:, self.Fstate == 1], self.Y_val)
            self.Y_pred = self.model.predict(self.X_val.iloc[:, self.Fstate == 1])
            macroF1 = f1_score(self.Y_val, self.Y_pred, average='macro')
            precision = precision_score(self.Y_val, self.Y_pred, average='macro')
            recall = recall_score(self.Y_val, self.Y_pred, average='macro')
            # print('3')
            corr = self.X_val.corr().abs()
            try:
                ave_corr = (corr.iloc[:, t].sum())/ (self.X_val.shape[1])
            except:
                # print(t)
                continue

            # ave_corr = X_val.corr().abs().sum().sum() / (X_val.shape[0] * X_val.shape[1])
            r = (accuracy - ave_corr)
            # r = accuracy
            # print('4')
            self.dqn.store_transition(s, self.Faction, r, s_)

            if self.dqn.memory_counter > self.MEMORY_CAPACITY:
                self.dqn.learn()
            print('{}/{}: r = {:.2f}, accuracy = {:.2f}%'.format(i+1, self.EXPLORE_STEPS, r, accuracy), end='\r')
            self.vs = self.s_
            result.append([accuracy, precision, recall, macroF1, self.Fstate])
            # print('5')

        print('done')    
        