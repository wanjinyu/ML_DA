# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 17:04:08 2021

@author: wanjinyu
"""

import numpy as np
import sklearn.svm as svm
from sklearn.model_selection import train_test_split,cross_val_score
import scipy.io as sio
import time
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from keras.layers import Dense,LSTM,Dropout
from keras.models import Sequential
from keras import optimizers
from sklearn.gaussian_process import GaussianProcessClassifier

class MLmodels(object):
    def __init__(self):
        pass

    def initialTSVM(self, C=1.5, kernel='linear', Cl = 1.5, Cu=0.0001):
        '''
        C: penalty coefficient
        kernel: kernel of svm
        '''
        self.C = C
        self.Cl, self.Cu = Cl, Cu
        self.kernel = kernel
        self.TSVM = svm.SVC(C=self.C, kernel=self.kernel)
        
    def initialSVM(self, C=1.5, kernel='linear'):
        '''
        C: penalty coefficient
        kernel: kernel of svm
        '''
        self.C = C
        self.kernel = kernel
        self.SVM = svm.SVC(C=1.5, kernel=self.kernel)
    
    def initialRF(self, initial_state=0):
        self.rfc = RandomForestClassifier(random_state=initial_state)
        
    def initialGP(self, kernel, initial_state=0):
        self.gpc = GaussianProcessClassifier(kernel=kernel,random_state=initial_state)
        
    def initialNN(self, struct, Afuc = 'tanh'):
        '''
        sturct = [L1, L2, L3, ...]: numeber of nuerons in each layer
        Afuc: activation function of hidden neurons
        '''
        NL = len(struct)
        self.NN = Sequential()
        for i in range(NL):
            if i==0:
                self.NN.add(Dense(struct[0], input_dim = struct[0], activation = Afuc))
            else:
                self.NN.add(Dense(struct[i], activation = Afuc))
        self.NN.add(Dense(2, activation = 'softmax'))
        self.NN.compile(optimizer = optimizers.adam(lr = 0.001),loss='mse',metrics = ['mse'])
        
    def trainTSVM(self, X1, Y1, X2):
        '''
        X1: Labeled training data
        Y1: Labels of X1
        X2: Unlabeled training data
        '''
        max_step = 50
        N = len(X1) + len(X2)
        sample_weight = np.ones(N)
        sample_weight[len(X1):] = self.Cu
        self.TSVM.fit(X1, np.ravel(Y1))
        Y2 = self.TSVM.predict(X2)
        Y2 = np.expand_dims(Y2, 1)
        X2_id = np.arange(len(X2))
        X3 = np.vstack([X1, X2])
        Y3 = np.vstack([Y1, Y2])     
        step_1 = 1
        while self.Cu < self.Cl:
#            print('step1: '+str(step_1))
            step_1 = step_1+1
            step_2 = 1
            self.TSVM.fit(X3, np.ravel(Y3), sample_weight=sample_weight)
            if step_1 > max_step:
                break
            while True:
                Y2_d = self.TSVM.decision_function(X2)    # linear: w^Tx + b
                Y2 = Y2.reshape(-1)
                epsilon = 1 - Y2 * Y2_d   # calculate function margin
                positive_set, positive_id = epsilon[Y2 > 0], X2_id[Y2 > 0]
                negative_set, negative_id = epsilon[Y2 < 0], X2_id[Y2 < 0]
                if len(positive_set) >0 and len(negative_set)>0:
                    positive_max_id = positive_id[np.argmax(positive_set)]
                    a = epsilon[positive_max_id]
                    negative_max_id = negative_id[np.argmax(negative_set)]
                    b = epsilon[negative_max_id]
                    if a > 0 and b > 0 and a + b > 2.0:
                        Y2[positive_max_id] = Y2[positive_max_id] * -1
                        Y2[negative_max_id] = Y2[negative_max_id] * -1
                        Y2 = np.expand_dims(Y2, 1)
                        Y3 = np.vstack([Y1, Y2])
                        self.TSVM.fit(X3, np.ravel(Y3), sample_weight=sample_weight)
#                        print('step2: '+str(step_2))
                        step_2 = step_2+1
                        if step_2 > max_step:
                            break
                    else:
                        break
                else:
                    break

            self.Cu = min(2*self.Cu, self.Cl)
            sample_weight[len(X1):] = self.Cu
            
    def trainSVM(self, X1, Y1):
        '''
        X1: Labeled training data
        Y1: Labels of X1
        '''
        self.SVM.fit(X1, np.ravel(Y1))
        
    def trainRF(self, X1, Y1):
        '''
        X1: Labeled training data
        Y1: Labels of X1
        '''
        self.rfc.fit(X1,np.ravel(Y1))
        
    def trainGP(self, X1, Y1):
        '''
        X1: Labeled training data
        Y1: Labels of X1
        '''
        self.gpc.fit(X1,np.ravel(Y1))
        
    def trainNN(self, X1, Y1, ephochs = 3000):
        '''
        X1: Labeled training data
        Y1: Labels of X1
        '''
        self.NNhistory = self.NN.fit(X1, Y1, epochs = ephochs)

    def predict(self, model, X):
        '''
        Feed X and predict Y
        '''
        return model.predict(X)
