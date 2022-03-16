import numpy as np
from keras.layers import Dense
from keras.models import Model, Sequential
from keras.layers import RepeatVector, multiply, dot, Lambda, Reshape, Concatenate, LSTM,  Bidirectional, merge, Permute
import warnings
warnings.filterwarnings('ignore')
from sklearn.neural_network import BernoulliRBM
from scipy.linalg import orth
import warnings
warnings.filterwarnings("ignore")

class Deep_Believe_Network:
    def __init__(self,layer_list):
        self.layer_list = layer_list
        self.RBM_list = []
        self.weight_list = []
        self.batch_size = 0
        self.model = 0
    
    def pretraining(self,traindata,epochs1,batch_size):
        for i in range(len(self.layer_list)): 
            rbm_input = traindata
            for j in range(i): 
                rbm_input = self.RBM_list[j].transform(rbm_input)
            rbm = BernoulliRBM(n_components = self.layer_list[i], n_iter = epochs1, verbose = False)
            rbm.fit(rbm_input)
            self.RBM_list.append(rbm)
            if rbm.components_.shape[0] >= rbm.components_.shape[1]:
                self.weight_list.append(orth(rbm.components_).T)
            else:
                self.weight_list.append(orth(rbm.components_.T))
#            print()       
     
    def build_DBN(self,traindata,trainlabel):
        self.model = Sequential()
        i = 0
        for layer_nodes in self.layer_list:
            if i == 0:
                self.model.add(Dense(layer_nodes,
                                     input_shape=(traindata.shape[1],),
                                     activation='sigmoid',use_bias=False))
            else:
                self.model.add(Dense(layer_nodes,
                                     activation='sigmoid',use_bias=False))
            self.model.layers[i].set_weights([self.weight_list[i]])
            i += 1
        self.model.add(Dense(trainlabel.shape[1],activation='sigmoid'))
    
    def fit(self,traindata,trainlabel,batch_size,epochs1,epochs2):
        self.pretraining(traindata,epochs1,batch_size)
        self.build_DBN(traindata,trainlabel)
        self.model.compile(optimizer='adam',loss='mean_squared_error')
        self.model.fit(traindata,trainlabel,verbose=0,batch_size=batch_size,shuffle=True,epochs=epochs2)
        self.batch_size = batch_size
    
    def predict(self,testdata):
        return self.model.predict(testdata,batch_size=self.batch_size)
    
    def evaluate(self,testdata,testlabel):
        return self.model.evaluate(testdata,testlabel,batch_size=self.batch_size)
