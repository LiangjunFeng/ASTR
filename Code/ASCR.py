import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import copy
from scipy.spatial.distance import cdist
import heapq
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from noisy_regressor import Nosiy_NN_regressor, Semisupervised_Nosiy_NN_regressor

class ASCR():
    def __init__(self,T,s,k1,k2,D1,D2):
        self.T = T
        self.s = s
        self.k1 = k1 
        self.k2 = k2
        self.D1 = D1
        self.D2 = D2

    def k_neighbors(self,x,data,label,k,d):
        d_value = cdist(np.mat(x),data,metric=d).tolist()[0]
        index_list = list(map(d_value.index,heapq.nsmallest(k,d_value)))
        return data[index_list],label[index_list]
        
    def fit(self,traindata,trainlabel,udata):
        num1,seq1,fea1 = traindata.shape
        traindata = np.resize(traindata,(num1,seq1*fea1))
        num2,seq2,fea2 = udata.shape
        udata = np.resize(udata,(num2,seq2*fea2))
        
        traindata1 = copy.deepcopy(traindata)
        trainlabel1 = copy.deepcopy(trainlabel)
        
        traindata2 = copy.deepcopy(traindata)
        trainlabel2 = copy.deepcopy(trainlabel)
        
        udata,udata_pool,_,_ = train_test_split(udata,udata,test_size = self.s)
        
        h1 = KNeighborsRegressor(n_neighbors = self.k1, metric = self.D1)
        h2 = KNeighborsRegressor(n_neighbors = self.k2, metric = self.D2)
        
        h1.fit(traindata1,trainlabel1)
        h2.fit(traindata2,trainlabel2)

        traindata_list = [traindata1,traindata2]
        trainlabel_list = [trainlabel1,trainlabel2]        
        h_list = [h1,h2]
        k_list = [self.k1,self.k2]
        d_list = [self.D1,self.D2]
        
        for t in range(self.T):
            for j in [0,1]:
                ulabel_list = []
                sigma_list = []
                for i in range(udata_pool.shape[0]):
                    neighbor_data, neighbor_label = self.k_neighbors(udata_pool[i],traindata,trainlabel,k_list[j],d_list[j])
                    ulabel_list.append(h_list[j].predict(np.mat(udata_pool[i])))
                    new_h = KNeighborsRegressor(n_neighbors = k_list[j], metric = d_list[j])                    
                    new_h.fit(np.vstack([traindata_list[j],np.mat(udata_pool[i])]),np.vstack([trainlabel_list[j],h_list[j].predict(np.mat(udata_pool[i]))]))
                    
                    error_old = mean_squared_error(neighbor_label,h_list[j].predict(neighbor_data))
                    error_new = mean_squared_error(neighbor_label,new_h.predict(neighbor_data))
                    sigma_list.append(error_old-error_new)
                
                if max(sigma_list) > 0:
                    useful_index = sigma_list.index(max(sigma_list))
                    
                    traindata_list[1-j] = np.vstack([traindata_list[1-j],np.mat(udata_pool[useful_index])])
                    trainlabel_list[1-j] = np.vstack([trainlabel_list[1-j],np.mat(ulabel_list[useful_index])])
                    
                    udata_pool = np.delete(udata_pool,useful_index,axis=0)
            
            if udata_pool.shape[0] == self.s:
                break
            else:
                h1 = KNeighborsRegressor(n_neighbors = k_list[0], metric = d_list[0]) 
                h2 = KNeighborsRegressor(n_neighbors = k_list[1], metric = d_list[1]) 
                
                h1.fit(traindata_list[0],trainlabel_list[0])
                h2.fit(traindata_list[1],trainlabel_list[1])
                
                h_list = [h1,h2]
                
                udata = np.vstack([udata,udata_pool])
                udata,udata_pool,_,_ = train_test_split(udata,udata,test_size = self.s)
        
        self.regressor1 = Semisupervised_Nosiy_NN_regressor(node_list=[1024,256],n_variable=34,n_output=1,noise_type='norm',reg = 0.001,noise_scale = 1)
        self.regressor2 = Semisupervised_Nosiy_NN_regressor(node_list=[1024,256],n_variable=34,n_output=1,noise_type='ruili',reg = 0.1,noise_scale = 1)
   
        num11  = traindata_list[0].shape[0]
        traindata1 = np.resize(traindata1,(num11,seq1,fea1))
        num12  = traindata_list[1].shape[0]
        traindata2 = np.resize(traindata2,(num12,seq1,fea1))        
        
        udata = np.concatenate([udata,udata_pool],axis=0)
        num2  = udata.shape[0]
        udata = np.resize(udata,(num2,seq2,fea2))   
        
        print(traindata1.shape,udata.shape)
        print(traindata2.shape,udata.shape)
    
        self.regressor1.fit(traindata1,trainlabel_list[0],udata,batch_size=32,epochs=50,lr=1e-4)
        # self.regressor2.fit(traindata2,trainlabel_list[1],udata,batch_size=32,epochs=50,lr=1e-4)


    def predict(self,testdata):
        res1 = self.regressor1.predict(testdata)
        # res2 = self.regressor2.predict(testdata)
        
        # return (res1 + res2)/2
        return res1


               
#from scipy.io import loadmat
#from sklearn import preprocessing
#import os
#import warnings
#os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#warnings.filterwarnings("ignore")
#
#table = loadmat('/Users/zhuxiaoxiansheng/Desktop/论文/ATR/cigarette/original_data.mat')
#datante = table['datante']
#
#label_index = [1,4,6]
##1,4,6
#data_index = [0,2,3,19,5,21,7,8,9,10,11,12,13,14,15,16,17,18,20,22]
#
#label = np.mat(datante[:,1]).T
#mi = np.min(label)
#di = (np.max(label)-np.min(label))
#label = (label-min(label))/(max(label)-min(label))
#
#data = datante[:,data_index]
#data = preprocessing.MinMaxScaler().fit_transform(data) 
#
#traindata = data[0:100,:]
#trainlabel = label[0:100]
#
#testdata = data[100:250,:]
#testlabel = label[100:250,:]
#
#testlabel = testlabel*di+mi
#ulabeldata = data[250:]
#ulabel = label[250:]
#ulabel = ulabel*di+mi
#print(traindata.shape,trainlabel.shape,testdata.shape,testlabel.shape,ulabeldata.shape,ulabel.shape)
#
##=====================================================================================================   
#
#clf = Semisupervised_Nosiy_NN_regressor(node_list=[512,256],noise_type='uniform',n_variable=20,n_output=1,reg = 1e-7,noise_scale = 1)
#clf.fit(traindata,trainlabel,ulabeldata,batch_size=128,epochs=200,lr=1e-3)
#res = clf.predict(testdata)
#res = res*di+mi
#rmse = np.sqrt(mean_squared_error(res,testlabel))
#r2 = r2_score(res,testlabel)
#mae = mean_absolute_error(res,testlabel)
#print('rmse: {:.4f}'.format(rmse),'r2: {:.4f}'.format(r2),'mae: {:.4f}'.format(mae))
#
#
#
#clf = Semisupervised_Nosiy_NN_regressor(node_list=[512,256],noise_type='norm',n_variable=20,n_output=1,reg = 1e-7,noise_scale = 1)
#clf.fit(traindata,trainlabel,ulabeldata,batch_size=128,epochs=200,lr=1e-3)
#res = clf.predict(testdata)
#res = res*di+mi
#rmse = np.sqrt(mean_squared_error(res,testlabel))
#r2 = r2_score(res,testlabel)
#mae = mean_absolute_error(res,testlabel)
#print('rmse: {:.4f}'.format(rmse),'r2: {:.4f}'.format(r2),'mae: {:.4f}'.format(mae))
#
#
#
#clf = ASR_Coreg(T=100,s=200,k1=3,k2=5,D1='euclidean',D2='minkowski')  
#clf.fit(traindata,trainlabel,ulabeldata)   
#res = clf.predict(testdata)    
#                    
#res = res*di+mi
#rmse = np.sqrt(mean_squared_error(res,testlabel))
#r2 = r2_score(res,testlabel)
#mae = mean_absolute_error(res,testlabel)
#print('rmse: {:.4f}'.format(rmse),'r2: {:.4f}'.format(r2),'mae: {:.4f}'.format(mae))
#


















































