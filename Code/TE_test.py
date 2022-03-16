import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.io as scio    
from sklearn.neighbors import KNeighborsRegressor   
from sklearn.svm import SVR
import lightgbm as lgb
from Coreg import Coreg
from DBN import Deep_Believe_Network
from sklearn.metrics import mean_absolute_error,mean_squared_error
from noisy_regressor import lstm_network,lstm_l2_network,Nosiy_NN_regressor,Semisupervised_Nosiy_NN_regressor
from ASCR import ASCR
import warnings
warnings.filterwarnings("ignore")

def make_seqdata(data,step=10):
    num = data.shape[0]
    data_list = []
    for i in range(num):
        seq_data = np.zeros((1,step,data.shape[1]))
        for k in range(step):
            if i-step+k+1 <= 0:
                seq_data[0,k,:] = data[0,:]
            else: 
                seq_data[0,k,:] = data[i-step+k+1,:]
        data_list.append(seq_data)  
    return np.concatenate(data_list,axis=0)
          

data1 = scio.loadmat('/Volumes/文档/数据集/TE_process/TE_mat_data/d00.mat')['data'].T
data2 = scio.loadmat('/Volumes/文档/数据集/TE_process/TE_mat_data/d00_te.mat')['data'].T

dataall = np.row_stack([data1,data2])

label = dataall[:,35]
data = np.delete(dataall,list(range(34,53)),axis=1)


np.random.seed(2019)
train_index = np.random.choice(1460,100,replace = False)
test_index = np.random.choice(list(set(list(np.arange(1460)))-set(train_index)),960,replace = False)
u_index = list(set(list(np.arange(1460)))-set(train_index)-set(test_index))

mi = np.min(label)
di = (np.max(label)-np.min(label))
label = (label-min(label))/(max(label)-min(label))
data = preprocessing.MinMaxScaler().fit_transform(data) 


traindata = data[train_index,:]
trainlabel = np.mat(label[train_index]).T

testdata = data[test_index,:]
testlabel = np.mat(label[test_index]).T
testlabel = testlabel*di+mi

udata = data[u_index,:]


result = pd.DataFrame(testlabel)

#=============================================
print('Coreg')
clf = Coreg(T=20,s=100,k1=3,k2=5,D1='euclidean',D2='minkowski')  
clf.fit(traindata,trainlabel,udata)   
res = clf.predict(testdata)    
res = res*di+mi 
rmse = np.sqrt(mean_squared_error(res,testlabel))
mae = mean_absolute_error(res,testlabel)
print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))
result['Coreg_res'] = res



print('DBN')
# clf =  Deep_Believe_Network([1024,128])
# clf.fit(traindata,trainlabel,64,20,100)
clf = SVR()
clf.fit(traindata,trainlabel)
res = clf.predict(testdata) 
res = res*di+mi
rmse = np.sqrt(mean_squared_error(res,testlabel))
mae = mean_absolute_error(res,testlabel)
print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))
result['SSDBN_res'] = res

data = make_seqdata(data,step=10)
traindata = data[train_index]
testdata = data[test_index]
udata = data[u_index]
print(traindata.shape,testdata.shape,udata.shape)

print('ASCR')
clf = ASCR(T=20,s=100,k1=3,k2=5,D1='euclidean',D2='minkowski')  
clf.fit(traindata,trainlabel,udata)   
res = clf.predict(testdata)   
res = res*di+mi
rmse = np.sqrt(mean_squared_error(res,testlabel))
mae = mean_absolute_error(res,testlabel)
print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))

result['ASCR_res'] = res

result.to_excel('/Users/zhuxiaoxiansheng/Desktop/componentB.xlsx')


# print('LSTM')
# clf = lstm_network(node_list=[1024,256],noise_type='norm',n_variable=34,n_output=1,reg = 1e-1,noise_scale = 1)
# clf.fit(traindata,trainlabel,batch_size=32,epochs=50,lr=1e-4)
# res = clf.predict(testdata)
# res = res*di+mi
# rmse = np.sqrt(mean_squared_error(res,testlabel))
# mae = mean_absolute_error(res,testlabel)
# print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))


# print('LSTM_l2')
# clf = lstm_l2_network(node_list=[1024,256],noise_type='norm',n_variable=34,n_output=1,reg = 1e-1,noise_scale = 1)
# clf.fit(traindata,trainlabel,batch_size=32,epochs=50,lr=1e-4)
# res = clf.predict(testdata)
# res = res*di+mi
# rmse = np.sqrt(mean_squared_error(res,testlabel))
# mae = mean_absolute_error(res,testlabel)
# print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))


# print('LSTM_ASR')
# clf = Nosiy_NN_regressor(node_list=[1024,256],noise_type='norm',n_variable=34,n_output=1,reg = 1e-1,noise_scale = 1)
# clf.fit(traindata,trainlabel,batch_size=32,epochs=50,lr=1e-4)
# res3 = clf.predict(testdata)
# res3 = res3*di+mi
# rmse = np.sqrt(mean_squared_error(res3,testlabel))
# mae = mean_absolute_error(res3,testlabel)
# print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))


# print('LSTM_ASR_U')
# clf = Semisupervised_Nosiy_NN_regressor(node_list=[1024,256],noise_type='ruili',n_variable=34,n_output=1,reg = 1e-1,noise_scale = 1)
# clf.fit(traindata,trainlabel,udata,batch_size=32,epochs=55,lr=1e-4)
# res = clf.predict(testdata)
# res = res*di+mi
# rmse = np.sqrt(mean_squared_error(res,testlabel))
# mae = mean_absolute_error(res,testlabel)
# print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))























