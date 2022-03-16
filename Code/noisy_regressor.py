from torch import nn
import torch
from torch.nn import Linear, ReLU, Sequential, Sigmoid, Softmax
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
import torch.optim as optim
from torch.optim import lr_scheduler
import warnings
warnings.filterwarnings('ignore')


class MyDataset(Dataset):
    def __init__(self,data,label):
        self.data = data
        self.label = label

    def __getitem__(self,index):
        return self.data[index],self.label[index]
    
    def __len__(self):
        return self.data.shape[0]
    

class Nosiy_NN(nn.Module):
    def __init__(self,node_list,n_variable,n_output,noise_type='norm',noise_scale = 1):
        super(Nosiy_NN, self).__init__()
        self.noise_scale = noise_scale
        self.noise_type = noise_type
        
        self.layer_list = []
        for i in range(len(node_list)):
            if i == 0:
                self.layer_list.append(Sequential(nn.LSTM(n_variable,node_list[0],batch_first=True))) 
            else:
                self.layer_list.append(Sequential(nn.LSTM(node_list[i-1],node_list[i],batch_first=True)))
        self.layer_list.append(Sequential(Linear(node_list[i],n_output),ReLU())) 
        self.layer_list = nn.ModuleList(self.layer_list)
        

    def forward(self,traindata):
        if self.noise_type == 'norm':
            noise = torch.from_numpy(self.noise_scale*np.random.normal(loc = 0, scale = 1, size = traindata.shape))
            if torch.cuda.is_available():
                traindata = traindata.cuda()
                noise = noise.cuda()
            noisy_traindata = traindata.float() + noise.float()
        elif self.noise_type == 'uniform':
            noise = torch.from_numpy(self.noise_scale*(np.random.random(traindata.shape)))
            if torch.cuda.is_available():
                traindata = traindata.cuda()
                noise = noise.cuda()
            noisy_traindata = traindata.float() + noise.float()
        elif self.noise_type == 'ruili':
            noise = torch.from_numpy(self.noise_scale*np.sqrt((-1*np.log(1-np.random.random(traindata.shape)))))
            if torch.cuda.is_available():
                traindata = traindata.cuda()
                noise = noise.cuda()
            noisy_traindata = traindata.float() + noise.float()    
        
        noisy_traindata = np.clip(noisy_traindata,0,1)        

        for i in range(len(self.layer_list)):
            if torch.cuda.is_available():
                traindata = traindata.cuda()
                noisy_traindata = noisy_traindata.cuda()
                
            if i == len(self.layer_list) - 1:
                output = self.layer_list[i](traindata[:,-1,:].squeeze().float())
                noisy_output = self.layer_list[i](noisy_traindata[:,-1,:].squeeze().float())              
            else:
                traindata, _  = self.layer_list[i](traindata.float())
                noisy_traindata, _ = self.layer_list[i](noisy_traindata.float())                
        
        return output, noisy_output


class lstm_network():
        def __init__(self,node_list,n_variable,n_output,noise_type='norm',reg = 0,noise_scale = 1):
            self.reg = reg
            self.model = Nosiy_NN(node_list,n_variable,n_output,noise_type=noise_type,noise_scale=noise_scale)
        
        def fit(self,traindata,trainlabel,batch_size=128,epochs=200,lr=1e-3):
            traindata = torch.from_numpy(traindata)
            trainlabel = torch.from_numpy(trainlabel)
    
            self.model.train()
            if torch.cuda.is_available():
                self.model.cuda()
                traindata.cuda()
                trainlabel.cuda()
            
            train_dataset = MyDataset(traindata,trainlabel)
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)  

            optimizer = optim.Adam(self.model.parameters(),lr=lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            loss1 = nn.MSELoss(size_average=False)
            
            for epoch in range(epochs):
                acc_loss = 0

                for item in train_dataloader:
                    batch_data, batch_label = item[0], item[1]
                    if torch.cuda.is_available():
                        batch_data = batch_data.cuda()
                        batch_label = batch_label.cuda()
                    optimizer.zero_grad()
                    output, noisy_output = self.model(batch_data.float())
            
                    train_loss1 = loss1(output,batch_label.float())    
                    train_loss = train_loss1 
                    
                    train_loss.backward()
                    optimizer.step()    
                    acc_loss += train_loss

                scheduler.step() 
                print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.8f}'.format(acc_loss.item()))
            print("Optimization Finished!")            
        
        def predict(self,testdata):
            testdata = torch.from_numpy(testdata)
            
            self.model.eval()
            if torch.cuda.is_available():
                testdata = testdata.cuda()

                
            test_dataset = MyDataset(testdata,np.zeros((testdata.shape[0],1)))
            test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)                 
            
            res = []
            for item in test_dataloader:
                batch_data, _ = item[0], item[1]

                if torch.cuda.is_available():
                    batch_data = batch_data.cuda()
                output, _ = self.model(batch_data)
                res.append(output.detach().cpu().numpy())
            res = np.row_stack(res)
            return res




class lstm_l2_network():
        def __init__(self,node_list,n_variable,n_output,noise_type='norm',reg = 0,noise_scale = 1):
            self.reg = reg
            self.model = Nosiy_NN(node_list,n_variable,n_output,noise_type=noise_type,noise_scale=noise_scale)
        
        def fit(self,traindata,trainlabel,batch_size=128,epochs=200,lr=1e-3):
            traindata = torch.from_numpy(traindata)
            trainlabel = torch.from_numpy(trainlabel)
    
            self.model.train()
            if torch.cuda.is_available():
                self.model.cuda()
                traindata.cuda()
                trainlabel.cuda()
            
            train_dataset = MyDataset(traindata,trainlabel)
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)  

            optimizer = optim.Adam(self.model.parameters(),lr=lr,weight_decay=0.0001)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            loss1 = nn.MSELoss(size_average=False)
            
            for epoch in range(epochs):
                acc_loss = 0

                for item in train_dataloader:
                    batch_data, batch_label = item[0], item[1]
                    if torch.cuda.is_available():
                        batch_data = batch_data.cuda()
                        batch_label = batch_label.cuda()
                    optimizer.zero_grad()
                    output, noisy_output = self.model(batch_data.float())
            
                    train_loss1 = loss1(output,batch_label.float())    
                    train_loss = train_loss1 
                    
                    train_loss.backward()
                    optimizer.step()    
                    acc_loss += train_loss

                scheduler.step() 
                print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.8f}'.format(acc_loss.item()))
            print("Optimization Finished!")            
        
        def predict(self,testdata):
            testdata = torch.from_numpy(testdata)
            
            self.model.eval()
            if torch.cuda.is_available():
                testdata = testdata.cuda()

                
            test_dataset = MyDataset(testdata,np.zeros((testdata.shape[0],1)))
            test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)                 
            
            res = []
            for item in test_dataloader:
                batch_data, _ = item[0], item[1]

                if torch.cuda.is_available():
                    batch_data = batch_data.cuda()
                output, _ = self.model(batch_data)
                res.append(output.detach().cpu().numpy())
            res = np.row_stack(res)
            return res

class Nosiy_NN_regressor():
        def __init__(self,node_list,n_variable,n_output,noise_type='norm',reg = 0.1,noise_scale = 1):
            self.reg = reg
            self.model = Nosiy_NN(node_list,n_variable,n_output,noise_type=noise_type,noise_scale=noise_scale)
        
        def fit(self,traindata,trainlabel,batch_size=128,epochs=200,lr=1e-3):
            traindata = torch.from_numpy(traindata)
            trainlabel = torch.from_numpy(trainlabel)
    
            self.model.train()
            if torch.cuda.is_available():
                self.model.cuda()
                traindata.cuda()
                trainlabel.cuda()
            
            train_dataset = MyDataset(traindata,trainlabel)
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)  

            optimizer = optim.Adam(self.model.parameters(),lr=lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            loss1 = nn.MSELoss(size_average=False)
            loss2 = nn.MSELoss(size_average=False)

            
            for epoch in range(epochs):
                acc_loss = 0
                acc_loss1 = 0
                acc_loss2 = 0
                for item in train_dataloader:
                    batch_data, batch_label = item[0], item[1]
                    if torch.cuda.is_available():
                        batch_data = batch_data.cuda()
                        batch_label = batch_label.cuda()
                    optimizer.zero_grad()
                    output, noisy_output = self.model(batch_data.float())
            
                    train_loss1 = loss1(output,batch_label.float()) 
                    train_loss2 = loss2(output,noisy_output)      
                    train_loss = train_loss1 + self.reg*train_loss2
                    
                    train_loss.backward()
                    optimizer.step()    
                    acc_loss += train_loss
                    
                    acc_loss1 += train_loss1
                    acc_loss2 += train_loss2
                    
                scheduler.step() 
                print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.8f}'.format(acc_loss.item()),'loss_fit: {:.8f}'.format(acc_loss1.item()),'loss_reg: {:.8f}'.format(acc_loss2.item()))
            print("Optimization Finished!")            
        
        def predict(self,testdata):
            testdata = torch.from_numpy(testdata)
            
            self.model.eval()
            if torch.cuda.is_available():
                testdata = testdata.cuda()

                
            test_dataset = MyDataset(testdata,np.zeros((testdata.shape[0],1)))
            test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)                 
            
            res = []
            for item in test_dataloader:
                batch_data, _ = item[0], item[1]

                if torch.cuda.is_available():
                    batch_data = batch_data.cuda()
                output, _ = self.model(batch_data)
                res.append(output.detach().cpu().numpy())
            res = np.row_stack(res)
            return res
                

class Semisupervised_Nosiy_NN_regressor():
        def __init__(self,node_list,n_variable,n_output,noise_type='norm',reg = 1e-7,noise_scale = 1):
            self.reg = reg
            self.model = Nosiy_NN(node_list,n_variable,n_output,noise_type=noise_type,noise_scale=noise_scale)
        
        def fit(self,traindata,trainlabel,udata,batch_size=64,epochs=200,lr=1e-3):
            traindata = torch.from_numpy(traindata)
            trainlabel = torch.from_numpy(trainlabel)
            udata = torch.from_numpy(udata)

            self.model.train()
            if torch.cuda.is_available():
                self.model.cuda()
                traindata.cuda()
                trainlabel.cuda()
                udata.cuda()
 
            train_dataset = MyDataset(traindata,trainlabel)
            train_dataloader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)  
            
            u_dataset = MyDataset(udata,np.zeros((udata.shape[0],1)))
            u_dataloader = DataLoader(u_dataset,batch_size=batch_size,shuffle=True)  


            optimizer = optim.Adam(self.model.parameters(),lr=lr)
            scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
            loss1 = nn.MSELoss(size_average=False)
            loss2 = nn.MSELoss(size_average=False)
            
            for epoch in range(epochs):
                acc_loss = 0
                acc_loss1 = 0
                acc_loss2 = 0
                
                for item, u_item in zip(train_dataloader,u_dataloader):
                    batch_data, batch_label = item[0], item[1]
                    batch_udata, _ = u_item[0], u_item[1]
                    if torch.cuda.is_available():
                        batch_data = batch_data.cuda()
                        batch_label = batch_label.cuda()
                        batch_udata = batch_udata.cuda()
                    optimizer.zero_grad()
                    output, noisy_output = self.model(batch_data.float())
                    uoutput, noisy_uoutput = self.model(batch_udata.float())
            
                    train_loss1 = loss1(output,batch_label.float()) 
                    train_loss2 = loss2(torch.cat([output,uoutput],dim=0),torch.cat([noisy_output,noisy_uoutput],dim=0))      
                    train_loss = train_loss1 + self.reg*train_loss2
                    
                    train_loss.backward()
                    optimizer.step()    
                    acc_loss += train_loss
                    
                    acc_loss1 += train_loss1
                    acc_loss2 += train_loss2
                    
                scheduler.step() 
                print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.8f}'.format(acc_loss.item()),'loss_fit: {:.8f}'.format(acc_loss1.item()),'loss_reg: {:.8f}'.format(acc_loss2.item()))
            print("Optimization Finished!")            
        
        def predict(self,testdata):
            testdata = torch.from_numpy(testdata)
            
            self.model.eval()
            if torch.cuda.is_available():
                testdata = testdata.cuda()

                
            test_dataset = MyDataset(testdata,np.zeros((testdata.shape[0],1)))
            test_dataloader = DataLoader(test_dataset,batch_size=32,shuffle=False)                 
            
            res = []
            for item in test_dataloader:
                batch_data, _ = item[0], item[1]

                if torch.cuda.is_available():
                    batch_data = batch_data.cuda()
                output, _ = self.model(batch_data)
                res.append(output.detach().cpu().numpy())
            res = np.row_stack(res)
            return res
 

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
          

# from sklearn import preprocessing
# import scipy.io as scio    
# from sklearn.neighbors import KNeighborsRegressor   
# import lightgbm as lgb
                
# data1 = scio.loadmat('/Volumes/文档/数据集/TE_process/TE_mat_data/d00.mat')['data'].T
# data2 = scio.loadmat('/Volumes/文档/数据集/TE_process/TE_mat_data/d00_te.mat')['data'].T

# dataall = np.row_stack([data1,data2])

# label = dataall[:,34]
# data = np.delete(dataall,list(range(34,53)),axis=1)


# np.random.seed(2019)
# train_index = np.random.choice(1460,100,replace = False)
# test_index = np.random.choice(list(set(list(np.arange(1460)))-set(train_index)),960,replace = False)
# u_index = list(set(list(np.arange(1460)))-set(train_index)-set(test_index))

# mi = np.min(label)
# di = (np.max(label)-np.min(label))
# label = (label-min(label))/(max(label)-min(label))
# data = preprocessing.MinMaxScaler().fit_transform(data) 


# traindata = data[train_index,:]
# trainlabel = np.mat(label[train_index]).T

# testdata = data[test_index,:]
# testlabel = np.mat(label[test_index]).T
# testlabel = testlabel*di+mi

# #=============================================
# clf = lgb.LGBMRegressor()
# clf.fit(traindata,trainlabel)
# res1 = clf.predict(testdata)
# res1 = res1*di+mi
# rmse = np.sqrt(mean_squared_error(res1,testlabel))
# mae = mean_absolute_error(res1,testlabel)
# print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))

# clf = KNeighborsRegressor()
# clf.fit(traindata,trainlabel)
# res2 = clf.predict(testdata)
# res2 = res2*di+mi
# rmse = np.sqrt(mean_squared_error(res2,testlabel))
# mae = mean_absolute_error(res2,testlabel)
# print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))



# data = make_seqdata(data,step=10)

# traindata = data[train_index]
# testdata = data[test_index]
# udata = data[u_index]
# print(traindata.shape,testdata.shape,udata.shape)

# clf = Nosiy_NN_regressor(node_list=[1024,256],noise_type='norm',n_variable=34,n_output=1,reg = 1e-1,noise_scale = 1)
# clf.fit(traindata,trainlabel,batch_size=32,epochs=50,lr=1e-4)
# res3 = clf.predict(testdata)
# res3 = res3*di+mi
# rmse = np.sqrt(mean_squared_error(res3,testlabel))
# mae = mean_absolute_error(res3,testlabel)
# print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))



# clf = Semisupervised_Nosiy_NN_regressor(node_list=[1024,256],noise_type='norm',n_variable=34,n_output=1,reg = 1e-1,noise_scale = 1)
# clf.fit(traindata,trainlabel,udata,batch_size=32,epochs=50,lr=1e-4)
# res = clf.predict(testdata)
# res = res*di+mi
# rmse = np.sqrt(mean_squared_error(res,testlabel))
# mae = mean_absolute_error(res,testlabel)
# print('rmse: {:.4f}'.format(rmse),'mae: {:.4f}'.format(mae))




























