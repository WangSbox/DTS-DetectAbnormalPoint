# -*- coding: utf-8 -*-
from __future__ import print_function
from numpy.core.fromnumeric import reshape  # do not delete this line if you want to save your log file.
import pandas as pd
#from torchvision import models
import torch
from torch.utils.tensorboard import SummaryWriter
#import matplotlib.pyplot as plt
import torch
import os
from tqdm import tqdm
import numpy as np

# from sklearn.metrics import confusion_matrix
p=0
abmodelpath = './model/model.pth'   #          
model = torch.load(abmodelpath,map_location='cpu')
# para = sum([np.prod(list(p.size())) for p in model.parameters()])
# print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
writer = SummaryWriter('runs/')
images = torch.ones(1,100,3)
writer.add_graph(model,images)
writer.close()
model.cuda()
#print(model)

td = torch.zeros(1,200,3)
testdata = torch.zeros(1,200,3)
ot = torch.zeros(1,200)
tem = torch.zeros(1,200)
yre = np.zeros((200))
yre[81:86] = (1.0,)
yre[174:180] = (1.0,)
def maxmin(data):  # 将数据最大最小归一化
    data = data.numpy()
    d = np.zeros_like(data)
    for i in range(100):
        d[i] = (data[i]-np.min(data))/(np.max(data)-np.min(data))
    return d  

for root, dirs, files in os.walk(r'./evaldata'):#dirpath, dirnames, filenames  文件路径  文件夹名   文件名
    # 遍历所有的文件
    for f in files:   

        if float(os.path.join(root, f).split('__')[1].split('温度')[1][:]) >= 30  :
            p += 1
            file_data = pd.read_csv( os.path.join(root,f))
            file_data.columns = ['Stokes','Anti-Stokes']
            
            for i in range(200):
                testdata[0,i,0] = (file_data['Stokes'][i+47]-2300)/300
                testdata[0,i,1] = (file_data['Anti-Stokes'][i+47]-2300)/300
                testdata[0,i,2] = (i+1)/2001
            #testdata[0,19:26,:2] = testdata[0,172:179,:2]
            #model=abmodel.Model(0)
            td = torch.cat((td,testdata),dim=0)
            
            for j in range(2):  # data detected the  abnormal hot zone  
                with torch.no_grad():
                    model.eval()
                    outputs = model(testdata[0,j*100:(j+1)*100,:].unsqueeze(0).to('cuda')).detach().cpu()
                    if j==0:
                        for sig in range(100):
                            if outputs[0,sig]<0.6:
                                outputs[0,sig]=0
                            else:
                                outputs[0,sig]=1
                        scores = []
                        x = torch.from_numpy(maxmin(testdata[0,:100,0]))
                        y = torch.Tensor(outputs[:]).squeeze()
                        
                        for k in range(3):
                            if k==0:
                                 #  不移动位置
                                scores.append([ torch.cosine_similarity(x,y,dim=0).numpy(),k])
                            if k==1:
                                y[:99] = y[1:].clone()  #  左移一位
                                scores.append([ torch.cosine_similarity(x,y,dim=0).numpy(),k])
                            if k==2:
                                y[1:] = y[:99].clone()  #  右移一位
                                scores.append([ torch.cosine_similarity(x,y,dim=0).numpy(),k])
                        
                        if max(scores)[1] == 0:
                            tem[0,:100] = y[:]
                        if max(scores)[1] == 1:
                            y[:99] = y[1:100].clone()
                            tem[0,:100] = y[:]
                        if max(scores)[1] == 2:
                            y[1:100] = y[:99].clone()
                            tem[0,:100] = y[:]
                    if j==1:
                        for sig in range(100):
                            if outputs[0,sig]<0.6:
                                outputs[0,sig]=0
                            else:
                                outputs[0,sig]=1
                        scores = []
                        x = torch.from_numpy(maxmin(testdata[0,:100,0]))
                        y = torch.Tensor(outputs[:]).squeeze()
                        for k in range(3):
                            if k==0:
                                 #  不移动位置
                                scores.append([ torch.cosine_similarity(x,y,dim=0).numpy(),k])
                            if k==1:
                                y[:99] = y[1:100].clone()  #  左移一位
                                scores.append([ torch.cosine_similarity(x,y,dim=0).numpy(),k])
                            if k==2:
                                y[1:100] = y[:99].clone()  #  右移一位
                                scores.append([ torch.cosine_similarity(x,y,dim=0).numpy(),k])
                        if max(scores)[1] == 0:
                            tem[0,100:] = y[:]
                        if max(scores)[1] == 1:
                            y[:99] = y[1:100].clone()
                            tem[0,100:] = y[:]
                        if max(scores)[1] == 2:
                            y[1:100] = y[:99].clone()
                            tem[0,100:] = y[:]                  
                ''' 
                for sig in range(100):
                    if outputs[0,sig]<0.5:
                        outputs[0,sig]=0
                    else:
                        outputs[0,sig]=1
                '''  
                '''
                plt.plot(outputs[0]) 
                plt.plot(testdata[0,j*100:(j+1)*100,0])     
                plt.plot(testdata[0,j*100:(j+1)*100,1])  
                plt.title(os.path.join(root, f).split('__')[1].split('温度')[1][:])
                plt.show()  
                '''
            ot = torch.cat((ot,tem),dim=0)
# from scipy.io import savemat
# savemat('round_no_cos.mat',{'pred':ot[1:,:].numpy(),'abnor_raw':td[1:,:,:].numpy()})
print(p)
ot = ot[1:,:].numpy()
ytrue = np.zeros((ot.shape[0],200))
ytrue[:,81:86] = (1.0,)
ytrue[:,174:180] = (1.0,)
    # 遍历所有的文件夹
    #for d in dirs:
    #    print(os.path.join(root, d).split('\\')[-1])
def perf_measure(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
           TP += 1
        if y_true[i] == 0 and y_pred[i] == 1:
           FP += 1
        if y_true[i] == 0 and y_pred[i] == 0:
           TN += 1
        if y_true[i] == 1 and y_pred[i] == 0:
           FN += 1

    return TP, FP, TN, FN
def measure(TP,FP,TN,FN):
    Precision = TP/(TP+FP+1e-8)
    Recall = TP/(TP+FN+1e-8)
    Accuracy = (TP+TN)/(TP+TN+FP+FN+1e-8)
    print(Precision, Recall, Accuracy)
    print('F1 score:{:.4f}'.format(2 * (Precision * Recall) / (Precision + Recall + 1e-8)))

TP, FP, TN, FN = perf_measure(reshape(ytrue,(-1)),reshape(ot,(-1)))
# a = np.expand_dims()

print(TP, FP, TN, FN)
measure(TP,FP,TN,FN) 
