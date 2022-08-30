# -*- coding: utf-8 -*-
from __future__ import print_function  # do not delete this line if you want to save your log file.
import pandas as pd
#from torchvision import models
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import os,copy,csv,time,math,src.Data_gt
from src.tempDateset import tempDataset
#from torch.utils.tensorboard import SummaryWriter
#import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random as rand
import src.abmodel  
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

def model_abnormal(epochs,learn_ratio,criterion,bs,nw,traindata_tem,traindata_label,testdata_tem,testdata_label):
    model= src.abmodel.Model(0)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled, torch.backends.cudnn.benchmark, CUDA_LAUNCH_BLOCKING = True, True, 1
    model.to(device)
    torch.cuda.manual_seed(100)
    torch.manual_seed(100)
    np.random.seed(100)
    rand.seed(100)
    w , max_acc = torch.randn(100,100), 0.95
    nn.init.kaiming_normal_(w, mode='fan_in', nonlinearity='relu')
    para = sum([np.prod(list(p.size())) for p in model.parameters()])
    print('Model {} : params: {:4f}M'.format(model._get_name(), para * 4 / 1000 / 1000))
    time.sleep(0.5)
    optimizer =optim.Adam(model.parameters(),lr=learn_ratio)
    #scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=20,eta_min=1e-15)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False,threshold=0.1, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-6)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_acc1,acc,acct,total,tal = max_acc,0.0,0.0,testdata_label.size(0),traindata_label.size(0)

    train_set = tempDataset(traindata_tem,traindata_label,transforms=None)
    test_set = tempDataset(testdata_tem,testdata_label,transforms=None)
    
    train_data = DataLoader(dataset=train_set,batch_size=bs,shuffle=True,num_workers=nw,pin_memory=True)
    test_data = DataLoader(dataset=test_set,batch_size=bs,shuffle=False,num_workers=nw,pin_memory=True)
 
    try:
        with tqdm(range(epochs)) as t:
            for e in t :
                test_loss,all_loss=0.0, 0.0
                with torch.set_grad_enabled(True) :
                    model.train()
                    loss,right,right2 = 0.0,0,0 
                    for _,[signal_data,temp_label] in enumerate(train_data) :
                        optimizer.zero_grad()
                        outputs = model(signal_data.to(device))
                        loss = criterion(outputs,temp_label.to(device))
                        if (e+1)%5 == 0:
                            right += np.sum(torch.topk(torch.abs_(outputs.detach().cpu()-temp_label),1).values.numpy()<0.1)
                        loss.backward()
                        optimizer.step()
                        all_loss+=loss.item()
                    if (e+1)%5==0:
                        acct=right/tal
                del loss,outputs,right
                torch.cuda.empty_cache()   
                if (e+1)%5==0:
                    with torch.no_grad():
                        model.eval() 
                        loss,right,right2=0.0,0,0
                        for _,[signal_data,temp_label] in enumerate(test_data):
                            outputs = model(signal_data.to(device))
                            #loss=criterion(outputs,temp_label.to(device))
                            right += np.sum(torch.topk(torch.abs_(outputs.detach().cpu()-temp_label),1).values.numpy()<0.1)
                            #test_loss+=loss.item()
                        acc=right/total
                        del right,loss,outputs
                        if acc>max_acc1:
                            max_acc1 = acc
                            torch.save(model,os.path.join("./model/abmodel0."+str(acc).split('.')[1][:5]+".pth"))
                            print("model saved: {}".format("abmodel0."+str(acc).split('.')[1][:5]+".pth"))
                    torch.cuda.empty_cache()
                    print("Epoch:{},    TrL:{},     Trc:{},     Tec:{},     Lr:{}".format(e,all_loss,acct,acc,optimizer.state_dict()['param_groups'][0]['lr']))
                else:
                    print("Epoch:{},    TrL:{},     Lr:{}".format(e,all_loss,optimizer.state_dict()['param_groups'][0]['lr']))
                scheduler.step(all_loss)
                #t.set_postfix(tl=all_loss,vl=test_loss,vac=acc,vac2=acc2,tac=acct,tac2=acct2,lrc=optimizer.state_dict()['param_groups'][0]['lr'])
                #t.set_postfix(tl=all_loss,vl=test_loss,vac=acc,tac=acct,lrc=optimizer.state_dict()['param_groups'][0]['lr'])
                #print(str(acct)+"   "+str(acc))
    except KeyboardInterrupt:
        t.close()
        raise
    t.close()
    if max_acc<max_acc1:
        max_acc=max_acc1
    model.cpu()
    torch.cuda.empty_cache()
    time.sleep(5)
    del model
    return max_acc

def main():
    epochs=300
    bs=1024*2
    nw=64
    learn_ratio=1e-2
    channels=3


    print('准备加载数据！')
    testdata_label,testdata_tem,traindata_label,traindata_tem = src.Data_gt.get_data(r'./traindata/Alltem1.mat')
    print('数据加载完成！')

    
    criterion = nn.BCELoss()
    
    print('开始训练模型！')
    max_acc = model_abnormal(epochs,learn_ratio,criterion,bs,nw,traindata_tem,traindata_label,testdata_tem,testdata_label)
    print('模型训练完毕！获得最大训练正确率：{}'.format(max_acc))
    return 0

if __name__=='__main__':
    main()
