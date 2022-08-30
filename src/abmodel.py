import torch.nn as nn
def Model(model_num):
    return cm_num(model_num)
class abnormal(nn.Module):#998 ,
    def __init__(self): 
        super(abnormal,self).__init__()  
        self.sample=nn.Sequential(  
            nn.Conv1d(100,256,3,1,padding=1),#3  
            nn.BatchNorm1d(256),  
            nn.ReLU(inplace=True),  
            nn.Conv1d(256,256,3,1,padding=1),#3  
            nn.BatchNorm1d(256),  
            nn.ReLU(inplace=True),  
                  
            nn.Conv1d(256,256,2,1,padding=0),#2  
            nn.BatchNorm1d(256),  
            nn.ReLU(inplace=True),  
              
            nn.Conv1d(256,512,2,1,padding=0),#1  
            nn.BatchNorm1d(512),  
            nn.ReLU(inplace=True),  
            nn.Conv1d(512,512,1,1,padding=0),#1  
            nn.BatchNorm1d(512),  
            nn.ReLU(inplace=True),  
                )
        self.fc=nn.Sequential(
            nn.Linear(512,100),
            nn.Sigmoid(),
                )
    def forward(self,input):
        out=self.sample(input)
        out=out.view(-1,512)
        out=self.fc(out)
        return out   
def cm_num(model_num):
    if model_num==0:
        model=abnormal()
    return model