from torch.utils.data import Dataset
class tempDataset(Dataset):
    def __init__(self, tem_data,label_data,transforms=None):
        self.trans=transforms
        self.tem=tem_data
        self.label=label_data
    def __len__(self):
        return self.tem.size(0)
    def __getitem__(self,i):
        temp_data=self.tem[i]
        label=self.label[i]
        if self.trans is not None:
            temp_data=self.trans(temp_data)
        return (temp_data,label)