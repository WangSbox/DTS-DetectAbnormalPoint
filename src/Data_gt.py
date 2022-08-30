import psutil,  torch, time
import h5py as h5py
def get_data(filename):

    mem=psutil.virtual_memory()
    print('总内存为：{}MB'.format(float( mem.total/1024/1024)))
    
    traindata_tem = torch.Tensor(h5py.File(filename)['traindata_tem'][:,100:,:].transpose(2,1,0))
    time.sleep(10)
    

    traindata_label = torch.Tensor(h5py.File(filename)['traindata_label'][100:,:].transpose(1,0))
                                        
    testdata_tem = torch.Tensor(h5py.File(filename)['testdata_tem'][:,100:,:].transpose(2,1,0))
    time.sleep(10)
    
    testdata_label = torch.Tensor(h5py.File(filename)['testdata_label'][100:,:].transpose(1,0))
  
    return testdata_label*100,testdata_tem,traindata_label*100,traindata_tem