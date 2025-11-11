import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import pickle

class myDataset(Dataset):
    def __init__(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        tx_len = len(data['tx_list'])
        rx_len = len(data['rx_list'])
        idx = 0
        for tx_idx in range(tx_len):
            for rx_idx in range(rx_len):
                date_data = data['data'][tx_idx][rx_idx][0][0]  # shape = (800, 256, 2)
                
                # 初始化内存
                if tx_idx == 0 and rx_idx == 0:
                    # print(np.shape(date_data))
                    shape = np.shape(date_data)
                    self.data = np.empty([tx_len * rx_len * shape[0], shape[1], shape[2]])
                    self.label = np.empty([tx_len * rx_len * shape[0]])
                    # print(np.shape(self.data))
                for i in range(shape[0]):
                    self.data[idx] = date_data[i]
                    self.label[idx] = tx_idx
                    idx = idx + 1
                    
                 
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    def __len__(self):
        return np.shape(self.label)[0]

# 测试程序
if __name__ == "__main__":
    dataset = myDataset("SingleDay.pkl")
    print(dataset.__len__())
    print(dataset[0])