import numpy as np
import pickle
import os
import os.path
import pickle
import matplotlib.pyplot as plt
from pathlib import Path

from sktime.libs.vmdpy import VMD

def save_processed_data(dataset_path, name, X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params):
    """保存处理后的数据和参数"""
    save_data = {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'class_weights': class_weights,
        'params': params
    }
    SAVE_DIR = Path(f"{dataset_path}/data_prep")
    with open(SAVE_DIR/name, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Data saved to {SAVE_DIR}/{name}")


def load_processed_data(dataset_path, name):
    """加载已处理的数据"""
    SAVE_DIR = Path(f'{dataset_path}/data_prep')
    try:
        with open(SAVE_DIR/name, 'rb') as f:
            data = pickle.load(f)
        print("Loaded preprocessed iq data")
        return data['X_train'], data['y_train'], data['X_val'], data['y_val'], data['X_test'], data['y_test'], data['class_weights'], data['params']
    except FileNotFoundError:
        print("No preprocessed data found")
        return None, None, None, None, None, None, None, None
    
# VMD params
k = 6   # number of components
alpha = 2000
tau = 0.0 
DC = 0 
init = 1 
tol = 1e-7
type = 'raw' # 'iq_sum' or 'amp_phase' or 'iq_separate' or 'iq_sum_freq'


# VMD preprocessing
def Dataset_VMD_amp_phase(dataset, alpha, tau, k, DC, init, tol):
    # input: (N * len * 2) IQ data and params
    # output: (N * len * (k * 2)) VMD data
    vmd = np.empty([dataset.shape[0], dataset.shape[1], k*2])
    counter = 0
    for dat in dataset:
        # 分离I/Q通道
        I = dat[:, 0]
        Q = dat[:, 1]
        
        # 计算幅度和相位
        amplitude = np.sqrt(I**2 + Q**2)
        phase = np.arctan2(Q, I)
        
        # 幅度VMD分解
        imfs_amp, _, _ = VMD(amplitude, alpha, tau, k, DC, init, tol)
        
        # 相位VMD分解（注意相位需要unwrap处理）
        phase_unwrapped = np.unwrap(phase)
        imfs_phase, _, _ = VMD(phase_unwrapped, alpha, tau, k, DC, init, tol)

        u=np.concatenate([imfs_amp, imfs_phase], axis=0)

        vmd[counter] = u.transpose()
        counter = counter + 1
        if counter % 1000 == 0 or counter == dataset.shape[0]:
            print(f"VMD process: {counter}/{dataset.shape[0]}")
    return vmd

# VMD preprocessing
def Dataset_VMD_iqsum(dataset, alpha, tau, k, DC, init, tol):
    # input: (N * len * 2) IQ data and params
    # output: (N * len * k) VMD data
    vmd = np.empty([dataset.shape[0], dataset.shape[1], k])
    counter = 0
    for dat in dataset:
        # 分离I/Q通道
        I = dat[:, 0]
        Q = dat[:, 1]
        
        # 计算IQ和
        sum = I + Q
        
        # VMD分解
        imfs, _, _ = VMD(sum, alpha, tau, k, DC, init, tol)

        vmd[counter] = imfs.transpose()
        counter = counter + 1
        if counter % 1000 == 0 or counter == dataset.shape[0]:
            print(f"VMD process: {counter}/{dataset.shape[0]}")
    return vmd

def Dataset_VMD_iq_separate(dataset, alpha, tau, k, DC, init, tol):
    # input: (N * len * 2) IQ data and params
    # output: (N * len * (k * 2)) VMD data
    vmd = np.empty([dataset.shape[0], dataset.shape[1], k*2])
    counter = 0
    for dat in dataset:
        # 分离I/Q通道
        I = dat[:, 0]
        Q = dat[:, 1]

        imfs_i, _, _ = VMD(I, alpha, tau, k, DC, init, tol)
        imfs_q, _, _ = VMD(Q, alpha, tau, k, DC, init, tol)

        u=np.concatenate([imfs_i, imfs_q], axis=0)

        vmd[counter] = u.transpose()
        counter = counter + 1
        if counter % 1000 == 0 or counter == dataset.shape[0]:
            print(f"VMD process: {counter}/{dataset.shape[0]}")
    return vmd

def Dataset_VMD_iqsum_freq(dataset, alpha, tau, k, DC, init, tol):
    # input: (N * len * 2) IQ data and params
    # output: (N * len * k) VMD data
    vmd = np.empty([dataset.shape[0], dataset.shape[1], k])
    counter = 0
    for dat in dataset:
        # 分离I/Q通道
        I = dat[:, 0]
        Q = dat[:, 1]
        
        # 计算IQ和
        sum = I + Q
        
        # VMD分解
        _, imfs_freq, _ = VMD(sum, alpha, tau, k, DC, init, tol)

        vmd[counter] = abs(imfs_freq)
        counter = counter + 1
        if counter % 1000 == 0 or counter == dataset.shape[0]:
            print(f"VMD process: {counter}/{dataset.shape[0]}")
    return vmd

def topk_maximal(f,k):
    tmp = np.argsort(f)
    print(tmp)
    i = np.size(f) - 1
    ans = []
    while (len(ans)<k):
        ind = tmp[i]
        if (ind==0) and (f[ind]>f[ind+1]) or (ind==np.size(f)) and (f[ind]>f[ind-1]) or (f[ind]>f[ind+1]) and (f[ind]>f[ind-1]):
            ans.append(ind)
        i = i - 1
    return np.sort(ans)

def VMD_no_remain(f,k,omega_0=-1,refine=True,refine_lr=0.01,refine_eps=0.01):
    f_hat = (np.fft.fft(f))
    l = np.size(f)
    spec_mono = f_hat[0:int(l/2)+1]
    F_sqr = abs(spec_mono) ** 2
    if omega_0 <= 0:
        omegas = topk_maximal(F_sqr,k)
    else:
        omegas = omega_0 * np.arange(start=1,stop=k+1)

    if refine == True:
        pass

    result = np.zeros((int(l/2)+1,k),dtype=complex)
    for omega in range(int(l/2)+1):
        alpha=np.zeros(k)
        for i in range(k):
            dist = omega-omegas[i]
            if abs(dist)<0.0001:
                alpha[i]=1e8
            else:
                alpha[i]=dist ** (-2)
        alpha = alpha / np.sum(alpha)
        result[omega]=spec_mono[omega]*alpha
    result=np.concatenate([result, np.flip(np.conjugate(result)[1:int(l/2)],axis=0)]).transpose()
    result_time = np.fft.ifft(result,axis=1)
    return result_time, result

def Dataset_VMD_no_remain(dataset, k=6, omega_0=12.8,refine=True,refine_lr=0.01,refine_eps=0.01):
    # input: (N * len * 2) IQ data and params
    # output: (N * len * k) VMD data
    vmd = np.empty([dataset.shape[0], dataset.shape[1], k])
    counter = 0
    for dat in dataset:
        # 分离I/Q通道
        I = dat[:, 0]
        Q = dat[:, 1]
        
        # 计算IQ和
        sum = I + Q
        
        # VMD分解
        imfs, _ = VMD_no_remain(sum,k,omega_0,refine,refine_lr,refine_eps)

        vmd[counter] = abs(imfs).transpose()
        counter = counter + 1
        if counter % 1000 == 0 or counter == dataset.shape[0]:
            print(f"VMD process: {counter}/{dataset.shape[0]}")
    return vmd

def Dataset_raw(dataset):
    # input: (N * len * 2) IQ data
    # output: (N * len * 1) raw data

    result = np.sum(dataset,axis=2,keepdims=True)
    
    print(result.shape)
    return result

dataset_path = 'D:/git_workspace/RF-fingerprint-classification/ManyTx/'
name = 'iq.pkl'
X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params = load_processed_data(dataset_path, name)

if not (X_train is None):
    if type == 'iq_sum':
        X_train = Dataset_VMD_iqsum(X_train, alpha, tau, k, DC, init, tol)
        X_val = Dataset_VMD_iqsum(X_val, alpha, tau, k, DC, init, tol)
        X_test = Dataset_VMD_iqsum(X_test, alpha, tau, k, DC, init, tol)
        params = {'type': type, 'k': k}
        name = f'iq_sum_{k}.pkl'
        save_processed_data(dataset_path, name, X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params)
    elif type == 'amp_phase':
        X_train = Dataset_VMD_amp_phase(X_train, alpha, tau, k, DC, init, tol)
        X_val = Dataset_VMD_amp_phase(X_val, alpha, tau, k, DC, init, tol)
        X_test = Dataset_VMD_amp_phase(X_test, alpha, tau, k, DC, init, tol)
        params = {'type': type, 'k': k * 2}
        name = f'amp_phase_{k}.pkl'
        save_processed_data(dataset_path, name, X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params)
    elif type == 'iq_separate':
        X_train = Dataset_VMD_iq_separate(X_train, alpha, tau, k, DC, init, tol)
        X_val = Dataset_VMD_iq_separate(X_val, alpha, tau, k, DC, init, tol)
        X_test = Dataset_VMD_iq_separate(X_test, alpha, tau, k, DC, init, tol)
        params = {'type': type, 'k': k * 2}
        name = f'iq_separate_{k}.pkl'
        save_processed_data(dataset_path, name, X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params)
    elif type == 'iq_sum_freq':
        X_train = Dataset_VMD_iqsum_freq(X_train, alpha, tau, k, DC, init, tol)
        X_val = Dataset_VMD_iqsum_freq(X_val, alpha, tau, k, DC, init, tol)
        X_test = Dataset_VMD_iqsum_freq(X_test, alpha, tau, k, DC, init, tol)
        params = {'type': type, 'k': k}
        name = f'iq_sum_freq{k}.pkl'
        save_processed_data(dataset_path, name, X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params)
    elif type == 'iq_sum_no_remain':
        X_train = Dataset_VMD_no_remain(X_train)
        X_val = Dataset_VMD_no_remain(X_val)
        X_test = Dataset_VMD_no_remain(X_test)
        params = {'type': type, 'k': k}
        name = f'iq_no_remain_{k}.pkl'
        save_processed_data(dataset_path, name, X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params)
    elif type == 'raw':
        X_train = Dataset_raw(X_train)
        X_val = Dataset_raw(X_val)
        X_test = Dataset_raw(X_test)
        params = {'type': type, 'k': 1}
        name = f'raw.pkl'
        
        save_processed_data(dataset_path, name, X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params)
    else:
        print("Not a valid VMD type!")
