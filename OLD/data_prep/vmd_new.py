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
    

type = 'lossless' # 'None', 'ADMM' or 'lossless'
object = 'iq'   # 'sum', 'iq' or 'amp_phase'
k = 2   # number of components

# ADMM VMD params, valid if type == 'ADMM'
alpha = 2000
tau = 0.0 
DC = 0 
init = 1 
tol = 1e-7

# lossless VMD params, valid if type == 'lossless'

init_omegas = [25.6, 51.2]    # preferred: [0, 12.8, ... , 76.8] (k=7)
refine = False

# normalization, valid if object = 'amp_phase'
norm_constant = 1

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

def VMD_no_remain(f,k,omega_init = None,refine=True,refine_lr=0.01,refine_eps=0.01):
    f_hat = (np.fft.fft(f))
    l = np.size(f)
    spec_mono = f_hat[0:int(l/2)+1]
    F_sqr = abs(spec_mono) ** 2
    if (omega_init is None):
        omegas = topk_maximal(F_sqr,k)
    else:
        omegas = omega_init

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
    return result_time, result, omegas

def myVMD(f, type):
    if type == 'None':
        return f
    elif type == 'ADMM':
        return  VMD(f, alpha, tau, k, DC, init, tol)
    elif type == 'lossless':
        return VMD_no_remain(f,k,init_omegas,refine)
    else:
        print(f"No Type '{type}'.")
        return None

def Dataset_VMD(dataset, object, type):
    # input: (N * len * 2) IQ data and params
    # output: (N * len * (k or 2k)) VMD data
    if object == 'amp_phase' or object == 'iq':
        vmd = np.empty([dataset.shape[0], dataset.shape[1], k*2])
    elif object == 'sum':
        vmd = np.empty([dataset.shape[0], dataset.shape[1], k])
    else:
        print(f"Type Error. No object type '{object}'")
        return None
    
    counter = 0
    for dat in dataset:
        # 分离I/Q通道
        I = dat[:, 0]
        Q = dat[:, 1]
        
        if object == 'sum':
            comp1 = I + Q
            comp2 = None
        elif object == 'iq':
            comp1 = I
            comp2 = Q
        elif object == 'amp_phase':
            comp1 = abs(I ** 2 + Q ** 2)
            comp2 = norm_constant * np.unwrap(np.arctan2(Q, I))

        imfs1, _, _ = myVMD(comp1, type)
        
        imfs2 = None
        if (comp2 is None):
            vmd[counter] = imfs1.transpose()
        else:
            imfs2, _, _ = myVMD(comp2, type)
            u=np.concatenate([imfs1, imfs2], axis=0)
            vmd[counter] = u.transpose()
        
        counter = counter + 1
        if counter % 1000 == 0 or counter == dataset.shape[0]:
            print(f"VMD process: {counter}/{dataset.shape[0]}")
    return vmd




dataset_path = 'D:/git_workspace/RF-fingerprint-classification/ManyTx'
name = 'iq.pkl'
X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params = load_processed_data(dataset_path, name)

if not (X_train is None):
    X_train = Dataset_VMD(X_train, object, type)
    X_val = Dataset_VMD(X_val, object, type)
    X_test = Dataset_VMD(X_test, object, type)
    params = {'type': type, 'k': X_train.shape[2]}
    name = f"{type}_{object}_{k}.pkl"
    save_processed_data(dataset_path, name, X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params)

else:
    print("Not a valid VMD type!")
