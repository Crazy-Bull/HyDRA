import numpy as np
import pickle
import os
import os.path
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib import font_manager

from sktime.libs.vmdpy import VMD
import time

def load_processed_data(dataset_path, name):
    """加载已处理的数据"""
    SAVE_DIR = Path(f'{dataset_path}')
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
    return vmd

def Dataset_raw(dataset):
    # input: (N * len * 2) IQ data
    # output: (N * len * 1) raw data

    result = np.sum(dataset,axis=2,keepdims=True)
    
    print(result.shape)
    return result


dataset_path = '/home/qiu/桌面/RF-fingerprint-classification/raw and iq data/'
name = 'iq_SingleDay.pkl'
X_train, y_train, X_val, y_val, X_test, y_test, class_weights, params = load_processed_data(dataset_path, name)

for k in np.arange(1,10):
    start_time = time.time()
    X_train=Dataset_VMD_iqsum(X_test,alpha,tau,k,DC,init,tol)
    end_time = time.time()
    print(f"Average time of AMDD k={k}: {(end_time-start_time)/22.4}ms")
    start_time = time.time()
    X_train=Dataset_VMD_no_remain(X_test,k)
    end_time = time.time()
    print(f"Average time of lossless k={k}: {(end_time-start_time)/22.4}ms")


"""
time1=[1.991,2.725,6.589,10.23,17.47,28.16,45.38]
time2=[0.5955, 0.6366,0.6579,0.6951,0.7141,0.7426,0.7941]
ks=[3,4,5,6,7,8,9]
plt.figure()
plt.plot(ks,time1,marker='o',color='steelblue')
plt.plot(ks,time2,marker='^',color='chocolate')
#plt.yscale('log')
plt.xlabel('k',size=12)
plt.ylabel('average execution time (ms)',size=12)
plt.legend(['ADMM','lossless'])
plt.grid(visible=True, which='both')
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.show()
"""


"""
acc = [90.221, 90.397, 90.665, 90.561, 90.207, 90.471, 90.035]
f1 = [92.045, 92.263, 92.443, 92.276, 92.051, 92.374, 91.764]
ks = [1,2,3,4,5,6,7]
plt.figure()
plt.plot(ks,acc,marker='o',color='steelblue')
plt.plot(ks,f1,marker='^',color='chocolate')
plt.xlabel('k',size=12)
plt.ylabel('metric',size=12)
plt.legend(['Acc','F1'])
plt.grid(visible=True, which='both')

plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.show()
"""