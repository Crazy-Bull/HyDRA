from sktime.libs.vmdpy import VMD
import pickle
import numpy as np
import matplotlib.pyplot as plt

with open("SingleDay.pkl", 'rb') as f:
    data = pickle.load(f)

# extract a sample
iq_signal = data['data'][0][0][0][0][3]
i = iq_signal[:, 0]
q = iq_signal[:, 1]
sum = i

# param
alpha = 2000
tau = 0.0 
DC = 0 
init = 1 
tol = 1e-7

u, u_hat, omega = VMD(sum, alpha, tau, 6, DC, init, tol)  # k = 3




# Visualize decomposed modes
"""
f_hat3 = np.sum(u, axis = 0)
plt.figure()
plt.subplot(3, 1, 1)
plt.plot(phase)
plt.title("Original signal")
plt.xlabel("time (s)")
plt.subplot(3, 1, 2)
plt.plot(u.T)
plt.title("Decomposed modes (k=3)")
plt.xlabel("time (s)")
plt.legend(["Mode %d" % m_i for m_i in range(u.shape[0])])
plt.tight_layout()
plt.subplot(3, 1, 3)
plt.plot(phase)
plt.plot(f_hat3)
plt.legend(["Original", "k=3"])
plt.title("Recovered signal")
plt.xlabel("time (s)")
plt.show()"
"""

f_hat = np.sum(u_hat,axis=1)
freq = np.fft.fftshift(np.fft.fft(sum))
print(f_hat.shape)
plt.figure()
plt.plot(abs(freq),linewidth=0.5)
plt.plot(abs(f_hat),linewidth=0.5)
plt.plot(abs(freq-f_hat))
plt.legend(["Original spectrum", "Reconstructed spectrum", "Reconstruction error"])
# plt.title("Reconstruction Error due to ADMM (k=6)")
plt.xlabel("DFT index",size=12)
plt.ylabel("Amplitude",size=12)
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.show()


"""
plt.figure()
plt.subplot(6, 1, 1)
plt.plot(f_hat[0])
plt.title("Mode 1")
plt.xlabel("Freq")
plt.subplot(6, 1, 2)
plt.plot(f_hat[1])
plt.title("Mode 2")
plt.xlabel("Freq")
plt.subplot(6, 1, 3)
plt.plot(f_hat[2])
plt.title("Mode 3")
plt.xlabel("Freq")
plt.subplot(6, 1, 4)
plt.plot(f_hat[3])
plt.title("Mode 4")
plt.xlabel("Freq")
plt.subplot(6, 1, 5)
plt.plot(f_hat[4])
plt.title("Mode 5")
plt.xlabel("Freq")
plt.subplot(6, 1, 6)
plt.plot(f_hat[5])
plt.title("Mode 6")
plt.xlabel("Freq")
plt.show()
"""

"""
freq = (np.fft.fft(sum))
spec_mono = freq[128:256]


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

vmd, vmd_freq = VMD_no_remain(sum,6,12.8)

print(vmd.shape)




plt.subplot(4,1,1)
plt.plot(np.angle(freq))
plt.subplot(4,1,2)
# plt.plot(abs(spec_mono))
plt.plot(np.angle(np.sum(vmd_freq,axis=0)))


plt.subplot(4,1,3)
plt.plot(np.fft.ifft(vmd_freq[1]))
plt.subplot(4,1,4)
plt.plot(abs(freq-np.sum(vmd_freq,axis=0)))
plt.show()
"""