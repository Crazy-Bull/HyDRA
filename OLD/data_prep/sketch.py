import matplotlib.pyplot as plt
import pickle
import numpy as np


train_loss_list=[]
val_loss_list=[]
lr_list=[]

with open("train_dat_iq_separate_3.pkl",'rb') as f:
    dat = pickle.load(f)
    train_loss_list = dat['train_loss']
    val_loss_list = dat['val_loss']
    lr_list = dat['lr']

print(np.size(lr_list))

plt.figure()
plt.subplot(2,1,1)
plt.plot(train_loss_list)
plt.plot(val_loss_list)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')

plt.subplot(2,1,2)
plt.plot(lr_list)
plt.title('learning rate change')
plt.ylabel('lr')
plt.xlabel('epoch')
plt.legend(['lr'], loc='upper left')
plt.show()