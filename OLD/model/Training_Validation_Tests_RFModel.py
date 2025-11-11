import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pickle
import os
import os.path
import math  # Added for hyperparameter calculations
from RFModel import RFModel

# [Previous functions remain unchanged: load_from_full_dataset, load_compact_pkl_dataset, shuffle, norm, split3, split_set3, get_node_indices, parse_nodes, to_categorical, prepare_txid_and_weights, prepare_dataset, create_dataset_impl, merge_compact_dataset]
def load_from_full_dataset(full_dataset_path, capture_date,rx_name,prefix=None):
    src=full_dataset_path

    if prefix is None: 
        dataset_path = '{}pkl_wifi_{}/dataset_{}_node{}.pkl'.format(src,capture_date,capture_date,rx_name)
    else:
        dataset_path = '{}pkl_wifi_{}_{}/dataset_{}_node{}.pkl'.format(src,prefix,capture_date,capture_date,rx_name)
   
    if os.path.isfile(dataset_path) :
        with open(dataset_path,'rb') as f:
            dataset = pickle.load(f)
    else:
            dataset = None
#             print('Not Found')
#             print(dataset_path)
    return dataset


def load_compact_pkl_dataset(dataset_path,dataset_name):
    with open(dataset_path+dataset_name+'.pkl','rb') as f:
        dataset = pickle.load(f)
    return dataset


def shuffle(vec1,vec2,seed = 0):
    np.random.seed(0)
#     print(vec1.shape[0],vec2.shape[0])
    shfl_indx = np.arange(vec1.shape[0])
    np.random.shuffle(shfl_indx)
    shfl_indx = shfl_indx.astype('int')
    vec1 = vec1[shfl_indx]
    vec2 = np.copy(vec2[shfl_indx])
    return vec1,vec2


def norm(sig_u):
    if len(sig_u.shape)==3:
        pwr = np.sqrt(np.mean(np.sum(sig_u**2,axis = -1),axis = -1))
        sig_u = sig_u/pwr[:,None,None]
    if len(sig_u.shape)==2:
        pwr =  np.sqrt(np.mean(np.sum(sig_u**2,axis = -1),axis = -1))
        sig_u = sig_u/pwr
    # print(sig_u.shape)
    return sig_u

def split3(vec,n1,n2):
    vec1 = vec[0:n1]
    vec2 = vec[n1:n1+n2]
    vec3 = vec[n1+n2:]
    return vec3,vec1,vec2

def split_set3(st,f1,f2):
    [sig,txid] = st

    n_samples  = sig.shape[0]
    n1 = int(f1*n_samples)
    n2 = int(f2*n_samples)

    sig1,sig2,sig3 = split3(sig,n1,n2)
    txid1,txid2,txid3 = split3(txid,n1,n2)
    st1 = [sig1,txid1]
    st2 = [sig2,txid2]
    st3 = [sig3,txid3]
    return st1,st2,st3 

def get_node_indices(tx_name_list,node_name_list):
    op_list = []
    for tx in tx_name_list:
        if tx in node_name_list:
            op_list.append(node_name_list.index(tx))
        else:
            op_list.append(None)
    return op_list
    
def parse_nodes(dataset,node_list,seed = 0):
    cat_sig = []
    cat_txid = []
    data = dataset['data']
    
    
    for i,node in enumerate(node_list):
        if (not node  is  None) and  node < len(data):
            cat_sig.append(data[node])
            cat_txid.append(np.ones( (data[node].shape[0]) )*i)
    cat_sig = np.concatenate(cat_sig)
    cat_txid = np.concatenate(cat_txid)
    np.random.seed(seed)
    cat_sig,cat_txid = shuffle(cat_sig,cat_txid)
    cat_sig = norm(cat_sig)
    return (cat_sig,cat_txid)

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def prepare_txid_and_weights(st,n):
    sig,txid = st
    txid_oh = to_categorical(txid,n)
    stat= np.sum(txid_oh,axis=0)
    cls_weights = np.max(stat,axis=0)/stat 
    cls_weights = cls_weights.tolist()
    augset = [sig,txid,txid_oh,cls_weights]
    return augset

def prepare_dataset(dataset,tx_name_list,val_frac=0.1, test_frac=0.1):
    tx_list = get_node_indices(tx_name_list,dataset['node_list'])
    all_set = parse_nodes(dataset,tx_list,seed = 0)
    train_set,val_set,test_set = split_set3(all_set,val_frac, test_frac)
    train_augset = prepare_txid_and_weights(train_set,len(tx_list))
    val_augset = prepare_txid_and_weights(val_set,len(tx_list))
    test_augset = prepare_txid_and_weights(test_set,len(tx_list))
    return train_augset,val_augset,test_augset


def create_dataset_impl(tx_list,rx_list,capture_date_list,max_sig=None,equalized_list=[0],full_dataset_path = 'data/',op_dataset_file = None):
    dataset = {}
    dataset['tx_list'] = tx_list
    dataset['rx_list'] = rx_list
    dataset['capture_date_list']=capture_date_list
    dataset['equalized_list'] = equalized_list
    dataset['max_sig'] = max_sig
    
    n_tx = len(tx_list)
    n_rx = len(rx_list)
    n_day = len(capture_date_list)
    n_eq = len(equalized_list)
    
    prefix_lut = [None,'eq']
    
    prefix_list = [prefix_lut[tt] for tt in  equalized_list]
    
    dataset['data'] = [ [ [ [ [ ] for _ in range(n_eq)] for _ in range(n_day) ] for _ in range(n_rx) ]  for _ in range(n_tx)     ]
    
    
    missing_rx_dict = {}
    
    missing_files = False

    
    with open('IdSig_info.pkl','rb') as f:
        IdSig_info=pickle.load(f)
    
    slc = slice(None,max_sig)
    for day_i,capture_date in enumerate(capture_date_list):
        for rx_i,rx_train in enumerate(rx_list):
            for eq_i,prefix in enumerate(prefix_list):
                tdataset = load_from_full_dataset(full_dataset_path,capture_date,rx_train,prefix=prefix)
                if not tdataset is None:
                    for tx_i,tx in enumerate(tx_list):
                        if tx in tdataset['node_list']:
                            tx_indx = tdataset['node_list'].index(tx)
                            dataset['data'][tx_i][rx_i][day_i][eq_i]= tdataset['data'][tx_indx][slc]  
                        else:
                            dataset['data'][tx_i][rx_i][day_i][eq_i]=np.zeros((0,256,2))
                else:
                    missing_rx_name =rx_list[rx_i]  
                    eq_val = equalized_list[eq_i]
                    IdSig_info_sub  = IdSig_info[eq_val][capture_date]
                    if missing_rx_name  in IdSig_info_sub.keys():
                            missing_files = True
                            if not eq_val in  missing_rx_dict.keys():
                                missing_rx_dict[eq_val]={}
                            if not capture_date in  missing_rx_dict[eq_val].keys():
                                missing_rx_dict[eq_val][capture_date]=[]
                            missing_rx_info  = IdSig_info_sub[missing_rx_name]
                            missing_rx_dict[eq_val][capture_date].append(   (missing_rx_info['name'], missing_rx_info['link'],missing_rx_info['size']) )

    
    if missing_files:
        ii=1
        total_file_sizes = 0
        print('You have missing files that you need to download.')
        
        for eq_k  in missing_rx_dict.keys():  
            if len(missing_rx_dict[eq_val])>0:
                print('')
                if eq_k==0:
                    print('You need to download the following files for the non equalized dataset')
                else:
                    print('You need to download the following files for the equalized dataset')
                
                print('')
                
                for date_k  in missing_rx_dict[eq_k].keys():  
                    for missing_rx in missing_rx_dict[eq_val][date_k]:
                        print('{}) Name: {} , Size: {} MB'.format(ii,missing_rx[0],missing_rx[2]/1e6))
                        total_file_sizes=total_file_sizes+missing_rx[2]
                        ii=ii+1
                print('Links:')
                for date_k  in missing_rx_dict[eq_k].keys():  
                    for missing_rx in missing_rx_dict[eq_val][date_k]:
                        print('https://drive.google.com/u/0/uc?export=download&id={}'.format(missing_rx[1]))               
        print('')
        print('You need to dowlnoad {} GB'.format(total_file_sizes/1e9))
        print('Note the following:')
        print('1) The non-equalized and eqalized files need to be downloaded in different fodlers because they share the same exact names')
        print('2) The  non-equalized folders needs to be grouped by date and equalization using the same structure as the following google drive folder')
        print('https://drive.google.com/drive/folders/1r8cd4zZ7fwvN_iiyI_uDKbIFGZve49lw?usp=sharing')
        print('3) If you have already downloaded the files make sure that the full dataset path is configured correctly.')
        dataset = None
    else:
        if not op_dataset_file is None:
            with open(op_dataset_file,'wb') as f:
                pickle.dump(dataset,f)
                print('Dataset saved in {}'.format(op_dataset_file))

    return dataset


def merge_compact_dataset(compact_dataset,capture_date,tx_list,rx_list,max_sig=None,equalized=0):
    dataset = {}
    dataset['node_list'] = tx_list
    dataset['data'] = [ () for _ in range(len(tx_list))]
    
    if not type(capture_date) is list: 
        capture_date_list = [capture_date]
    else:
        capture_date_list = capture_date
    slc = slice(None,max_sig)
    for capture_date in capture_date_list:
        for rx_train in rx_list:
            for indx,tx in enumerate(tx_list):
                tx_i=compact_dataset['tx_list'].index(tx)
                rx_i=compact_dataset['rx_list'].index(rx_train)
                date_i=compact_dataset['capture_date_list'].index(capture_date)
                eq_i=compact_dataset['equalized_list'].index(equalized)
                dataset['data'][indx]  +=  (compact_dataset['data'][tx_i][rx_i][date_i][eq_i][slc],)
    for indx in range(len(tx_list)):
        if len(dataset['data'][indx])>0:
            dataset['data'][indx] =  np.concatenate(dataset['data'][indx])
        else:
            dataset['data'][indx] =np.zeros((0,256,2))
    return dataset

# Define parameters (replace with actual values from your dataset)
tx_list = ['1-11', '10-11', '10-7', '11-1', '11-17', '11-4', '11-7', '13-3', '14-10', '14-7', '15-1', '16-16', '2-19', '20-12', '20-15', '20-19', '20-7', '3-13', '3-18', '4-11', '5-5', '6-1', '6-15', '7-10', '7-11', '8-18', '8-20', '8-3']
rx_list = ['1-1', '13-13', '14-7', '2-1', '2-20', '20-1', '7-14', '7-7', '8-13', '8-8']
capture_date_list = ['2021_03_23']
equalized = 0
dataset_path = '/home/qiu/桌面/RF-fingerprint-classification/New Data/'
dataset_name = 'SingleDay'

# Step 1: Load the compact dataset
compact_dataset = load_compact_pkl_dataset(dataset_path, dataset_name)

# Step 2: Merge the dataset
dataset = merge_compact_dataset(compact_dataset, capture_date_list, tx_list, rx_list, max_sig=None, equalized=equalized)

# Step 3: Prepare the dataset
train_augset, val_augset, test_augset = prepare_dataset(dataset, tx_list, val_frac=0.1, test_frac=0.1)

# Step 4: Extract features and labels
X_train = train_augset[0]
y_train = train_augset[2]
X_val = val_augset[0]
y_val = val_augset[2]
X_test = test_augset[0]
y_test = test_augset[2]
class_weights = train_augset[3]

# Print shapes to verify
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation data shape:", X_val.shape)
print("Validation labels shape:", y_val.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
print("Class weights:", class_weights)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- Remove Transformer Model Definitions ---
# (PositionalEncoding and TransformerEncoderModel classes are no longer needed)

# --- Training and Evaluation ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define hyperparameters for RFModel
input_length = 256  # Matches dataset sample shape (sequence length)
channels = 2        # I and Q components
f = [1.27, 2.54, 3.71, 4.98, 6.25, 7.53]
periods = []
for value in f:
    period = math.ceil(input_length / value)
    if period % 2 != 0:
        period += 1
    periods.append(period)
print(f"Periods: {periods}")

N_list = [math.ceil(input_length / p) for p in periods]
embed_dim_list = [2 * p for p in periods]
output_dims = list(zip(N_list, embed_dim_list))
print(f"Output dims: {output_dims}")

# Model hyperparameters
num_heads = 4
k = 3
ff_dim = 512
N = 2
hidden_dim = 64
fused_dim = 256  # Adjust based on your RFModel's requirements
num_classes = len(tx_list)  # Number of transmitters

# Initialize RFModel (assuming RFModel is defined/imported elsewhere)
model = RFModel(
    input_length=input_length,
    channels=channels,
    periods=periods,
    output_dims=output_dims,
    num_heads=num_heads,
    k=k,
    ff_dim=ff_dim,
    N=N,
    hidden_dim=hidden_dim,
    fused_dim=fused_dim,
    num_classes=num_classes,
    device=device,
).to(device)
print("Model initialized")

# Loss and optimizer
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch.argmax(dim=1))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch.argmax(dim=1))
            val_loss += loss.item()

    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss/len(val_loader):.4f}')

# Testing
model.eval()
y_pred = []
y_true = []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(y_batch.argmax(dim=1).numpy())

# Convert predictions and true labels to numpy arrays
y_pred = np.array(y_pred)
y_true = np.array(y_true)

# Generate classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=tx_list))