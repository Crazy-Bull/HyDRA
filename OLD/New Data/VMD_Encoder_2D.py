import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pickle
import os
import os.path
from sktime.libs.vmdpy import VMD
from tqdm.auto import tqdm
import pickle
from pathlib import Path

def save_processed_data(X_train, X_val, X_test, params):
    """保存处理后的数据和参数"""
    save_data = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'params': params
    }
    with open(SAVE_DIR/'vmd_processed.pkl', 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Data saved to {SAVE_DIR}/vmd_processed.pkl")

def load_processed_data():
    """加载已处理的数据"""
    try:
        with open(SAVE_DIR/'vmd_processed.pkl', 'rb') as f:
            data = pickle.load(f)
        print("Loaded preprocessed data")
        return data['X_train'], data['X_val'], data['X_test']
    except FileNotFoundError:
        print("No preprocessed data found")
        return None, None, None

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
    stat= np.sum(txid_oh, axis=0) + 1e-6
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
tx_list = ['1-11', '10-11', '10-7', '11-1', '11-17', '11-4', '11-7', '13-3', '14-10', '14-7', '15-1', '16-16', '2-19', '20-12', '20-15', '20-19', '20-7', '3-13', '3-18', '4-11', '5-5', '6-1', '6-15', '7-10', '7-11', '8-18', '8-20', '8-3']  # List of transmitter names
rx_list = ['1-1', '13-13', '14-7', '2-1', '2-20', '20-1', '7-14', '7-7', '8-13', '8-8']                        # List of receiver names
capture_date_list = ['2021_03_23'] # List of capture dates
equalized = 0                                   # 0 for non-equalized, 1 for equalized
# dataset_path = '/home/qiu/桌面/RF-fingerprint-classification/liu/'               # Directory containing the compact dataset
dataset_path = 'E:/真正的桌面/学习/大学内容/enjoy/G1/Week5-6/SingleDay.pkl/'
dataset_name = 'SingleDay'                # Name of the pickle file (without .pkl)
SAVE_DIR = Path("./processed_data/")
SAVE_DIR.mkdir(exist_ok=True)

# Step 1: Load the compact dataset
compact_dataset = load_compact_pkl_dataset(dataset_path, dataset_name)

# Step 2: Merge the dataset for specified transmitters, receivers, and dates
dataset = merge_compact_dataset(compact_dataset, capture_date_list, tx_list, rx_list, max_sig=None, equalized=equalized)

# Step 3: Prepare the dataset for machine learning
train_augset, val_augset, test_augset = prepare_dataset(dataset, tx_list, val_frac=0.1, test_frac=0.1)

# Step 4: Extract features and labels
X_train = train_augset[0]  # Training signals
y_train = train_augset[2]  # Training one-hot labels
X_val = val_augset[0]      # Validation signals
y_val = val_augset[2]      # Validation one-hot labels
X_test = test_augset[0]    # Test signals
y_test = test_augset[2]    # Test one-hot labels
class_weights = train_augset[3]  # Class weights

# Print shapes to verify
print("Training data shape:", X_train.shape)
print("Training labels shape:", y_train.shape)
print("Validation data shape:", X_val.shape)
print("Validation labels shape:", y_val.shape)
print("Test data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)
print("Class weights:", class_weights)

# VMD preprocessing
def Dataset_VMD(dataset, alpha=2000, tau=0.0, K=5, DC=0, init=1, tol=1e-7):
    """
    改进的VMD处理流程：
    1. 分离I/Q通道
    2. 计算幅度和相位
    3. 分别进行VMD分解
    4. 合并特征
    """
    processed_features = []
    
    for i, sample in enumerate(tqdm(dataset, desc="Processing VMD")):
        try:
            # 分离I/Q通道
            I = sample[:, 0]
            Q = sample[:, 1]
            
            # 计算幅度和相位
            amplitude = np.sqrt(I**2 + Q**2)
            phase = np.arctan2(Q, I)
            
            # 幅度VMD分解
            imfs_amp, _, _ = VMD(amplitude, alpha, tau, K, DC, init, tol)
            
            # 相位VMD分解（注意相位需要unwrap处理）
            phase_unwrapped = np.unwrap(phase)
            imfs_phase, _, _ = VMD(phase_unwrapped, alpha, tau, K, DC, init, tol)
            
            # 合并特征 (K*2, 256)
            combined = np.concatenate([imfs_amp.T, imfs_phase.T], axis=1)
            
            # 转置为(256, K*2) 适配Transformer输入
            processed_features.append(combined.T)
            
        except Exception as e:
            print(f"Error processing sample {i}: {str(e)}")
            processed_features.append(np.zeros((256, K*2)))  # 异常时填充零值
    
    return np.array(processed_features)
        

# VMD params
VMD_PARAMS = {
    'alpha': 2000,
    'tau': 0.0,
    'K': 5,       # 每个模态分解5个IMF
    'DC': 0,
    'init': 1,
    'tol': 1e-7
}

train_augset, val_augset, test_augset = prepare_dataset(dataset, tx_list, val_frac=0.1, test_frac=0.1)

X_train_raw = train_augset[0]  # 原始训练信号 (179200, 256, 2)
y_train = train_augset[2]      # 训练标签
X_val_raw = val_augset[0]      # 原始验证信号 (22400, 256, 2)
y_val = val_augset[2]          # 验证标签
X_test_raw = test_augset[0]    # 原始测试信号 (22400, 256, 2)
y_test = test_augset[2]        # 测试标签

# 加载已处理数据（如果存在）
X_train_loaded, X_val_loaded, X_test_loaded = load_processed_data()

if X_train_loaded is None:
    # 使用原始数据生成处理后的数据
    X_train = Dataset_VMD(X_train_raw, **VMD_PARAMS)  
    X_val = Dataset_VMD(X_val_raw,**VMD_PARAMS)
    X_test = Dataset_VMD(X_test_raw, **VMD_PARAMS)
    
    # 保存处理后的数据
    save_processed_data(X_train, X_val, X_test, VMD_PARAMS)
else:
    # 直接加载已处理数据
    X_train = X_train_loaded.transpose(0, 2, 1)
    X_val = X_val_loaded.transpose(0, 2, 1)
    X_test = X_test_loaded.transpose(0, 2, 1)
    assert X_train.shape == (179200, 10, 256), "数据形状应为 (samples, 10, 256)"
    
def compute_accuracy(outputs, labels):
    """计算分类准确率"""
    _, preds = torch.max(outputs, dim=1)
    correct = (preds == labels).sum().item()
    return correct / labels.size(0)

# 验证数据维度
print("\nProcessed Data Dimensions:")
print(f"Train: {X_train.shape} (samples, timesteps, features)")
print(f"Valid: {X_val.shape}")
print(f"Test:  {X_test.shape}")

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader for batching
train_dataset = TensorDataset(X_train, y_train)
val_dataset = TensorDataset(X_val, y_val)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# Now X_train, y_train, etc., are ready for model training

# --- Transformer Model ---

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim=256, num_classes=28, d_model=128, nhead=8, num_layers=3, dropout=0.1): # 若要改k，要重新规划input_dim
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model, 
            nhead, 
            dim_feedforward=256,
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.dropout = nn.Dropout(dropout)  # 定义 Dropout 层
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # Global average pooling
        x = self.dropout(x)  
        x = self.fc(x)
        return x

# --- Training and Evaluation ---

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model parameters
input_dim = VMD_PARAMS['K'] * 2  # 5 * 2=10 (5个幅度IMF + 5个相位IMF)
num_classes = len(tx_list)

# 添加维度验证
assert X_train.shape[1] == input_dim, f"输入特征维度{X_train.shape[1]}与模型定义{input_dim}不匹配"

model = TransformerEncoderModel(
    input_dim=256,  # 输入特征维度=256
    num_classes=len(tx_list),
    d_model=128,
    nhead=8,
    num_layers=3
).to(device)

class_weights = np.array(class_weights)  # 转换为numpy数组
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
best_val_acc = 0.0

for epoch in range(num_epochs):
    # Training Phase
    model.train()
    total_train_loss = 0.0
    total_train_acc = 0.0
    
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        labels = y_batch.argmax(dim=1)  # 将one-hot转为索引
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        total_train_acc += compute_accuracy(outputs, labels)
    
    # Validation Phase
    model.eval()
    total_val_loss = 0.0
    total_val_acc = 0.0
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            labels = y_batch.argmax(dim=1).to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, labels)
            
            total_val_loss += loss.item()
            total_val_acc += compute_accuracy(outputs, labels)
    
    # 计算平均指标
    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_acc = total_train_acc / len(train_loader)
    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_val_acc / len(val_loader)
    
    # 打印结果
    print(f"\nEpoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {avg_train_loss:.4f} | Acc: {avg_train_acc:.2%}")
    print(f"Val Loss: {avg_val_loss:.4f} | Acc: {avg_val_acc:.2%}")
    print("-"*60)
    
    # 保存最佳模型
    if avg_val_acc > best_val_acc:
        best_val_acc = avg_val_acc
        torch.save(model.state_dict(), SAVE_DIR/'best_model.pth')
        print(f"New best model saved with val acc {avg_val_acc:.2%}")

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