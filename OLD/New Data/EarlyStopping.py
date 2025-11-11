import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report
import pickle
import os
import os.path

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

# Step 1: Load the compact dataset
dataset_path = '/home/qiu/桌面/RF-fingerprint-classification/'
dataset_name = 'SingleDay'
compact_dataset = load_compact_pkl_dataset(dataset_path, dataset_name)
# Define parameters (replace with actual values from your dataset)
tx_list = compact_dataset['tx_list']
rx_list = compact_dataset['rx_list']
capture_date_list = compact_dataset['capture_date_list']
equalized = 0

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

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Union
import math
from einops import rearrange, repeat, einsum

# --- Mamba Components ---

@dataclass
class ModelArgs:
    d_model: int
    n_layer: int
    vocab_size: int
    d_state: int = 16
    expand: int = 2
    dt_rank: Union[int, str] = 'auto'
    d_conv: int = 4 
    pad_vocab_size_multiple: int = 8
    conv_bias: bool = True
    bias: bool = False
    
    def __post_init__(self):
        self.d_inner = int(self.expand * self.d_model)
        if self.dt_rank == 'auto':
            self.dt_rank = math.ceil(self.d_model / 16)
        if self.vocab_size % self.pad_vocab_size_multiple != 0:
            self.vocab_size += (self.pad_vocab_size_multiple - self.vocab_size % self.pad_vocab_size_multiple)

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.in_proj = nn.Linear(args.d_model, args.d_inner * 2, bias=args.bias)
        self.conv1d = nn.Conv1d(
            in_channels=args.d_inner,
            out_channels=args.d_inner,
            bias=args.conv_bias,
            kernel_size=args.d_conv,
            groups=args.d_inner,
            padding=args.d_conv - 1,
        )
        self.x_proj = nn.Linear(args.d_inner, args.dt_rank + args.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(args.dt_rank, args.d_inner, bias=True)
        A = repeat(torch.arange(1, args.d_state + 1), 'n -> d n', d=args.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(args.d_inner))
        self.out_proj = nn.Linear(args.d_inner, args.d_model, bias=args.bias)

    def forward(self, x):
        (b, l, d) = x.shape
        x_and_res = self.in_proj(x)
        (x, res) = x_and_res.split(split_size=[self.args.d_inner, self.args.d_inner], dim=-1)
        x = rearrange(x, 'b l d_in -> b d_in l')
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, 'b d_in l -> b l d_in')
        x = F.silu(x)
        y = self.ssm(x)
        y = y * F.silu(res)
        output = self.out_proj(y)
        return output

    def ssm(self, x):
        (d_in, n) = self.A_log.shape
        A = -torch.exp(self.A_log.float())
        D = self.D.float()
        x_dbl = self.x_proj(x)
        (delta, B, C) = x_dbl.split(split_size=[self.args.dt_rank, n, n], dim=-1)
        delta = F.softplus(self.dt_proj(delta))
        y = self.selective_scan(x, delta, A, B, C, D)
        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]
        deltaA = torch.exp(einsum(delta, A, 'b l d_in, d_in n -> b l d_in n'))
        deltaB_u = einsum(delta, B, u, 'b l d_in, b l n, b l d_in -> b l d_in n')
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], 'b d_in n, b n -> b d_in')
            ys.append(y)
        y = torch.stack(ys, dim=1)
        y = y + u * D
        return y

class ResidualBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.mixer = MambaBlock(args)
        self.norm = RMSNorm(args.d_model)
        
    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

# --- CNN Components ---

class ResConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super(ResConv1d, self).__init__()
        padding_t = (kernel_size // 2) * dilation
        padding_f = 7  # Fixed for kernel_size=15
        self.conv_t = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding_t, dilation=dilation)
        self.conv_f = nn.Conv1d(in_channels, out_channels, kernel_size=15, padding=padding_f)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.shortcut = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out_t = self.conv_t(x)
        out_f = self.conv_f(x)
        out = out_t + out_f
        out = self.bn(out)
        out = self.relu(out)
        return out + identity

class CNNFeatureExtractor(nn.Module):
    def __init__(self, input_channels=2, d_model=64, split_channels=None):
        super(CNNFeatureExtractor, self).__init__()
        if split_channels is not None:
            assert input_channels == sum(split_channels), "Input channels must match the sum of split channels"
            self.split = True
            self.split_channels = split_channels
            self.mag_cnn = nn.Sequential(
                ResConv1d(split_channels[0], 32),
                ResConv1d(32, 64)
            )
            self.phase_cnn = nn.Sequential(
                ResConv1d(split_channels[1], 32),
                ResConv1d(32, 64)
            )
            self.combine = nn.Conv1d(128, d_model, kernel_size=1)
        else:
            self.split = False
            self.layers = nn.Sequential(
                ResConv1d(input_channels, 32),
                ResConv1d(32, 32, dilation=3),
                ResConv1d(32, d_model)
            )

    def forward(self, x):
        if self.split:
            mag = x[:, :, :self.split_channels[0]]
            phase = x[:, :, self.split_channels[0]:]
            mag = mag.permute(0, 2, 1)
            phase = phase.permute(0, 2, 1)
            mag_feat = self.mag_cnn(mag)
            phase_feat = self.phase_cnn(phase)
            combined = torch.cat((mag_feat, phase_feat), dim=1)
            combined = self.combine(combined)
            return combined.permute(0, 2, 1)
        else:
            x = x.permute(0, 2, 1)
            x = self.layers(x)
            return x.permute(0, 2, 1)

# --- Mamba Encoder Model (Without Positional Encoding) ---

class MambaEncoderModel(nn.Module):
    def __init__(self, d_model=64, n_layer=2, d_state=16, expand=2, dt_rank='auto', d_conv=4, conv_bias=True, bias=False):
        super().__init__()
        args = ModelArgs(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=1,  # Dummy value
            d_state=d_state,
            expand=expand,
            dt_rank=dt_rank,
            d_conv=d_conv,
            conv_bias=conv_bias,
            bias=bias
        )
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.layers = nn.ModuleList([ResidualBlock(args) for _ in range(args.n_layer)])
        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.cls_token, 0, 0.02)

    def forward(self, x):
        batch_size = x.size(0)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((x, cls_tokens), dim=1)  # CLS token at end
        for layer in self.layers:
            x = layer(x)
        return x

# --- Classification Head and Full Model ---

class ClassificationHead(nn.Module):
    def __init__(self, d_model=64, num_classes=10):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        cls_output = x[:, -1, :]  # Use last token
        return self.fc(cls_output)

class CNNTransformerModel(nn.Module):
    def __init__(self, input_channels=2, num_classes=10, d_model=64, num_layers=2, 
                 split_channels=None, d_state=16, expand=2, dt_rank='auto', d_conv=4, conv_bias=True, bias=False):
        super(CNNTransformerModel, self).__init__()
        self.cnn = CNNFeatureExtractor(input_channels, d_model, split_channels)
        self.mamba_encoder = MambaEncoderModel(
            d_model=d_model, 
            n_layer=num_layers, 
            d_state=d_state, 
            expand=expand, 
            dt_rank=dt_rank, 
            d_conv=d_conv, 
            conv_bias=conv_bias, 
            bias=bias
        )
        self.classifier = ClassificationHead(d_model, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.cnn(x)
        x = self.mamba_encoder(x)
        logits = self.classifier(x)
        return logits

# --- Example Initialization ---

# Note: Replace 'k' and 'tx_list' with your actual values
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_channels = 2  # Example value; adjust based on your data
num_classes = len(tx_list)    # Example value; adjust based on your data
model = CNNTransformerModel(
    input_channels=input_channels, 
    num_classes=num_classes, 
    split_channels=None  # Or use split_channels=(5, 5) if splitting magnitude and phase
).to(device)


# model = CNNTransformerModel(input_channels=input_channels, num_classes=num_classes, split_channels=(5, 5)).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(device))
optimizer = optim.Adam(model.parameters(), lr=0.001)


class EarlyStopping:
    def __init__(self, patience=5, verbose=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            # 保存最佳模型
            if self.verbose:
                print(f'Validation loss improved ({self.best_loss:.4f} → {val_loss:.4f}). Saving model...')
            torch.save(model.state_dict(), 'best_model.pth')
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

# --- 训练和验证过程封装 ---
def train_epoch(model, loader, criterion, optimizer, device, class_weights):
    model.train()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels.argmax(dim=1))
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels.argmax(dim=1)).sum().item()
        total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1))
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels.argmax(dim=1)).sum().item()
            total_samples += labels.size(0)
    
    avg_loss = total_loss / total_samples
    accuracy = correct / total_samples
    return avg_loss, accuracy

# --- 训练流程 ---
def enhanced_training(model, train_loader, val_loader, criterion, optimizer, device, class_weights, num_epochs=100):
    early_stopping = EarlyStopping()
    
    # 初始化记录列表
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, class_weights
        )
        
        # 验证阶段
        val_loss, val_acc = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # 记录数据
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印日志
        print(f'Epoch {epoch+1:03d} | '
              f'Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} | '
              f'Val Loss: {val_loss:.4f} Acc: {val_acc:.3f}')
        
        # 早停检查
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))
    
    # 哈希验证
    print("\n[Debug] Model Hash Verification:")
    initial_hash = hash(frozenset(model.state_dict().items()))
    model.load_state_dict(torch.load('best_model.pth'))
    loaded_hash = hash(frozenset(model.state_dict().items()))
    print(f"Initial Hash: {initial_hash}\nLoaded Hash: {loaded_hash}")
    
    # 可视化
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(val_losses, label='Validation')
        plt.title('Loss Curves')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train')
        plt.plot(val_accs, label='Validation')
        plt.title('Accuracy Curves')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.close()
    except ImportError:
        print("Matplotlib not installed, skip visualization")
    
    print("Loaded best model weights")

enhanced_training(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    class_weights = class_weights,
    num_epochs=100  
)

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