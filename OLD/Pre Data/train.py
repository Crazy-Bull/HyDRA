import torch
from torch import nn, optim

# 模型设置
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# model = ModelName().to(device)
model = 0
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        correct_num = 0
        total_num = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  

            loss_sum += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_num += labels.size(0)
            correct_num += (predicted == labels).sum().item()
        
        train_loss = loss_sum / len(train_loader)
        train_accuracy = correct_num / total_num * 100
        
        val_loss, val_accuracy = evaluate_model(model, val_loader, criterion)
        
        # 输出训练结果
        print("Epoch {}/{}, Train Loss: {:.4f}, Train Accuracy: {:.2f}%, Val Loss: {:.4f}, Val Accuracy: {:.2f}%".format( \
                epoch+1, epochs, train_loss, train_accuracy, val_loss, val_accuracy))

# 评估模型
def evaluate_model(model, loader, criterion):
    model.eval()
    loss_sum = 0.0
    correct_num = 0
    total_num = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs) 
            loss = criterion(outputs, labels)  
            loss_sum += loss.item()  
            _, predicted = torch.max(outputs, 1)  
            total_num += labels.size(0)  
            correct_num += (predicted == labels).sum().item()  

    loss = loss_sum / len(loader)
    accuracy = 100 * correct_num / total_num
    return loss, accuracy

