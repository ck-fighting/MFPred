import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn import utils as nn_utils
from Data_Create import k_mer
import pandas as pd
import os

PATH_Model = 'Trained_model/ablation/k_mer'
os.environ["CUDA_VISIBLE_DEVICES"]="0"
class MinimalDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label
    def __len__(self):
        return len(self.data)
def collate_fn(batch_data):
    batch_data.sort(key = lambda xi: len(xi[0]), reverse = True)
    data_length = [len(xi[0]) for xi in batch_data]
    sent_seq = [xi[0] for xi in batch_data]
    label = [xi[1] for xi in batch_data]
    padden_sent_seq = pad_sequence([torch.from_numpy(x) for x in sent_seq], batch_first=True, padding_value=0)
    return padden_sent_seq, data_length, torch.tensor(label, dtype=torch.float32)
Test_Data, Test_Label = k_mer.test_data()
model = torch.load(PATH_Model)
if torch.cuda.is_available():
    model = model.cuda()
test_data = MinimalDataset(Test_Data, Test_Label)
criterion = nn.CrossEntropyLoss()
optimer = optim.Adam(model.parameters(), lr=0.0001)
data_loader_test = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
model.eval()
max_acc = 0
model.eval()
List_Data = []
with torch.no_grad():
    correct = 0
    total = 0
    loss_totall = 0
    iii = 0
    for item_train in data_loader_test:
        train_data, train_length, train_label = item_train
        num = train_label.shape[0]
        train_data = train_data.float()
        train_label = train_label.long()
        train_data = Variable(train_data)
        train_label = Variable(train_label)
        if torch.cuda.is_available():
            train_data = train_data.cuda()
            train_label = train_label.cuda()
        pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)
        outputs = model(pack)
        loss = criterion(outputs, train_label)
        loss_totall += loss.data.item()
        iii += 1
        _, pred_acc = torch.max(outputs.data, 1)
        correct += (pred_acc == train_label).sum()
        #print(pred_acc[0].item())
        for i in range(num):
            List_Tem = []
            List_Tem.append(train_label[i].item())
            List_Tem.append(pred_acc[i].item())
            #print(List_Tem)
            List_Data.append(List_Tem)
        total += train_label.size(0)
    print('Accuracy of the test Data:{}%'.format(100 * correct / total))
    print('Loss of the test Data:{}%'.format(loss_totall / iii))
num = 0
Index_List = []
for i in List_Data:
    num = num + 1
    if(i[0] != i[1]):
        Index_List.append(num-1)

name = ['Real Label', 'Predict Label']

Pre_Data = pd.DataFrame(columns=name, data = List_Data)

Pre_Data.to_csv('F:/桌面/ck 返稿修改/MFPred-master/Pre_Data_new/Ablation/k_mer', encoding='gbk')
