import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn import utils as nn_utils
from Data_Create import data_process_nRC2
import pandas as pd
import os
import numpy as np




PATH_Model_word2vec = 'Trained_Model/GRU/word2vec_nRC5'
PATH_Model_DCN = 'Trained_Model/GRU/DCN_nRC5'
PATH_Model_kmer = 'Trained_Model/GRU/kmer_nRC5'
PATH_Model_GCN = 'Trained_Model/GRU/GCN_nRC5'
PATH_Model_CNN = 'Trained_Model/CNN/ResNet_SE_nRC5'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
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

Test_Data_word2vec, Test_Data_DCN, Test_Data_kmer, Test_Data_GCN, Test_Label = data_process_nRC2.test_data()  # 获取RNA标签信息
lstm_word2vec = torch.load(PATH_Model_word2vec)  # 调用模型BI-GRU+AM+Densenet
lstm_DCN = torch.load(PATH_Model_DCN)
lstm_kmer = torch.load(PATH_Model_kmer)
lstm_GCN = torch.load(PATH_Model_GCN)
model = torch.load(PATH_Model_CNN)

if torch.cuda.is_available():
    lstm_word2vec = lstm_word2vec.cuda()
    lstm_DCN = lstm_DCN.cuda()
    lstm_kmer = lstm_kmer.cuda()
    lstm_GCN = lstm_GCN.cuda()
    model = model.cuda()  # 判断gpu是否可用

test_data_word2vec = MinimalDataset(Test_Data_word2vec, Test_Label)
test_data_DCN = MinimalDataset(Test_Data_DCN, Test_Label)
test_data_kmer = MinimalDataset(Test_Data_kmer, Test_Label)
test_data_GCN = MinimalDataset(Test_Data_GCN, Test_Label)

criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimer_word2vec = optim.Adam(lstm_word2vec.parameters(), lr=0.0001, weight_decay=0.0001)  # 优化器用于更新参数权重
optimer_DCN = optim.Adam(lstm_DCN.parameters(), lr=0.0001, weight_decay=0.0001)
optimer_kmer = optim.Adam(lstm_kmer.parameters(), lr=0.0001, weight_decay=0.0001)
optimer_GCN = optim.Adam(lstm_GCN.parameters(), lr=0.0001, weight_decay=0.0001)
optimer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  # 优化器用于更新参数权重

data_loader_test_word2vec = DataLoader(test_data_word2vec, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数

data_loader_test_DCN = DataLoader(test_data_DCN, batch_size=32, shuffle=True,collate_fn=collate_fn)  # 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数

data_loader_test_kmer = DataLoader(test_data_kmer, batch_size=32, shuffle=True,collate_fn=collate_fn)  # 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数

data_loader_test_GCN = DataLoader(test_data_GCN, batch_size=32, shuffle=True,collate_fn=collate_fn)  # 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数

lstm_word2vec.eval()
lstm_DCN.eval()
lstm_kmer.eval()
lstm_GCN.eval()
model.eval()  # pytorch框架自动设置Dropout层和BN层，该语句是不启用，因为首先进行预测没有进行训练模型
List_Data = []
with torch.no_grad():  # 以下是为了保存模型
    correct = 0
    total = 0
    loss_totall = 0
    iii = 0
    preds = np.empty((0, 13))
    targets = np.empty((0, 1))
    for item_test_word2vec, item_test_DCN, item_test_kmer, item_test_GCN in zip(data_loader_test_word2vec, data_loader_test_DCN,
                                                                 data_loader_test_kmer,data_loader_test_GCN):
        test_data_word2vec, test_length_word2vec, test_label = item_test_word2vec  # 提取data_loader中每个批次的数据进行计算
        test_data_DCN, test_length_DCN, h = item_test_DCN
        test_data_kmer, test_length_kmer, h2 = item_test_kmer
        test_data_GCN, test_length_GCN , h3 = item_test_GCN
        num = test_label.shape[0]

        test_data_word2vec = test_data_word2vec.float()
        test_data_DCN = test_data_DCN.float()
        test_data_kmer = test_data_kmer.float()
        test_data_GCN = test_data_GCN.float()
        test_label = test_label.long()

        test_data_word2vec = Variable(test_data_word2vec)
        test_data_DCN = Variable(test_data_DCN)
        test_data_kmer = Variable(test_data_kmer)  # 将train_data里的数据变成Variable形式，用于反向传播
        test_data_GCN = Variable(test_data_GCN)
        test_label = Variable(test_label)  # 将train_label里的数据变成Variable形式，用于反向传播

        if torch.cuda.is_available():  # 判断gpu是否可用，可用的话在gpu上处理train_data和train_label数据
            test_data_word2vec = test_data_word2vec.cuda()
            test_data_DCN = test_data_DCN.cuda()
            test_data_kmer = test_data_kmer.cuda()
            test_data_GCN = test_data_GCN.cuda()
            test_label = test_label.cuda()

        pack_word2vec = nn_utils.rnn.pack_padded_sequence(test_data_word2vec, test_length_word2vec,
                                                          batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_word2vec = lstm_word2vec(pack_word2vec)  # 将pack数据放入到model模型中

        pack_DCN = nn_utils.rnn.pack_padded_sequence(test_data_DCN, test_length_DCN,
                                                     batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_DCN = lstm_DCN(pack_DCN)  # 将pack数据放入到model模型中

        pack_kmer = nn_utils.rnn.pack_padded_sequence(test_data_kmer, test_length_kmer,
                                                      batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_kmer = lstm_kmer(pack_kmer)  # 将pack数据放入到model模型中

        pack_GCN = nn_utils.rnn.pack_padded_sequence(test_data_GCN, test_length_GCN,
                                                      batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_GCN = lstm_GCN(pack_GCN)

        merged_feature = torch.cat((output_word2vec, output_DCN, output_kmer, output_GCN), dim=1)
        outputs = model(merged_feature)
        loss = criterion(outputs, test_label)  # 根据输出结果和原来的rna标签计算损失率
        loss_totall += loss.data.item()  # 将每一批次的损失率加起来
        iii += 1  # 共有多少批次
        _, pred_acc = torch.max(outputs.data, 1)  # pred_acc返回预测结果
        correct += (pred_acc == test_label).sum()  # 将pred_acc和correct相比，计算出预测正确的rna个数，之后将每一批次预测正确的相加
        total += test_label.size(0)  # 共有多少rna
        for i in range(num):
            List_Tem = []
            List_Tem.append(test_label[i].item())
            List_Tem.append(pred_acc[i].item())
            #print(List_Tem)
            List_Data.append(List_Tem)
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

Pre_Data.to_csv('Pre_Data/nRC/fold_nRC5.csv', encoding='gbk')
