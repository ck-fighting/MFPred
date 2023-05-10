import torch
from torch.utils.data import Dataset
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
from torch.nn import utils as nn_utils
import os
from Model import LSTM_Attention, ResNet_SE
from Data_Create import data_process_nRC2
import pandas as pd

PATH_Model_word2vec = 'Trained_Model/GRU/word2vec_nRC5'
PATH_Model_DCN = 'Trained_Model/GRU/DCN_nRC5'
PATH_Model_kmer = 'Trained_Model/GRU/kmer_nRC5'
PATH_Model_DenseNet = 'Trained_Model/CNN/ResNet_SE_nRC5'
PATH_Model_GCN = 'Trained_Model/GRU/GCN_nRC5'
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
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)  # 将传进来的32个rna序列按照序列长度从大到小排序
    data_length = [len(xi[0]) for xi in batch_data]  # 存放32个RNA的序列长度
    sent_seq = [xi[0] for xi in batch_data]  # 存放32个RNA序列的每一个序列
    label = [xi[1] for xi in batch_data]  # 存放32个RNA序列的类别
    padden_sent_seq = pad_sequence([torch.from_numpy(x) for x in sent_seq], batch_first=True, padding_value=0)  # 将所有RNA序列填充到32个RNA序列里面最长的长度
    return padden_sent_seq, data_length, torch.tensor(label, dtype=torch.float32)

Train_Data_word2vec, Train_Data_DCN, Train_Data_kmer, Train_Data_GCN, Train_Label = data_process_nRC2.train_data()  # 获取RNA碱基信息
Test_Data_word2vec, Test_Data_DCN, Test_Data_kmer, Test_Data_GCN, Test_Label = data_process_nRC2.test_data()  # 获取RNA标签信息
lstm_word2vec = LSTM_Attention.LSTM_word2vec()  # 调用模型BI-GRU+AM+Densenet
lstm_DCN = LSTM_Attention.LSTM_DCN()
lstm_kmer = LSTM_Attention.LSTM_kmer()
lstm_GCN = LSTM_Attention.LSTM_GCN()
model = ResNet_SE.ResNet_50()

if torch.cuda.is_available():
    lstm_word2vec = lstm_word2vec.cuda()
    lstm_DCN = lstm_DCN.cuda()
    lstm_kmer = lstm_kmer.cuda()
    lstm_GCN = lstm_GCN.cuda()
    model = model.cuda()# 判断gpu是否可用
train_data_word2vec = MinimalDataset(Train_Data_word2vec, Train_Label)
test_data_word2vec = MinimalDataset(Test_Data_word2vec, Test_Label)

train_data_DCN = MinimalDataset(Train_Data_DCN, Train_Label)
test_data_DCN = MinimalDataset(Test_Data_DCN, Test_Label)

train_data_kmer = MinimalDataset(Train_Data_kmer, Train_Label)
test_data_kmer = MinimalDataset(Test_Data_kmer, Test_Label)

train_data_GCN = MinimalDataset(Train_Data_GCN, Train_Label)
test_data_GCN = MinimalDataset(Test_Data_GCN, Test_Label)

criterion = nn.CrossEntropyLoss()  # 定义损失函数
optimer_word2vec = optim.Adam(lstm_word2vec.parameters(), lr=0.0001, weight_decay=0.0001)  # 优化器用于更新参数权重
optimer_DCN = optim.Adam(lstm_DCN.parameters(), lr=0.0001, weight_decay=0.0001)
optimer_kmer = optim.Adam(lstm_kmer.parameters(), lr=0.0001, weight_decay=0.0001)
optimer_GCN = optim.Adam(lstm_GCN.parameters(),  lr=0.0001, weight_decay=0.0001)
optimer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)  # 优化器用于更新参数权重

data_loader_word2vec = DataLoader(train_data_word2vec, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将train_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数
data_loader_test_word2vec = DataLoader(test_data_word2vec, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数

data_loader_DCN = DataLoader(train_data_DCN, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将train_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数
data_loader_test_DCN = DataLoader(test_data_DCN, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数

data_loader_kmer = DataLoader(train_data_kmer, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将train_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数
data_loader_test_kmer = DataLoader(test_data_kmer, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数

data_loader_GCN = DataLoader(train_data_GCN, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将train_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数
data_loader_test_GCN = DataLoader(test_data_GCN, batch_size=32, shuffle=True, collate_fn=collate_fn)  # 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数

lstm_word2vec.eval()
lstm_DCN.eval()
lstm_kmer.eval()
lstm_GCN.eval()
model.eval()# pytorch框架自动设置Dropout层和BN层，该语句是不启用，因为首先进行预测没有进行训练模型
max_acc = 0
with torch.no_grad():  # 加速节省GPU空间
    correct = 0  # 预测正确的个数
    total = 0  # 改批次下的RNA总数
    loss_totall = 0
    iii = 0
    for item_train_word2vec, item_train_DCN, item_train_kmer, item_train_GCN in zip(data_loader_word2vec, data_loader_DCN, data_loader_kmer, data_loader_GCN):
        train_data_word2vec, train_length_word2vec, train_label = item_train_word2vec  # 提取data_loader中每个批次的数据进行计算
        train_data_DCN, train_length_DCN, h = item_train_DCN
        train_data_kmer, train_length_kmer, h2 = item_train_kmer
        train_data_GCN, train_length_GCN, h3 = item_train_GCN
        num = train_label.shape[0]

        train_data_word2vec = train_data_word2vec.float()
        train_data_DCN = train_data_DCN.float()
        train_data_kmer = train_data_kmer.float()
        train_data_GCN = train_data_GCN.float()
        train_label = train_label.long()

        train_data_word2vec = Variable(train_data_word2vec)
        train_data_DCN = Variable(train_data_DCN)
        train_data_kmer = Variable(train_data_kmer)# 将train_data里的数据变成Variable形式，用于反向传播
        train_data_GCN = Variable(train_data_GCN)
        train_label = Variable(train_label)  # 将train_label里的数据变成Variable形式，用于反向传播

        if torch.cuda.is_available():  # 判断gpu是否可用，可用的话在gpu上处理train_data和train_label数据
            train_data_word2vec = train_data_word2vec.cuda()
            train_data_DCN = train_data_DCN.cuda()
            train_data_kmer = train_data_kmer.cuda()
            train_data_GCN = train_data_GCN.cuda()
            train_label = train_label.cuda()

        pack_word2vec = nn_utils.rnn.pack_padded_sequence(train_data_word2vec, train_length_word2vec, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_word2vec = lstm_word2vec(pack_word2vec)  # 将pack数据放入到model模型中

        pack_DCN = nn_utils.rnn.pack_padded_sequence(train_data_DCN, train_length_DCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_DCN = lstm_DCN(pack_DCN)  # 将pack数据放入到model模型中

        pack_kmer = nn_utils.rnn.pack_padded_sequence(train_data_kmer, train_length_kmer, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_kmer = lstm_kmer(pack_kmer)  # 将pack数据放入到model模型中

        pack_GCN = nn_utils.rnn.pack_padded_sequence(train_data_GCN, train_length_GCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_GCN = lstm_GCN(pack_GCN)

        merged_feature = torch.cat((output_word2vec, output_DCN, output_kmer, output_GCN), dim=1)

        outputs = model(merged_feature)

        loss = criterion(outputs, train_label)  # 根据输出结果和原来的rna标签计算损失率
        loss_totall += loss.data.item()  # 将每一批次的损失率加起来
        iii += 1  # 共有多少批次
        _, pred_acc = torch.max(outputs.data, 1)  # pred_acc返回预测结果
        correct += (pred_acc == train_label).sum()  # 将pred_acc和correct相比，计算出预测正确的rna个数，之后将每一批次预测正确的相加
        total += train_label.size(0)  # 共有多少rna
    print('Accuracy of the train Data:{}%'.format(100 * correct / total))
    print('Loss of the train Data:{}%'.format(loss_totall / iii))  # 没有经过训练的模型的准确率和损失率

lstm_word2vec.eval()
lstm_DCN.eval()
lstm_kmer.eval()
lstm_GCN.eval()
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    loss_totall = 0
    iii = 0
    for item_test_word2vec, item_test_DCN, item_test_kmer, item_test_GCN in zip(data_loader_test_word2vec, data_loader_test_DCN, data_loader_test_kmer, data_loader_test_GCN):
        test_data_word2vec, test_length_word2vec, test_label = item_test_word2vec  # 提取data_loader中每个批次的数据进行计算
        test_data_DCN, test_length_DCN, h = item_test_DCN
        test_data_kmer, test_length_kmer, h2 = item_test_kmer
        test_data_GCN, test_length_GCN ,h3 = item_test_GCN
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

        pack_word2vec = nn_utils.rnn.pack_padded_sequence(test_data_word2vec, test_length_word2vec, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_word2vec = lstm_word2vec(pack_word2vec)  # 将pack数据放入到model模型中

        pack_DCN = nn_utils.rnn.pack_padded_sequence(test_data_DCN, test_length_DCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_DCN = lstm_DCN(pack_DCN)  # 将pack数据放入到model模型中

        pack_kmer = nn_utils.rnn.pack_padded_sequence(test_data_kmer, test_length_kmer, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_kmer = lstm_kmer(pack_kmer)  # 将pack数据放入到model模型中

        pack_GCN = nn_utils.rnn.pack_padded_sequence(test_data_GCN, test_length_GCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_GCN = lstm_GCN(pack_GCN)

        merged_feature = torch.cat((output_word2vec, output_DCN, output_kmer,output_GCN), dim=1)
        outputs = model(merged_feature)

        loss = criterion(outputs, test_label)  # 根据输出结果和原来的rna标签计算损失率
        loss_totall += loss.data.item()  # 将每一批次的损失率加起来
        iii += 1  # 共有多少批次
        _, pred_acc = torch.max(outputs.data, 1)  # pred_acc返回预测结果
        correct += (pred_acc == test_label).sum()  # 将pred_acc和correct相比，计算出预测正确的rna个数，之后将每一批次预测正确的相加
        total += test_label.size(0)  # 共有多少rna

    print('Accuracy of the test Data:{}%'.format(100 * correct / total))
    print('Loss of the test Data:{}%'.format(loss_totall / iii))
Train_Accuracy = []
Train_Loss = []
Test_Accuracy = []
Test_Loss = []
for j in range(100):  # 开始模型训练，一共训练100次
    i = 0
    lstm_word2vec.train()
    lstm_DCN.train()
    lstm_GCN.train()
    lstm_kmer.train()
    model.train()# 标志着模型开始训练，自动开始设置Dropout层和BN层
    for item_train_word2vec, item_train_DCN, item_train_kmer, item_train_GCN in zip(data_loader_word2vec, data_loader_DCN, data_loader_kmer, data_loader_GCN):
        i += 1
        train_data_word2vec, train_length_word2vec, train_label = item_train_word2vec  # 提取data_loader中每个批次的数据进行计算
        train_data_DCN, train_length_DCN, h = item_train_DCN
        train_data_kmer, train_length_kmer, h2 = item_train_kmer
        train_data_GCN, train_length_GCN, h3 = item_train_GCN
        num = train_label.shape[0]

        train_data_word2vec = train_data_word2vec.float()
        train_data_DCN = train_data_DCN.float()
        train_data_kmer = train_data_kmer.float()
        train_data_GCN = train_data_GCN.float()
        train_label = train_label.long()

        train_data_word2vec = Variable(train_data_word2vec)
        train_data_DCN = Variable(train_data_DCN)
        train_data_kmer = Variable(train_data_kmer)  # 将train_data里的数据变成Variable形式，用于反向传播
        train_data_GCN = Variable(train_data_GCN)
        train_label = Variable(train_label)  # 将train_label里的数据变成Variable形式，用于反向传播

        if torch.cuda.is_available():  # 判断gpu是否可用，可用的话在gpu上处理train_data和train_label数据
            train_data_word2vec = train_data_word2vec.cuda()
            train_data_DCN = train_data_DCN.cuda()
            train_data_kmer = train_data_kmer.cuda()
            train_data_GCN = train_data_GCN.cuda()
            train_label = train_label.cuda()

        pack_word2vec = nn_utils.rnn.pack_padded_sequence(train_data_word2vec, train_length_word2vec, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_word2vec = lstm_word2vec(pack_word2vec)  # 将pack数据放入到model模型中

        pack_DCN = nn_utils.rnn.pack_padded_sequence(train_data_DCN, train_length_DCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_DCN = lstm_DCN(pack_DCN)  # 将pack数据放入到model模型中

        pack_kmer = nn_utils.rnn.pack_padded_sequence(train_data_kmer, train_length_kmer, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_kmer = lstm_kmer(pack_kmer)  # 将pack数据放入到model模型中

        pack_GCN = nn_utils.rnn.pack_padded_sequence(train_data_GCN, train_length_GCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        output_GCN = lstm_GCN(pack_GCN)  # 将pack数据放入到model模型中

        merged_feature = torch.cat((output_word2vec, output_DCN, output_kmer, output_GCN), dim=1)
        outputs = model(merged_feature)

        _, pred_acc = torch.max(outputs.data, 1)  # pred_acc返回预测结果
        correct = (pred_acc == train_label).sum()  # 将pred_acc和correct相比，计算出预测正确的rna个数，之后将每一批次预测正确的相加
        loss = criterion(outputs, train_label)  # 根据输出结果和原来的rna标签计算损失率
        optimer_word2vec.zero_grad()   # 清空过往梯度
        optimer_DCN.zero_grad()
        optimer_kmer.zero_grad()
        optimer_GCN.zero_grad()
        optimer.zero_grad()
        loss.backward()  # 计算当前梯度，反向传播
        optimer_word2vec.step()
        optimer_DCN.step()
        optimer_kmer.step()
        optimer_GCN.step()
        optimer.step()# 模型更新
        if (i % 10 == 0 or i == 178):  # 每十批次显示一次该次模型训练的准确率和损失率
            print(('Epoch:[{}/{}], Step[{}/{}], loss:{:.4f}, Accuracy:{:.4f}'.format(j + 1, 100, i, 178, loss.data.item(), 100 * correct / num)))

    lstm_word2vec.eval()
    lstm_DCN.eval()
    lstm_kmer.eval()
    lstm_GCN.eval()
    model.eval()
    with torch.no_grad():  # 和上面一样，只是上面是显示该次训练每十个批次的详细信息，下面是显示该次训练的最终结果
        correct = 0  # 预测正确的个数
        total = 0  # 改批次下的RNA总数
        loss_totall = 0
        iii = 0
        for item_train_word2vec, item_train_DCN, item_train_kmer , item_train_GCN in zip(data_loader_word2vec, data_loader_DCN,
                                                                        data_loader_kmer, data_loader_GCN):
            train_data_word2vec, train_length_word2vec, train_label = item_train_word2vec  # 提取data_loader中每个批次的数据进行计算
            train_data_DCN, train_length_DCN, h = item_train_DCN
            train_data_kmer, train_length_kmer, h2 = item_train_kmer
            train_data_GCN,train_length_GCN,h3 = item_train_GCN
            num = train_label.shape[0]

            train_data_word2vec = train_data_word2vec.float()
            train_data_DCN = train_data_DCN.float()
            train_data_kmer = train_data_kmer.float()
            train_data_GCN = train_data_GCN.float()
            train_label = train_label.long()

            train_data_word2vec = Variable(train_data_word2vec)
            train_data_DCN = Variable(train_data_DCN)
            train_data_kmer = Variable(train_data_kmer)
            train_data_GCN = Variable(train_data_GCN)# 将train_data里的数据变成Variable形式，用于反向传播
            train_label = Variable(train_label)  # 将train_label里的数据变成Variable形式，用于反向传播

            if torch.cuda.is_available():  # 判断gpu是否可用，可用的话在gpu上处理train_data和train_label数据
                train_data_word2vec = train_data_word2vec.cuda()
                train_data_DCN = train_data_DCN.cuda()
                train_data_kmer = train_data_kmer.cuda()
                train_data_GCN = train_data_GCN.cuda()
                train_label = train_label.cuda()

            pack_word2vec = nn_utils.rnn.pack_padded_sequence(train_data_word2vec, train_length_word2vec,batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
            output_word2vec = lstm_word2vec(pack_word2vec)  # 将pack数据放入到model模型中

            pack_DCN = nn_utils.rnn.pack_padded_sequence(train_data_DCN, train_length_DCN,batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
            output_DCN = lstm_DCN(pack_DCN)  # 将pack数据放入到model模型中

            pack_kmer = nn_utils.rnn.pack_padded_sequence(train_data_kmer, train_length_kmer, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
            output_kmer = lstm_kmer(pack_kmer)  # 将pack数据放入到model模型中

            pack_GCN = nn_utils.rnn.pack_padded_sequence(train_data_GCN, train_length_GCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
            output_GCN = lstm_GCN(pack_GCN)

            merged_feature = torch.cat((output_word2vec, output_DCN, output_kmer, output_GCN), dim=1)
            outputs = model(merged_feature)

            loss = criterion(outputs, train_label)  # 根据输出结果和原来的rna标签计算损失率
            loss_totall += loss.data.item()  # 将每一批次的损失率加起来
            iii += 1  # 共有多少批次
            _, pred_acc = torch.max(outputs.data, 1)  # pred_acc返回预测结果
            correct += (pred_acc == train_label).sum()  # 将pred_acc和correct相比，计算出预测正确的rna个数，之后将每一批次预测正确的相加
            total += train_label.size(0)  # 共有多少rna
        print('Accuracy of the train Data:{}%'.format(100 * correct / total))
        print('Loss of the train Data:{}%'.format(loss_totall / iii))  # 没有经过训练的模型的准确率和损失率

    lstm_word2vec.eval()
    lstm_DCN.eval()
    lstm_kmer.eval()
    lstm_GCN.eval()
    model.eval()
    List_Data = []

    with torch.no_grad():  # 以下是为了保存模型
        correct = 0
        total = 0
        loss_totall = 0
        iii = 0
        for item_test_word2vec, item_test_DCN, item_test_kmer, item_test_GCN in zip(data_loader_test_word2vec, data_loader_test_DCN, data_loader_test_kmer, data_loader_test_GCN):
            test_data_word2vec, test_length_word2vec, test_label = item_test_word2vec  # 提取data_loader中每个批次的数据进行计算
            test_data_DCN, test_length_DCN, h = item_test_DCN
            test_data_kmer, test_length_kmer, h2 = item_test_kmer
            test_data_GCN, test_length_GCN, h3 = item_test_GCN
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

            pack_word2vec = nn_utils.rnn.pack_padded_sequence(test_data_word2vec, test_length_word2vec, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
            output_word2vec = lstm_word2vec(pack_word2vec)  # 将pack数据放入到model模型中

            pack_DCN = nn_utils.rnn.pack_padded_sequence(test_data_DCN, test_length_DCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
            output_DCN = lstm_DCN(pack_DCN)  # 将pack数据放入到model模型中

            pack_kmer = nn_utils.rnn.pack_padded_sequence(test_data_kmer, test_length_kmer, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
            output_kmer = lstm_kmer(pack_kmer)  # 将pack数据放入到model模型中

            pack_GCN = nn_utils.rnn.pack_padded_sequence(test_data_GCN, test_length_GCN, batch_first=True)  # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
            output_GCN = lstm_GCN(pack_GCN)  # 将pack数据放入到model模型中

            merged_feature = torch.cat((output_word2vec, output_DCN, output_kmer, output_GCN), dim=1)
            outputs = model(merged_feature)

            loss = criterion(outputs, test_label)  # 根据输出结果和原来的rna标签计算损失率
            loss_totall += loss.data.item()  # 将每一批次的损失率加起来
            iii += 1  # 共有多少批次
            _, pred_acc = torch.max(outputs.data, 1)  # pred_acc返回预测结果
            correct += (pred_acc == test_label).sum()  # 将pred_acc和correct相比，计算出预测正确的rna个数，之后将每一批次预测正确的相加
            total += test_label.size(0)  # 共有多少rna
        print('Accuracy of the test Data:{}%'.format(100 * correct / total))
        print('Loss of the test Data:{}%'.format(loss_totall / iii))
        if (100 * correct / total > max_acc):
            max_acc = 100 * correct / total
            torch.save(model, PATH_Model_DenseNet)
            torch.save(lstm_word2vec, PATH_Model_word2vec)
            torch.save(lstm_DCN, PATH_Model_DCN)
            torch.save(lstm_GCN, PATH_Model_GCN)
            torch.save(lstm_kmer, PATH_Model_kmer)
