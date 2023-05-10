import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from itertools import product
import gensim
from gensim.models import word2vec as wv
import itertools
from gensim.models import Word2Vec
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

train_path = "F:/桌面/ncRNA_Family_Prediction/nRC_Ten_Fold_Data/train_5.xlsx"
test_path = "F:/桌面/ncRNA_Family_Prediction/nRC_Ten_Fold_Data/test_5.xlsx"
all_path = "F:/桌面/ncRNA_Family_Prediction/nRC_Ten_Fold_Data/ALL_nRC.xlsx"
Word_model_path = "F:/桌面/ncRNA_Family_Prediction/Trained_Model/word2vec_nRC.txt"
# def getRNASequence_3kmer():
#     data = pd.read_excel(all_path)
#     data = data.values
#     RNA_Sequence = []
#     for i in range(len(data)):
#         RNA_Sequence.append(data[i][2])
#     RNA_Sequence =[x.strip() for x in RNA_Sequence]
#     k = 3  # k-mer的长度
#     alphabet = ['A', 'C', 'G', 'T']
#     kmers = [''.join(p) for p in itertools.product(alphabet, repeat=k)]
#     sentences = []
#     for sequence in RNA_Sequence:
#         sentence = []
#         for i in range(len(sequence) - k + 1):
#             kmer = sequence[i:i + k]
#             if kmer in kmers:
#                 sentence.append(kmer)
#         sentences.append(sentence)
#     return sentences
#
# model = Word2Vec(getRNASequence_3kmer(), sg=0, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
# model.wv.save_word2vec_format(Word_model_path, binary=False)  # 将GloVe模型保存到文件中
model = KeyedVectors.load_word2vec_format(Word_model_path, binary=False)

def cal(c, cb, i):
    bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0], 'T': [0, 0, 1]}
    p = bases[c]
    p.append(np.round(cb / float(i + 1), 2))
    return p

def calculate(s):
    f=[]
    cba = cbc = cbt = cbg = 0
    for i, c in enumerate(s):
        if c == 'A':
            cba += 1
            p = cal(c, cba, i)
        elif c == 'T':
            cbt += 1
            p = cal(c, cbt, i)
        elif c == 'C':
            cbc += 1
            p = cal(c, cbc, i)
        elif c == 'G':
            cbg += 1
            p = cal(c, cbg, i)
        else:
            p = [0, 0, 0, 0]
        f.append(p)
    f = np.array(f)
    return f

def kmer_encode(sequence, k):
    base_list = []
    kmers = []
    for i in range(len(sequence) - k):
        kmer = sequence[i:i + k]
        kmers.append(kmer)
    for i in kmers:
        binary_dict = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
        binary_list = []
        for nucleotide in i:
            binary_list += binary_dict[nucleotide]
        base_list.append(binary_list)
    base_list = np.array(base_list)
    return base_list

class GCN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, output_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def build_graph(seq):
    kmer_list = [''.join(kmer) for kmer in itertools.product('ACGT', repeat=3)]
    kmer_dict = {kmer: i for i, kmer in enumerate(kmer_list)}
    kmer_list = np.array(kmer_list)
    edge_index = []
    kmers = []
    for i in range(len(seq) - 3 + 1):
        kmer = seq[i:i + 3]
        kmers.append(kmer)
    kmers_code = []
    for kmer in kmers:
        kmers_code.append(kmer_dict[kmer])
    num_nodes = len(kmers_code)
    for i in range(num_nodes-1):
        if kmers_code[i] < num_nodes and kmers_code[i+1] < num_nodes:
            edge_index.append([kmers_code[i], kmers_code[i+1]])
        if kmers_code[num_nodes-i-1] < num_nodes and kmers_code[num_nodes-i-2] < num_nodes:
            edge_index.append([kmers_code[num_nodes-i-1] ,kmers_code[num_nodes-i-2]])
    edge_index = torch.tensor(edge_index).t().contiguous()
    x = torch.zeros(num_nodes, 64)
    for i, kmer in enumerate(kmers):
        j = np.where(kmer_list == kmer)[0][0]
        x[i][j] = 1
    data = Data(x=x, edge_index=edge_index)

    return data

def train_data():
    data = pd.read_excel(train_path)
    data1 = data.values
    Train_Matrix_word2vec = []
    Train_Matrix2_DCN_NP = []
    Train_Matrix3_kmer = []
    Train_Matrix4_GCN = []
    Train_label = []
    for i in range(len(data1)):
        if data1[i][3] == "5S_rRNA":
            Train_label.append(0)
        if data1[i][3] == "5_8S_rRNA":
            Train_label.append(1)
        if data1[i][3] == "tRNA":
            Train_label.append(2)
        if data1[i][3] == "ribozyme":
            Train_label.append(3)
        if data1[i][3] == "CD-box":
            Train_label.append(4)
        if data1[i][3] == "miRNA":
            Train_label.append(5)
        if data1[i][3] == "Intron_gpI":
            Train_label.append(6)
        if data1[i][3] == "Intron_gpII":
            Train_label.append(7)
        if data1[i][3] == "HACA-box":
            Train_label.append(8)
        if data1[i][3] == "riboswitch":
            Train_label.append(9)
        if data1[i][3] == "IRES":
            Train_label.append(10)
        if data1[i][3] == "leader":
            Train_label.append(11)
        if data1[i][3] == "scaRNA":
            Train_label.append(12)
        line = data1[i][2].strip()
        allowed_chars = set("AGCT")
        line = "".join(c for c in line if c in allowed_chars)
        kmers = [line[i:i + 3] for i in range(0, len(line) - 2)]
        # 训练Word2vec模型
        embedded_sequence = np.array([model[kmer] for kmer in kmers])
        Train_Matrix_word2vec.append(embedded_sequence)

        Tem_List = calculate(line[0:len(line)])
        Train_Matrix2_DCN_NP.append(Tem_List)

        Tem_List3 = kmer_encode(line, 3)
        Train_Matrix3_kmer.append(Tem_List3)

        rna_seq = line.strip()
        data = build_graph(rna_seq)
        model1 = GCN(input_channels=len(data.x[0]), hidden_channels=32, output_channels=16)
        out = model1(data.x, data.edge_index)
        Tem_List = out.detach().numpy()
        Train_Matrix4_GCN.append(Tem_List)

    Train_Matrix_word2vec = np.array(Train_Matrix_word2vec)
    Train_Matrix2_DCN_NP = np.array(Train_Matrix2_DCN_NP)
    Train_Matrix3_kmer = np.array(Train_Matrix3_kmer)
    Train_Matrix4_GCN = np.array(Train_Matrix4_GCN)
    Train_label = np.array(Train_label)
    return Train_Matrix_word2vec, Train_Matrix2_DCN_NP, Train_Matrix3_kmer, Train_Matrix4_GCN, Train_label

def test_data():
    data = pd.read_excel(test_path)
    data1 = data.values
    Test_Matrix_word2vec = []
    Test_Matrix2_DCN_NP = []
    Test_Matrix3_kmer = []
    Test_Matrix4_GCN = []
    Test_label = []
    for i in range(len(data)):
        if data1[i][3] == "5S_rRNA":
            Test_label.append(0)
        if data1[i][3] == "5_8S_rRNA":
            Test_label.append(1)
        if data1[i][3] == "tRNA":
            Test_label.append(2)
        if data1[i][3] == "ribozyme":
            Test_label.append(3)
        if data1[i][3] == "CD-box":
            Test_label.append(4)
        if data1[i][3] == "miRNA":
            Test_label.append(5)
        if data1[i][3] == "Intron_gpI":
            Test_label.append(6)
        if data1[i][3] == "Intron_gpII":
            Test_label.append(7)
        if data1[i][3] == "HACA-box":
            Test_label.append(8)
        if data1[i][3] == "riboswitch":
            Test_label.append(9)
        if data1[i][3] == "IRES":
            Test_label.append(10)
        if data1[i][3] == "leader":
            Test_label.append(11)
        if data1[i][3] == "scaRNA":
            Test_label.append(12)
        line = data1[i][2].strip()
        allowed_chars = set("AGCT")
        line = "".join(c for c in line if c in allowed_chars)
        kmers = [line[i:i + 3] for i in range(0, len(line) - 2)]
        # 训练Word2vec模型
        embedded_sequence = np.array([model[kmer] for kmer in kmers])
        Test_Matrix_word2vec.append(embedded_sequence)

        Tem_List = calculate(line[0:len(line)])
        Test_Matrix2_DCN_NP.append(Tem_List)

        Tem_List3 = kmer_encode(line, 3)
        Test_Matrix3_kmer.append(Tem_List3)

        rna_seq = line.strip()
        data = build_graph(rna_seq)
        model1 = GCN(input_channels=len(data.x[0]), hidden_channels=32, output_channels=16)
        out = model1(data.x, data.edge_index)
        Tem_List = out.detach().numpy()
        Test_Matrix4_GCN.append(Tem_List)

    Test_Matrix_word2vec = np.array(Test_Matrix_word2vec)
    Test_Matrix2_DCN_NP = np.array(Test_Matrix2_DCN_NP)
    Test_Matrix3_kmer = np.array(Test_Matrix3_kmer)
    Test_Matrix4_GCN = np.array(Test_Matrix4_GCN)
    Test_label = np.array(Test_label)
    return Test_Matrix_word2vec, Test_Matrix2_DCN_NP, Test_Matrix3_kmer, Test_Matrix4_GCN, Test_label

