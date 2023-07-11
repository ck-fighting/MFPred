import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
from itertools import product
import gensim
from gensim.models import word2vec as wv
import itertools
from gensim.models import Word2Vec
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import itertools
all_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/All_categories"
train_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/old_ten_fold/train_1"
test_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/new_ten_fold/test1.xlsx"
all_path2 = "F:/桌面/ncRNA_Family_Prediction/new_Ten_Fold_Data/All.fasta"
Word_model_path = "F:/桌面/hao/Trained_Model/word2vec_new.txt"

# def getRNASequence_3kmer():
#     RNA_Sequence = []
#     for line in open(all_path):
#         if line[0] != ">":
#             RNA_Sequence.append(line)
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

# 数据预处理
def preprocess_sequences(sequences, k):
    kmers = []
    node_features = {}
    edge_features = {}

    # 构建图的节点和节点特征
    node_id = 0
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmers.append(kmer)
            if kmer in node_features:
                node_features[kmer] += 1
            else:
                node_features[kmer] = 1
            node_id += 1

    # 构建图的边和边特征
    edges = []
    for i in range(len(kmers) - 1):
        edge = (kmers[i], kmers[i+1])
        if edge in edge_features:
            edge_features[edge] += 1
        else:
            edge_features[edge] = 1
        edges.append(edge)

    return node_features, edges, edge_features

# 构建图
def build_graph(node_features, edge_features):
    graph = nx.Graph()

    # 添加节点
    for node, freq in node_features.items():
        graph.add_node(node, freq=freq)

    # 添加边
    for edge, freq in edge_features.items():
        graph.add_edge(edge[0], edge[1], freq=freq)

    # 将NetworkX图转换为DGL图
    dgl_graph = dgl.from_networkx(graph)

    return dgl_graph

# 图卷积神经网络模型
class GCN(nn.Module):
    def __init__(self, in_feats, hidden_size, out_feats):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(in_feats, hidden_size)
        self.conv2 = GraphConv(hidden_size, out_feats)

    def forward(self, g, features):
        h = features.float()
        h = self.conv1(g, h)
        h = torch.relu(h)
        h = self.conv2(g, h)
        return h

def getRNASequence():
    RNA_Sequence = []
    for line in open(all_path):
        if line[0] != ">":
            RNA_Sequence.append(line)
    RNA_Sequence =[x.strip() for x in RNA_Sequence]

    return RNA_Sequence
sequences = getRNASequence()

k = 3
node_features, edges, edge_features = preprocess_sequences(sequences, k)

# 构建图
graph = build_graph(node_features, edge_features)

# 创建GCN模型
gcn_model = GCN(in_feats=1, hidden_size=16, out_feats=32)

# 执行GCN消息传递和节点更新
output = gcn_model(graph, torch.tensor(list(graph.nodes()))[:, None])

k = 3  # k-mer的长度
alphabet = ['A', 'C', 'G', 'T']
kmers = [''.join(p) for p in itertools.product(alphabet, repeat=k)]

def train_data():
    Train_Matrix_word2vec = []
    Train_Matrix2_NCP_NP = []
    Train_Matrix3_kmer = []
    Train_Matrix4_GCN = []
    Train_label = []
    for line in open(train_path):
        if line[0] == ">":
            if line.split("_")[1] == "5S-rRNA\n":
                Train_label.append(0)
            if (line.split("_")[1] == '5.8S-rRNA\n'):
                Train_label.append(1)
            if (line.split("_")[1] == 'tRNA\n'):
                Train_label.append(2)
            if (line.split("_")[1] == 'Ribozyme\n'):
                Train_label.append(3)
            if (line.split("_")[1] == 'CD-box\n'):
                Train_label.append(4)
            if (line.split("_")[1] == 'miRNA\n'):
                Train_label.append(5)
            if (line.split("_")[1] == 'Intron-gp-I\n'):
                Train_label.append(6)
            if (line.split("_")[1] == 'Intron-gp-II\n'):
                Train_label.append(7)
            if (line.split("_")[1] == 'HACA-box\n'):
                Train_label.append(8)
            if (line.split("_")[1] == 'Riboswitch\n'):
                Train_label.append(9)
            if (line.split("_")[1] == 'Y-RNA\n'):
                Train_label.append(10)
            if (line.split("_")[1] == 'Leader\n'):
                Train_label.append(11)
            if (line.split("_")[1] == 'Y-RNA-like\n'):
                Train_label.append(12)
        else:
            line = line.strip()
            kmers = [line[i:i + 3] for i in range(0, len(line) - 2)]
            # 训练Word2vec模型
            embedded_sequence = np.array([model[kmer] for kmer in kmers])
            Train_Matrix_word2vec.append(embedded_sequence)

            Tem_List = calculate(line[0:len(line)-1])
            Train_Matrix2_NCP_NP.append(Tem_List)

            Tem_List3 = kmer_encode(line, 3)
            Train_Matrix3_kmer.append(Tem_List3)

            line = line.strip()
            kmers = [line[i:i + 3] for i in range(0, len(line) - 2)]
            # 训练Word2vec模型
            alphabet = ['A', 'C', 'G', 'T']
            kmers2 = [''.join(p) for p in itertools.product(alphabet, repeat=3)]
            Tem_List = []
            for kmer in kmers:
                for i in range(len(kmers2)):
                    if kmer == kmers2[i]:
                        Tem_List.append(output[i].detach().numpy())
            Tem_List = np.array(Tem_List)
            Train_Matrix4_GCN.append(Tem_List)


    Train_Matrix_word2vec = np.array(Train_Matrix_word2vec)
    Train_Matrix2_NCP_NP = np.array(Train_Matrix2_NCP_NP)
    Train_Matrix3_kmer = np.array(Train_Matrix3_kmer)
    Train_Matrix4_GCN = np.array(Train_Matrix4_GCN)
    Train_label = np.array(Train_label)
    return Train_Matrix_word2vec, Train_Matrix2_NCP_NP, Train_Matrix3_kmer, Train_Matrix4_GCN, Train_label
Train_Matrix_word2vec, Train_Matrix2_NCP_NP, Train_Matrix3_kmer, Train_Matrix4_GCN, Train_label = train_data()
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/word2vec_train.npy",Train_Matrix_word2vec)
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/NCP_ND_train.npy",Train_Matrix2_NCP_NP)
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/kmer_train.npy",Train_Matrix3_kmer)
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/GCN_train.npy",Train_Matrix4_GCN)
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/train_label.npy",Train_label)
def test_data():
    data = pd.read_excel(test_path)
    data1 = data.values
    Test_Matrix_word2vec = []
    Test_Matrix2_NCP_ND = []
    Test_Matrix3_kmer = []
    Test_Matrix4_GCN = []
    Test_label = []
    for i in range(len(data)):
        Test_label.append(data1[i][2])
        line = data1[i][1].strip()
        allowed_chars = set("AGCT")
        line = "".join(c for c in line if c in allowed_chars)
        kmers = [line[i:i + 3] for i in range(0, len(line) - 2)]
        # 训练Word2vec模型
        embedded_sequence = np.array([model[kmer] for kmer in kmers])
        Test_Matrix_word2vec.append(embedded_sequence)

        Tem_List = calculate(line[0:len(line)])
        Test_Matrix2_NCP_ND.append(Tem_List)

        Tem_List3 = kmer_encode(line, 3)
        Test_Matrix3_kmer.append(Tem_List3)

        line = line.strip()
        kmers = [line[i:i + 3] for i in range(0, len(line) - 2)]
        # 训练Word2vec模型
        alphabet = ['A', 'C', 'G', 'T']
        kmers2 = [''.join(p) for p in itertools.product(alphabet, repeat=3)]
        Tem_List = []
        for kmer in kmers:
            for i in range(len(kmers2)):
                if kmer == kmers2[i]:
                    Tem_List.append(output[i].detach().numpy())
        Tem_List = np.array(Tem_List)
        Test_Matrix4_GCN.append(Tem_List)

    Test_Matrix_word2vec = np.array(Test_Matrix_word2vec)
    Test_Matrix2_NCP_NP = np.array(Test_Matrix2_NCP_ND)
    Test_Matrix3_kmer = np.array(Test_Matrix3_kmer)
    Test_Matrix4_GCN = np.array(Test_Matrix4_GCN)
    Test_label = np.array(Test_label)
    return Test_Matrix_word2vec, Test_Matrix2_NCP_NP, Test_Matrix3_kmer, Test_Matrix4_GCN, Test_label
Test_Matrix_word2vec, Test_Matrix2_NCP_NP, Test_Matrix3_kmer, Test_Matrix4_GCN, Test_label= test_data()
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/word2vec_test.npy",Test_Matrix_word2vec)
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/NCP_ND_test.npy",Test_Matrix2_NCP_NP)
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/kmer_test.npy",Test_Matrix3_kmer)
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/GCN_test.npy",Test_Matrix4_GCN)
np.save("F:/桌面/ck 返稿修改/MFPred-master/Encoded_data/NCY/fold1/test_label.npy",Test_label)