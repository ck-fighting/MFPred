# import networkx as nx
# import dgl
# import torch
# import torch.nn as nn
# import dgl.function as fn
# from dgl.nn.pytorch import GraphConv
#
# # 数据预处理
# def preprocess_sequences(sequences, k):
#     kmers = []
#     node_features = {}
#     edge_features = {}
#
#     # 构建图的节点和节点特征
#     node_id = 0
#     for seq in sequences:
#         for i in range(len(seq) - k + 1):
#             kmer = seq[i:i+k]
#             kmers.append(kmer)
#             if kmer in node_features:
#                 node_features[kmer] += 1
#             else:
#                 node_features[kmer] = 1
#             node_id += 1
#
#     # 构建图的边和边特征
#     edges = []
#     for i in range(len(kmers) - 1):
#         edge = (kmers[i], kmers[i+1])
#         if edge in edge_features:
#             edge_features[edge] += 1
#         else:
#             edge_features[edge] = 1
#         edges.append(edge)
#
#     return node_features, edges, edge_features
#
# # 构建图
# def build_graph(node_features, edges, edge_features):
#     graph = nx.Graph()
#
#     # 添加节点
#     for node, freq in node_features.items():
#         graph.add_node(node, freq=freq)
#
#     # 添加边
#     for edge, freq in edge_features.items():
#         graph.add_edge(edge[0], edge[1], freq=freq)
#
#     # 将NetworkX图转换为DGL图
#     dgl_graph = dgl.from_networkx(graph)
#
#     return dgl_graph
#
# # 图卷积神经网络模型
# class GCN(nn.Module):
#     def __init__(self, in_feats, hidden_size, out_feats):
#         super(GCN, self).__init__()
#         self.conv1 = GraphConv(in_feats, hidden_size)
#         self.conv2 = GraphConv(hidden_size, out_feats)
#
#     def forward(self, g, features):
#         h = features.float()
#         h = self.conv1(g, h)
#         h = torch.relu(h)
#         h = self.conv2(g, h)
#         return h
#
# # 数据准备
# sequences = ["GACTCTCGGCAACGGATATCTCGGCTCTCGCATCGATGAAGAACGTAGCGTAAATGTGAATTTCACTAGAAAGGACCTAGTAGTAAACTTTGAAAGTATGATGGGGAAATGTGTGATG", "AAATTTGCCTATACTCTTGGCTCCTGTCACCATGAAGAACATGGTAATATGCGATGCATGGTGTTAATTACAGAATCGTGTGAATCATCAAGCATGTGCATGCAATTTGTGCCCAAGGCTGTCAGGCTGGAAAACAGCCCTGTCTGGGCGCCA", "GACTCTCGGCAACAAATATCTCGGCTCTTGGATCAATGAAGAACGTAGCGAAATGCGATACTTGTGAGCACATGATTTTTGTTTTACGCGACAATCACTCCAAAAGAAATA"]
# k = 3
# node_features, edges, edge_features = preprocess_sequences(sequences, k)
#
# # 构建图
# graph = build_graph(node_features, edges, edge_features)
#
# # 创建GCN模型
# gcn_model = GCN(in_feats=1, hidden_size=16, out_feats=16)
#
# # 执行GCN消息传递和节点更新
# output = gcn_model(graph, torch.tensor(list(graph.nodes()))[:, None])
#
# # 打印最终结果
# print(output)
import networkx as nx
import dgl
import torch
import torch.nn as nn
import dgl.function as fn
from dgl.nn.pytorch import GraphConv
import itertools
import numpy as np

all_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/All_categories"
train_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/training-set-corrected.fasta"
test_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/test-set-corrected.fasta"
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
    Train_Matrix = []
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
            alphabet = ['A', 'C', 'G', 'T']
            kmers2 = [''.join(p) for p in itertools.product(alphabet, repeat=3)]
            Tem_List = []
            for kmer in kmers:
                for i in range(len(kmers2)):
                    if kmer == kmers2[i]:
                        Tem_List.append(output[i].detach().numpy())
            Tem_List = np.array(Tem_List)
            Train_Matrix.append(Tem_List)

    Train_Matrix = np.array(Train_Matrix)
    Train_label =np.array(Train_label)
    return Train_Matrix, Train_label

def test_data():
    Test_Matrix = []
    Test_label = []
    for line in open(test_path):
        if line[0] == ">":
            if line.split("_")[1] == "5S-rRNA\n":
                Test_label.append(0)
            if (line.split("_")[1] == '5.8S-rRNA\n'):
                Test_label.append(1)
            if (line.split("_")[1] == 'tRNA\n'):
                Test_label.append(2)
            if (line.split("_")[1] == 'Ribozyme\n'):
                Test_label.append(3)
            if (line.split("_")[1] == 'CD-box\n'):
                Test_label.append(4)
            if (line.split("_")[1] == 'miRNA\n'):
                Test_label.append(5)
            if (line.split("_")[1] == 'Intron-gp-I\n'):
                Test_label.append(6)
            if (line.split("_")[1] == 'Intron-gp-II\n'):
                Test_label.append(7)
            if (line.split("_")[1] == 'HACA-box\n'):
                Test_label.append(8)
            if (line.split("_")[1] == 'Riboswitch\n'):
                Test_label.append(9)
            if (line.split("_")[1] == 'Y-RNA\n'):
                Test_label.append(10)
            if (line.split("_")[1] == 'Leader\n'):
                Test_label.append(11)
            if (line.split("_")[1] == 'Y-RNA-like\n'):
                Test_label.append(12)
        else:
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
            Test_Matrix.append(Tem_List)
    Test_Matrix = np.array(Test_Matrix)
    Test_label = np.array(Test_label)
    return Test_Matrix, Test_label

a , b = test_data()
