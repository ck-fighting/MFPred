import numpy as np
from gensim.models import KeyedVectors
from itertools import product
import gensim
from gensim.models import word2vec as wv
import itertools
from gensim.models import Word2Vec
import networkx as nx
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

train_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/training-set-corrected.fasta"
test_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/test-set-corrected.fasta"
all_path = "F:/桌面/ncRNA_Family_Prediction/new_Ten_Fold_Data/All.fasta"
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

# class GCN(torch.nn.Module):
#     def __init__(self, in_features, hidden_features, out_features):
#         super(GCN, self).__init__()
#         self.gc1 = torch.nn.Linear(in_features, hidden_features)
#         self.gc2 = torch.nn.Linear(hidden_features, out_features)
#     def forward(self, adj, x):
#         x = F.relu(self.gc1(torch.matmul(adj, x)))
#         x = self.gc2(torch.matmul(adj, x))
#         x = nn.functional.normalize(x, p=2, dim=1)
#         return x
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
    Train_Matrix_word2vec = []
    Train_Matrix2_DCN_NP = []
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
            Train_Matrix2_DCN_NP.append(Tem_List)

            Tem_List3 = kmer_encode(line, 3)
            Train_Matrix3_kmer.append(Tem_List3)

            # rna_seq = line
            # seq_len = len(rna_seq)
            # graph = nx.Graph()
            #
            # for i in range(seq_len):
            #     graph.add_node(i, feature=np.array([1 if rna_seq[i] == 'A' else 0,
            #                                         1 if rna_seq[i] == 'C' else 0,
            #                                         1 if rna_seq[i] == 'G' else 0,
            #                                         1 if rna_seq[i] == 'T' else 0]))
            #
            # for i in range(seq_len):
            #     for j in range(i + 1, seq_len):
            #         if (rna_seq[i] == 'A' and rna_seq[j] == 'T') or (rna_seq[i] == 'T' and rna_seq[j] == 'A'):
            #             graph.add_edge(i, j)
            #         elif (rna_seq[i] == 'C' and rna_seq[j] == 'G') or (rna_seq[i] == 'G' and rna_seq[j] == 'C'):
            #             graph.add_edge(i, j)
            #
            # # 构建RNA序列的邻接矩阵和特征矩阵
            # adj_matrix = nx.to_numpy_array(graph)
            # feature_matrix = np.array([node[1]['feature'] for node in graph.nodes(data=True)])
            # adj_matrix = torch.FloatTensor(adj_matrix)
            # feature_matrix = torch.FloatTensor(feature_matrix)
            #
            # gcn = GCN(in_features=4, hidden_features=16, out_features=8)
            # output = gcn(adj_matrix, feature_matrix)
            # output = output.detach().numpy()
            # Train_Matrix4_GCN.append(output)
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
    Test_Matrix_word2vec = []
    Test_Matrix2_DCN_NP = []
    Test_Matrix3_kmer = []
    Test_Matrix4_GCN = []
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
            embedded_sequence = np.array([model[kmer] for kmer in kmers])
            Test_Matrix_word2vec.append(embedded_sequence)

            Tem_List = calculate(line[0:len(line) - 1])
            Test_Matrix2_DCN_NP.append(Tem_List)

            Tem_List3 = kmer_encode(line, 3)
            Test_Matrix3_kmer.append(Tem_List3)

            # rna_seq = line
            # seq_len = len(rna_seq)
            # graph = nx.Graph()
            #
            # for i in range(seq_len):
            #     graph.add_node(i, feature=np.array([1 if rna_seq[i] == 'A' else 0,
            #                                         1 if rna_seq[i] == 'C' else 0,
            #                                         1 if rna_seq[i] == 'G' else 0,
            #                                         1 if rna_seq[i] == 'T' else 0]))
            #
            # for i in range(seq_len):
            #     for j in range(i + 1, seq_len):
            #         if (rna_seq[i] == 'A' and rna_seq[j] == 'T') or (rna_seq[i] == 'T' and rna_seq[j] == 'A'):
            #             graph.add_edge(i, j)
            #         elif (rna_seq[i] == 'C' and rna_seq[j] == 'G') or (rna_seq[i] == 'G' and rna_seq[j] == 'C'):
            #             graph.add_edge(i, j)
            #
            # # 构建RNA序列的邻接矩阵和特征矩阵
            # adj_matrix = nx.to_numpy_array(graph)
            # feature_matrix = np.array([node[1]['feature'] for node in graph.nodes(data=True)])
            # adj_matrix = torch.FloatTensor(adj_matrix)
            # feature_matrix = torch.FloatTensor(feature_matrix)
            #
            # gcn = GCN(in_features=4, hidden_features=16, out_features=8)
            # output = gcn(adj_matrix, feature_matrix)
            # output = output.detach().numpy()
            # Test_Matrix4_GCN.append(output)
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

