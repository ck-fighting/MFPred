import numpy as np
from gensim.models import KeyedVectors
import gensim
from gensim.models import word2vec as wv
import itertools
from gensim.models import Word2Vec


train_path = "C:/Users/Administrator/Desktop/ncRNA_Family_Prediction/new_Ten_Fold_Data/train_0"
test_path = "C:/Users/Administrator/Desktop/ncRNA_Family_Prediction/new_Ten_Fold_Data/test_0"
all_path = "C:/Users/Administrator/Desktop/ncRNA_Family_Prediction/new_Ten_Fold_Data/All.fasta"
Gmodel_path = "C:/Users/Administrator/Desktop/ncRNA_Family_Prediction/Trained-Model/glove.txt"
Wmodel_path = "C:/Users/Administrator/Desktop/ncRNA_Family_Prediction/Trained-Model/glove.word2vec"

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
# model.wv.save_word2vec_format(Gmodel_path, binary=False)  # 将GloVe模型保存到文件中
model = KeyedVectors.load_word2vec_format(Gmodel_path, binary=False)
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
            embedded_sequence = np.array([model[kmer] for kmer in kmers])
            Train_Matrix.append(embedded_sequence)

    Train_Matrix = np.array(Train_Matrix)
    Train_label =np.array(Train_label)
    return Train_Matrix, Train_label
train_data()
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
            embedded_sequence = np.array([model[kmer] for kmer in kmers])
            Test_Matrix.append(embedded_sequence)
    Test_Matrix = np.array(Test_Matrix)
    Test_label =np.array(Test_label)

    return Test_Matrix, Test_label

