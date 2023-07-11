import numpy as np
from gensim.models import KeyedVectors
from itertools import product
import gensim
from gensim.models import word2vec as wv
import itertools
from gensim.models import Word2Vec
train_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/training-set-corrected.fasta"
test_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/test-set-corrected.fasta"
# all_path = "C:/Users/Administrator/Desktop/ncRNA_Family_Prediction/new_Ten_Fold_Data/All.fasta"
Word_model_path = "F:/桌面/ck 返稿修改/MFPred-master/Trained_model/word2vec_new.txt"

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
model = KeyedVectors.load_word2vec_format(Word_model_path, binary=False)

def cal(c, cb, i):
    bases = {'A': [1, 1, 1], 'C': [0, 1, 0], 'G': [1, 0, 0, ], 'T': [0, 0, 1]}
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
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmer = sequence[i:i + k]
        kmers.append(kmer)
    bases = 'ATCG'
    combinations = [''.join(c) for c in product(bases, repeat=k)]
    combinations = np.array(combinations)
    encoding = np.zeros((len(kmers), len(combinations)))
    for i, kmer in enumerate(kmers):
        j = np.where(combinations == kmer)[0][0]
        encoding[i][j] = 1
    return encoding


def train_data():
    Train_Matrix_word2vec = []
    Train_Matrix2_DCN_NP = []
    Train_Matrix3_kmer = []
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

    Train_Matrix_word2vec = np.array(Train_Matrix_word2vec)
    Train_Matrix2_DCN_NP = np.array(Train_Matrix2_DCN_NP)
    Train_Matrix3_kmer = np.array(Train_Matrix3_kmer)
    Train_label = np.array(Train_label)
    return Train_Matrix_word2vec, Train_Matrix2_DCN_NP, Train_Matrix3_kmer, Train_label

def test_data():
    Test_Matrix_word2vec = []
    Test_Matrix2_DCN_NP = []
    Test_Matrix3_kmer = []
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

    Test_Matrix_word2vec = np.array(Test_Matrix_word2vec)
    Test_Matrix2_DCN_NP = np.array(Test_Matrix2_DCN_NP)
    Test_Matrix3_kmer = np.array(Test_Matrix3_kmer)
    Test_label = np.array(Test_label)

    return Test_Matrix_word2vec, Test_Matrix2_DCN_NP, Test_Matrix3_kmer, Test_label

