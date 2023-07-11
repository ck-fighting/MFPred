import numpy as np
from itertools import product

train_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/training-set-corrected.fasta"
test_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/test-set-corrected.fasta"
def kmer_encode(sequence, k):
    kmers = []
    for i in range(len(sequence) - k):
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
            Tem_List = kmer_encode(line, 3)
            Train_Matrix.append(Tem_List)


    Train_Matrix = np.array(Train_Matrix)
    Train_label =np.array(Train_label)

    return Train_Matrix,Train_label


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
            Tem_List = kmer_encode(line, 3)
            Test_Matrix.append(Tem_List)


    Test_Matrix = np.array(Test_Matrix)
    Test_label =np.array(Test_label)

    return Test_Matrix,Test_label





