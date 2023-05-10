import numpy as np

train_path = "/Users/chenkai/Desktop/ncPREDICT/代码/ncRNA_Family_Prediction/new_Ten_Fold_Data/train_0"
test_path = "/Users/chenkai/Desktop/ncPREDICT/代码/ncRNA_Family_Prediction/new_Ten_Fold_Data/test_0"

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
            Tem_List = calculate(line[0:len(line)-1])
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
            Tem_List = calculate(line[0:len(line)-1])
            Test_Matrix.append(Tem_List)

    Test_Matrix = np.array(Test_Matrix)
    Test_label =np.array(Test_label)

    return Test_Matrix,Test_label

