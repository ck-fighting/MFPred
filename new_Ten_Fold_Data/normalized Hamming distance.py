import pandas as pd

def normalized_hamming_distance(seq1, seq2):
    distance = sum(c1 != c2 for c1, c2 in zip(seq1, seq2))
    if len(seq1) > len(seq2):
        normalized_hamming_distance = distance / len(seq2)
    else:
        normalized_hamming_distance = distance / len(seq1)
    return normalized_hamming_distance

filtered_X_train = []
filtered_X_test = []
filtered_X_train_label = []
filtered_X_test_label = []
train_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/old_ten_fold/train_3"
test_path = "F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/old_ten_fold/test_3"
def train_data():
    Train_label = []
    Train_sequence = []
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
            Train_sequence.append(line)
    return  Train_sequence, Train_label

def test_data():
    Test_label = []
    Test_sequence = []
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
            Test_sequence.append(line)
    return  Test_sequence, Test_label


X_train, X_train_label = train_data()
X_test, X_test_label = test_data()
count = 0
for seq_test in X_test:
    count = count + 1
    is_valid = True
    for seq_train in X_train:
        distance = normalized_hamming_distance(seq_train, seq_test)
        if distance < 0.3:
            is_valid = False
            break
    if is_valid:
        filtered_X_test.append(seq_test)
        filtered_X_test_label.append(X_test_label[count-1])

List_Data = []
for i in range(len(filtered_X_test)):
    List_TEM = []
    List_TEM.append(filtered_X_test[i])
    List_TEM.append(filtered_X_test_label[i])
    List_Data.append(List_TEM)

name = ['Sequence', 'Label']

Pre_Data = pd.DataFrame(columns=name, data = List_Data)

Pre_Data.to_csv('F:/桌面/ck 返稿修改/MFPred-master/new_Ten_Fold_Data/new_ten_fold/test3.csv', encoding='gbk')

