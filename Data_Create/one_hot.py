import numpy as np
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
file_train = 'F:/桌面/ck 返稿修改/MFPred-master/nRC_Ten_Fold_Data/nRC_public/dataset_nRC_train_new.fasta'
file_test = "F:/桌面/ck 返稿修改/MFPred-master/nRC_Ten_Fold_Data/nRC_public/dataset_nRC_test_new.fasta"

List_A_Eight = [1, 0, 0, 0]
List_U_Eight = [0, 0, 1, 0]
List_G_Eight = [0, 1, 0, 0]
List_C_Eight = [0, 0, 0, 1]
List_N_Eight = [0, 0, 0, 0] #The coding rules

# List_A_Eight = [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0]
# List_U_Eight = [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1]
# List_G_Eight = [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]
# List_C_Eight = [0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0]
# List_N_Eight = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] #The coding rules
# List_A_Eight = [1, 0, 0, 0, 0, 0, 1, 0]
# List_U_Eight = [0, 0, 1, 0, 1, 0, 0, 0]
# List_G_Eight = [0, 1, 0, 0, 0, 0, 0, 1]
# List_C_Eight = [0, 0, 0, 1, 0, 1, 0, 0]
# List_N_Eight = [0, 0, 0, 0, 0, 0, 0, 0]


# def train_data_500():  # 将训练集的碱基放入
#     Train_Matrix = []
#     Train_label = []
#     for line in open(file_train):
#         if line[0] == ">":
#             if line.split("_")[1] == "5S-rRNA\n":
#                 Train_label.append(0)
#             if (line.split("_")[1] == '5.8S-rRNA\n'):
#                 Train_label.append(1)
#             if (line.split("_")[1] == 'tRNA\n'):
#                  Train_label.append(2)
#             if (line.split("_")[1] == 'Ribozyme\n'):
#                 Train_label.append(3)
#             if (line.split("_")[1] == 'CD-box\n'):
#                 Train_label.append(4)
#             if (line.split("_")[1] == 'miRNA\n'):
#                 Train_label.append(5)
#             if (line.split("_")[1] == 'Intron-gp-I\n'):
#                 Train_label.append(6)
#             if (line.split("_")[1] == 'Intron-gp-II\n'):
#                 Train_label.append(7)
#             if (line.split("_")[1] == 'HACA-box\n'):
#                 Train_label.append(8)
#             if (line.split("_")[1] == 'Riboswitch\n'):
#                 Train_label.append(9)
#             if (line.split("_")[1] == 'Y-RNA\n'):
#                 Train_label.append(10)
#             if (line.split("_")[1] == 'Leader\n'):
#                 Train_label.append(11)
#             if (line.split("_")[1] == 'Y-RNA-like\n'):
#                 Train_label.append(12)
#         else:
#             Tem_List = []
#             for i in range(200):
#                 if (i < len(line) - 1):
#                     if (line[i] == 'A' or line[i] == 'a'):
#                         Tem_List.append(List_A_Eight)
#                     elif (line[i] == 'T' or line[i] == 't'):
#                         Tem_List.append(List_U_Eight)
#                     elif (line[i] == 'C' or line[i] == 'c'):
#                         Tem_List.append(List_C_Eight)
#                     elif (line[i] == 'G' or line[i] == 'g'):
#                         Tem_List.append(List_G_Eight)
#                     else:
#                         Tem_List.append(List_N_Eight)
#                 else:
#                     Tem_List.append(List_N_Eight)
#             Tem_List = np.array(Tem_List)
#             Train_Matrix.append(Tem_List)
#     Train_Matrix = np.array(Train_Matrix)
#     Train_label = np.array(Train_label)
#     return Train_Matrix,Train_label
#
#
#
# def test_data_50():
#     Test_Matrix = []  # Save the matrix of ncRNAs
#     Test_label = []
#     for line in open(file_test):
#         if(line[0] == ">"):
#             if line.split("_")[1] == "5S-rRNA\n":
#                 Test_label.append(0)
#             if (line.split("_")[1] == '5.8S-rRNA\n'):
#                 Test_label.append(1)
#             if (line.split("_")[1] == 'tRNA\n'):
#                 Test_label.append(2)
#             if (line.split("_")[1] == 'Ribozyme\n'):
#                 Test_label.append(3)
#             if (line.split("_")[1] == 'CD-box\n'):
#                 Test_label.append(4)
#             if (line.split("_")[1] == 'miRNA\n'):
#                 Test_label.append(5)
#             if (line.split("_")[1] == 'Intron-gp-I\n'):
#                 Test_label.append(6)
#             if (line.split("_")[1] == 'Intron-gp-II\n'):
#                 Test_label.append(7)
#             if (line.split("_")[1] == 'HACA-box\n'):
#                 Test_label.append(8)
#             if (line.split("_")[1] == 'Riboswitch\n'):
#                 Test_label.append(9)
#             if (line.split("_")[1] == 'Y-RNA\n'):
#                 Test_label.append(10)
#             if (line.split("_")[1] == 'Leader\n'):
#                 Test_label.append(11)
#             if (line.split("_")[1] == 'Y-RNA-like\n'):
#                 Test_label.append(12)
#         else:
#             Tem_List = []
#             for i in range(200):
#                 if (i < len(line) - 1):
#                     if (line[i] == 'A' or line[i] == 'a'):
#                         Tem_List.append(List_A_Eight)
#                     elif (line[i] == 'T' or line[i] == 't'):
#                         Tem_List.append(List_U_Eight)
#                     elif (line[i] == 'C' or line[i] == 'c'):
#                         Tem_List.append(List_C_Eight)
#                     elif (line[i] == 'G' or line[i] == 'g'):
#                         Tem_List.append(List_G_Eight)
#                     else:
#                         Tem_List.append(List_N_Eight)
#                 else:
#                     Tem_List.append(List_N_Eight)
#             Tem_List = np.array(Tem_List)
#             Test_Matrix.append(Tem_List)
#     Test_Matrix = np.array(Test_Matrix)
#     Test_label = np.array(Test_label)
#     return Test_Matrix, Test_label
#
def train_data():
    Train_Matrix = []  # Save the matrix of ncRNAs in test
    Train_label = []  # Save the label of ncRNAs in test
    for line in open(file_train):
        if(line[0] == '>'):
            #print(line.split()[-1])
            if (line.strip().lstrip('>') == '5S_rRNA'):
                Train_label.append(0)
            if (line.strip().lstrip('>') == '5_8S_rRNA'):
                Train_label.append(1)
            if (line.strip().lstrip('>') == 'tRNA'):
                Train_label.append(2)
            if (line.strip().lstrip('>') == 'ribozyme'):
                Train_label.append(3)
            if (line.strip().lstrip('>') == 'CD-box'):
                Train_label.append(4)
            if (line.strip().lstrip('>') == 'miRNA'):
                Train_label.append(5)
            if (line.strip().lstrip('>') == 'Intron_gpI'):
                Train_label.append(6)
            if (line.strip().lstrip('>') == 'Intron_gpII'):
                Train_label.append(7)
            if (line.strip().lstrip('>') == 'HACA-box'):
                Train_label.append(8)
            if (line.strip().lstrip('>') == 'riboswitch'):
                Train_label.append(9)
            if (line.strip().lstrip('>') == 'IRES'):
                Train_label.append(10)
            if (line.strip().lstrip('>') == 'leader'):
                Train_label.append(11)
            if (line.strip().lstrip('>') == 'scaRNA'):
                Train_label.append(12)
        else:
            Tem_List = []
            for i in range(len(line[0: -1])):
                if (i < len(line) - 1):
                    if (line[i] == 'A' or line[i] == 'a'):
                        Tem_List.append(List_A_Eight)
                    elif (line[i] == 'T' or line[i] == 't'):
                        Tem_List.append(List_U_Eight)
                    elif (line[i] == 'C' or line[i] == 'c'):
                        Tem_List.append(List_C_Eight)
                    elif (line[i] == 'G' or line[i] == 'g'):
                        Tem_List.append(List_G_Eight)
                    else:
                        Tem_List.append(List_N_Eight)
                else:
                    Tem_List.append(List_N_Eight)
            Tem_List = np.array(Tem_List)
            Train_Matrix.append(Tem_List)

    Train_Matrix = np.array(Train_Matrix)
    Train_label = np.array(Train_label)
    return Train_Matrix, Train_label
def test_data():
    Test_Matrix = []  # Save the matrix of ncRNAs in test
    Test_label = []  # Save the label of ncRNAs in test
    for line in open(file_test):
        if(line[0] == '>'):
            #print(line.split()[-1])
            if (line.strip().lstrip('>') == '5S_rRNA'):
                Test_label.append(0)
            if (line.strip().lstrip('>') == '5_8S_rRNA'):
                Test_label.append(1)
            if (line.strip().lstrip('>') == 'tRNA'):
                Test_label.append(2)
            if (line.strip().lstrip('>') == 'ribozyme'):
                Test_label.append(3)
            if (line.strip().lstrip('>') == 'CD-box'):
                Test_label.append(4)
            if (line.strip().lstrip('>') == 'miRNA'):
                Test_label.append(5)
            if (line.strip().lstrip('>') == 'Intron_gpI'):
                Test_label.append(6)
            if (line.strip().lstrip('>') == 'Intron_gpII'):
                Test_label.append(7)
            if (line.strip().lstrip('>') == 'HACA-box'):
                Test_label.append(8)
            if (line.strip().lstrip('>') == 'riboswitch'):
                Test_label.append(9)
            if (line.strip().lstrip('>') == 'IRES'):
                Test_label.append(10)
            if (line.strip().lstrip('>') == 'leader'):
                Test_label.append(11)
            if (line.strip().lstrip('>') == 'scaRNA'):
                Test_label.append(12)
        else:
            Tem_List = []
            for i in range(len(line[0: -1])):

                if (i < len(line) - 1):
                    if (line[i] == 'A' or line[i] == 'a'):
                        Tem_List.append(List_A_Eight)
                    elif (line[i] == 'T' or line[i] == 't'):
                        Tem_List.append(List_U_Eight)
                    elif (line[i] == 'C' or line[i] == 'c'):
                        Tem_List.append(List_C_Eight)
                    elif (line[i] == 'G' or line[i] == 'g'):
                        Tem_List.append(List_G_Eight)
                    else:
                        Tem_List.append(List_N_Eight)
                else:
                    Tem_List.append(List_N_Eight)

            Tem_List = np.array(Tem_List )
            Test_Matrix.append(Tem_List)

    Test_Matrix = np.array(Test_Matrix)
    Test_label = np.array(Test_label)
    return Test_Matrix,Test_label
