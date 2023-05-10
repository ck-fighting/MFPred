import numpy as np
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
file = "/Ten_Fold_Data/All.fasta"

def train_data_500():  # 将训练集的碱基放入
    Train_Matrix = []
    Train_label = []
    Tem_List = []
    Label_List = []
    for line in open(file):
        if line[0] == ">":
            Label_List.append(line)
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
            Tem_List.append(line)
    return Tem_List,Train_label,Label_List
a,b,c = train_data_500()
num = 0
s_rRNA = []
es_rRNA = []
tRNA = []
Ribozyme = []
CD_box = []
miRNA = []
Intron_gp_I = []
Intron_gp_II = []
HACA_box = []
Riboswitch = []
Y_RNA = []
Leader = []
Y_RNA_like = []

for i in b:
    if i == 0:
        s_rRNA.append(num)
        num = num + 1
    if i == 1:
        es_rRNA.append(num)
        num = num + 1
    if i == 2:
        tRNA.append(num)
        num = num + 1
    if i == 3:
        Ribozyme.append(num)
        num = num + 1
    if i == 4:
        CD_box.append(num)
        num = num + 1
    if i == 5:
        miRNA.append(num)
        num = num + 1
    if i == 6:
        Intron_gp_I.append(num)
        num = num + 1
    if i == 7:
        Intron_gp_II.append(num)
        num = num + 1
    if i == 8:
        HACA_box.append(num)
        num = num + 1
    if i == 9:
        Riboswitch.append(num)
        num = num + 1
    if i == 10:
        Y_RNA.append(num)
        num = num + 1
    if i == 11:
        Leader.append(num)
        num = num + 1
    if i == 12:
        Y_RNA_like.append(num)
        num = num + 1

s_rRNA1 = []
es_rRNA1 = []
tRNA1 = []
Ribozyme1 = []
CD_box1 = []
miRNA1 = []
Intron_gp_I1 = []
Intron_gp_II1 = []
HACA_box1 = []
Riboswitch1 = []
Y_RNA1 = []
Leader1 = []
Y_RNA_like1 = []

for i in s_rRNA:
    s_rRNA1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = s_rRNA1)
Pre_Data.to_csv('RNA_classify/5s_rRNA.csv', encoding='gbk', index=False)

for i in es_rRNA:
    es_rRNA1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = es_rRNA1)
Pre_Data.to_csv('RNA_classify/5.8s_rRNA.csv', encoding='gbk', index=False)

for i in tRNA:
    tRNA1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = tRNA1)
Pre_Data.to_csv('RNA_classify/tRNA.csv', encoding='gbk', index=False)

for i in Ribozyme:
    Ribozyme1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = Ribozyme1)
Pre_Data.to_csv('RNA_classify/Ribozyme.csv', encoding='gbk', index=False)

for i in Riboswitch:
    Riboswitch1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = Riboswitch1)
Pre_Data.to_csv('RNA_classify/Riboswitch.csv', encoding='gbk', index=False)

for i in CD_box:
    CD_box1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = CD_box1)
Pre_Data.to_csv('RNA_classify/CD_box.csv', encoding='gbk', index=False)

for i in miRNA:
    miRNA1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = miRNA1)
Pre_Data.to_csv('RNA_classify/miRNA.csv', encoding='gbk', index=False)

for i in Intron_gp_I:
    Intron_gp_I1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = Intron_gp_I1)
Pre_Data.to_csv('RNA_classify/Intron_gp_I.csv', encoding='gbk', index=False)

for i in Intron_gp_II:
    Intron_gp_II1.append(c[i]+a[i])

Pre_Data = pd.DataFrame(data = Intron_gp_II1)
Pre_Data.to_csv('RNA_classify/Intron_gp_II.csv', encoding='gbk', index=False)

for i in HACA_box:
    HACA_box1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = HACA_box1)
Pre_Data.to_csv('RNA_classify/HACA_box.csv', encoding='gbk', index=False)

for i in Y_RNA:
    Y_RNA1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = Y_RNA1)
Pre_Data.to_csv('RNA_classify/Y_RNA.csv', encoding='gbk', index=False)

for i in Leader:
    Leader1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = Leader1)
Pre_Data.to_csv('RNA_classify/Leader.csv', encoding='gbk', index=False)

for i in Y_RNA_like:
    Y_RNA_like1.append(c[i]+a[i])
Pre_Data = pd.DataFrame(data = Y_RNA_like1)
Pre_Data.to_csv('RNA_classify/Y_RNA_like.csv', encoding='gbk', index=False)

























