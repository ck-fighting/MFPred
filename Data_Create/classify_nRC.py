import pandas as pd
import numpy as np

path = "../nRC_Ten_Fold_Data/ALL_nRC.csv"
data = pd.read_csv(path)
data = data.values

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
IRES = []
Leader = []
scaRNA = []
for i in range(len(data)):
    if data[i][3] == "5S_rRNA":
        s_rRNA.append(data[i])
    if data[i][3] == "5_8S_rRNA":
        es_rRNA.append(data[i])
    if data[i][3] == "tRNA":
        tRNA.append(data[i])
    if data[i][3] == "ribozyme":
        Ribozyme.append(data[i])
    if data[i][3] == "CD-box":
        CD_box.append(data[i])
    if data[i][3] == "miRNA":
        miRNA.append(data[i])
    if data[i][3] == "Intron_gpI":
        Intron_gp_I.append(data[i])
    if data[i][3] == "Intron_gpII":
        Intron_gp_II.append(data[i])
    if data[i][3] == "HACA-box":
        HACA_box.append(data[i])
    if data[i][3] == "riboswitch":
        Riboswitch.append(data[i])
    if data[i][3] == "IRES":
        IRES.append(data[i])
    if data[i][3] == "leader":
        Leader.append(data[i])
    if data[i][3] == "scaRNA":
        scaRNA.append(data[i])

Pre_Data = pd.DataFrame(data = s_rRNA)
Pre_Data.to_csv('../RNA_classify/nRC/5s_rRNA.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = es_rRNA)
Pre_Data.to_csv('../RNA_classify/nRC/5.8s_rRNA.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = tRNA)
Pre_Data.to_csv('../RNA_classify/nRC/tRNA.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = Ribozyme)
Pre_Data.to_csv('../RNA_classify/nRC/Ribozyme.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = Riboswitch)
Pre_Data.to_csv('../RNA_classify/nRC/Riboswitch.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = CD_box)
Pre_Data.to_csv('../RNA_classify/nRC/CD_box.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = miRNA)
Pre_Data.to_csv('../RNA_classify/nRC/miRNA.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = Intron_gp_I)
Pre_Data.to_csv('../RNA_classify/nRC/Intron_gp_I.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = Intron_gp_II)
Pre_Data.to_csv('../RNA_classify/nRC/Intron_gp_II.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = HACA_box)
Pre_Data.to_csv('../RNA_classify/nRC/HACA_box.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = IRES)
Pre_Data.to_csv('../RNA_classify/nRC/IRES.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = Leader)
Pre_Data.to_csv('../RNA_classify/nRC/Leader.csv', encoding='gbk', index=False)

Pre_Data = pd.DataFrame(data = scaRNA)
Pre_Data.to_csv('../RNA_classify/nRC/scaRNA.csv', encoding='gbk', index=False)

from sklearn.model_selection import KFold

# 定义数据文件名和折数
data_filenames = ['5s_rRNA.csv', '5.8s_rRNA.csv', 'tRNA.csv', 'Ribozyme.csv', 'Riboswitch.csv', 'CD_box.csv',
                  'miRNA.csv', 'Intron_gp_I.csv', 'Intron_gp_II.csv', 'HACA_box.csv', 'IRES.csv',
                  'Leader.csv', 'scaRNA.csv']
import numpy as np
import pandas as pd

# 生成13个一维数组，每个数组包含10个元素
arrays = [s_rRNA,es_rRNA,tRNA,Ribozyme,Riboswitch,CD_box,miRNA,Intron_gp_I,Intron_gp_II,HACA_box,IRES,Leader,scaRNA]

# 将每个数组分成10等分，并进行拼接
for i in range(10):
    # 得到一个13 x 1的numpy数组
    concatenated_splits = np.concatenate([np.array_split(a, 10)[i] for a in arrays], axis=0)

    # 将结果转换为DataFrame
    df = pd.DataFrame(concatenated_splits)

    # 将结果存放为csv格式文件
    filename = f"result_{i + 1}.csv"
    df.to_csv(filename, index=False, header=False)