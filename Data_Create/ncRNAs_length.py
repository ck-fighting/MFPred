import pandas as pd
import numpy as np

train_path = "../nRC_Ten_Fold_Data/ALL_nRC.xlsx"



def train_data():
    data = pd.read_excel(train_path)
    data1 = data.values
    List = []
    Tem_List = []
    count1b,count2b,count3b,count4b,count5b,count6b,count7b,count8b,count9b,count10b,count11b,count12b,count13b = 0,0,0,0,0,0,0,0,0,0,0,0,0
    count1s, count2s, count3s, count4s, count5s, count6s, count7s, count8s, count9s, count10s, count11s, count12s, count13s = 0,0,0,0,0,0,0,0,0,0,0,0,0

    for i in range(len(data1)):
        if data1[i][3] == "5S_rRNA":
            if len(data1[i][2].strip())>200:
                count1b = count1b + 1
            else:
                count1s = count1s + 1
        if data1[i][3] == "5_8S_rRNA":
            if len(data1[i][2].strip()) > 200:
                count2b = count2b + 1
            else:
                count2s = count2s + 1
        if data1[i][3] == "tRNA":
            if len(data1[i][2].strip()) > 200:
                count3b = count3b + 1
            else:
                count3s = count3s + 1
        if data1[i][3] == "ribozyme":
            if len(data1[i][2].strip()) > 200:
                count4b = count4b + 1
            else:
                count4s = count4s + 1
        if data1[i][3] == "CD-box":
            if len(data1[i][2].strip()) > 200:
                count5b = count5b + 1
            else:
                count5s = count5s + 1
        if data1[i][3] == "miRNA":
            if len(data1[i][2].strip()) > 200:
                count6b = count6b + 1
            else:
                count6s = count6s + 1
        if data1[i][3] == "Intron_gpI":
            if len(data1[i][2].strip()) > 200:
                count7b = count7b + 1
            else:
                count7s = count7s + 1
        if data1[i][3] == "Intron_gpII":
            if len(data1[i][2].strip()) > 200:
                count8b = count8b + 1
            else:
                count8s = count8s + 1
        if data1[i][3] == "HACA-box":
            if len(data1[i][2].strip()) > 200:
                count9b = count9b + 1
            else:
                count9s = count9s + 1
        if data1[i][3] == "riboswitch":
            if len(data1[i][2].strip()) > 200:
                count10b = count10b + 1
            else:
                count10s = count10s + 1
        if data1[i][3] == "IRES":
            if len(data1[i][2].strip()) > 200:
                count11b = count11b + 1
            else:
                count11s = count11s + 1
        if data1[i][3] == "leader":
            if len(data1[i][2].strip()) > 200:
                count12b = count12b + 1
            else:
                count12s = count12s + 1
        if data1[i][3] == "scaRNA":
            if len(data1[i][2].strip()) > 200:
                count13b = count13b + 1
            else:
                count13s = count13s + 1
    # Tem_List.append(count3b/700)
    # Tem_List.append(count3s/700)
    # Tem_List.append(count4b/700)
    # Tem_List.append(count4s/700)
    # Tem_List.append(count5b/700)
    # Tem_List.append(count5s/700)
    Tem_List.append(count6b/700)
    Tem_List.append(count6s/700)
    Tem_List.append(count7b/700)
    Tem_List.append(count7s/700)
    Tem_List.append(count8b/700)
    Tem_List.append(count8s/700)
    Tem_List.append(count9b/700)
    Tem_List.append(count9s/700)
    # Tem_List.append(count10b/700)
    # Tem_List.append(count10s/700)
    # Tem_List.append(count11b/700)
    # Tem_List.append(count11s/700)
    # Tem_List.append(count12b/700)
    # Tem_List.append(count12s/700)
    # Tem_List.append(count13b/700)
    # Tem_List.append(count13s/700)
    print(Tem_List)







train_data()


