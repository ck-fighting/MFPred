
new_Ten_Flod_Data: store the NCY data set, ten-fold cross-validation contains ten test sets and ten training sets,includes dataset for setting Hamming distance
nRC_Ten_Fold_Data: store the nRC data set
Pre_Data_new: Deposit experimental data including MFPred in nRC and NCY datasets as well as ablation experiments, comparison experiments, noise addition experiments
Trained_model: Storing the trained model
Encoded_data:Deposit of ncRNAs sequences after model processing
Data_create: data processing, put the base information of each RNA sequence into Train_Matrix, put the classification information of each RNA sequence into Train_Label
Main_program: Main program is used for model training and data processing and displaying training results and saving the best training model
Model: Model BiGRU ResNet+SE
Preformance: The best model will be saved for the prediction of the final results due to the result prediction of the test set.
Code running environment
pytorch
networkx
dgl
gensim


