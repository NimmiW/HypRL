from sklearn.model_selection import KFold # import KFold
import numpy as np
import pandas as pd
import os



DATASET_PATH = "metadata/all_datasets"
files = os.listdir(DATASET_PATH)

for idx,file in enumerate(files):
    print(file)
    index = 0

    file_ = os.path.join(DATASET_PATH,file)

    data = pd.read_csv(file_, header=None, delimiter=' ')
    print(data.head())
    data = data.to_numpy()
    print(len(data))
    kf = KFold(n_splits=5)  # Define the split - into 2 folds
    kf.get_n_splits(data)  # returns the number of splitting iterations in the cross-validator


    for train_index, test_index in kf.split(data):
     print("------------------------------")
     print("Train", len(data[train_index]))
     df = pd.DataFrame(data[train_index])
     df.to_csv("metadata/nn-meta/split-"+str(index)+"/train/"+file, sep=" ")
     print("Test", len(data[test_index]))
     df = pd.DataFrame(data[test_index])
     df.to_csv("metadata/nn-meta/split-" + str(index) + "/test/"+file, sep=" ")
     index += 1
