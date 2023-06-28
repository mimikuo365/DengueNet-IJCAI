import numpy as np
import skimage
from pathlib import Path
from torch.utils.data import Dataset
import pandas as pd
from pathlib import Path
import pickle
import sys

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / 'code/util'))
sys.path.insert(0, str(root_dir / 'code/model'))

from DengueNet import *
from DataPreprocess import *
from ModelUtil import getModelList, getInput, createTuner, getCallback, createModel, getResultLocation
from ForFeatureEngineering import getFeatureList, getThresholdPairList
from CalculateMetric import *

class DengueDataset(Dataset):
    def __init__(self, annotations_file, img_dir, feature_dir, transform=None, target_transform=None):
        self.labels = pd.read_csv(annotations_file)
        self.img_dir = Path(img_dir)
        self.feature_dir = Path(feature_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.img_dir / (str(self.labels.iloc[idx, 0]) + '.tiff')
        image = skimage.io.imread(img_path)
        feature_path = self.feature_dir / (str(self.labels.iloc[idx, 0]) + '.pickle')
        with open(feature_path, 'rb') as handle:
            f = pickle.load(handle)
            feature = np.array(f.values())
            print(feature)

        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, feature, label

def createDataFrame(path):
    df = pd.read_csv(path / "label.csv")
    features_ls = []
    new_df = None
    new_df_cols = ["epiweek"]
    for i, row in df.iterrows():
        epiweek = str(row["epiweek"])
        print(epiweek)
        feature_path = path / "feature" / epiweek + ".pickle"
        with open(feature_path, 'rb') as handle:
            feature_dic = pickle.load(handle)
        if i == 0:
            features_ls = list(feature_dic[0].keys())
            for tile in 256:
                for feature in features_ls:
                    new_df_cols.append(f"{tile}_{feature}")
            new_df = pd.DataFrame(columns=new_df_cols)
        
        tmp_df = pd.DataFrame(columns=new_df_cols)
        for tile in feature_dic.keys():
            for feature in features_ls:
                tmp_df.at[0, f"{tile}_{feature}"] = feature
        new_df = pd.concat(new_df, tmp_df)
        print(new_df)
        
def main():
    for city in getCityList():
        createDataFrame(Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}/"))
    
if __name__ == "__main__":
    main()