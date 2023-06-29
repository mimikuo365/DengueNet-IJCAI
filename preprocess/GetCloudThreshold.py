import os
from pathlib import Path
import pandas as pd
import sys
import argparse
import tensorflow as tf
from datetime import datetime

root_dir = Path(__file__).parents[2]
sys.path.insert(0, str(root_dir / "code/util"))
sys.path.insert(0, str(root_dir / "code/model"))

from DataPreprocess import *


def main():
    for city in getCityList():
        path = Path(f"/mnt/usb/jupyter-mimikuo/dataset/sorted/{city}")
        train_ratio = 0.8

        threshold_path = path / "threshold_11.pickle"
        print(threshold_path)
        if os.path.exists(threshold_path):
            with open(threshold_path, "rb") as f:
                threshold_dic = pickle.load(f)
                print(threshold_dic.values())
        else:
            df = pd.read_csv(path / "label.csv")
            train_img_ls, _ = readImgs(df, path)
            threshold_dic = getThresholds(train_img_ls)
            with open(threshold_path, "wb") as f:
                pickle.dump(threshold_dic, f)

        for band_index, band_name in zip([0, 1, 2], ["1", "2", "3"]):
            threshold_path = path / f"threshold_{band_name}.pickle"
            print(threshold_path)
            if os.path.exists(threshold_path):
                with open(threshold_path, "rb") as f:
                    threshold_dic = pickle.load(f)
                    print(threshold_dic.values())
            else:
                df = pd.read_csv(path / "label.csv")
                img_ls = np.load(path / "rgb.npy")
                train_img_ls = img_ls[: int(train_ratio * len(df)), band_index]
                threshold_dic = getThresholds(train_img_ls)
                with open(threshold_path, "wb") as f:
                    pickle.dump(threshold_dic, f)


if __name__ == "__main__":
    main()
